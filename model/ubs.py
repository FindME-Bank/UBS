import math
import time
import copy

from scipy.integrate import quad
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from model.base_model.basic_net import MLP
from model.base_model.cnf import build_cnf
from model.base_model.container import FraudContainer
from model.base_model.ode import ODEnet


def quad_func(t, c, w):
    """This is the t * f(t) function calculating the mean time to next event,
    given c, w."""
    return c * t * np.exp(-w * t + (c / w) * (np.exp(-w * t) - 1))


class UBS(FraudContainer):
    def __init__(self, type_num, query_feature_dim, support_feature_dim,  # input
                 type_embed_dim=32, cnf_hidden_dim=32, flow_layer_num=3, flow_layer_type="ignore",
                 flow_nonlinearity="softplus", time_length=1.0,  # etm
                 influence_embed_dim=32, time_normalize=10000, ode_layer_num=2, ode_layer_type="ignore",
                 ode_nonlinearity="sigmoid",  # ode-rnn
                 query_embedding_dim=32,
                 predictor_hidden_dim=32,  # predictor
                 sample_num=100,  # tpm
                 max_seq_len=60, beta=-1, distance_hidden_dim=32,  # hdt
                 etm=True, tpm=True, hdt=True,  # ablation
                 lr=1e-3, weight_decay=0, comment="-",
                 **kwargs):
        """
        Args:
            type_num: number of event type
            query_feature_dim: feature dimension of the query loan event
            support_feature_dim: feature dimension of the support event
            %
            type_embed_dim: embedding dimension of event type
            cnf_hidden_dim: hidden dimension of continuous normalizing flow
            flow_layer_num: layer number of continuous normalizing flow, in which the last layer dimension equals type_embed_dim
            flow_layer_type: layer type of of continuous normalizing flow
            flow_nonlinearity: activation function of odenet in continuous normalizing
            %
            distance_hidden_dim: hidden dimension of the distance network in metric learning
            predictor_hidden_dim: hidden dimension of default prediction network
            %
            influence_embed_dim:  embedding dimension of historical cumulative influence
            ode_layer_num: layer number of the ode network for calculating cumulative influence
            ode_layer_type: layer type of each ode layer for calculating cumulative influence
            ode_nonlinearity: activation function of the ode network for calculating cumulative influence
            %
            sample_num: MCMC integral sample number in time prediciton model
            %
            max_seq_len: maximum historical sequence length for adaptive margin calculation in metric learning
            beta: margin in metric learning. Default: -1 refer to adaptive margin calculation
            %
            etm: boolean indicator for ablation study UBS-w/o-ETM, default True
            tpm: boolean indicator for ablation study UBS-w/o-TPM, default True
            hdt: boolean indicator for ablation study UBS-w/o-HDT, default True
            %
            lr: learning rate
            weight_decay: decay rate
        """
        super(UBS, self).__init__(lr=lr, weight_decay=weight_decay, comment=comment)
        self.beta = beta
        self.type_num = type_num
        self.query_feature_dim = query_feature_dim
        self.support_feature_dim = support_feature_dim
        self.type_embed_dim = type_embed_dim
        self.time_normalize = time_normalize
        self.sample_num = sample_num
        self.max_seq_len = max_seq_len
        self.influence_embed_dim = influence_embed_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.distance_hidden_dim = distance_hidden_dim
        self.query_embedding_dim = query_embedding_dim


        # ablation
        self.etm = etm
        self.tpm = tpm
        self.hdt = hdt

        # event_type_distribution_model
        self.encoders = nn.ModuleList([IreregularCNN(in_dim=support_feature_dim, pool_dim=30, hidden_dim=type_embed_dim)
                                       for _ in range(type_num)])
        self.mean_linears = nn.ModuleList([nn.Linear(type_embed_dim, type_embed_dim) for _ in range(type_num)])
        self.std_linears = nn.ModuleList([nn.Linear(type_embed_dim, type_embed_dim) for _ in range(type_num)])
        self.cnfs = nn.ModuleList([build_cnf(input_dim=type_embed_dim, layer_type=flow_layer_type,
                                             hidden_dims=[cnf_hidden_dim] * (flow_layer_num - 1) + [type_embed_dim],
                                             nonlinearity=flow_nonlinearity, time_length=time_length)
                                   for _ in range(type_num)])

        # event_type_attention
        self.type_atten_linear = nn.ModuleList(
            [nn.Linear(2 * type_embed_dim, type_embed_dim) for _ in range(type_num)])

        # cumulated_historical_influence_module
        if self.etm:
            self.rnn = nn.GRUCell(1 + support_feature_dim + type_embed_dim, influence_embed_dim)
        else:
            self.rnn = nn.GRUCell(1 + support_feature_dim, influence_embed_dim)
        self.ode_func = ODEnet(in_dim=influence_embed_dim, hidden_dims=[influence_embed_dim] * ode_layer_num,
                               out_dim=influence_embed_dim, layer_type=ode_layer_type, nonlinearity=ode_nonlinearity)

        # time_prediction_module
        self.v = nn.Parameter(torch.rand(influence_embed_dim, 1))
        self.w = nn.Parameter(torch.rand(1))  # beta
        self.b = nn.Parameter(torch.rand(1))  # base_intensity, lambda_0

        # query_embedding_module
        self.query_embedding_net = MLP(query_feature_dim, [query_embedding_dim])

        # default_event_prediction_module
        self.predictor = MLP(query_embedding_dim + influence_embed_dim + 1, [predictor_hidden_dim, 1], activation="relu", last_activation=None)

        self.distance = MLP(2 * (query_embedding_dim + influence_embed_dim + 1), [distance_hidden_dim, 1], activation="relu",
                            last_activation=None)

        self.first_validation = True
        self.is_weighted_loss = True

    def forward(self, batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb,
                behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, is_test, gt=None):
        """
        Variable:
            batch_size: int
            support_emb_after_group: list of tensor, [batch_size * type_num][event_num, dims]
            support_type_after_group: list of tensor, [batch_size][type_num]
            behavior_seq_emb: llist of tensor, [historical_behavior_num, embedding_dim]
            behavior_seq_type: list of tensor, [batch_size][historical_behavior_num]
            behavior_seq_time: list of tensor, [batch_size][historical_behavior_num]
            behavior_seq_len: list, [batch_size]
            query_time: torch.tensor, [batch_size]
            settle_time: torch.tensor, [batch_size]
            is_test: bool
            gt: ground_truth. It will not be passed during the test phase.
        """
        if self.etm:
            event_type_distribution, mus, logvars, delta_logps = self.model_event_type_distribution(
                support_emb_after_group, support_type_after_group)  # [batch_size][type_num, type_embed_dim]
            event_type_representations = self.event_type_attention(event_type_distribution, support_type_after_group,
                                                                   batch_size)  # [batch_size][type_num, type_embed_dim]
        else:
            mus, logvars, delta_logps, event_type_representations = None, None, None, None
        behavior_seq_representations = self.get_behavior_seq_representations(batch_size, behavior_seq_emb,
                                                                             behavior_seq_type, behavior_seq_time,
                                                                             event_type_representations)
        cumulated_influence, h = self.model_cumulated_influence(batch_size, behavior_seq_representations,
                                                             behavior_seq_time, behavior_seq_len, query_time)
        f, loglike_tp = self.predict_time(h, query_time, settle_time, is_test)

        query_emb = self.query_embedding_net(query_emb)
        default_prob = self.predict_default(query_emb, h, f)

        if is_test:
            return default_prob
        else:
            if self.hdt:
                distance = self.calculate_distance(query_emb, h, f)
                metric_pairs, beta = self.hard_discriminated_sample(distance, behavior_seq_len, gt)
            else:
                metric_pairs, distance, beta = None, None, None
            return default_prob, mus, logvars, delta_logps, loglike_tp, \
                   metric_pairs, distance, beta

    def calculate_distance(self, query_emb, h, f):
        h = torch.cat([query_emb, h, f.unsqueeze(-1)], dim=-1)
        input = torch.cat([h.repeat_interleave(h.shape[0], dim=0), h.repeat((h.shape[0], 1))], dim=-1)  # [b*b,2*h]
        distance = self.distance(input).squeeze(-1).reshape(h.shape[0], h.shape[0])
        distance = torch.sigmoid(distance)
        return distance

    def predict_default(self, query_emb, h, f):
        if self.tpm:
            h = torch.cat([query_emb, h, f.unsqueeze(-1)], dim=-1)
            default_prob = self.predictor(h).squeeze(-1)
        else:
            h = torch.cat([query_emb, h], dim=-1)
            default_prob = self.predictor(h).squeeze(-1)
        return torch.sigmoid(default_prob)

    def predict_time(self, h, query_time, settle_time, is_test):
        """
        Args:
            h: tensor, [b, h]
            query_time: tensor, [b]
            settle_time: tensor, [b]
            is_test: Bool
        """
        h = h.matmul(self.v).squeeze(-1)  # [b]
        beta = - F.softplus(self.w)

        repayment_willingness = torch.zeros(h.shape[0]).to(self.device)
        f = torch.zeros(h.shape[0]).to(self.device)

        samples_t = [torch.randint(query_time[i] + 1, settle_time[i] + 1, [self.sample_num]) for i in range(h.shape[0])]
        samples_t = torch.stack(samples_t, dim=0).to(self.device)  # [b,sample_num]

        for i in range(self.sample_num):
            delta_t = samples_t[:, i] - query_time

            log_lambda_ = h + beta * delta_t + self.b
            lambda_ = torch.exp(torch.min(log_lambda_, torch.ones_like(h) * 2.))
            f += lambda_ / torch.exp((1.0 / beta) * torch.exp(
                torch.min(torch.ones_like(h) * 10., h + self.b) + lambda_ / beta))
            repayment_willingness += lambda_
        return repayment_willingness, f

    def quad_worker(self, h, query_time):
        """
        Args:
            h: [b,h]
            query_time: [b]
        Return:
            preds_t: list
        """
        c = (torch.exp(h.matmul(self.v).squeeze(-1) + self.b).reshape(-1)).cpu().detach().numpy()  # [b]
        for i, c_ in enumerate(c):
            args = (c_, self.w.cpu().detach().numpy())
            upbound = np.inf
            val, _err = quad(quad_func, 0, upbound, args=args)
            query_time[i] += int(val)

        return query_time

    def model_cumulated_influence(self, batch_size, behavior_seq, behavior_seq_time, behavior_seq_len, query_time):
        """
        Args:
            batch_size: int
            behavior_seq: list of tensor, [batch_size][historical_behavior_num???type_embed_dim]
            behavior_seq_time: list of tensor, [batch_size][historical_behavior_num]
            behavior_seq_len: list, [batch_size]
            query_time: tensor, [batch_size]
        Returns:
            cumulated_historical_influence: tensor, [batch_size, hidden_dim]
        """
        # 1. ?????????????????????irregular??????????????????concat??????????????????????????????????????????????????????????????????
        max_seq_len = max(behavior_seq_len)
        aligned_behavior_seq = torch.zeros([batch_size, max_seq_len, behavior_seq[0].shape[-1]]).to(self.device)
        for i, seq in enumerate(behavior_seq):
            aligned_behavior_seq[i, :behavior_seq_len[i]] = seq
            aligned_behavior_seq[i, behavior_seq_len[i]:] = seq[-1]

        # 2. ??????behavior_seq?????????cumulated_influence
        rnn_h = torch.zeros(batch_size, max_seq_len, self.influence_embed_dim).to(self.device)
        ode_h = torch.zeros(batch_size, max_seq_len, self.influence_embed_dim).to(self.device)
        for s in range(max_seq_len):
            if s == 0:
                rnn_h[:, s] = self.rnn(aligned_behavior_seq[:, s])  # [batch_size, hidden_dim]
            else:
                rnn_h[:, s] = self.rnn(aligned_behavior_seq[:, s], rnn_h[:, s-1].data)  # [batch_size, hidden_dim]

            if s != max_seq_len - 1:
                target_time = aligned_behavior_seq[:, s, 0] / self.time_normalize
                unique_target_time = torch.cat([torch.tensor([0.]).to(self.device), target_time],
                                               dim=0).unique().sort().values
                ode_out = odeint(func=self.ode_func, y0=rnn_h[:, s], t=unique_target_time,
                                 adjoint_options={"norm": "seminorm"})

                target_time = target_time.detach().cpu().numpy()
                unique_target_time = unique_target_time.detach().cpu().numpy()
                time_idx = torch.tensor((unique_target_time[:, None] == target_time).argmax(axis=0)).to(self.device)
                time_idx = time_idx.unsqueeze(0).unsqueeze(-1).expand(1, batch_size, self.influence_embed_dim)
                ode_h[:, s] = ode_out.gather(0, time_idx).squeeze(0)  # [b,h]

        # 3. ???????????????????????????query????????????cumulated_influence
        select_index = torch.tensor(behavior_seq_len).unsqueeze(-1).\
            unsqueeze(-1).expand(batch_size, 1, self.influence_embed_dim).to( self.device)
        h = rnn_h.gather(1,  select_index-1).squeeze(1)
        last_event_time = torch.cat([x[-1:] for x in behavior_seq_time], dim=0) / self.time_normalize
        query_time = query_time / self.time_normalize
        delta_time = query_time - last_event_time
        unique_delta_time = torch.cat([torch.tensor([0.]).to(self.device), delta_time],
                                      dim=0).unique().sort().values

        ode_out = odeint(func=self.ode_func, y0=h, t=unique_delta_time,
                         adjoint_options={"norm": "seminorm"})  # [t,b,h]

        delta_time = delta_time.cpu().numpy()
        unique_delta_time = unique_delta_time.cpu().numpy()
        time_idx = torch.tensor((unique_delta_time[:, None] == delta_time).argmax(axis=0)).to(self.device)
        time_idx = time_idx.unsqueeze(0).unsqueeze(-1).expand(1, batch_size, self.influence_embed_dim)
        cumulated_influence = ode_out.gather(0, time_idx).squeeze(0)  # [b,h]

        return cumulated_influence, h

    def get_behavior_seq_representations(self, batch_size, behavior_seq_emb, behavior_seq_type, behavior_seq_time,
                                         event_type_representations):
        behavior_seq_representations = []
        for i in range(batch_size):
            time_representation = behavior_seq_time[i][1:] - behavior_seq_time[i][:-1]
            time_representation = torch.cat([torch.zeros([1]).to(time_representation.device), time_representation])
            time_representation = time_representation.unsqueeze(-1)

            if event_type_representations is not None:
                type_representation = event_type_representations[i][behavior_seq_type[i]]
                representation = torch.cat([time_representation, behavior_seq_emb[i], type_representation], dim=-1)
            else:
                representation = torch.cat([time_representation, behavior_seq_emb[i]], dim=-1)
            behavior_seq_representations.append(representation)
        return behavior_seq_representations  # [batch_size][historical_behavior_num???1+support_feature_dim+type_embed_dim]

    def event_type_attention(self, event_type_distribution, support_type_after_group, batch_size):
        """
        Args:
            event_type_distribution: list of tensor, [batch_size][type_num, hidden_dim]
            support_type_after_group: list of tensor, [batch_size][type_num]
        Returns:
            behavior_seq_representations: list of tensor, [batch_size][type_num, hidden_dim]
        """
        behavior_seq_representations = torch.zeros([batch_size, self.type_num, self.type_embed_dim]).to(self.device)
        for i in range(batch_size):
            distributions, types = event_type_distribution[i], support_type_after_group[i]
            l = len(types)
            concat_distributions = torch.cat([distributions.repeat_interleave(l, dim=0), distributions.repeat((l, 1))],
                                             dim=-1)
            concat_distributions = concat_distributions.chunk(l, 0)

            for j, t in enumerate(types):
                unnorm_weights = torch.tanh(self.type_atten_linear[t](
                    concat_distributions[j]))  # [type_num, hidden_dim * 2] -> [type_num, hidden_dim]
                weights = torch.softmax(unnorm_weights, dim=0)  # [type_num, hidden_dim]
                behavior_seq_representations[i][j] = torch.sum(weights * distributions, dim=0)
        return behavior_seq_representations

    def model_event_type_distribution(self, support_emb_after_group, support_type_after_group):
        """
        Args:
            support_emb_after_group: list of tensor, [batch_size * type_num][event_num, dims]
            support_type_after_group: list of tensor, [batch_size][type_num]
        Returns:
            event_type_distribution: list of tensor, [batch_size][type_num, type_embed_dim]
        """
        all_type = torch.cat(support_type_after_group, dim=0)
        event_type_distribution = torch.zeros([len(all_type), self.type_embed_dim]).to(all_type.device)

        mus = []
        logvars = []
        delta_logps = []

        for i in range(self.type_num):
            if (all_type == i).sum() != 0:
                # encoder
                index = (all_type == i).nonzero().squeeze(-1)
                support_emb = [support_emb_after_group[i] for i in index]  # [one_type_num][event_num, dims]

                hidden = self.encoders[i](support_emb)  # [one_type_num, hidden_dim]
                mu = self.mean_linears[i](hidden)
                logvar = self.std_linears[i](hidden)
                z = self.reparameterize(mu, logvar)

                zero = torch.zeros(z.shape[0], 1).to(z)
                event_type_distribution[index], delta_logp = self.cnfs[i](z, zero)  # [b,h], [b, 1]

                mus.append(mu)
                logvars.append(logvar)
                delta_logps.append(delta_logp.squeeze(-1))

        mus = torch.cat(mus, dim=0)
        logvars = torch.cat(logvars, dim=0)
        delta_logps = torch.cat(delta_logps, dim=0)

        # ??????batch_size * type
        type_len = [len(x) for x in support_type_after_group]
        event_type_distribution = event_type_distribution.split(type_len, 0)
        return event_type_distribution, mus, logvars, delta_logps

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps
        return z

    def hard_discriminated_sample(self, distance, seq_len, gt):
        positive_points = (gt == 1).nonzero().squeeze(-1)
        negetive_points = (gt == 0).nonzero().squeeze(-1)

        seq_len = torch.tensor(seq_len).to(self.device)
        if self.beta == -1:
            beta = 1 - seq_len / self.max_seq_len  # [b]
        else:
            beta = torch.ones(distance.shape[0]).to(self.device) * self.beta

        positive_points_num = len(positive_points)
        positive_paris = torch.stack([positive_points.repeat_interleave(positive_points_num, dim=0),
                                      positive_points.repeat(positive_points_num)], dim=1)  # [b*b,2]
        positive_paris = positive_paris[positive_paris[:, 0] != positive_paris[:, 1]]  # [b*(b-1),2]

        metric_pairs = []
        for i, j in positive_paris:
            k_index = negetive_points[(distance[i, negetive_points] < distance[i, j] + beta[i])]
            l_index = negetive_points[(distance[j, negetive_points] < distance[i, j] + beta[j])]
            metric_pairs.append((i, j, k_index, l_index))

        return metric_pairs, beta

    def calculate_hard_discriminated_loss(self, metric_pairs, distance, beta):
        J = 0
        for i, j, k_index, l_index in metric_pairs:
            J_overline = 0
            if len(k_index) != 0:
                J_overline += (beta[i] - distance[i, k_index]).exp().mean()
            if len(l_index) != 0:
                J_overline += (beta[j] - distance[j, l_index]).exp().mean()
            if J_overline != 0:
                J_overline = J_overline.log()
            J_overline = J_overline + distance[i, j]
            J += max(0, J_overline.pow(2))
        J = J / len(metric_pairs)
        return J

    @staticmethod
    def weighted_loss(loss):
        if loss > 10:
            return loss / math.pow(10, math.floor(math.log10(abs(loss.item()) + 1e-9)))
        else:
            return loss

    def calculate_loss(self, mus, logvars, delta_logps, like_tp, metric_pairs, distance, beta, default_prob, gt, query_time, settle_time):
        # ??????MLP???loss
        bce = F.binary_cross_entropy(default_prob, gt.to(default_prob.dtype))
        bactch_size = len(gt)

        loss = bce
        if self.etm:
            loss_atae = torch.sum(
                (- 0.5 * (1 + logvars - mus.pow(2) - logvars.exp()).sum(-1)) + delta_logps) / bactch_size
            loss = loss + self.weighted_loss(loss_atae) if self.is_weighted_loss else loss + loss_atae

        if self.tpm:
            loss_tp = torch.mean(- (1 - gt) * torch.log(like_tp) - gt * torch.log(
                torch.max(settle_time - query_time - like_tp, torch.ones_like(gt) * 1e-9)))
            loss = loss + self.weighted_loss(loss_tp) if self.is_weighted_loss else loss + loss_tp

        if self.hdt:
            J = self.calculate_hard_discriminated_loss(metric_pairs, distance, beta)
            loss = loss + self.weighted_loss(J) if self.is_weighted_loss else loss + J

        return loss

    def formulate_for_metric(self, p_prob):
        p = ((p_prob >= 0.5) * 1).to(torch.int)
        return p

    def training_step(self, data, batch_idx, optimizer_idx=None):
        self.first_validation = False

        data = self.data_merge(data)
        batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, \
        behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt = \
            self.data_prepare(data, is_test=False)

        default_prob, mus, logvars, delta_logps, like_tp, metric_pairs, distance, beta = \
            self.forward(batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb,
                         behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time,
                         is_test=False, gt=gt)

        loss = self.calculate_loss(mus, logvars, delta_logps, like_tp, metric_pairs, distance, beta, default_prob,
                                   gt, query_time, settle_time)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(gt))
        return loss

    def validation_step(self, data, batch_idx):
        if self.first_validation:
            gt = data[1]["target_event_fraud"]
            default_prob = torch.rand(gt.shape)

        else:
            batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, \
            behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt = \
                self.data_prepare(data, is_test=True)
            default_prob = self.forward(batch_size, query_emb, support_emb_after_group, support_type_after_group,
                                        behavior_seq_emb, behavior_seq_type, behavior_seq_time, behavior_seq_len,
                                        query_time, settle_time, is_test=True)
        p = self.formulate_for_metric(default_prob)
        self.update_metric("val", gt, p, p_prob=default_prob)

    def on_test_epoch_start(self) -> None:
        super(UBS, self).on_test_epoch_start()
        self.on_validation_epoch_start()

    def test_step(self, data, batch_idx):
        batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, \
        behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt = \
            self.data_prepare(data, is_test=True)
        default_prob = self.forward(batch_size, query_emb, support_emb_after_group, support_type_after_group,
                                    behavior_seq_emb, behavior_seq_type, behavior_seq_time, behavior_seq_len,
                                    query_time, settle_time, is_test=True)
        p = self.formulate_for_metric(default_prob)
        self.update_metric("test", gt, p, p_prob=default_prob)

    def data_merge(self, data):
        data, data_fraud = data.values()
        for k in data[0].keys():
            if type(data[0][k]) is list:
                data[0][k] += data_fraud[0][k]
            elif type(data[0][k]) is torch.Tensor:
                data[0][k] = torch.cat([data[0][k], data_fraud[0][k]], dim=0)
        for k, v in data[1].items():
            data[1][k] = torch.cat([v, data_fraud[1][k]], dim=0)
        return data

    def data_prepare(self, data, is_test):
        """
        Args:
            data: ??????test??????????????????dataloader; ??????train???????????????dataloader???????????????
                - for test: (x,y)
                    - x: dict, contains all context data that ubs need.
                            - 'historical_behavior_embedding': list of tensor. It's length is event_num, and each tensor shape is [historical_behavior_num, embedding_dim].
                            - 'historical_behavior_type': list of tensor. It's length is event_num, and each tensor shape is [historical_behavior_num].
                            - 'historical_behavior_time': list of tensor. It's length is event_num, and each tensor shape is [historical_behavior_num].
                        - y: dict, ground truth.
                            - 'target_event_fraud': torch.Tensor. Shape is [event_num]. 1 means fraud while 0 means normal.
                            - 'target_event_time': torch.Tensor. Shape is [event_num]
                - for train: dict
                    - keys: {"benign", "fraud"}
                    - values: {(benign_x, benign_y), (fraud_x, fraud_y)}, the struct of x and y is same as above
            is_test: bool
        Returns:
            batch_size: int
            support_emb_after_group: list of tensor, [batch_size * type_num][event_num, dims]
            support_type_after_group: list of tensor, [batch_size][type_num]
            behavior_seq_type: list of tensor, [batch_size][historical_behavior_num]
            behavior_seq_time: list of tensor, [batch_size][historical_behavior_num]
            behavior_seq_len: list, [batch_size]
            query_time: torch.tensor, [batch_size]
        """
        # get data
        query_emb, query_time = data[0]['target_event_embedding'], data[0]['target_event_time']
        behavior_seq_emb, behavior_seq_type, behavior_seq_time = data[0]['historical_behavior_embedding'], \
                                                                 data[0]['historical_behavior_type'], \
                                                                 data[0]['historical_behavior_time']

        gt, gt_time = data[1]["target_event_fraud"], data[1]["target_event_settle_time"]
        behavior_seq_len = [len(x) for x in behavior_seq_emb]
        batch_size = len(gt)

        # ???type????????????
        support_emb_after_group = []  # [batch_size * type][event_num, dims]
        support_type_after_group = []  # [batch_size][type]
        for i, emb in enumerate(behavior_seq_emb):
            support_type_after_group.append(behavior_seq_type[i].unique())
            for j in range(self.type_num):
                type_len = (behavior_seq_type[i] == j).sum()
                if type_len != 0:
                    support_emb_after_group.append(emb[behavior_seq_type[i] == j])

        if is_test:
            settle_time = query_time + 60
        else:
            settle_time = copy.deepcopy(gt_time)
            settle_time[gt == 1] = settle_time[gt == 1] + 60

        return batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, \
               behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt


class IreregularTimeAwareSelfAtten(nn.Module):
    def __init__(self, in_dim):
        super(IreregularTimeAwareSelfAtten, self).__init__()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, input, seq_len):
        """
        Args:
            input: tensor.Tensor, [batch_size, seq_len, in_dim]
            seq_len: list, [batch_size]
        """
        out = torch.sigmoid(self.linear(input))
        weight = torch.zeros_like(input)
        for i in range(len(input)):
            weight[i, :seq_len[i]] = out[i, :seq_len[i]] / out[i, :seq_len[i]].sum(dim=1, keepdim=True)
        out = input * weight
        return out


class IreregularCNN(nn.Module):
    def __init__(self, in_dim, pool_dim, hidden_dim):
        super(IreregularCNN, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((pool_dim, None))
        self.cnn = nn.Sequential(nn.Conv2d(1, hidden_dim, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden_dim, hidden_dim, kernel_size=((pool_dim - 3 + 1), (in_dim - 3 + 1))),
                                 nn.ReLU(inplace=True))

    def forward(self, input):
        """
        Args:
            input: list of tensor, [batch_size][seq_len, in_dim]
        """
        out = torch.cat([self.pool(x.unsqueeze(0).unsqueeze(0)) for x in input],
                        dim=0)  # [batch_size,1,pool_dim,in_dim]
        out = self.cnn(out)  # [b, hidden_dim, 1, 1]
        out = out.squeeze(-1).squeeze(-1)
        return out
