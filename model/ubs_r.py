import math
from multiprocessing import Pool

import scipy.optimize as opt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from model.base_model.basic_net import MLP, GAT
from model.ubs import UBS


class UBSR(UBS):
    def __init__(self, type_num, query_feature_dim, support_feature_dim, walk_length, walk_emb_dim,  # input
                 time_encoder_dim=1, risk_embed_dim=32, conduction_hidden_dim=32, match_hidden_dim=32, graph_layers=2,
                 **kwargs):
        '''
        Args:
            walk_length: path length of the time-aware risk conduction paths extraction layer
            walk_emb_dim: the input walk embedding dimension in the risk conduction embedding layer
            time_encoder_dim: time encoder dimension for walk embedding
            conduction_hidden_dim: hidden dimension of the risk conduction embedding layer
            match_hidden_dim: hidden dimension of the graph affinity matrix
            graph_layers: number of the cross graph risk diffusion layer
            query_embedding_dim: dimension of query embedding
            risk_embed_dim: risk embedding dimension for default prediction
            **kwargs:
        '''
        super(UBSR, self).__init__(type_num=type_num, query_feature_dim=query_feature_dim,
                                   support_feature_dim=support_feature_dim, **kwargs)
        # risk_conduction_effect_learning_module
        self.time_encoder = TimeEncode(expand_dim=time_encoder_dim)
        self.s_position_encoder = nn.Sequential(nn.Linear(walk_length + 1, conduction_hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(conduction_hidden_dim, conduction_hidden_dim))
        self.t_position_encoder = nn.Sequential(nn.Linear(walk_length + 1, conduction_hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(conduction_hidden_dim, conduction_hidden_dim))
        self.feature_layer = nn.LSTM(time_encoder_dim + conduction_hidden_dim * 2 + walk_emb_dim, conduction_hidden_dim,
                                     batch_first=True,
                                     bidirectional=True)
        self.position_layer = nn.LSTM(conduction_hidden_dim * 2, conduction_hidden_dim, batch_first=True,
                                      bidirectional=True)
        self.projector = nn.Linear(2 * conduction_hidden_dim + 2 * conduction_hidden_dim, conduction_hidden_dim)
        self.risk_layer = nn.Linear(conduction_hidden_dim, risk_embed_dim)

        # default_event_prediction_module
        self.predictor = MLP(self.query_embedding_dim + self.influence_embed_dim +  1 + risk_embed_dim,
                             [self.predictor_hidden_dim, 1], activation="relu", last_activation=None)
        self.distance = MLP(2 * (self.query_embedding_dim + self.influence_embed_dim +  1 + risk_embed_dim),
                            [self.distance_hidden_dim, 1], activation="relu", last_activation=None)

        # match_module
        self.affinity = Affinity(match_hidden_dim)
        self.sinkhorn = Sinkhorn(batched_operation=True)
        self.sage = GAT(in_dim=match_hidden_dim, hidden_dims=[match_hidden_dim] * graph_layers,
                        heads=[8] * graph_layers, activation="relu", last_activation="relu")
        self.similarity = MLP(2 * match_hidden_dim, [match_hidden_dim, 1], activation="relu", last_activation=None)

    def forward(self, batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb,
                behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time,
                caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len, caw_edge_index,
                is_test, gt=None):
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
            caw_s: list of tensor, [batch_size][node_num, walk_num, walk_length, 2, walk_length]
            caw_t: list of tensor, [batch_size][node_num, walk_num, walk_length, 2, walk_length]
            caw_time: list of tensor, [batch_size][node_num, walk_num, walk_length]
            caw_edge_emb: list of tensor, [batch_size][node_num, walk_num, walk_length, edge_emb_dim]
            caw_mask_len: list of tensor, [batch_size][node_num, walk_num]
            caw_edge_index: list of tensor, [batch_size][2, edge_num]
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
        repayment_willingness, like_tp = self.predict_time(h, query_time, settle_time, is_test)

        risk_embedding = self.learning_risk_conduction_effect(caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len)

        match_pairs = self.get_match_pairs(batch_size=batch_size)
        target_risk_embedding, walk_distance = self.match(match_pairs, risk_embedding, caw_edge_index)
        query_emb = self.query_embedding_net(query_emb)
        default_prob = self.predict_default(h, repayment_willingness, query_emb, target_risk_embedding)

        if is_test:
            return default_prob
        else:
            distance = self.calculate_distance(cumulated_influence, repayment_willingness, query_emb, target_risk_embedding)
            metric_pairs, beta = self.hard_discriminated_sample(distance, behavior_seq_len, gt)
            return default_prob, mus, logvars, delta_logps, like_tp, \
                   metric_pairs, distance, beta

    def calculate_distance(self, h, willingness, query_emb, target_risk_embedding):
        input = torch.cat([query_emb, h, willingness.unsqueeze(-1), target_risk_embedding], dim=-1)
        input = torch.cat([input.repeat_interleave(input.shape[0], dim=0), input.repeat((input.shape[0], 1))],
                          dim=-1)  # [b*b,2*h]
        distance = self.distance(input).squeeze(-1).reshape(h.shape[0], h.shape[0])
        distance = torch.sigmoid(distance)
        return distance

    def predict_default(self, h, willingness, query_emb, target_risk_embedding):
        input = torch.cat([query_emb, h, willingness.unsqueeze(-1), target_risk_embedding], dim=-1)
        default_prob = self.predictor(input).squeeze(-1)
        return torch.sigmoid(default_prob)

    def match(self, match_pairs, risk_embedding, graph_enclosing):
        """
        Args:
            match_pairs: [match_num, 2]
            risk_embedding: [batch_size][node_num, _dim]
            graph_enclosing: [batch_size][2, edge_num]
        """
        node_num = torch.tensor([len(x) for x in risk_embedding]).to(self.device)
        max_node_num = node_num.max().item()

        batch_size, emb_dim = len(risk_embedding), risk_embedding[0].shape[-1]
        align_risk_embedding = torch.zeros(batch_size, max_node_num, emb_dim).to(self.device)
        for b, emb in enumerate(risk_embedding):
            align_risk_embedding[b, :len(emb)] = emb

        x = align_risk_embedding[match_pairs[:, 0]]  # [match_num, max_node_num, emb_dim]
        x_len = node_num[match_pairs[:, 0]]  # [match_num]
        y = align_risk_embedding[match_pairs[:, 1]]  # [match_num, max_node_num, emb_dim]
        y_len = node_num[match_pairs[:, 1]]  # [match_num]

        s = self.affinity(x, y)  # [match_num, max_node_num, max_node_num]
        s = self.sinkhorn(s, nrows=x_len, ncols=y_len)  # [match_num, max_node_num, max_node_num]
        s = hungarian(s, x_len, y_len)

        target_node = torch.zeros([1]).to(torch.int64).to(self.device)
        target_node = torch.cat([target_node, torch.cumsum(node_num, dim=0)[:-1]], dim=0)  # [batch_size]

        x = torch.cat(risk_embedding, dim=0)  # [node_num, dim]
        edge_index = [edge - 1 + target_node[i] for i, edge in enumerate(graph_enclosing)]  # [2,]
        for idx, (i, j) in enumerate(match_pairs):
            fake_edge_index = dense_to_sparse(s[idx])[0]
            fake_edge_index[0] += target_node[i]
            fake_edge_index[1] += target_node[j]
            edge_index.append(fake_edge_index)
        edge_index = torch.cat(edge_index, dim=-1)
        edge_index = edge_index.unique(dim=-1)

        target_node_emb = self.sage(x, edge_index)[target_node]  # [target_node, dim]
        input = torch.cat([target_node_emb.repeat_interleave(target_node_emb.shape[0], dim=0),
                           target_node_emb.repeat((target_node_emb.shape[0], 1))], dim=-1)  # [b*b,2*h]
        similarity = self.similarity(input).squeeze(-1).reshape(batch_size, batch_size)
        similarity = torch.sigmoid(similarity)
        return target_node_emb, similarity

    def get_match_pairs(self, metric_pairs=None, batch_size=0):
        if metric_pairs is not None:
            match_pairs = []
            for i, j, k_index, l_index in metric_pairs:
                if len(k_index) != 0: match_pairs.append(torch.stack([torch.ones_like(k_index) * i, k_index], dim=1))
                if len(l_index) != 0: match_pairs.append(torch.stack([torch.ones_like(l_index) * j, l_index], dim=1))
            match_pairs = torch.cat(match_pairs, dim=0).unique(dim=0)
            return match_pairs

        if batch_size != 0:
            points = torch.arange(batch_size)
            paris = torch.stack([points.repeat_interleave(batch_size, dim=0), points.repeat(batch_size)],
                                dim=1)  # [b*b,2]
            match_pairs = paris[paris[:, 0] != paris[:, 1]]  # [b*(b-1),2]
            return match_pairs

        raise IOError("Please check input!")

    def learning_risk_conduction_effect(self, caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len):
        """
        Args:
            caw_s: [batch_size][node_num, walk_num, walk_length, 2, walk_length]
            caw_t: [batch_size][node_num, walk_num, walk_length, 2, walk_length]
            caw_time: [batch_size][node_num, walk_num, walk_length]
            caw_edge_emb: [batch_size][node_num, walk_num, walk_length, edge_emb_dim]
            caw_mask_len: [batch_size][node_num, walk_num]
        """
        node_num = [len(x) for x in caw_s]
        caw_s = torch.cat(caw_s, dim=0)
        caw_t = torch.cat(caw_t, dim=0)
        caw_edge_emb = torch.cat(caw_edge_emb, dim=0)
        caw_time = torch.cat(caw_time, dim=0)
        caw_mask_len = torch.cat(caw_mask_len, dim=0)

        # 1. get the feature matrix shaped [batch, n_walk, len_walk + 1, time_encoder_dim + pos_dim + edge_dim]
        time_emb = self.time_encoder(caw_time)
        s_position_emb = self.s_position_encoder(caw_s).sum(dim=-2)
        t_position_emb = self.t_position_encoder(caw_t).sum(dim=-2)
        features = torch.cat([time_emb, s_position_emb, t_position_emb, caw_edge_emb], dim=-1)

        # 2. feed the matrix forward to LSTM, then transformer, now shaped [batch, n_walk, transformer_model_dim]
        batch_size, walk_num, walk_length, feat_dim = features.shape
        caw_mask_len = caw_mask_len.view(batch_size, walk_num, 1, 1) - 1

        features = features.view(-1, walk_length, feat_dim)
        features = self.feature_layer(features)[0].view(batch_size, walk_num, walk_length, -1)
        features = features.gather(2, caw_mask_len.expand(batch_size, walk_num, 1, features.shape[-1])).squeeze(
            2)  # [batch, n_walk, *_dim]

        position_emb = torch.cat([s_position_emb, t_position_emb], dim=-1)
        position_features = position_emb.view(-1, walk_length, position_emb.shape[-1])
        position_features = self.position_layer(position_features)[0].view(batch_size, walk_num, walk_length, -1)
        position_features = position_features.gather(2, caw_mask_len.expand(batch_size, walk_num, 1,
                                                                            position_features.shape[-1])).squeeze(
            2)  # [batch, n_walk, *_dim]

        combined_features = torch.cat([features, position_features], dim=-1)
        combined_features = torch.relu(self.projector(combined_features))

        # 3. aggregate and collapse dim=1 (using set operation), now shaped [batch, out_dim]
        combined_features = combined_features.mean(1)
        risk_embedding = self.risk_layer(combined_features)  # [batch_size, *_dim]
        risk_embedding = risk_embedding.split(node_num, 0)  # [batch_size][node_num, _dim]
        return risk_embedding

    def training_step(self, data, batch_idx, optimizer_idx=None):
        self.first_validation = False

        data = self.data_merge(data)
        batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, \
        behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt, \
        caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len, caw_edge_index = self.data_prepare(data, is_test=False)

        default_prob, mus, logvars, delta_logps, like_tp, metric_pairs, distance, beta = \
            self.forward(batch_size, query_emb, support_emb_after_group, support_type_after_group,
                         behavior_seq_emb, behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time,
                         settle_time, caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len, caw_edge_index,
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
            behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt, \
            caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len, caw_edge_index = self.data_prepare(data, is_test=True)

            default_prob = self.forward(batch_size, query_emb, support_emb_after_group, support_type_after_group,
                                        behavior_seq_emb, behavior_seq_type, behavior_seq_time, behavior_seq_len,
                                        query_time, settle_time, caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len,
                                        caw_edge_index, is_test=True)
        p = self.formulate_for_metric(default_prob)
        self.update_metric("val", gt, p, p_prob=default_prob)

    def test_step(self, data, batch_idx):
        batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, \
        behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt, \
        caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len, caw_edge_index = self.data_prepare(data, is_test=True)

        default_prob = self.forward(batch_size, query_emb, support_emb_after_group, support_type_after_group,
                                    behavior_seq_emb, behavior_seq_type, behavior_seq_time, behavior_seq_len,
                                    query_time, settle_time, caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len,
                                    caw_edge_index, is_test=True)

        p = self.formulate_for_metric(default_prob)
        self.update_metric("test", gt, p, p_prob=default_prob)

    def data_prepare(self, data, is_test):
        batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, behavior_seq_type, \
        behavior_seq_time, behavior_seq_len, query_time, settle_time, gt = super(UBSR, self).data_prepare(data, is_test)

        caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len, caw_edge_index = data[0]['caw_s'], \
                                                                             data[0]['caw_t'], \
                                                                             data[0]['caw_time'], \
                                                                             data[0]['caw_edge_emb'], \
                                                                             data[0]['caw_mask_len'], \
                                                                             data[0]['caw_edge_index']

        return batch_size, query_emb, support_emb_after_group, support_type_after_group, behavior_seq_emb, \
               behavior_seq_type, behavior_seq_time, behavior_seq_len, query_time, settle_time, gt, \
               caw_s, caw_t, caw_time, caw_edge_emb, caw_mask_len, caw_edge_index


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        self.time_encoder_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_encoder_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_encoder_dim).float())

    def forward(self, time):  # [batch_size, walk_num, walk_length]
        batch_size, walk_num, walk_length = time.shape

        time = time.view(batch_size, -1).unsqueeze(-1)  # [batch_size, walk_num * walk_length, 1]
        map_time = time * self.basis_freq.view(1, 1, -1)  # [batch_size, walk_num * walk_length, time_encoder_dim]
        map_time += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_time)
        harmonic = harmonic.view(batch_size, walk_num, walk_length, -1)
        return harmonic


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: dim
    Input: feature X, Y
    Output: affinity matrix M
    """

    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = nn.Parameter(torch.Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        """
        input: [batch_size, node_num, node_num]
        """
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, self.A)  # [b,n,d] x [d, d] -> [b,n,d]
        M = torch.matmul(M, Y.transpose(1, 2))  # [b,n,d] x [b,d,m] -> [b,n,m]
        return M  # [batch_size, node_num1, node_num2]


class Sinkhorn(nn.Module):
    r"""
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.
    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:
    .. math::
        \mathbf{S}_{i,j} = \exp \left(\frac{\mathbf{s}_{i,j}}{\tau}\right)
    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:
    .. math::
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top \cdot \mathbf{S}) \\
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{S} \cdot \mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top)
    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}_n` means a column-vector with length :math:`n`
    whose elements are all :math:`1`\ s.
    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param log_forward: apply log-scale computation for better numerical stability (default: ``True``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)
    .. note::
        ``tau`` is an important hyper parameter to be set for Sinkhorn algorithm. ``tau`` controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
        :func:`~src.lap_solvers.hungarian.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
        Hungarian, at the cost of slower convergence speed and reduced numerical stability.
    .. note::
        We recommend setting ``log_forward=True`` because it is more numerically stable. It provides more precise
        gradient in back propagation and helps the model to converge better and faster.
    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient.
    """

    def __init__(self, max_iter: int = 10, tau: float = 1., epsilon: float = 1e-4,
                 log_forward: bool = True, batched_operation: bool = False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.batched_operation = batched_operation  # batched operation may cause instability in backward computation,
        # but will boost computation.

    def forward(self, s: torch.Tensor, nrows=None, ncols=None, dummy_row=False) -> torch.Tensor:
        """
        Compute sinkhorn with row/column normalization in the log space.
        Args:
            s: [batch_size, n_1, n_2]
            nrows: [batch_size]
            ncols: [batch_size]
            dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
        Return:
            the computed doubly-stochastic matrix, [batch_size, n_1, n_2]
        Note:
            - We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.
            - We assume row number <= column number. If not, the input matrix will be transposed.
        """
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose(1, 2)
            nrows, ncols = ncols, nrows
            transposed = True

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # operations are performed on log_s
        s = s / self.tau

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            s = torch.cat((s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                s[b, nrows[b]:, :] = -float('inf')
                s[b, :, ncols[b]:] = -float('inf')

        if self.batched_operation:
            log_s = s

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')

                # ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s = s[b, row_slice, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                        log_s = log_s - log_sum
                    else:
                        log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                        log_s = log_s - log_sum

                ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if transposed:
                ret_log_s = ret_log_s.transpose(1, 2)
            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

        # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        # for b in range(batch_size):
        #    row_slice = slice(0, nrows[b])
        #    col_slice = slice(0, ncols[b])
        #    log_s = s[b, row_slice, col_slice]


def hungarian(s: torch.Tensor, n1=None, n2=None, nproc: int = 1):
    r"""
    Solve optimal LAP permutation by hungarian algorithm. The time cost is :math:`O(n^3)`.
    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param n1: :math:`(b)` number of objects in dim1
    :param n2: :math:`(b)` number of objects in dim2
    :param nproc: number of parallel processes (default: ``nproc=1`` for no parallel)
    :return: :math:`(b\times n_1 \times n_2)` optimal permutation matrix
    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded.
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat


def _hung_kernel(s: torch.Tensor, n1=None, n2=None):
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    row, col = opt.linear_sum_assignment(s[:n1, :n2])
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat
