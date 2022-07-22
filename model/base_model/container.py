import torch
import pytorch_lightning as pl
from torchmetrics import AUROC, PrecisionRecallCurve, AUC

__all__ = ["Container", "FraudContainer"]


class Container(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=0, metric_mode=['val', 'test'], comment="-", **kwargs):
        super(Container, self).__init__()
        # learning
        self.lr = lr
        self.weight_decay = weight_decay

        # metric
        self.default_metric_names = []
        self.metric_mode = metric_mode
        if "train" in metric_mode:
            self.train_metric_dict = self.init_metric()
            assert type(self.train_metric_dict) is dict
        if "val" in metric_mode:
            self.val_metric_dict = self.init_metric()
            assert type(self.val_metric_dict) is dict
        if "test" in metric_mode:
            self.test_metric_dict = self.init_metric()
            assert type(self.test_metric_dict) is dict

        # unique comment for containter to print log
        self.comment = comment

    def init_metric(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def print_log(mode, current_epoch, candidate_metric_dict, comment="-", **kwargs):
        assert mode in ["train", "val", "test"]
        flag = True
        for name, value in candidate_metric_dict.items():
            if flag:
                print("\nepoch:{}: metric_{}_{}:{}: comment: {}".format(current_epoch, mode, name, value.compute(),
                                                                        comment))
                flag = False
            else:
                print("epoch:{}: metric_{}_{}:{}: comment: {}".format(current_epoch, mode, name, value.compute(),
                                                                      comment))

    def update_metric(self, mode, t, p, metric_list=None, p_prob=None, **kwargs):
        """
        Args:
            mode: "train"/"val"/"test"
            p: predict
            t: truth
            metric_list: the metric names that u wanna update.
                         Note that the metric name must in the keys of mode_metric_dict.
                         The default is all metrics in the corresponding mode_metric_dict.
        """
        assert mode in ["train", "val", "test"]

        metric_dict = getattr(self, mode + "_metric_dict")
        if metric_list is None:
            metric_list = self.default_metric_names if self.default_metric_names else metric_dict.keys()

        for metric_name in metric_list:
            metric = metric_dict[metric_name]
            metric(p.cpu(), t.cpu())
            self.log(mode + "_" + metric_name, metric, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=p.shape[0])

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, data, batch_idx, optimizer_idx=None):
        raise NotImplementedError

    def on_train_epoch_end(self, metric_list=None) -> None:
        super(Container, self).on_train_epoch_end()
        if "train" in self.metric_mode:
            if metric_list is None: metric_list = self.default_metric_names
            current_metric_dict = {metric: self.train_metric_dict[metric] for metric in
                                   metric_list} if metric_list else self.train_metric_dict
            self.print_log("train", self.current_epoch, current_metric_dict, comment=self.comment)

            for key in self.train_metric_dict.keys():
                self.train_metric_dict[key].reset()

    def validation_step(self, data, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        super(Container, self).on_validation_epoch_end()
        if "val" in self.metric_mode:
            metric_list = self.default_metric_names
            current_metric_dict = {metric: self.val_metric_dict[metric] for metric in
                                   metric_list} if metric_list else self.val_metric_dict
            self.print_log("val", self.current_epoch, current_metric_dict, comment=self.comment)

            for key in self.val_metric_dict.keys():
                self.val_metric_dict[key].reset()

    def test_step(self, data, batch_idx):
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        super(Container, self).on_test_epoch_end()
        if "test" in self.metric_mode:
            metric_list = self.default_metric_names
            current_metric_dict = {metric: self.test_metric_dict[metric] for metric in
                                   metric_list} if metric_list else self.test_metric_dict
            self.print_log("test", self.current_epoch, current_metric_dict, comment=self.comment)

            for key in self.test_metric_dict.keys():
                self.test_metric_dict[key].reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class FraudContainer(Container):
    def __init__(self, lr=1e-3, weight_decay=0, metric_mode=['val', 'test'], comment="-", **kwargs):
        super(FraudContainer, self).__init__(lr=lr, weight_decay=weight_decay, metric_mode=metric_mode, comment=comment)
        # learning
        self.lr = lr
        self.weight_decay = weight_decay

        # metric
        # self.default_metric_names = ["acc", "auroc", "auprc", "bin_f1", "bin_recall", "bin_precision"]
        self.default_metric_names = ["auroc",  "auprc"]

    def init_metric(self):
        auroc = AUROC(pos_label=1, compute_on_step=False)
        prcurve = PrecisionRecallCurve(pos_label=1, compute_on_step=False)
        auprc = AUC(reorder=True, compute_on_step=False)
        return {"auroc": auroc, "prcurve": prcurve, "auprc": auprc}

    def update_metric(self, mode, t, p=None, metric_list=None, p_prob=None, **kwargs):
        """

        Args:
            mode:
            t: tensor, [n]
            p: tensor, [n]
            metric_list:
            p_prob: tensor, [n]
            **kwargs:

        Returns:

        """
        assert mode in ["train", "val", "test"]

        metric_dict = getattr(self, mode + "_metric_dict")
        if metric_list is None:
            metric_list = self.default_metric_names

        for metric_name in metric_list:
            metric = metric_dict[metric_name]

            if metric_name == "auroc" or metric_name == "auprc":
                p = p_prob.squeeze()

            if metric_name == "auprc":
                metric_dict["prcurve"](p.cpu(), t.cpu())
                y, x, _ = metric_dict["prcurve"].compute()
                metric.reset()
                metric(x, y)
            else:
                metric(p.cpu(), t.cpu())

            self.log(mode + "_" + metric_name, metric.compute(), prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=p.shape[0])
