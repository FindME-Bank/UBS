import torch
from torch.utils.data import Dataset


def get_data(data, index):
    if type(int) is list:
        print()
    if type(index) is list and type(data) is list:
        return [data[i] for i in index]
    else:
        try:
            return data[index]
        except:
            print()
        # return data[index]


class UBSDataset(Dataset):
    """
    Data:
    x: dict, sequence feature including ["historical_behavior_embedding", "historical_behavior_type",
                                         "historical_behavior_time", "target_event_embedding",
                                         "target_event_time", "target_user_id"]
    y: dict, target ground truth including ["target_event_fraud", "target_event_settle_time"]
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x, y = {}, {}
        for key, value in self.x.items():
            x[key] = get_data(value, index)

        for key, value in self.y.items():
            y[key] = get_data(value, index)

        return x, y

    def __len__(self):
        return len(self.x.get(next(iter(self.x))))


class UBSRDataset(UBSDataset):
    """
    Data:
    x: dict, sequence feature including ["historical_behavior_embedding", "historical_behavior_type",
                                         "historical_behavior_time", "target_event_embedding",
                                         "target_event_time", "target_user_id"]
    y: dict, target ground truth including ["target_event_fraud", "target_event_settle_time"]
    z: tuple, caw data, i.e., (caw_source, caw_target, time, edge_emb, mask_len, edge_index),
                              note that each item is list of tensor.
    """
    def __init__(self, x, y, z):
        super(UBSRDataset, self).__init__(x, y)
        self.z = z

    def __getitem__(self, index):
        x, y = super(UBSRDataset, self).__getitem__(index)

        x["caw_s"] = get_data(self.z[0], index)
        x["caw_t"] = get_data(self.z[1], index)
        x["caw_time"] = get_data(self.z[2], index)
        x["caw_edge_emb"] = get_data(self.z[3], index)
        x["caw_mask_len"] = get_data(self.z[4], index)
        x["caw_edge_index"] = get_data(self.z[5], index)
        return x, y


data_type = {"historical_behavior_embedding": "list", "historical_behavior_time": "list",
             "historical_behavior_type": "list", "historical_behavior_fraud": "list",
             "target_event_time": "tensor", "target_event_embedding": "tensor", "target_event_fraud": "tensor",
             "target_event_settle_time": "tensor", "target_user_id": "tensor",
             "caw_s": "list", "caw_t": "list", "caw_time": "list",
             "caw_edge_emb": "list", "caw_mask_len": "list", "caw_edge_index": "list",}


def multi_list_collate(batch):
    """
    功能：multi feature collate
    返回：(x, y), x和y都是dict
    根据给定data_type返回指定类型的数据
    """
    x_feature = batch[0][0].keys()
    y_feature = batch[0][1].keys()

    x, y = {}, {}
    for feature in x_feature:
        x[feature] = [data[0][feature] for data in batch]
        if data_type[feature] == "tensor":
            x[feature] = torch.stack(x[feature])

    for feature in y_feature:
        y[feature] = [data[1][feature] for data in batch]
        if data_type[feature] == "tensor":
            y[feature] = torch.stack(y[feature])

    return (x, y)