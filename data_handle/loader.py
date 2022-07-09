import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_handle.dataset import UBSDataset, UBSRDataset, multi_list_collate
from data_handle.sampler import SequentialSampler

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PATH_FILE = Path(PARENT_DIR + "/config.json")


def load_from_sequence_for_ubs(server, data_name, benign_train_batch_size, fraud_train_batch_size,
                                 train_batch_num, test_batch_size, benign_train_data_len):
    path = json.load(open(PATH_FILE))["server"][server]["data_path"] + data_name + "/data_ubs.pt"
    benign_train_x, benign_train_y, fraud_train_x, fraud_train_y, val_x, val_y, test_x, test_y = torch.load(path)

    benign_train_sampler = SequentialSampler(benign_train_batch_size, train_batch_num, benign_train_data_len)
    benign_train_loader = DataLoader(UBSDataset(benign_train_x, benign_train_y),
                                    sampler=benign_train_sampler, collate_fn=lambda x: x[0])
    fraud_train_loader = DataLoader(UBSDataset(fraud_train_x, fraud_train_y),
                                    batch_size=fraud_train_batch_size, collate_fn=multi_list_collate)

    valid_loader = DataLoader(UBSDataset(val_x, val_y), batch_size=test_batch_size, collate_fn=multi_list_collate)
    test_loader = DataLoader(UBSDataset(test_x, test_y), batch_size=test_batch_size, collate_fn=multi_list_collate)

    in_dim = 13
    return benign_train_loader, fraud_train_loader, valid_loader, test_loader, in_dim


def load_from_sequence_for_ubs_r(server, data_name, benign_train_batch_size, fraud_train_batch_size,
                                 train_batch_num, test_batch_size, benign_train_data_len):
    # 数据加载
    path = json.load(open(PATH_FILE))["server"][server]["data_path"] + data_name + "/data_ubs_r.pt"
    benign_train_x, benign_train_y, benign_train_z, fraud_train_x, fraud_train_y, fraud_train_z, \
    val_x, val_y, val_z, test_x, test_y, test_z = torch.load(path)

    benign_train_sampler = SequentialSampler(benign_train_batch_size, train_batch_num, benign_train_data_len)
    benign_train_loader = DataLoader(UBSRDataset(benign_train_x, benign_train_y, benign_train_z),
                                    sampler=benign_train_sampler, collate_fn=lambda x: x[0])
    fraud_train_loader = DataLoader(UBSRDataset(fraud_train_x, fraud_train_y, fraud_train_z),
                                    batch_size=fraud_train_batch_size, collate_fn=multi_list_collate)

    valid_loader = DataLoader(UBSRDataset(val_x, val_y, val_z), batch_size=test_batch_size,
                              collate_fn=multi_list_collate)
    test_loader = DataLoader(UBSRDataset(test_x, test_y, test_z), batch_size=test_batch_size,
                             collate_fn=multi_list_collate)

    in_dim = 13
    return benign_train_loader, fraud_train_loader, valid_loader, test_loader, in_dim




