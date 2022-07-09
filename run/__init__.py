import os
import sys
import time
import json

import argparse

import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR + "/model")
sys.path.append(PARENT_DIR + "/data_handle")


def save_config(args, model_name):
    time_hash = str(time.time())
    model_path = json.load(open(PARENT_DIR + "/config.json"))["server"][args.server]["model_path"]
    model_file_name = args.dataset + "_" + time_hash

    if args.save_model:
        model_dir = os.path.join(model_path, "ubs/" + model_name + "/")
        print("model save in {}.ckpt".format(model_dir + model_file_name))
    else:
        model_dir = os.path.join(model_path, "ubs/trash/" + model_name + "/")
    os.makedirs(model_dir, exist_ok=True)

    result_path = json.load(open(PARENT_DIR + "/config.json"))["server"][args.server]["result_path"]
    result_version_name = args.dataset + "_" + time_hash
    if args.save_result:
        result_dir = os.path.join(result_path, "ubs_experiment/" + model_name + "/")
        print("result save in {}/".format(result_dir + result_version_name))
    else:
        result_dir = os.path.join(result_path, "ubs_experiment/trash/" + model_name + "/")
    os.makedirs(result_dir, exist_ok=True)
    return model_dir, model_file_name, result_dir, result_version_name

def process_args(args, dataset, server):
    args.dataset = dataset

    # device
    args.server = server
    args.checkpoint_mode = "max"
    args.checkpoint_monitor = "val_auroc"

    if torch.cuda.is_available() is False:
        args.cuda = -1

    return args

parser = argparse.ArgumentParser()
parser.add_argument('--comment', dest='comment', type=str, help='comment for ubs model', default="-")
parser.add_argument('--run_mode', dest='run_mode', type=str,
                    help='select exp [dup, beta, walk_len, without_etm, without_tpm, without_hdt]', default="dup")


parser.add_argument('--cuda', dest='cuda', type=int, help='gpu parameter of pl.Trainer', default=1)
parser.add_argument('--save_model', dest='save_model', type=int,
                    help='save the model in your setting path or in trash path', default=0)
parser.add_argument('--save_result', dest='save_result', type=int,
                    help='save the result in your setting path or in trash path', default=0)
parser.add_argument('--is_early_stopping', dest='is_early_stopping', type=bool, help='parameter in pl.Trainer',
                    default=False)
parser.add_argument('--max_epochs', dest='max_epochs', type=int, help='parameter in pl.Trainer', default=100)

parser.add_argument('--lr', dest='lr', type=float, help='learning rate.', default=1e-3)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, help='weight_decay of learning model',
                    default=1e-5)

parser.add_argument('--beta', dest='beta', type=float, help='parameter of metric learning for ubs', default=-1)
parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, help='hidden_dim in ', default=32)
parser.add_argument('--event_embed_dim', dest='event_embed_dim', type=int, help='event_embed_dim in ATAE module',
                    default=32)
parser.add_argument('--flow_layer_num', dest='flow_layer_num', type=int, help='flow_layer_num in ATAE module',
                    default=2)
parser.add_argument('--ode_layer_num', dest='ode_layer_num', type=int, help='ode_layer_num in ATAE module', default=8)
parser.add_argument('--sample_num', dest='sample_num', type=int, help='parameter in TP module', default=100)
parser.add_argument('--walk_emb_dim', dest='walk_emb_dim', type=int, help='parameter in UBS-R', default=10)
parser.add_argument('--walk_length', dest='walk_length', type=int, help='parameter in UBS-R', default=2)
parser.add_argument('--time_dim', dest='time_dim', type=int, help='parameter in UBS-R', default=1)
parser.add_argument('--position_dim', dest='position_dim', type=int, help='parameter in UBS-R', default=8)
parser.add_argument('--graph_layers', dest='graph_layers', type=int, help='parameter in UBS-R', default=2)

parser.add_argument('--etm', dest='etm', type=bool, help='for ablation analysis', default=True)
parser.add_argument('--tpm', dest='tpm', type=bool, help='for ablation analysis', default=True)
parser.add_argument('--hdt', dest='hdt', type=bool, help='for ablation analysis', default=True)

parser.add_argument('--benign_train_batch_size', dest='benign_train_batch_size', type=int,
                    help='num of benign in a batch of 128', default=123)
parser.add_argument('--fraud_train_batch_size', dest='fraud_train_batch_size', type=int,
                    help='num of fraud in a batch of 128', default=5)
parser.add_argument('--train_batch_num', dest='train_batch_num', type=int, help='total num of train batch', default=171)
parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, help='batch size of test', default=128)
parser.add_argument('--benign_train_data_len', dest='benign_train_data_len', type=int,
                    help='total num of benign event in train', default=19613)

parser.add_argument('--type_num', dest='type_num', type=int, help='total num of event type', default=5)
parser.add_argument('--time_normalize', dest='time_normalize', type=int, help='a normalizing process for time to (0,1)',
                    default=743)
parser.add_argument('--query_feature_dim', dest='query_feature_dim', type=int, help='query event\'s feature_dim',
                    default=12)
parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, help='max_seq_len', default=15)

parser.add_argument('--pretrained_file_path', dest='pretrained_file_path', type=str, help='the pretrained file path', default=None)

