import os
import sys
import time
import json

import argparse

import torch
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def save_config(args, model_name):
    time_hash = str(time.time())

    model_dir = json.load(open(PARENT_DIR + "/config.json"))["server"][args.server]["model_path"] + "model_name/"
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = "model_" + time_hash
    print("The model will save in {}.ckpt".format(model_dir + model_file_name))


    result_dir = json.load(open(PARENT_DIR + "/config.json"))["server"][args.server]["result_path"] + "model_name/"
    os.makedirs(result_dir, exist_ok=True)
    result_version_name = "result_" + time_hash
    print("result save will in {}/".format(result_dir + result_version_name))

    return model_dir, model_file_name, result_dir, result_version_name


def process_args(args, server):
    # device
    args.server = server
    args.checkpoint_mode = "max"
    args.checkpoint_monitor = "val_auroc"

    if torch.cuda.is_available() is False:
        args.cuda = -1

    return args


parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', dest='run_mode', type=str,
                    help='select exp [dup, beta, walk_len, without_etm, without_tpm, without_hdt]', default="dup")

parser.add_argument('--cuda', dest='cuda', type=int, help='gpu parameter of pl.Trainer', default=1)
parser.add_argument('--max_epochs', dest='max_epochs', type=int, help='parameter in pl.Trainer', default=100)
parser.add_argument('--pretrained_file_path', dest='pretrained_file_path', type=str, help='the pretrained file path',
                    default=None)

# dataset
parser.add_argument('--benign_train_batch_size', dest='benign_train_batch_size', type=int,
                    help='num of benign in a batch of 128', default=64 - 5)
parser.add_argument('--fraud_train_batch_size', dest='fraud_train_batch_size', type=int,
                    help='num of fraud in a batch of 128', default=5)
parser.add_argument('--train_batch_num', dest='train_batch_num', type=int, help='total num of train batch', default=20)
parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, help='batch size of test', default=64)
parser.add_argument('--benign_train_data_len', dest='benign_train_data_len', type=int,
                    help='total num of benign event in train', default=1100)

# input
parser.add_argument('--type_num', dest='type_num', type=int, help='total num of event type', default=5)

# ETM
parser.add_argument('--type_embed_dim', dest='type_embed_dim', type=int, help='type_embed_dim in ETM module',
                    default=2)
parser.add_argument('--cnf_hidden_dim', dest='cnf_hidden_dim', type=int, help='type_embed_dim in ATAE module',
                    default=32)
parser.add_argument('--flow_layer_num', dest='flow_layer_num', type=int, help='flow_layer_num in ATAE module',
                    default=2)

# ode-rnn
parser.add_argument('--influence_embed_dim', dest='influence_embed_dim', type=int, help='influence embed after ode_rnn',
                    default=4)
parser.add_argument('--time_normalize', dest='time_normalize', type=int, help='a normalizing process for time to (0,1)',
                    default=743)
parser.add_argument('--ode_layer_num', dest='ode_layer_num', type=int, help='ode_layer_num in ATAE module', default=2)

# query_embedding_net
parser.add_argument('--query_embedding_dim', dest='query_embedding_dim', type=int,
                    help='query_embedding_dim after query_embedding_net', default=16)

# predictor
parser.add_argument('--predictor_hidden_dim', dest='predictor_hidden_dim', type=int, help='hidden dim in predictor',
                    default=64)

# tpm
parser.add_argument('--sample_num', dest='sample_num', type=int, help='parameter in TP module', default=100)

# hdt
parser.add_argument('--distance_hidden_dim', dest='distance_hidden_dim', type=int, help='hidden_dim in tpm', default=128)
parser.add_argument('--beta', dest='beta', type=float, help='parameter of metric learning for ubs', default=-1)
parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, help='max_seq_len', default=15)

# ubs-r
parser.add_argument('--walk_length', dest='walk_length', type=int, help='parameter in UBS-R', default=2)
parser.add_argument('--walk_emb_dim', dest='walk_emb_dim', type=int, help='parameter in UBS-R', default=10)
parser.add_argument('--time_encoder_dim', dest='time_encoder_dim', type=int, help='parameter in UBS-R', default=1)
parser.add_argument('--conduction_hidden_dim', dest='conduction_hidden_dim', type=int, help='parameter in UBS-R', default=32)
parser.add_argument('--risk_embed_dim', dest='risk_embed_dim', type=int, help='parameter in UBS-R', default=32)
parser.add_argument('--match_hidden_dim', dest='match_hidden_dim', type=int, help='parameter in UBS-R', default=32)
parser.add_argument('--graph_layers', dest='graph_layers', type=int, help='parameter in UBS-R', default=2)

# ablation
parser.add_argument('--etm', dest='etm', type=bool, help='for ablation analysis', default=True)
parser.add_argument('--tpm', dest='tpm', type=bool, help='for ablation analysis', default=True)
parser.add_argument('--hdt', dest='hdt', type=bool, help='for ablation analysis', default=True)

# learning
parser.add_argument('--lr', dest='lr', type=float, help='learning rate.', default=1e-3)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, help='weight_decay of learning model',
                    default=1e-5)

# comment
parser.add_argument('--comment', dest='comment', type=str, help='comment for ubs model', default="-")

