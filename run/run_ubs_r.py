import os
import sys
import random

device = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = device
os.environ['CUDA_LAUNCH_BLOCKING'] = device

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_DIR + "/model")
sys.path.append(PARENT_DIR + "/data_handle")
sys.path.append(PARENT_DIR + "/run")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from model.ubs_r import UBSR
from run import save_config, parser, process_args
from data_handle.loader import load_from_sequence_for_ubs_r


def run(args):
    if args.seed:
        pl.seed_everything(args.seed)

    # ========================= save ======================= #
    model_dir, model_file_name, result_dir, result_version_name = save_config(args, "ubs-r")

    # ====================== load data ======================= #
    benign_train_loader, fraud_train_loader, valid_loader, test_loader, query_feature_dim, support_feature_dim = \
        load_from_sequence_for_ubs_r( args.server, args.benign_train_batch_size,
                                      args.fraud_train_batch_size, args.train_batch_num, args.test_batch_size,
                                      args.benign_train_data_len)


    # ====================== config trainer ================= #
    callbacks = [ModelCheckpoint(save_weights_only=True,
                                 dirpath=model_dir,
                                 filename=model_file_name,
                                 mode=args.checkpoint_mode,
                                 monitor=args.checkpoint_monitor),
                 LearningRateMonitor(log_momentum=True)]

    trainer = pl.Trainer(callbacks=callbacks,
                         logger=[TensorBoardLogger(save_dir=result_dir, name="", version=result_version_name)],
                         gpus=args.cuda if args.cuda != -1 else 0,
                         max_epochs=args.max_epochs)

    # Check whether pretrained model exists. If yes, load it and skip training
    if args.pretrained_file_path and os.path.isfile(args.pretrained_file_path):
        print("Found pretrained model, loading...")
        model = UBSR.load_from_checkpoint(args.pretrained_file_path,
                    type_num=args.type_num, query_feature_dim=query_feature_dim,
                    support_feature_dim=support_feature_dim,  # input
                    walk_length=args.walk_length, walk_emb_dim=args.walk_emb_dim,
                    time_encoder_dim=args.time_encoder_dim, risk_embed_dim=args.risk_embed_dim,
                    conduction_hidden_dim=args.conduction_hidden_dim, match_hidden_dim=args.match_hidden_dim,
                    graph_layers=args.graph_layers, # ubs-r param
                    type_embed_dim=args.type_embed_dim, cnf_hidden_dim=args.cnf_hidden_dim,
                    flow_layer_num=args.flow_layer_num,  # etm
                    influence_embed_dim=args.influence_embed_dim, time_normalize=args.time_normalize,
                    ode_layer_num=args.ode_layer_num,  # ode-rnn
                    predictor_hidden_dim=args.predictor_hidden_dim,  # predictor
                    sample_num=args.sample_num,  # tpm
                    beta=args.beta, max_seq_len=args.max_seq_len, distance_hidden_dim=args.distance_hidden_dim,  # hdt
                    etm=args.etm, tpm=args.tpm, hdt=args.hdt, # ablation
                    lr=args.lr, weight_decay=args.weight_decay,
                    comment=args.comment)
        trainer.test(model=model, dataloaders=test_loader)
    else:
        model = UBSR(type_num=args.type_num, query_feature_dim=query_feature_dim,
                    support_feature_dim=support_feature_dim,  # input
                    walk_length=args.walk_length, walk_emb_dim=args.walk_emb_dim,
                    time_encoder_dim=args.time_encoder_dim, risk_embed_dim=args.risk_embed_dim,
                    conduction_hidden_dim=args.conduction_hidden_dim, match_hidden_dim=args.match_hidden_dim,
                    graph_layers=args.graph_layers, # ubs-r param
                    type_embed_dim=args.type_embed_dim, cnf_hidden_dim=args.cnf_hidden_dim,
                    flow_layer_num=args.flow_layer_num,  # etm
                    influence_embed_dim=args.influence_embed_dim, time_normalize=args.time_normalize,
                    ode_layer_num=args.ode_layer_num,  # ode-rnn
                    predictor_hidden_dim=args.predictor_hidden_dim,  # predictor
                    sample_num=args.sample_num,  # tpm
                    beta=args.beta, max_seq_len=args.max_seq_len, distance_hidden_dim=args.distance_hidden_dim,  # hdt
                    etm=args.etm, tpm=args.tpm, hdt=args.hdt, # ablation
                    lr=args.lr, weight_decay=args.weight_decay,
                    comment=args.comment)
        trainer.fit(model, train_dataloaders={"benign": benign_train_loader, "fraud": fraud_train_loader},
                    val_dataloaders=valid_loader)
        trainer.test(dataloaders=test_loader)

def run_dup(server, args):
    duplicate = 10
    for i in range(duplicate):
        args = process_args(args, server)
        args.seed = random.randint(0, 10000)
        print("seed:", args.seed)
        print("duplicate begin:", i)
        run(args)
        print("duplicate over.")


if __name__ == '__main__':
    server = "local"
    args = parser.parse_args()

    run_dup(server, args)
