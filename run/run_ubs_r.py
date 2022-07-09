import os
import random

import torch
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
    model_dir, model_file_name, result_dir, result_version_name = save_config(args, "ubs")

    # ====================== load data ======================= #
    benign_train_loader, fraud_train_loader, valid_loader, test_loader, in_dim = load_from_sequence_for_ubs_r(
        args.server, args.dataset, args.benign_train_batch_size, args.fraud_train_batch_size,
        args.train_batch_num, args.test_batch_size, args.benign_train_data_len)


    # ====================== config trainer ================= #
    callbacks = [ModelCheckpoint(save_weights_only=True,
                                 dirpath=model_dir,
                                 filename=model_file_name,
                                 mode=args.checkpoint_mode,
                                 monitor=args.checkpoint_monitor),
                 LearningRateMonitor(log_momentum=True)]
    if args.is_early_stopping:
        callbacks.append(EarlyStopping(patience=args.patience,
                                       mode="min",
                                       monitor="train_loss"))

    trainer = pl.Trainer(callbacks=callbacks,
                         logger=[TensorBoardLogger(save_dir=result_dir, name="", version=result_version_name)],
                         gpus=args.cuda if args.cuda != -1 else 0,
                         max_epochs=args.max_epochs)

    # Check whether pretrained model exists. If yes, load it and skip training
    if args.pretrained_file_path and os.path.isfile(args.pretrained_file_path):
        print("Found pretrained model, loading...")
        model = UBSR.load_from_checkpoint(args.pretrained_file_path, type_num=args.type_num,
                                          query_feature_dim=args.query_feature_dim, support_feature_dim=in_dim,
                                          time_dim=args.time_dim, position_dim=args.position_dim,
                                          walk_emb_dim=args.walk_emb_dim, hidden_dim=args.hidden_dim,
                                          event_embed_dim=args.event_embed_dim, time_normalize=args.time_normalize,
                                          flow_layer_num=args.flow_layer_num, ode_layer_num=args.ode_layer_num,
                                          sample_num=args.sample_num,
                                          max_seq_len = args.max_seq_len, lr=args.lr, weight_decay=args.weight_decay,
                                          comment=args.comment)
        trainer.test(model=model, dataloaders=test_loader)
    else:
        model = UBSR(type_num=args.type_num, query_feature_dim=args.query_feature_dim, support_feature_dim=in_dim,
                     time_dim=args.time_dim, position_dim=args.position_dim,  walk_emb_dim=args.walk_emb_dim,
                     walk_length=args.walk_length, graph_layers=args.graph_layers,
                     hidden_dim=args.hidden_dim, event_embed_dim=args.event_embed_dim, time_normalize=args.time_normalize,
                     flow_layer_num=args.flow_layer_num, ode_layer_num=args.ode_layer_num,
                     sample_num=args.sample_num, max_seq_len = args.max_seq_len,
                     lr=args.lr, weight_decay=args.weight_decay, comment=args.comment)
        trainer.fit(model, train_dataloaders={"benign": benign_train_loader, "fraud": fraud_train_loader},
                    val_dataloaders=valid_loader)
        trainer.test(dataloaders=test_loader)


def run_dup(dataset, server, args):
    duplicate = 10
    for i in range(duplicate):
        args = process_args(args, dataset, server)
        args.seed = random.randint(0, 10000)
        print("seed:", args.seed)
        print("duplicate begin:", i)
        run(args)
        print("duplicate over.")


if __name__ == '__main__':
    dataset = "paysim"
    server = "33"
    args = parser.parse_args()
    run_dup(dataset, server, args)

