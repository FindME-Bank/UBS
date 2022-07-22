import os
import sys
import random

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_DIR + "/model")
sys.path.append(PARENT_DIR + "/data_handle")
sys.path.append(PARENT_DIR + "/run")
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from run import save_config, parser, process_args
from data_handle.loader import load_from_sequence_for_ubs
from model.ubs import UBS


def run(args):
    if args.seed:
        pl.seed_everything(args.seed)

    # ========================= save ======================= #
    model_dir, model_file_name, result_dir, result_version_name = save_config(args, "ubs")

    # ====================== load data ======================= #
    benign_train_loader, fraud_train_loader, valid_loader, test_loader, query_feature_dim, support_feature_dim \
        = load_from_sequence_for_ubs(args.server, args.benign_train_batch_size,
                                     args.fraud_train_batch_size, args.train_batch_num,
                                     args.test_batch_size, args.benign_train_data_len)

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


    if args.etm is False and args.tpm is True and args.hdt is True:
        comment = "UBS-w/o-ETM" if args.comment == "-" else "UBS-w/o-ETM" + args.comment
    elif args.etm is True and args.tpm is False and args.hdt is True:
        comment = "UBS-w/o-TPM" if args.comment == "-" else "UBS-w/o-TPM" + args.comment
    elif args.etm is True and args.tpm is True and args.hdt is False:
        comment = "UBS-w/o-HDT" if args.comment == "-" else "UBS-w/o-HDT" + args.comment
    else:
        comment = args.comment

    # Check whether pretrained model exists. If yes, load it and skip training
    if args.pretrained_file_path and os.path.isfile(args.pretrained_file_path):
        print("Found pretrained model, loading...")
        model = UBS.load_from_checkpoint(args.pretrained_file_path,
                                         type_num=args.type_num, query_feature_dim=query_feature_dim,
                                         support_feature_dim=support_feature_dim,  # input
                                         type_embed_dim=args.type_embed_dim, cnf_hidden_dim=args.cnf_hidden_dim,
                                         flow_layer_num=args.flow_layer_num,  # etm
                                         influence_embed_dim=args.influence_embed_dim,
                                         time_normalize=args.time_normalize,
                                         ode_layer_num=args.ode_layer_num,  # ode-rnn
                                         query_embedding_dim=args.query_embedding_dim,  # query_embedding_net
                                         predictor_hidden_dim=args.predictor_hidden_dim,  # predictor
                                         sample_num=args.sample_num,  # tpm
                                         beta=args.beta, max_seq_len=args.max_seq_len,
                                         distance_hidden_dim=args.distance_hidden_dim,  # hdt
                                         etm=args.etm, tpm=args.tpm, hdt=args.hdt,  # ablation
                                         lr=args.lr, weight_decay=args.weight_decay,
                                         comment=comment)
        trainer.test(model=model, dataloaders=test_loader)
    else:
        model = UBS(type_num=args.type_num, query_feature_dim=query_feature_dim,
                    support_feature_dim=support_feature_dim,  # input
                    type_embed_dim=args.type_embed_dim, cnf_hidden_dim=args.cnf_hidden_dim,
                    flow_layer_num=args.flow_layer_num,  # etm
                    influence_embed_dim=args.influence_embed_dim, time_normalize=args.time_normalize,
                    ode_layer_num=args.ode_layer_num,  # ode-rnn
                    query_embedding_dim=args.query_embedding_dim,  # query_embedding_net
                    predictor_hidden_dim=args.predictor_hidden_dim,  # predictor
                    sample_num=args.sample_num,  # tpm
                    beta=args.beta, max_seq_len=args.max_seq_len, distance_hidden_dim=args.distance_hidden_dim,  # hdt
                    etm=args.etm, tpm=args.tpm, hdt=args.hdt,  # ablation
                    lr=args.lr, weight_decay=args.weight_decay,
                    comment=comment)

        trainer.fit(model, train_dataloaders={"benign": benign_train_loader, "fraud": fraud_train_loader},
                    val_dataloaders=valid_loader)
        trainer.test(dataloaders=test_loader)


def run_tune_beta_exp(server, args):
    beta_list = [0.2, 0.4, 0.6, 0.8, 1]
    duplicate = 10
    for beta in beta_list:
        for i in range(duplicate):
            args = process_args(args, server)
            args.seed = random.randint(0, 10000)
            args.beta = beta
            args.comment = "beta:" + str(beta)
            print("seed:", args.seed)
            print("beta:", args.beta)
            print("duplicate begin:", i)
            run(args)
            print("duplicate over.")


def run_ubs_without_etm(server, args):
    duplicate = 10
    for i in range(duplicate):
        args = process_args(args, server)
        args.etm = False
        args.comment = "UBS-w/o-ETM" if args.comment == "-" else "UBS-w/o-ETM" + args.comment
        args.seed = random.randint(0, 10000)
        print("seed:", args.seed)
        print("duplicate UBS-w/o-ETM begin:", i)
        run(args)
        print("duplicate UBS-w/o-ETM over.")


def run_ubs_without_tpm(server, args):
    duplicate = 10
    for i in range(duplicate):
        args = process_args(args, server)
        args.tpm = False
        args.seed = random.randint(0, 10000)
        print("seed:", args.seed)
        print("duplicate UBS-w/o-TPM begin:", i)
        run(args)
        print("duplicate UBS-w/o-TPM over.")


def run_ubs_without_hdt(server, args):
    duplicate = 10
    for i in range(duplicate):
        args = process_args(args, server)
        args.hdt = False
        args.seed = random.randint(0, 10000)
        print("seed:", args.seed)
        print("duplicate UBS-w/o-HDT begin:", i)
        run(args)
        print("duplicate UBS-w/o-HDT over.")


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

    if args.run_mode == "dup":
        run_dup(server, args)
    elif args.run_mode == "beta":
        run_tune_beta_exp(server, args)
    elif args.run_mode == "without_etm":
        run_ubs_without_etm(server, args)
    elif args.run_mode == "without_tpm":
        run_ubs_without_tpm(server, args)
    elif args.run_mode == "without_hdt":
        run_ubs_without_hdt(server, args)