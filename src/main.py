# coding=utf-8

import argparse
import logging
import sys
import numpy as np
import os


def main():
    # =========================
    # 1. Initial args
    # =========================
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--rank', type=int, default=1,
                             help='1=ranking, 0=rating/click')
    init_parser.add_argument('--data_loader', type=str, default='DataLoader',
                             help='Choose data_loader')
    init_parser.add_argument('--model_name', type=str, default='BaseModel',
                             help='Choose model to run.')
    init_parser.add_argument('--runner', type=str, default='BaseRunner',
                             help='Choose runner')
    init_parser.add_argument('--data_processor', type=str, default='DataProcessor',
                             help='Choose data_processor')
    init_parser.add_argument('--gpu', type=str, default='0',
                             help='GPU id. Use 0 for first GPU, -1 for CPU.')

    init_args, init_extras = init_parser.parse_known_args()

    # =========================
    # 2. Set CUDA_VISIBLE_DEVICES before importing torch/models
    # =========================
    if init_args.gpu == '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = init_args.gpu

    import torch

    # Important for GRU4Rec:
    # keeps CUDA/GPU enabled, but disables cuDNN backend for GRU/LSTM
    # to avoid crashes in older PyTorch/CUDA/cuDNN combinations.
    if init_args.model_name == 'GRU4Rec':
        torch.backends.cudnn.enabled = False

    from utils import utils

    from data_loaders.DataLoader import DataLoader

    from models.BaseModel import BaseModel
    from models.RecModel import RecModel
    from models.NCR import NCR
    from models.NCRExplicitFeedback import NCRExplicitFeedback
    from models.NCREwoLR import NCREwoLR
    from models.CompareModel import CompareModel
    from models.GRU4Rec import GRU4Rec
    from models.GRU4RecHidasi import GRU4RecHidasi
    from models.SASRec import SASRec
    from models.BERT4Rec import BERT4Rec
    from models.STAMP import STAMP
    from models.BiasedMF import BiasedMF
    from models.SVDPP import SVDPP
    from models.DMF import DMF
    from models.NeuMF import NeuMF

    from runners.BaseRunner import BaseRunner
    from runners.ProLogicRunner import ProLogicRunner

    from data_processor.DataProcessor import DataProcessor
    from data_processor.ProLogicRecDP import ProLogicRecDP
    from data_processor.HisDataProcessor import HisDataProcessor

    # =========================
    # 3. Available classes
    # =========================
    data_loader_dict = {
        'DataLoader': DataLoader
    }

    model_dict = {
        'BaseModel': BaseModel,
        'RecModel': RecModel,
        'NCR': NCR,
        'NCRExplicitFeedback': NCRExplicitFeedback,
        'NCR-E': NCRExplicitFeedback,
        'NCRE': NCRExplicitFeedback,
        'NCR-E-WoLR': NCREwoLR,
        'NCREwoLR': NCREwoLR,
        'CompareModel': CompareModel,
        'GRU4Rec': GRU4Rec,
        'GRU4RecHidasi': GRU4RecHidasi,
        'SASRec': SASRec,
        'BERT4Rec': BERT4Rec,
        'STAMP': STAMP,
        'BiasedMF': BiasedMF,
        'BPRMF': BiasedMF,
        'BPR-MF': BiasedMF,
        'SVDPP': SVDPP,
        'SVD++': SVDPP,
        'DMF': DMF,
        'NeuMF': NeuMF
    }

    runner_dict = {
        'BaseRunner': BaseRunner,
        'ProLogicRunner': ProLogicRunner
    }

    data_processor_dict = {
        'DataProcessor': DataProcessor,
        'ProLogicRecDP': ProLogicRecDP,
        'HisDataProcessor': HisDataProcessor
    }

    # =========================
    # 4. Choose data loader
    # =========================
    if init_args.data_loader not in data_loader_dict:
        print("Unknown DataLoader: %s" % init_args.data_loader)
        return
    data_loader_name = data_loader_dict[init_args.data_loader]

    # =========================
    # 5. Choose model
    # =========================
    if init_args.model_name not in model_dict:
        print("Unknown Model: %s" % init_args.model_name)
        return
    model_name = model_dict[init_args.model_name]

    # =========================
    # 6. Choose runner
    # =========================
    if init_args.model_name in ['NCR', 'NCRExplicitFeedback', 'NCR-E', 'NCRE', 'NCR-E-WoLR', 'NCREwoLR']:
        init_args.runner_name = 'ProLogicRunner'
    else:
        init_args.runner_name = 'BaseRunner'

    runner_name = runner_dict[init_args.runner_name]

    # =========================
    # 7. Choose data processor
    # =========================
    if init_args.model_name in ['SVDPP', 'SVD++']:
        init_args.data_processor = 'HisDataProcessor'
    elif init_args.model_name in ['NCR', 'NCRExplicitFeedback', 'NCR-E', 'NCRE', 'NCR-E-WoLR', 'NCREwoLR', 'RNNModel',
                                  'CompareModel', 'GRU4Rec', 'GRU4RecHidasi', 'SASRec', 'BERT4Rec', 'STAMP']:
        init_args.data_processor = 'ProLogicRecDP'

    if init_args.data_processor not in data_processor_dict:
        print("Unknown DataProcessor: %s" % init_args.data_processor)
        return
    data_processor_name = data_processor_dict[init_args.data_processor]

    # =========================
    # 8. Full parser
    # =========================
    parser = argparse.ArgumentParser(description='')
    parser = utils.parse_global_args(parser)
    parser = data_loader_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)

    args, extras = parser.parse_known_args()

    # =========================
    # 9. Log/model/result filenames
    # =========================
    log_file_name = [
        str(init_args.rank),
        init_args.model_name,
        args.dataset,
        str(args.random_seed),
        'optimizer=' + args.optimizer,
        'lr=' + str(args.lr),
        'l2=' + str(args.l2),
        'dropout=' + str(args.dropout),
        'batch_size=' + str(args.batch_size)
    ]

    log_file_name = '__'.join(log_file_name).replace(' ', '__')

    if args.log_file == '../log/log.txt':
        args.log_file = '../log/%s.txt' % log_file_name

    if args.result_file == '../result/result.npy':
        args.result_file = '../result/%s.npy' % log_file_name

    if args.model_path == '../model/%s/%s.pt' % (init_args.model_name, init_args.model_name):
        args.model_path = '../model/%s/%s.pt' % (init_args.model_name, log_file_name)

    # Create folders if they do not exist
    for file_path in [args.log_file, args.result_file, args.model_path]:
        folder = os.path.dirname(file_path)
        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

    # =========================
    # 10. Logging
    # =========================
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(vars(init_args))
    logging.info(vars(args))

    logging.info('DataLoader: ' + init_args.data_loader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner_name)
    logging.info('DataProcessor: ' + init_args.data_processor)

    # =========================
    # 11. Random seed
    # =========================
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # =========================
    # 12. Device
    # =========================
    use_gpu = init_args.gpu != '-1' and torch.cuda.is_available()

    if use_gpu:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        logging.info("# cuda devices: %d" % torch.cuda.device_count())
        logging.info("Using GPU: %s" % torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        logging.info("# cuda devices: %d" % torch.cuda.device_count())
        logging.info("Using CPU")

    # =========================
    # 13. Create data loader
    # =========================
    data_loader = data_loader_name(
        path=args.path,
        dataset=args.dataset,
        label=args.label,
        sep=args.sep
    )

    features, feature_dims, feature_min, feature_max = data_loader.feature_info(
        include_id=model_name.include_id,
        include_item_features=model_name.include_item_features,
        include_user_features=model_name.include_user_features
    )

    # =========================
    # 14. Create model
    # =========================
    if init_args.model_name in ['BaseModel']:
        model = model_name(
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=len(features),
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    elif init_args.model_name in ['RecModel', 'BiasedMF', 'BPRMF', 'BPR-MF', 'SVDPP', 'SVD++']:
        model = model_name(
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    elif init_args.model_name in ['NeuMF']:
        model = model_name(
            neumf_mlp_layers=args.neumf_mlp_layers,
            neumf_mlp_vector_size=args.neumf_mlp_vector_size,
            neumf_mf_vector_size=args.neumf_mf_vector_size,
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    elif init_args.model_name in ['DMF']:
        model = model_name(
            train_df=data_loader.train_df,
            label=args.label,
            dmf_user_embedding_size=args.dmf_user_embedding_size,
            dmf_item_embedding_size=args.dmf_item_embedding_size,
            dmf_user_layers=args.dmf_user_layers,
            dmf_item_layers=args.dmf_item_layers,
            dmf_inter_matrix_type=args.dmf_inter_matrix_type,
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    elif init_args.model_name in ['GRU4Rec', 'STAMP']:
        sequential_kwargs = dict(
            neg_emb=args.neg_emb,
            neg_layer=args.neg_layer,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            p_layers=args.p_layers,
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            random_seed=args.random_seed,
            model_path=args.model_path
        )
        if init_args.model_name == 'STAMP':
            sequential_kwargs['attention_size'] = args.attention_size
        model = model_name(**sequential_kwargs)

    elif init_args.model_name == 'GRU4RecHidasi':
        model = model_name(
            hidasi_layers=args.hidasi_layers,
            hidasi_loss=args.hidasi_loss,
            hidasi_embedding=args.hidasi_embedding,
            hidasi_constrained_embedding=args.hidasi_constrained_embedding,
            hidasi_dropout_embed=args.hidasi_dropout_embed,
            hidasi_dropout_hidden=args.hidasi_dropout_hidden,
            hidasi_momentum=args.hidasi_momentum,
            hidasi_bpreg=args.hidasi_bpreg,
            hidasi_elu_param=args.hidasi_elu_param,
            hidasi_adagrad_eps=args.hidasi_adagrad_eps,
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    elif init_args.model_name == 'SASRec':
        model = model_name(
            sasrec_num_blocks=args.sasrec_num_blocks,
            sasrec_num_heads=args.sasrec_num_heads,
            sasrec_norm_first=args.sasrec_norm_first,
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    elif init_args.model_name == 'BERT4Rec':
        model = model_name(
            bert_num_blocks=args.bert_num_blocks,
            bert_num_heads=args.bert_num_heads,
            bert_intermediate_size=args.bert_intermediate_size,
            bert_pooling=args.bert_pooling,
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    elif init_args.model_name in ['NCR', 'NCRExplicitFeedback', 'NCR-E', 'NCRE', 'NCR-E-WoLR', 'NCREwoLR', 'CompareModel']:
        model = model_name(
            label_min=data_loader.label_min,
            label_max=data_loader.label_max,
            feature_num=0,
            user_num=data_loader.user_num,
            item_num=data_loader.item_num,
            u_vector_size=args.u_vector_size,
            i_vector_size=args.i_vector_size,
            r_weight=args.r_weight,
            ppl_weight=args.ppl_weight,
            pos_weight=args.pos_weight,
            random_seed=args.random_seed,
            model_path=args.model_path
        )

    else:
        logging.error('Unknown Model: ' + init_args.model_name)
        return

    # =========================
    # 15. Init model parameters
    # =========================
    if not getattr(model, 'uses_custom_initialization', False):
        model.apply(model.init_paras)

    # =========================
    # 16. Move model to GPU/CPU
    # =========================
    model = model.to(device)

    # =========================
    # 17. Append user interaction history
    # =========================
    if init_args.model_name in ['NCR', 'NCRExplicitFeedback', 'NCR-E', 'NCRE', 'NCR-E-WoLR', 'NCREwoLR', 'RNNModel',
                                'CompareModel', 'GRU4Rec', 'GRU4RecHidasi', 'SASRec', 'BERT4Rec', 'STAMP']:
        data_loader.append_his(
            last_n=args.max_his,
            supply=False,
            neg=True,
            neg_column=False
        )

    # =========================
    # 18. Ranking mode: remove observed negative labels
    # =========================
    if init_args.rank == 1:
        data_loader.drop_neg()

    # =========================
    # 19. Create data processor
    # =========================
    if init_args.data_processor in ['ProLogicRecDP']:
        data_processor = data_processor_name(
            data_loader,
            model,
            rank=init_args.rank,
            test_neg_n=args.test_neg_n,
            max_his=args.max_his,
            sup_his=0,
            sparse_his=0
        )

    elif init_args.data_processor in ['HisDataProcessor']:
        data_processor = data_processor_name(
            data_loader,
            model,
            rank=init_args.rank,
            test_neg_n=args.test_neg_n,
            sup_his=args.sup_his,
            max_his=args.max_his,
            sparse_his=args.sparse_his
        )

    else:
        data_processor = data_processor_name(
            data_loader,
            model,
            rank=init_args.rank,
            test_neg_n=args.test_neg_n
        )

    # =========================
    # 20. Create runner
    # =========================
    if init_args.runner_name in ['BaseRunner', 'ProLogicRunner']:
        runner_args = dict(
            optimizer=args.optimizer,
            learning_rate=args.lr,
            epoch=args.epoch,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            dropout=args.dropout,
            l2=args.l2,
            metrics=args.metric,
            check_epoch=args.check_epoch,
            early_stop=args.early_stop
        )
        if hasattr(args, 'fast_eval'):
            runner_args['fast_eval'] = args.fast_eval
        runner = runner_name(**runner_args)
    else:
        logging.error('Unknown Runner: ' + init_args.runner_name)
        return

    # =========================
    # 21. Test before training
    # =========================
    if not getattr(args, 'fast_eval', 0):
        logging.info(
            'Test Before Training = ' +
            utils.format_metric(
                runner.evaluate(model, data_processor.get_test_data(), data_processor)
            ) +
            ' ' +
            ','.join(runner.metrics)
        )
    else:
        logging.info('Test before training skipped (fast_eval=1)')

    # =========================
    # 22. Load model if requested
    # =========================
    if args.load > 0:
        model.load_model()

    # =========================
    # 23. Train
    # =========================
    if args.train > 0:
        runner.train(model, data_processor, skip_eval=args.skip_eval)

    # =========================
    # 24. Test after training
    # =========================
    logging.info(
        'Test After Training = ' +
        utils.format_metric(
            runner.evaluate(model, data_processor.get_test_data(), data_processor)
        ) +
        ' ' +
        ','.join(runner.metrics)
    )

    # =========================
    # 25. Save test predictions
    # =========================
    predictions = runner.predict(model, data_processor.get_test_data(), data_processor)
    np.save(args.result_file, predictions)

    logging.info('Save Test Results to ' + args.result_file)

    logging.debug(runner.evaluate(model, data_processor.get_test_data(), data_processor))
    logging.debug(runner.evaluate(model, data_processor.get_test_data(), data_processor))

    return


if __name__ == '__main__':
    main()