#encoding:utf-8
import torch
import warnings
from pyernie.train.metrics import F1Score
from pyernie.train.losses import CrossEntropy
from pyernie.train.trainer import Trainer
from torch.utils.data import DataLoader
from pyernie.io.dataset import CreateDataset
from pyernie.utils.logginger import init_logger
from pyernie.utils.utils import seed_everything
from pyernie.callback.lrscheduler import BertLR
from pyernie.model.nn.ernie_fine import ErnieFine
from pyernie.io.data_transformer import DataTransformer
from pyernie.train.metrics import ClassReport,Accuracy
from pyernie.config.basic_config import configs as config
from pyernie.callback.modelcheckpoint import ModelCheckpoint
from pyernie.callback.trainingmonitor import TrainingMonitor
from pyernie.model.ernie.optimization import BertAdam
warnings.filterwarnings("ignore")

# 主函数
def main():
    # **************************** 基础信息 ***********************
    logger = init_logger(log_name=config['arch'], log_dir=config['log_dir'])
    logger.info("seed is %d"%config['seed'])
    device = 'cuda:%d' % config['n_gpu'][0] if len(config['n_gpu']) else 'cpu'
    seed_everything(seed=config['seed'],device=device)
    logger.info('starting load data from disk')
    config['id_to_label'] = {v:k for k,v in config['label_to_id'].items()}
    target_names = [config['id_to_label'][x] for x in range(len(config['label_to_id']))]
    # **************************** 数据生成 ***********************
    # data_transformer = DataTransformer(logger      = logger,
    #                                    label_to_id = config['label_to_id'],
    #                                    train_file  = config['train_file_path'],
    #                                    valid_file  = config['valid_file_path'],
    #                                    valid_size  = config['valid_size'],
    #                                    seed        = config['seed'],
    #                                    shuffle     = True,
    #                                    skip_header = False,
    #                                    preprocess  = None,
    #                                    raw_data_path=config['raw_data_path'])
    # 读取数据集以及数据划分
    # data_transformer.read_data()
    # train
    train_dataset   = CreateDataset(data_path    = config['train_file_path'],
                                    vocab_path   = config['vocab_path'],
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'train')
    # valid
    valid_dataset   = CreateDataset(
                                    data_path    = config['valid_file_path'],
                                    vocab_path   = config['vocab_path'],
                                    max_seq_len  = config['max_seq_len'],
                                    seed         = config['seed'],
                                    example_type = 'valid'
                                    )
    #加载训练数据集
    train_loader = DataLoader(dataset     = train_dataset,
                              batch_size  = config['batch_size'],
                              num_workers = config['num_workers'],
                              shuffle     = True,
                              drop_last   = False,
                              pin_memory  = False)
    # 验证数据集
    valid_loader = DataLoader(dataset     = valid_dataset,
                              batch_size  = config['batch_size'],
                              num_workers = config['num_workers'],
                              shuffle     = False,
                              drop_last   = False,
                              pin_memory  = False)

    # **************************** 模型 ***********************
    logger.info("initializing model")
    model = ErnieFine.from_pretrained(config['ernie_model_dir'],
                                    num_classes = len(config['label_to_id']))

    # ************************** 优化器 *************************
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_steps = int(
        len(train_dataset.examples) / config['batch_size'] / config['gradient_accumulation_steps'] * config['epochs'])
    # t_total: total number of training steps for the learning rate schedule
    # warmup: portion of t_total for the warmup
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = config['learning_rate'],
                         warmup = config['warmup_proportion'],
                         t_total = num_train_steps)

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    # 模型保存
    model_checkpoint = ModelCheckpoint(checkpoint_dir   = config['checkpoint_dir'],
                                       mode = config['mode'],
                                       monitor = config['monitor'],
                                       save_best_only = config['save_best_only'],
                                       arch = config['arch'],
                                       logger = logger)
    # 监控训练过程
    train_monitor = TrainingMonitor(file_dir = config['figure_dir'],arch = config['arch'])
    # 学习率机制
    lr_scheduler = BertLR(optimizer=optimizer,
                          learning_rate = config['learning_rate'],
                          t_total = num_train_steps,
                          warmup = config['warmup_proportion'])

    # **************************** training model ***********************
    logger.info('training model....')
    train_configs = {
        'model': model,
        'logger': logger,
        'optimizer': optimizer,
        'resume': config['resume'],
        'epochs': config['epochs'],
        'n_gpu': config['n_gpu'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'epoch_metrics':[F1Score(average='macro',task_type='multiclass'),
                         ClassReport(target_names=target_names)],
        'batch_metrics':[Accuracy(topK=1)],
        'criterion': CrossEntropy(),
        'model_checkpoint': model_checkpoint,
        'training_monitor': train_monitor,
        'lr_scheduler': lr_scheduler,
        'early_stopping': None,
        'verbose': 1
    }
    trainer = Trainer(train_configs=train_configs)
    # 拟合模型
    trainer.train(train_data = train_loader,valid_data=valid_loader)
    # 释放显存
    if len(config['n_gpu']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
