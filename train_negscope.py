import configparser
import argparse
from transformers import BertTokenizer
import os

from util import create_logger, bool_flag
from models.tokenization import setup_customized_tokenizer
from models.MTLModel import MultiTaskDataset,MTLModel,Task, MultiTaskBatchSampler
from models.optimization import get_scheduler, get_optimizer
from mydatasets.NegScopeDataset import NegScopeDataset, read_examples
from torch.utils.data import DataLoader

import random
import numpy as np
import torch

import uuid




def main(args):

    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # find out device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)


    # check if output dir exists. if so, assign a new one
    if os.path.isdir(args.outdir):
        # create new output dir
        outdir = os.path.join(args.outdir, str(uuid.uuid4()))
    else:
        outdir = args.outdir


    # make the output dir
    os.makedirs(outdir)
    if args.save_best:
        os.makedirs(os.path.join(outdir, 'best_model'))

    # create a logger
    logger = create_logger(__name__, to_disk=True, log_file='{}/{}'.format(outdir, args.logfile))
    logger.info('Created new output dir {}'.format(outdir))
    logger.info('Running experiments on {}'.format(device))
    # get config with all data locations
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    tokenizer = setup_customized_tokenizer(model='bert-base-multilingual-cased', do_lower_case=False, config=config,
                                           tokenizer_class=BertTokenizer)
    task1 = Task(type='seqlabeling', num_labels=5, dropout_prob=0.1, splits=['train', 'dev', 'test'], dataset='iula', split_seqs=False, max_seq_len=512)
    task2 = Task(type='seqlabeling', num_labels=5, dropout_prob=0.1, splits=['train', 'dev', 'test'], dataset='french', split_seqs=False, max_seq_len=512)
    tasks = [task1, task2]

    train_datasets = {}
    dev_dataloaders = {}
    test_dataloaders = {}

    for task_id, task in enumerate(tasks):

        task.set_task_id(task_id)
        if 'train' in task.splits:
            train_datasets[task_id] = NegScopeDataset(
                read_examples(os.path.join(config.get('Files', 'preproc_data'), '{}_train.tsv'.format(task.dataset))),
                tokenizer=tokenizer, max_seq_len=task.max_seq_len, split_seqs=task.split_seqs)
            train_datasets[task_id].set_task_id(task_id)
            task.label_map = train_datasets[task_id].label_map
            task.reverse_label_map = {val: key for key, val in train_datasets[task_id].label_map.items()}
        if 'dev' in task.splits:
            dev_data = NegScopeDataset(
                read_examples(os.path.join(config.get('Files', 'preproc_data'), '{}_dev.tsv'.format(task.dataset))),
                tokenizer=tokenizer, max_seq_len=task.max_seq_len, split_seqs=task.split_seqs)
            dev_data.set_task_id(task_id)
            dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.bs_prediction, collate_fn=dev_data.collate_fn)
            dev_dataloaders[task_id] = dev_dataloader

        if 'test' in task.splits:
            test_data = NegScopeDataset(
                read_examples(os.path.join(config.get('Files', 'preproc_data'), '{}_test.tsv'.format(task.dataset))),
                tokenizer=tokenizer, max_seq_len=task.max_seq_len, split_seqs=task.split_seqs)
            test_data.set_task_id(task_id)
            test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs_prediction, collate_fn=test_data.collate_fn)
            test_dataloaders[task_id] = test_dataloader

    padding_label = train_datasets[0].padding_label

    sorted_train_datasets = [ds for _, ds in sorted(train_datasets.items())]

    mtl_dataset = MultiTaskDataset(sorted_train_datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(sorted_train_datasets, batch_size=args.bs, mix_opt=args.mix_opt, extra_task_ratio=args.extra_task_ratio,
                                                     annealed_sampling=args.annealed_sampling, max_epochs=args.epochs)
    mtl_train_dataloader = DataLoader(mtl_dataset, batch_sampler=multi_task_batch_sampler,
                                      collate_fn=train_datasets[0].collate_fn, pin_memory=False)

    model = MTLModel(checkpoint='small_bert', device=device, tasks=tasks, padding_label_idx=padding_label)

    # get optimizer
    optimizer = get_optimizer(model, lr=args.lr, eps=args.eps, decay=args.decay)



    # get lr schedule
    total_steps = len(mtl_dataset) * args.epochs
    warmup_steps = args.warmup_frac * total_steps
    logger.info('Scheduler: {} with {} warmup steps'.format('warmuplinear', warmup_steps))
    scheduler = get_scheduler(optimizer, scheduler='warmuplinear', warmup_steps=warmup_steps, t_total=total_steps)

    model.fit(tasks, optimizer, scheduler, train_dataloader=mtl_train_dataloader,
              dev_dataloaders=dev_dataloaders, test_dataloaders=test_dataloaders, epochs=args.epochs,
              evaluation_step=args.eval_step, save_best=args.save_best, outdir=outdir, predict=args.predict)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default='preprocessing/config.cfg')
    parser.add_argument("--save_best", type=bool_flag, default=True)
    parser.add_argument("--predict", type=bool_flag, default=True)
    parser.add_argument("--outdir", type=str, default='output')
    parser.add_argument("--logfile", type=str, default='ner-models.log')
    parser.add_argument("--bert_model", type=str, default='test_config',
                        choices=['bert-base-multilingual-cased', 'distilbert', 'test_config'])
    parser.add_argument("--model_path", type=str, default='/home/mareike/PycharmProjects/anydomainbert/code/anydomainbert/out')
    parser.add_argument("--model_name", type=str,
                        default='checkpoint-20')
    parser.add_argument("--load_from_disk", type=bool_flag,  default=False)
    parser.add_argument("--split_seqs", type=bool_flag, default=False, help='if set to true, sequences are split rather than truncated')
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=20)
    parser.add_argument("--bs_prediction", type=int, default=8)
    parser.add_argument("--ds", type=str, default='iula',
                        choices=['biofull', 'bioabstracts', 'bio',
                'sherlocken', 'sherlockzh',
                'iula', 'sfuen', 'sfues',
                 'ita', 'socc', 'dtneg', 'nubes',
                'biofullall', 'bioabstractsall', 'bioall',
                'sherlockenall', 'sherlockzhall',
                'iulaall',
                'sfuenall','sfuesall',
                'itaall',
                'nubesall',
                'dtnegall', 'soccall',
                'french', 'frenchall'])
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--annealed_sampling', type=float, default=0,
                        help='Factor for annealed sampling, where tasks are sampled proportional to ds size towards beginning of training. If 0, sampling strategy is to draw proportional to ds size')

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default='warmuplinear')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument('--evaluation_step', type=int, default=10,
                        help="Evaluate every n training steps")
    args = parser.parse_args()
    main(args)
