import torch

import random

import numpy as np

import logging

from models.optimization import get_optimizer, get_scheduler
from util import dump_json, bool_flag
from mydatasets.MultiTaskDataset import MultiTaskDataset, MultiTaskBatchSampler
from mydatasets.load_data import get_data
import configparser
from transformers import BertTokenizer
import os
from util import create_logger
from models.tokenization import setup_customized_tokenizer

from torch.utils.data import DataLoader
import argparse
import uuid
from models.tasks import load_task
from models.MTLModel import  MTLModel


def main(args):
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    tasks = []
    for task_name in args.tasks.split(','):
        task = load_task(os.path.join(args.task_spec, '{}.yml'.format(task_name)))
        tasks.append(task)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if not args.load_checkpoint:
        tokenizer = setup_customized_tokenizer(model=args.tokenizer, do_lower_case=False, config=config,
                                               tokenizer_class=BertTokenizer)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.checkpoint)




    train_datasets = {}
    dev_dataloaders = {}
    test_dataloaders = {}


    for task_id, task in enumerate(tasks):
        task.set_task_id(task_id)
        logging.info('Task {}: {} on {}'.format(task_id, task.task_type, task.dataset))
        if 'train' in task.splits:
            train_datasets[task_id] = get_data(task=task, split='train', config=config, tokenizer=tokenizer)
            train_datasets[task_id].set_task_id(task_id)
            task.set_label_map(train_datasets[task_id].label_map)

        if 'dev' in task.splits:
            dev_data = get_data(task=task, split='dev', config=config, tokenizer=tokenizer)
            dev_data.set_task_id(task_id)
            dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=8, collate_fn=dev_data.collate_fn)
            dev_dataloaders[task_id] = dev_dataloader

        if 'test' in task.splits:
            test_data = get_data(task=task, split='test', config=config, tokenizer=tokenizer)
            test_data.set_task_id(task_id)
            test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8, collate_fn=test_data.collate_fn)
            test_dataloaders[task_id] = test_dataloader


    padding_label = train_datasets[0].padding_label

    sorted_train_datasets =  [ds for _, ds in sorted(train_datasets.items())]

    mtl_dataset = MultiTaskDataset(sorted_train_datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(sorted_train_datasets, batch_size=args.bs, mix_opt=args.mix_opt, extra_task_ratio=args.extra_task_ratio, annealed_sampling=args.annealed_sampling, max_epochs=args.epochs)
    mtl_train_dataloader = DataLoader(mtl_dataset, batch_sampler=multi_task_batch_sampler,
                                      collate_fn=mtl_dataset.collate_fn, pin_memory=False)



    model = MTLModel(bert_encoder=args.bert_model, device=device, tasks=tasks, padding_label_idx=padding_label, load_checkpoint=args.load_checkpoint, checkpoint=os.path.join(args.checkpoint, 'model.pt'), tokenizer=tokenizer)

    # get optimizer
    # TODO: in case of loading from checkpoint, initialize optimizer using saved optimizer state dict
    optimizer = get_optimizer(optimizer_name='adamw', model=model, lr=args.lr, eps=args.eps, decay=args.decay)



    # get lr schedule
    total_steps = (len(mtl_dataset)/args.grad_accumulation_steps) * args.epochs
    warmup_steps = args.warmup_frac * total_steps
    logger.info('Bs_per_device={}, gradient_accumulation_steps={} --> effective bs= {}'.format(args.bs, args.grad_accumulation_steps, args.bs*args.grad_accumulation_steps))
    logger.info('Total steps: {}'.format(total_steps))
    logger.info('Scheduler: {} with {} warmup steps'.format('warmuplinear', warmup_steps))

    scheduler = get_scheduler(optimizer, scheduler='warmuplinear', warmup_steps=warmup_steps, t_total=total_steps)

    model.fit(tasks, optimizer, scheduler, gradient_accumulation_steps=args.grad_accumulation_steps, train_dataloader=mtl_train_dataloader,
              dev_dataloaders=dev_dataloaders, test_dataloaders=test_dataloaders, epochs=args.epochs,
              evaluation_step=args.evaluation_steps, save_best=args.save_best, outdir=outdir, predict=args.predict)



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Train MTL model')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    parser.add_argument('--bert_model', type=str,
                        default='small_bert',
                        choices=['bert-base-multilingual-cased', 'bert-base-cased', 'small_bert', 'xlm-roberta-base'],
                        help="The pre-trained encoder used to encode the entities of the analogy")
    parser.add_argument('--tokenizer', type=str,
                        default='bert-base-cased',
                        help="The tokenizer. Should be the same as bert_model. When loading from a trained checkpoint, the tokenizer is loaded from the checkpoint as well.")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/test',
                        help="A trained model checkpoint")
    parser.add_argument('--load_checkpoint', type=bool_flag, default=False,
                        help="Load a trained model checkpoint")
    parser.add_argument('--config',
                        default='../preprocessing/config.cfg')
    parser.add_argument('--task_spec',
                        default='../task_specs', help='Directory with task specifications')
    parser.add_argument('--tasks', type=str,
                        help="Yaml file specifying the training/testing tasks", default='bioconv')
    parser.add_argument('--test_tasks', type=str,
                        help="Yaml file specifying the additional testing tasks for zero-shot experiments. By default, the output layer of the first training task is used for prediction.", default='nubes')
    parser.add_argument('--outdir', type=str,
                        help="output path", default='checkpoints')
    parser.add_argument('--logfile', type=str,
                        help="name of log file", default='mtl_model.log')
    parser.add_argument('--bs', type=int, default=2,
                        help="Batch size")
    parser.add_argument('--grad_accumulation_steps', type=int, default=4,
                        help="Steps over which the gradient is accumulated")
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help="Max seq length")
    parser.add_argument('--epochs', type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument('--evaluation_steps', type=int, default=6,
                        help="Evaluate every n training steps")
    parser.add_argument("--save_best", type=bool_flag, default=True)
    parser.add_argument("--predict", type=bool_flag, default=True)

    #MTL options
    parser.add_argument('--extra_task_ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--annealed_sampling', type=float, default=0)

    # Optimization parameters
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default='warmuplinear')
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()

    main(args)