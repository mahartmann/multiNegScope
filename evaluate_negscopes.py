import torch
import random
import numpy as np

from util import dump_json
from mydatasets.load_data import get_data
import configparser
from transformers import BertTokenizer
import os
from util import create_logger

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


    outdir = args.outdir
    if not os.path.isdir(outdir):
        # make the output dir
        os.makedirs(outdir)

    # create a logger
    logger = create_logger(__name__, to_disk=True, log_file='{}/{}'.format(outdir, args.logfile))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = BertTokenizer.from_pretrained('/'.join(args.model_checkpoint.split('/')[:-1]))

    test_dataloaders = []
    tasks = []
    ds_names = []
    for ds in args.test_datasets.split(','):

        task_name = ''.join(ds.split('_')[:-1])

        splt = ds.split('_')[-1]
        ds_names.append(ds)
        task = load_task(os.path.join(args.task_spec, '{}.yml'.format(task_name)))
        task.task_id = 0
        task.num_labels = 5

        test_data = get_data(task=task, split=splt, config=config, tokenizer=tokenizer)
        task.set_label_map(test_data.label_map)
        tasks.append(task)


        test_data.set_task_id(task.task_id)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs, collate_fn=test_data.collate_fn)

        test_dataloaders.append(test_dataloader)


    padding_label = test_data.padding_label

    model = MTLModel(bert_encoder='small_bert', device='cpu', tasks=tasks, padding_label_idx=padding_label, load_checkpoint=True,
                     checkpoint=args.model_checkpoint, tokenizer=tokenizer)


    for task, dl, ds in zip(tasks, test_dataloaders, ds_names):
        logger.info('Evaluating {} with output layer {}'.format(task.dataset, task.task_id))
        results = model.evaluate_on_dev(dl, task)
        test_score, test_report, test_predictions = results['score'], results['results'], results[
            'predictions']
        # dump to file
        logger.info('Dumping results to {}'.format(os.path.join(outdir, 'results_{}.json'.format(ds))))
        dump_json(fname=os.path.join(outdir, 'results_{}.json'.format(ds)),
                  data={'f1': test_score, 'report': test_report, 'predictions': test_predictions})




if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Train MTL model')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    parser.add_argument('--model_checkpoint', type=str,
                        default='./models/checkpoints/7196c9d2-ee59-4572-b39c-298e7c6c4b1b/best_model/model_0.pt',
                        help="The trained model used to predict the test data")
    parser.add_argument('--config',
                        default='./preprocessing/config.cfg')
    parser.add_argument('--test_datasets', type=str,
                        help="List of test datasets to be predicted", default='iulaconv_test')
    parser.add_argument('--task_spec',
                        default='./task_specs', help='Directory with task specifications')
    parser.add_argument('--outdir', type=str,
                        help="output path", default='checkpoints/results')
    parser.add_argument('--logfile', type=str,
                        help="name of log file", default='mtl_model.log')
    parser.add_argument('--bs', type=int, default=8,
                        help="Batch size")

    args = parser.parse_args()

    main(args)