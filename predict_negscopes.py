import torch

import random
from models.evaluation import batch_to_device
import numpy as np

from mydatasets.NegScopeDataset import NegScopeDataset, read_examples
from mydatasets.load_data import get_data
import configparser
from transformers import BertTokenizer
import os
from util import create_logger

from torch.utils.data import DataLoader
import argparse
import uuid
from models.tasks import load_test_task
from models.MTLModel import  MTLModel
from heatmap import html_heatmap

def load_data(fname, task, tokenizer):
    data = NegScopeDataset(read_examples(fname, with_labels=False), tokenizer=tokenizer, split_seqs=task.split_seqs, max_seq_len=task.max_seq_len, with_labels=False)
    return data


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

    # create a logger
    logger = create_logger(__name__, to_disk=True, log_file='{}/{}'.format(outdir, args.logfile))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = BertTokenizer.from_pretrained('/'.join(args.model_checkpoint.split('/')[:-1]))

    test_dataloaders = []
    tasks = []
    for ds in args.test_datasets.split(','):

        task = load_test_task(args.task_spec)
        task.task_id = 0
        task.num_labels = 5

        test_data = load_data(os.path.join(args.datapath, ds), task=task, tokenizer=tokenizer)
        tasks.append(task)


        test_data.set_task_id(task.task_id)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs, collate_fn=test_data.collate_fn)

        test_dataloaders.append(test_dataloader)


    padding_label = test_data.padding_label
    model = MTLModel(bert_encoder='small_bert', device='cpu', tasks=tasks, padding_label_idx=padding_label, load_checkpoint=True,
                     checkpoint=args.model_checkpoint, tokenizer=tokenizer)


    for task, dl in zip(tasks, test_dataloaders):

        logger.info('Predicting negation scopes for {}, writing results to {}'.format(task.dataset, outdir))
        scopes, seqs = predict_negation_scopes(dl, model, task)
        with open(os.path.join(outdir, '{}_scopes.html'.format(task.dataset)), 'w') as fout:
            for scope, seq in zip(scopes, seqs):
                final_labels, final_seq = rejoin_subwords(model.tokenizer.convert_ids_to_tokens(seq), scope)
                #get the label_map associated with the output layer used for classification
                label_map = {val: key for key, val in model.label_map[task.task_id].items()}
                final_labels = [label_map[elm]  if elm in label_map else elm for elm in final_labels]
                scores = []
                for l in final_labels:
                    if l == 'O':
                        scores.append(0)
                    elif l == 'CUE':
                        scores.append(1)
                    else:
                        scores.append(-1)
                print(final_seq)
                print(final_labels)
                print('\n')
                html_str = html_heatmap(words=final_seq + ['<br>'], scores = scores + [0])
                fout.write(html_str + '\n')
        fout.close()




def rejoin_subwords(seq, labels):
    """
    strip cls and sep tokens
    assign cue label for cue tokens
    join subword tokens. assign the label of the first subtoken as label for the re-joined token
    """
    seq = seq[1:-1]
    labels = labels[1:-1]
    final_labels = []
    final_toks = []
    i = 0
    while i < len(seq):
        tok = seq[i]
        label = labels[i]
        if tok == '[CUE]':
            i += 1
            final_toks.append(seq[i])
            final_labels.append('CUE')
            i += 1
        elif tok.startswith('##'):
            final_toks[-1] = final_toks[-1] + tok.split('##')[-1]
            i += 1
        else:
            final_toks.append(tok)
            final_labels.append(label)
            i += 1
    assert len(final_labels) == len(final_toks)
    return final_labels, final_toks





def predict_negation_scopes(data_loader, model, task):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cropped_preds = []
    seqs = []

    for step, batch in enumerate(data_loader):
        # batch to device
        batch = batch_to_device(batch, device)

        # perform forward pass
        with torch.no_grad():
            loss, output = model(task=task, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            out_probs = output.data.cpu()
            out_probs = out_probs.numpy()

            mask = batch['attention_mask']

            preds = np.argmax(out_probs, axis=-1).reshape(mask.size()).tolist()

            # only take into account the predictions for non PAD tokens
            valid_length = mask.sum(1).tolist()

            final_predict = []
            for idx, p in enumerate(preds):
                final_predict.append(p[:int(valid_length[idx])])

            seq = []
            for idx, s in enumerate(batch['input_ids']):
                seq.append(s[:int(valid_length[idx])])

            cropped_preds.extend(final_predict)
            seqs.extend(seq)

    return cropped_preds, seqs


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Train MTL model')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    parser.add_argument('--model_checkpoint', type=str,
                        default='/home/mareike/PycharmProjects/multiNegScope/models/checkpoints/7196c9d2-ee59-4572-b39c-298e7c6c4b1b/best_model/model_0.pt',
                        help="The trained model used to predict the test data")
    parser.add_argument('--config',
                        default='./preprocessing/config.cfg')
    parser.add_argument('--datapath', type=str,
                        help="Folder containing datasets to be predicted",
                        default='/home/mareike/PycharmProjects/multiNegScope/data/preprocessed')
    parser.add_argument('--test_datasets', type=str,
                        help="Dataset to be predicted", default='iulaconv_test#cues.jsonl')
    parser.add_argument('--task_spec',
                        default='./task_specs/iulaconv_test#cues.yml', help='Yml file with task specification')
    parser.add_argument('--outdir', type=str,
                        help="output path", default='checkpoints/results')
    parser.add_argument('--logfile', type=str,
                        help="name of log file", default='mtl_model.log')
    parser.add_argument('--bs', type=int, default=1,
                        help="Batch size")
    args = parser.parse_args()

    main(args)