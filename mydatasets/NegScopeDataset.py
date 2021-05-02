from torch.utils.data import Dataset
from typing import List
import torch
import numpy as np
import logging
from tqdm import tqdm
from mydatasets.InputExample import SeqLabelingInputExample
from transformers import PreTrainedTokenizer
import itertools
from torch.utils.data import DataLoader
import json

class NegScopeDataset(Dataset):

    def __init__(self, examples: List[SeqLabelingInputExample], tokenizer: PreTrainedTokenizer, max_seq_len: int, label_map=None, show_progress_bar: bool = None,
                 split_seqs: bool=None, with_labels:bool=True):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_seq_len
        self.split_seqs = split_seqs
        self.convert_input_examples(examples, tokenizer, max_seq_len)
        self.padding_label = -1
        self.task_id = None
        self.with_labels = with_labels

    def set_task_id(self, task_id):
        self.task_id = task_id

    def get_task_id(self):
        return self.task_id

    def convert_input_examples(self, examples: List[SeqLabelingInputExample], tokenizer: PreTrainedTokenizer, max_seq_len:int):
        """
        Converts input examples to a SmartBatchingDataset usable to train the models with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param models
            the Sentence BERT models for the conversion
        :return: a SmartBatchingDataset usable to train the models with SentenceTransformer.smart_batching_collate as the collate_fn
            for the DataLoader
        """


        iterator = examples

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")

        if self.label_map is None:
            # get all labels in the dataset
            all_labels = list(set(list(itertools.chain.from_iterable([example.label for example in examples]))))
            # add special labels
            all_labels.extend(['CLS', 'SEP', 'X'])
            all_labels.sort()
            self.label_map = {label: idx for idx, label in enumerate(all_labels)}

        tokenized_seqs = []
        extended_labels = []
        c = 0
        for ex_index, example in enumerate(iterator):

            subtoks = []
            # labels need to be extended. a subtoken that starts with '#' gets label 'X'
            sublabels = []
            for tid, tok in enumerate(example.seq):
                for sid, subtok in enumerate(tokenizer.tokenize(tok)):
                    subtoks.append(subtok)
                    if sid == 0:
                        sublabels.append(example.label[tid])
                    else:
                        sublabels.append('X')

            # split the sequence if it is longer than max length. make sure to not split within a scope or within a word

            if len(subtoks) > max_seq_len-2:
                c += 1
                if self.split_seqs:
                    smaller_subtoks, smaller_sublabels = self.split_seq(subtoks, sublabels)

                else:
                    smaller_subtoks, smaller_sublabels = [subtoks[:max_seq_len-2]], [sublabels[:max_seq_len-2]]

            else:
                smaller_subtoks, smaller_sublabels = [subtoks], [sublabels]

            for subtoks, sublabels in zip(smaller_subtoks, smaller_sublabels):
                assert len(subtoks) == len(sublabels)
                # add [CLS] and [SEP] tokens

                subtoks =  [self.tokenizer.cls_token] + subtoks + [self.tokenizer.sep_token]
                sublabels = ['CLS'] + sublabels + ['SEP']

                if ex_index < 5:
                    logging.info('Ex {}'.format(ex_index))
                    logging.info('Input seq: {}'.format(subtoks))
                    logging.info('--> {}'.format(tokenizer.convert_tokens_to_ids(subtoks)))
                    logging.info('Labels: {}'.format(sublabels))
                    logging.info('--> {}'.format([self.label_map[l] for l in sublabels]))


                tokenized_seqs.append(tokenizer.convert_tokens_to_ids(subtoks))
                extended_labels.append([self.label_map[l] for l in sublabels])



        self.seqs = tokenized_seqs
        self.labels = extended_labels
        logging.info('Found {} sequences longer than max_len of {}'.format(c, self.max_len))


    def split_seq(self, subtoks, sublabels):
        """
        split sequences longer than max_length. make sure to not split within word or within entity
        :param subtoks:
        :param sublabels:
        :return:
        """

        smaller_subtoks, smaller_sublabels = [], []
        abs_end_positon = self.max_len - 2
        abs_start_position = 0
        while abs_start_position < len(subtoks):

            t_chunk = subtoks[ abs_start_position:abs_end_positon]
            l_chunk = sublabels[abs_start_position:abs_end_positon]
            start_label = '[]'
            if abs_end_positon < len(sublabels):
                start_label = sublabels[abs_end_positon]

            if start_label == 'X' or start_label == 'I':
                while start_label == 'X' or start_label == 'I':
                    start_label = l_chunk.pop()
                    t_chunk.pop()
            # this is to have a stopping criterion in case the entity is as long as the seq len (should never happen in a realistic setup)
            if len(t_chunk) == 0:
                break
            smaller_subtoks.append(t_chunk)
            smaller_sublabels.append(l_chunk)
            abs_start_position = sum([len(elm) for elm in smaller_subtoks])
            abs_end_positon = abs_start_position + self.max_len - 2

        return smaller_subtoks, smaller_sublabels


    def __getitem__(self, item):
        return {'seq':self.seqs[item], 'labels':self.labels[item], 'task_id': self.task_id}

    def __len__(self):
        return len(self.seqs)

    def collate_fn(self, data):
        seqs = [elm['seq'] for elm in data]
        labels = [elm['labels'] for elm in data]
        max_len = np.max([len(seq) for seq in seqs])
        padded_seqs = []
        padded_attention_masks = []
        padded_labels = []

        for seq, label in zip(seqs, labels):
            valid_length = np.sum([1 for elm in label if elm != self.padding_label])
            seq = seq[:valid_length]
            label = label[:valid_length]

            assert -1 not in label

            # pad the sequence and labels to max len, build attention mask
            attention_mask = [1]*len(seq)
            padded_seq = seq
            padded_label = label


            while len(padded_seq) < max_len:
                padded_seq.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
                padded_label.append(self.padding_label)
                attention_mask.append(0)
                assert len([1 for elm in attention_mask if elm == 0]) == len([1 for elm in padded_label if elm == -1])

            assert len([1 for elm in attention_mask if elm == 0]) == len([1 for elm in padded_label if elm == -1])

            padded_seqs.append(padded_seq)
            padded_labels.append(padded_label)
            padded_attention_masks.append(attention_mask)
            assert len(padded_seq) == len(padded_label) == len(attention_mask)
            assert len([1 for elm in attention_mask if elm == 0]) == len([1 for elm in padded_label if elm == -1])


        if self.with_labels:
            return {'input_ids': torch.LongTensor(padded_seqs),
                    'attention_mask': torch.LongTensor(padded_attention_masks),
                    'labels': torch.LongTensor(padded_labels),
                    'task_id': torch.LongTensor([self.task_id]*len(padded_seqs))}
        else:
            return {'input_ids': torch.LongTensor(padded_seqs),
                    'attention_mask': torch.LongTensor(padded_attention_masks),
                    'task_id': torch.LongTensor([self.task_id] * len(padded_seqs))}



def read_examples(fname, with_labels = True):
    data = []
    with open(fname) as f:
        for line in f:
            j = json.loads(line.strip())
            if with_labels:
                data.append(SeqLabelingInputExample(guid=j['uid'], text=j['seq'], label=j['labels']))
            else:
                data.append(SeqLabelingInputExample(guid=j['uid'], text=j['seq'], label=['0']*len(j['seq'])))
    return data





if __name__=="__main__":

    import configparser
    from transformers import BertTokenizer
    import json
    import os
    from util import create_logger
    from models.tokenization import setup_customized_tokenizer

    cfg = '../preprocessing/config.cfg'

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    # create a logger
    logger = create_logger(__name__, to_disk=True, log_file='{}/{}'.format('.', 'log'))


    ds = 'iula'
    split = 'test'

    fname = os.path.join(config.get('Files', 'preproc_data'), '{}_{}.tsv'.format(ds, split))
    data = read_examples(fname)

    tokenizer = setup_customized_tokenizer(model='bert-base-multilingual-cased', do_lower_case=False, config=config,
                                     tokenizer_class=BertTokenizer)

    train_data = NegScopeDataset(data, tokenizer=tokenizer, max_seq_len=512, split_seqs=False)
    train_data.set_task_id(0)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8, collate_fn=train_data.collate_fn)
    for elm in train_dataloader:
        print(elm)
