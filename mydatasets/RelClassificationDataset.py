from torch.utils.data import Dataset
from typing import List
import torch
import numpy as np
import json
import logging
from tqdm import tqdm
from mydatasets.InputExample import SeqClassificationInputExample
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
import itertools
from transformers.data.data_collator import DataCollatorWithPadding



class RelClassificationDataset(Dataset):

    def __init__(self, examples: List[SeqClassificationInputExample], tokenizer: PreTrainedTokenizer, max_seq_len: int,
                 label_map=None, show_progress_bar: bool = None):
        """
        Create a new Dataset with the tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = (
                        logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.convert_input_examples(examples, tokenizer)
        self.padding_label = -1
        self.task_id = None

    def set_task_id(self, task_id):
        self.task_id = task_id

    def get_task_id(self):
        return self.task_id

    def convert_input_examples(self, examples: List[SeqClassificationInputExample], tokenizer: PreTrainedTokenizer):
        """
        """

        inputs = []
        labels = []

        label_type = torch.long
        iterator = examples

        if self.label_map is None:
            # get all labels in the dataset
            all_labels = list(set([example.label for example in examples]))
            # add special labels
            all_labels.sort()
            self.label_map = {label: idx for idx, label in enumerate(all_labels)}
            logging.info('Dataset has {} labels: {}'.format(len(self.label_map), self.label_map))

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")

        for ex_index, example in enumerate(iterator):


            labels.append(self.label_map[example.label])
            print(tokenizer.vocab_size)
            tokenized_seq = tokenizer.encode_plus(text=example.seq,  padding='longest', max_length=self.max_seq_len)
            for elm in tokenized_seq['input_ids']:
                if elm > 28996:
                    for tid, t in zip(tokenized_seq['input_ids'], tokenizer.convert_ids_to_tokens(tokenized_seq['input_ids'])):
                        print(tid, t)
            inputs.append(tokenized_seq)

            if ex_index < 5:
                logging.info('Ex {}'.format(ex_index))
                logging.info('Input seq: {}'.format(example.seq))
                logging.info('--> {}'.format(tokenized_seq))
                logging.info('--> {}'.format(tokenizer.convert_ids_to_tokens(tokenized_seq['input_ids'])))
                logging.info('Label: {}'.format(example.label))
                logging.info('--> {}'.format(self.label_map[example.label]))

        tensor_labels = torch.tensor(labels, dtype=label_type)

        logging.info("Num sentences: %d" % (len(examples)))
        self.tokens = inputs
        self.labels = tensor_labels

    def __getitem__(self, i):
        elm = dict(self.tokens[i])
        elm.update({'task_id': self.task_id})
        elm.update({'label': self.labels[i]})
        return elm

    def __len__(self):
        return len(self.tokens)

    def collate_fn(self, data):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)(data)




def read_relation_classification_examples(fname):
    data = []

    with open(fname) as f:
        for line in f:
            j = json.loads(line.strip())
            data.append(SeqClassificationInputExample(guid=j['uid'], text=j['seq'], label=j['label']))
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


    ds = 'gad'
    split = 'test'

    fname = os.path.join(config.get('Files', 'preproc_data'), '{}_{}.jsonl'.format(ds, split))
    data = read_relation_classification_examples(fname)

    tokenizer = setup_customized_tokenizer(model='bert-base-multilingual-cased', do_lower_case=False, config=config,
                                     tokenizer_class=BertTokenizer)

    train_data = RelClassificationDataset(data, tokenizer=tokenizer, max_seq_len=128)
    train_data.set_task_id(0)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8, collate_fn=train_data.collate_fn)
    for elm in train_dataloader:
        print(elm)