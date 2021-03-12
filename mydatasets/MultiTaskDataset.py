"""
this code is adapted from the mt-dnn implementation
"""
import logging
# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.

import random
import logging
import numpy as np

from torch.utils.data import Dataset,  BatchSampler
from models.optimization import get_scheduler, get_optimizer
from collections import Counter
from mydatasets.RelClassificationDataset import read_relation_classification_examples, RelClassificationDataset


logger = logging.getLogger(__name__)

class Task():
    def __init__(self, type, dataset, num_labels, dropout_prob, splits, max_seq_len, split_seqs=False):
        self.dataset=dataset
        self.splits = splits
        self.task_type = type
        self.num_labels = num_labels
        self.hidden_dropout_prob = dropout_prob
        self.label_map = None
        self.reverse_label_map = None
        self.max_seq_len = max_seq_len
        self.split_seqs = split_seqs
        self.task_id = None

    def set_task_id(self, tid):
        self.task_id = tid


class MultiTaskBatchSampler(BatchSampler):

    def __init__(self, datasets, batch_size, mix_opt, extra_task_ratio, annealed_sampling=0, max_epochs=2400):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        self.annealed_sampling_factor = annealed_sampling
        self.current_epoch = -1
        self.max_epochs = max_epochs
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(self._get_shuffled_index_batches(len(dataset), batch_size))
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i + batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        self.current_epoch += 1
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list, self._mix_opt, self._extra_task_ratio,
                                             self.annealed_sampling_factor, self.current_epoch, self.max_epochs)
        self.sampling_stats(all_indices)
        if self.annealed_sampling_factor == 0:
            for local_task_idx in all_indices:
                task_id = self._datasets[local_task_idx].get_task_id()
                batch = next(all_iters[local_task_idx])
                yield [(task_id, sample_id) for sample_id in batch]
        else:
            for local_task_idx, bid in all_indices:
                task_id = self._datasets[local_task_idx].get_task_id()

                batch = self._train_data_list[task_id][bid]

                yield [(task_id, sample_id) for sample_id in batch]

    def sampling_stats(self, all_indices):
        alpha = 1 - self.annealed_sampling_factor * ((self.current_epoch - 1.) / (self.max_epochs - 1.))
        if isinstance(all_indices[0], int):
            tids = [elm for elm in all_indices]
        else:
            tids = [elm[0] for elm in all_indices]
        logger.info(
            'Epoch {}, annealed sampling factor {}, alpha={}'.format(self.current_epoch, self.annealed_sampling_factor,
                                                                     alpha))
        c = Counter(tids)
        for key, val in c.most_common():
            logger.info(
                '{:.2f}% ({}) of sampled batches for task {}'.format(val / np.sum([elm for elm in c.values()]) * 100,
                                                                     val, key))

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio, annealed_sampling_factor, current_epoch,
                          max_epochs):
        all_indices = []
        num_updates = int(np.sum([len(train_data_list[i]) for i in range(len(train_data_list))]))
        if annealed_sampling_factor > 0:
            #  compute alpha for annealed sampling according to Stickland and Murray 2019
            alpha = 1 - annealed_sampling_factor * ((current_epoch - 1.) / (max_epochs - 1.))
            # factor used to make control the total number of updates
            scaling_factor = int(np.ceil(
                float(num_updates) / np.sum([len(train_data_list[i]) ** alpha for i in range(len(train_data_list))])))
            for i in range(1, len(train_data_list)):
                print('train data list {} has {} samples'.format(i, len(train_data_list[i])))
                _all_task_indices = [i] * int(np.ceil(len(train_data_list[i]) ** alpha)) * scaling_factor
                print('_all task indices {} has {} samples'.format(i, len(_all_task_indices)))
                _all_batch_indices = []
                tid = 0
                for elm in _all_task_indices:
                    # append from start if the end is reached (this is approximates shuffling with replacement)
                    if tid >= len(train_data_list[i]):
                        tid = 0
                    _all_batch_indices.append(tid)
                    tid += 1
                all_indices += [(tid, bid) for tid, bid in zip(_all_task_indices, _all_batch_indices)]
                print('task {} has {} samples'.format(i, len(all_indices)))
            if mix_opt > 0:
                random.shuffle(all_indices)
            _all_task_indices = [0] * int(np.ceil(len(train_data_list[0]) ** alpha)) * scaling_factor
            _all_batch_indices = []
            print('task 0 has {} samples'.format(i, len(_all_task_indices)))
            tid = 0
            for elm in _all_task_indices:
                # append from start if the end is reached (this approximates shuffling with replacement)
                if tid >= len(train_data_list[0]):
                    tid = 0
                _all_batch_indices.append(tid)
                tid += 1
            all_indices += [(tid, bid) for tid, bid in zip(_all_task_indices, _all_batch_indices)]
            if mix_opt < 1:
                random.shuffle(all_indices)
            # restrict the number of batches per epoch to the number resulting from sampling with alpha=0
            num_updates = int(np.sum([len(train_data_list[i]) for i in range(len(train_data_list))]))
            all_indices = all_indices[:num_updates + 1]

            print('Num updates {}'.format(num_updates))
        elif len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices)))
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]

    def collate_fn(self, batch):
        task_id = batch[0]['task_id']
        return self._task_id_2_data_set_dic[task_id].collate_fn(batch)








if __name__ == "__main__":
    import configparser
    from transformers import BertTokenizer
    import os
    from util import create_logger
    from models.tokenization import setup_customized_tokenizer
    from mydatasets.NegScopeDataset import NegScopeDataset, read_examples
    from torch.utils.data import DataLoader

    cfg = '../preprocessing/config.cfg'

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    # create a logger
    logger = create_logger(__name__, to_disk=True, log_file='{}/{}'.format('.', 'log'))

    ds = 'iula'
    split = 'test'
    tokenizer = setup_customized_tokenizer(model='bert-base-multilingual-cased', do_lower_case=False, config=config,
                                           tokenizer_class=BertTokenizer)


    task1 = Task(type='seqlabeling', num_labels=5, dropout_prob=0.1, splits=['train', 'dev', 'test'], dataset='iula',
                 split_seqs=False, max_seq_len=512)
    # task1 = Task(type='seqclassification', num_labels=3, dropout_prob=0.1, splits=['train', 'dev', 'test'], dataset='gad', max_seq_len=512)

    task2 = Task(type='seqclassification', num_labels=3, dropout_prob=0.1, splits=['train', 'dev', 'test'],
                 dataset='gad', max_seq_len=512)

    tasks = [task2, task1]


    dev_dataloaders = {}
    test_dataloaders = {}
    train_datasets = {}

    def get_data(task, split, config, tokenizer):
        if task.dataset == 'gad':
            data = RelClassificationDataset(read_relation_classification_examples(
                os.path.join(config.get('Files', 'preproc_data'), '{}_{}.jsonl'.format(task.dataset, split))),
                tokenizer=tokenizer, max_seq_len=task.max_seq_len)
        elif task.dataset in ['iula', 'french']:
            data = NegScopeDataset(
                read_examples(
                    os.path.join(config.get('Files', 'preproc_data'), '{}_{}.jsonl'.format(task.dataset, split))),
                tokenizer=tokenizer, max_seq_len=task.max_seq_len, split_seqs=task.split_seqs)
        return data


    for task_id, task in enumerate(tasks):
        task.set_task_id(task_id)
        logging.info('Setting task id {}'.format(task_id))
        logging.info('Task {}: {} on {}'.format(task_id, task.task_type, task.dataset))
        for i, t in enumerate(tasks):
            logging.info('Passing task {} inner'.format(t.task_id))
            # assert i == t.task_id
        if 'train' in task.splits:
            train_datasets[task_id] = get_data(task=task, split='train', config=config, tokenizer=tokenizer)
            train_datasets[task_id].set_task_id(task_id)
            task.label_map = train_datasets[task_id].label_map
            task.reverse_label_map = {val: key for key, val in train_datasets[task_id].label_map.items()}
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
    train_datasets = [train_datasets[i] for i in range(len(train_datasets))]
    mtl_dataset = MultiTaskDataset(train_datasets)

    multi_task_batch_sampler = MultiTaskBatchSampler(train_datasets, batch_size=4, mix_opt=0, extra_task_ratio=0,
                                                     annealed_sampling=0, max_epochs=5)

    mtl_train_dataloader = DataLoader(mtl_dataset, batch_sampler=multi_task_batch_sampler, collate_fn=mtl_dataset.collate_fn,
                                      pin_memory=False)
    for elm in mtl_train_dataloader:
        print(elm)

    #model = MTLModel(checkpoint='small_bert', device='cpu', tasks=tasks, padding_label_idx=padding_label)



    """
    # get optimizer
    optimizer = get_optimizer(model, lr=2e-5, eps=1e-6, decay=0.01)

    epochs = 1
    warmup_frac = 0

    # get lr schedule
    total_steps = len(mtl_dataset) * epochs
    warmup_steps = warmup_frac * total_steps
    logger.info('Scheduler: {} with {} warmup steps'.format('warmuplinear', warmup_steps))
    scheduler = get_scheduler(optimizer, scheduler='warmuplinear', warmup_steps=warmup_steps, t_total=total_steps)

    model.fit(tasks, optimizer, scheduler, train_dataloader=mtl_train_dataloader,
              dev_dataloaders=dev_dataloaders, test_dataloaders=test_dataloaders, epochs=epochs,
              evaluation_step=20, save_best=False, outdir='.', predict=True)
    """
