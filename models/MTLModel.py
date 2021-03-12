import torch
import os

from transformers import BertForSequenceClassification, BertConfig, BertModel
import random
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import logging
import metrics
import json
import test_config
from models.optimization import get_optimizer, get_scheduler
from util import dump_json, bool_flag
from mydatasets.MultiTaskDataset import MultiTaskDataset, MultiTaskBatchSampler
from mydatasets.RelClassificationDataset import RelClassificationDataset, read_relation_classification_examples
import configparser
from transformers import BertTokenizer
import os
from util import create_logger
from models.tokenization import setup_customized_tokenizer
from mydatasets.NegScopeDataset import NegScopeDataset, read_examples
from torch.utils.data import DataLoader
import argparse
import yaml
import uuid



logger = logging.getLogger(__name__)


def print_steps(ts):
    s = ''
    for key, val in ts.items():
        s += ('task{}:{}|'.format(key, val))
    return s[:-1]

class Task():
    def __init__(self, type, dataset,  dropout_prob, splits, max_seq_len, split_seqs=False):
        self.dataset=dataset
        self.splits = splits
        self.task_type = type
        self.hidden_dropout_prob = dropout_prob
        self.label_map = None
        self.reverse_label_map = None
        self.max_seq_len = max_seq_len
        self.split_seqs = split_seqs
        self.task_id = None

    def set_task_id(self, tid):
        self.task_id = tid

    def set_label_map(self, label_map):
        self.label_map = label_map
        self.reverse_label_map = {val: key for key, val in label_map.items()}
        self.num_labels = len(label_map)



class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class OutputLayerSeqLabeling(nn.Module):
    def __init__(self, task, hidden_size, padding_idx):
        super(OutputLayerSeqLabeling, self).__init__()
        self.task_type = task.task_type
        self.dropout = nn.Dropout(task.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, task.num_labels)
        self.criterion = CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, last_hidden_states, labels):
        # get a classifcation for each token in the sequence
        last_hidden_states = self.dropout(last_hidden_states)
        logits = self.classifier(last_hidden_states)

        # logits: bs x seq_len x num_labels -> (bs x seq_len) x num_labels
        flattened_logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])

        # labels: bs x seq_len -> (bs x seq_len)
        flattened_labels = labels.view(-1)

        loss = self.criterion(flattened_logits, flattened_labels)
        return loss, logits

class OutputLayerSeqClassification(nn.Module):
    def __init__(self, task, hidden_size, padding_idx):
        super(OutputLayerSeqClassification, self).__init__()
        self.task_type = task.task_type
        self.pooler = BertPooler(hidden_size=hidden_size)

        self.dropout = nn.Dropout(task.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, task.num_labels)
        self.criterion = CrossEntropyLoss(ignore_index=padding_idx)


    def forward(self, input, labels):
        pooled_input = self.pooler(input)
        pooled_input = self.dropout(pooled_input)
        logits = self.classifier(pooled_input)
        loss = self.criterion(logits, labels)
        return loss, logits


class MTLModel(nn.Module):

    def __init__(self, checkpoint, device,  tasks, padding_label_idx):
        super(MTLModel, self).__init__()

        # load encoder
        # small bert is a toy model for debugging
        if checkpoint == 'small_bert':
            bert_config = BertConfig.from_dict(test_config.test_config)
            self.encoder = BertModel(bert_config)
        else:
            self.encoder = BertModel.from_pretrained(pretrained_model_name_or_path=checkpoint)
        self.encoder.to(device)
        self.hidden_size = self.encoder.config.hidden_size
        self.device = device
        self.padding_label_idx = padding_label_idx


        self.output_layers = nn.ModuleList()

        # add the task specific output layers
        for task in tasks:
            if task.task_type == 'seqlabeling':
                output_layer = OutputLayerSeqLabeling(task, self.hidden_size, padding_idx=self.padding_label_idx)
                self.output_layers.insert(task.task_id, output_layer)
                logging.info('Adding task {} output layer for {}'.format(task.task_id, task.task_type))
            if task.task_type == 'seqclassification':
                output_layer = OutputLayerSeqClassification(task, self.hidden_size, padding_idx=self.padding_label_idx)
                self.output_layers.insert(task.task_id, output_layer)
                logging.info('Adding task {} output layer for {}'.format(task.task_id, task.task_type))


        headline = '############# Model Arch of MTL Model #############'
        logger.info('\n{}\n{}\n'.format(headline, self))
        logger.info("Total number of params: {}".format(sum([p.nelement() for p in self.parameters() if p.requires_grad])))



    def forward(self, task, input_ids, attention_mask,  labels=None):
        # feed input ids through transformer
        encoded_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # access the last hidden states
        last_hidden_states = encoded_output['last_hidden_state']
        # feed the last hidden state throuhh the task specific output layers. Pooling, etc and loss computation is handled by those
        loss, logits = self.output_layers[task.task_id](last_hidden_states, labels)
        return loss, logits


    def fit(self, tasks, optimizer, scheduler, train_dataloader, dev_dataloaders, test_dataloaders, epochs, evaluation_step, save_best, outdir, predict):

        # get lr schedule
        total_steps = len(train_dataloader) * epochs

        loss_values = []
        global_step = 0

        best_dev_score = 0
        epoch = -1

        task_specific_forward = {t.task_id: 0 for t in tasks}
        for epoch in range(epochs):
            logger.info('Starting epoch {}'.format(epoch))

            total_loss = 0

            for step, batch in enumerate(train_dataloader):

                self.train()

                # batch to device
                for key, val in batch.items():
                    val.to(self.device)

                # clear gradients
                self.zero_grad()

                task_id=batch['task_id'][0]

                # perform forward pass
                output = self(tasks[task_id], input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])

                # compute loss
                loss = output[0]

                total_loss += loss.item()

                # perform backward pass
                loss.backward()

                # clip the gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.

                # take a step and update the model
                optimizer.step()

                # Update the learning rate.
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                # keep track of the task specific steps
                task_specific_forward[task_id.item()] += 1


                # evaluate on dev
                if step > 0 and step % evaluation_step == 0:
                    self.eval()
                    dev_results = self.evaluate_on_dev(data_loader=dev_dataloaders[0], task=tasks[0])
                    logger.info('Epoch {}, global step {}/{}\ttask_specific steps: {}\ttrain loss: {:.5f}\t dev score: {}'.format(epoch,
                                                                                                         global_step,
                                                                                                         total_steps,
                                                                                                        print_steps(
                                                                                                        task_specific_forward),
                                                                                                        total_loss/step,
                                                                                                        dev_results['score']))


            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            # evaluate on dev after epoch is finished
            self.eval()
            for task in tasks:
                dev_results = self.evaluate_on_dev(data_loader=dev_dataloaders[task.task_id], task=task)
                dev_score = dev_results['score']
                logger.info('End of epoch {}, global step {}/{}\ttrain loss: {:.5f}\t task {} dev score: {:.5f}\ndev report {}'.format(epoch,
                                                                                                         global_step,
                                                                                                         total_steps,
                                                                                                         total_loss / step,
                                                                                                          task.task_id,
                                                                                                         dev_results['score'],
                                                                                                        json.dumps(dev_results['results'], indent=4)))
                # use task 0 dev score for model selection
                if task.task_id == 0:
                    if dev_score >= best_dev_score:
                        logger.info('New task 0 dev score {:.5f} > {:.5f}'.format(dev_score, best_dev_score))
                        best_dev_score = dev_score
                        if save_best:
                            #save model
                            logger.info('Saving model after epoch {} as best model to {}'.format(epoch, os.path.join(outdir, 'best_model')))
                            self.save(os.path.join(outdir, 'best_model/model_{}.pt'.format(epoch)))




            if predict:
                for task in tasks:
                    logger.info('Predicting {} test data at end of epoch {}'.format(task.dataset, epoch))
                    self.eval()
                    test_results = self.evaluate_on_dev(data_loader=test_dataloaders[task.task_id], task=task)
                    test_score, test_report, test_predictions = test_results['score'], None, test_results['predictions']
                    # dump to file
                    dump_json(fname=os.path.join(outdir, 'test_preds_{}.json'.format(epoch)),
                              data={'f1': test_score, 'report': test_report, 'predictions': test_predictions})



        if not save_best:
            # save model
            logger.info('Saving model after epoch {} to {}'.format(epoch, os.path.join(outdir, 'model_{}.pt'.format(epoch))))
            self.save(os.path.join(outdir, 'model_{}.pt'.format(epoch)))


    def evaluate_on_dev(self, data_loader, task):
        self.eval()

        if task.task_type == 'seqlabeling':
            results = evaluate_seq_labeling(data_loader, self, task)
        elif task.task_type == 'seqclassification':
            results = evaluate_seq_classification(data_loader, self, task)
        return results

    def save(self, outpath):
        outpath = '/'.join(outpath.split('/')[:-1])
        self.save_pretrained(outpath)

def evaluate_seq_labeling(data_loader, model, task):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cropped_preds = []
    cropped_golds = []

    for step, batch in enumerate(data_loader):
        # batch to device
        for key, val in batch.items():
            val.to(device)

        # perform forward pass
        with torch.no_grad():
            loss, output = model(task=task, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                    labels=batch['labels'])

            out_probs = output.data.cpu()
            out_probs = out_probs.numpy()

            mask = batch['attention_mask']
            golds = batch['labels'].data.cpu().numpy()

            preds = np.argmax(out_probs, axis=-1).reshape(mask.size()).tolist()

            # only take into account the predictions for non PAD tokens
            valid_length = mask.sum(1).tolist()

            final_predict = []
            final_golds = []
            for idx, p in enumerate(preds):
                final_predict.append(p[:int(valid_length[idx])])
                final_golds.append(golds[idx][:int(valid_length[idx])])

            cropped_preds.extend(final_predict)
            cropped_golds.extend(final_golds)

    score = metrics.compute_pcs(cropped_preds, cropped_golds, task.reverse_label_map, dataset=task.dataset)
    results = {'score': score, 'predictions': cropped_preds, 'results': None}
    return results

def evaluate_seq_classification(data_loader, model, task):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    preds = None
    out_label_ids = None

    for step, batch in enumerate(data_loader):
        # batch to device
        batch.to(device)

        # perform forward pass
        with torch.no_grad():
            output = model(task=task, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                    labels=batch['labels'])
            loss, logits = output[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    def compute_metrics(golds, preds, label_map):
        labels = [i for i in range(len(label_map))]
        target_names = [label_map[key] for key in labels]
        report = classification_report(y_true=golds, y_pred=preds, labels=labels,
                                       target_names=target_names, sample_weight=None,
                                       output_dict=True, zero_division='warn')
        return report

    results = compute_metrics(golds=out_label_ids, preds=preds, label_map=task.reverse_label_map)
    return {'score': results['macro avg']['f1-score'], 'results': results, 'predictions': preds.tolist()}

def load_task(fname):
    task_def = yaml.safe_load(open(fname))
    task = Task(type=task_def['task_type'], dropout_prob=task_def['dropout_prob'], splits=task_def['splits'], dataset=task_def['dataset'],
         split_seqs=task_def['split_seqs'], max_seq_len=task_def['max_seq_len'])
    return task


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = setup_customized_tokenizer(model=args.tokenizer, do_lower_case=False, config=config,
                                           tokenizer_class=BertTokenizer)
    tasks = []
    for task_name in args.tasks.split(','):
        task = load_task(os.path.join(args.task_spec, '{}.yml'.format(task_name)))
        tasks.append(task)


    train_datasets = {}
    dev_dataloaders = {}
    test_dataloaders = {}


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
              evaluation_step=args.evaluation_steps, save_best=args.save_best, outdir=outdir, predict=args.predict)


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Train SentenceBert with analogy data')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    parser.add_argument('--bert_model', type=str,
                        default='small_bert',
                        choices=['bert-base-multilingual-cased', 'bert-base-uncased', 'small_bert', 'xlm-roberta-base'],
                        help="The pre-trained encoder used to encode the entities of the analogy")
    parser.add_argument('--tokenizer', type=str,
                        default='bert-base-multilingual-cased',
                        help="The tokenizer")
    parser.add_argument('--config',
                        default='../preprocessing/config.cfg')
    parser.add_argument('--task_spec',
                        default='../task_specs', help='Directory with task specifications')
    parser.add_argument('--tasks', type=str,
                        help="Yaml file specifying the training/testing tasks", default='gad')
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
    parser.add_argument('--evaluation_steps', type=int, default=2,
                        help="Evaluate every n training steps")
    parser.add_argument("--save_best", type=bool_flag, default=True)
    parser.add_argument("--predict", type=bool_flag, default=False)

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