import torch
import os

from transformers import  BertConfig, BertModel, BertTokenizer, PreTrainedTokenizer
from models.tasks import load_task
from eval.evaluation import evaluate_seq_labeling, evaluate_seq_classification
from models.task_specific_layers import OutputLayerSeqLabeling, OutputLayerSeqClassification
from models.optimization import get_optimizer
from torch import nn
from models.tokenization import setup_customized_tokenizer
import logging
import json
import test_config

from util import dump_json






logger = logging.getLogger(__name__)


def batch_to_device(batch, device):
    moved_batch = {}
    for key, val in batch.items():
        moved_val = val.to(device)
        moved_batch[key] = moved_val
    return moved_batch

def print_steps(ts):
    s = ''
    for key, val in ts.items():
        s += ('task{}:{}|'.format(key, val))
    return s[:-1]


class MTLModel(nn.Module):

    def __init__(self, bert_encoder, device, tasks, padding_label_idx, tokenizer, load_checkpoint=False, checkpoint=None):
        super(MTLModel, self).__init__()

        # load encoder
        if load_checkpoint is True:
            logger.info('Loading config of trained model from {}'.format(checkpoint))
            if torch.cuda.is_available():
                ckpt = torch.load(checkpoint)
            else:
                ckpt = torch.load(checkpoint,map_location=torch.device('cpu') )
            config = ckpt['config']
            state_dict = ckpt['model_state_dict']
            self.label_map = ckpt['label_map']
            if bert_encoder == 'small_bert':
                bert_config = BertConfig.from_dict(test_config.test_config)
                self.encoder = BertModel(bert_config)
            else:
                self.encoder = BertModel.from_pretrained(config['encoder'])
        # small bert is a toy model for debugging
        elif bert_encoder == 'small_bert':
            bert_config = BertConfig.from_dict(test_config.test_config)
            self.encoder = BertModel(bert_config)
        else:
            self.encoder = BertModel.from_pretrained(pretrained_model_name_or_path=bert_encoder)

        self.hidden_size = self.encoder.config.hidden_size
        self.device = device
        self.padding_label_idx = padding_label_idx

        self.tasks = tasks

        # we only keep track of the tokenizer to save it along with the model
        self.tokenizer = tokenizer

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

        # initialize weights
        if load_checkpoint is True:
            model_params = set()
            for key, val in self.named_parameters():
                model_params.add(key)
            logger.info('Initializing weights from trained model checkpoint {}'.format(checkpoint))
            print('Initializing weights from trained model checkpoint {}'.format(checkpoint))
            for key in state_dict.keys():
                if key not in model_params:
                    logger.info('Additional parameter keys: {}'.format(key))
                    print('Additional parameter keys: {}'.format(key))
            for key, val in self.named_parameters():
                if key not in state_dict:
                    print('Missing parameter keys: {}'.format(key))
                    logger.info('Missing parameter keys: {}'.format(key))

            self.load_state_dict(state_dict, strict=False)


        headline = '############# Model Arch of MTL Model #############'
        logger.info('\n{}\n{}\n'.format(headline, self))
        logger.info("Total number of params: {}".format(sum([p.nelement() for p in self.parameters() if p.requires_grad])))
        self.to(device)


    def forward(self, task, input_ids, attention_mask,  labels=None):
        # feed input ids through transformer
        encoded_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # access the last hidden states

        last_hidden_states = encoded_output['last_hidden_state']
        # feed the last hidden state through the task specific output layers. Pooling, etc and loss computation is handled by those
        loss, logits = self.output_layers[task.task_id](last_hidden_states, labels)
        return loss, logits


    def fit(self, tasks, optimizer, scheduler, gradient_accumulation_steps, train_dataloader, dev_dataloaders, test_dataloaders, epochs, evaluation_step, save_best, outdir, predict):

        # get lr schedule
        total_steps = (len(train_dataloader)/gradient_accumulation_steps) * epochs

        loss_values = []
        global_step = 0

        best_dev_score = 0
        epoch = -1

        task_specific_forward = {t.task_id: 0 for t in tasks}

        accumulated_steps = 0
        for epoch in range(epochs):
            logger.info('Starting epoch {}'.format(epoch))

            total_loss = 0

            for step, batch in enumerate(train_dataloader):


                self.train()

                # batch to device

                batch = batch_to_device(batch, self.device)

                task_id=batch['task_id'][0]

                # perform forward pass
                output = self(tasks[task_id], input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])

                # compute loss
                loss = output[0]

                total_loss += loss.item()

                # scale the loss before doing backward pass
                loss = loss/gradient_accumulation_steps

                # perform backward pass
                loss.backward()

                # clip the gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                accumulated_steps += 1

                # keep track of the task specific steps
                task_specific_forward[task_id.item()] += 1

                #print(accumulated_steps)
                if accumulated_steps > 0 and accumulated_steps% gradient_accumulation_steps == 0:
                    #logger.info('Performing update after accumulating {} batches'.format(accumulated_steps))
                    # take a step and update the model
                    optimizer.step()

                    # Update the learning rate.
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    accumulated_steps = 0

                    # evaluate on dev
                    if global_step > 0 and global_step % evaluation_step == 0:
                        self.eval()
                        dev_results = self.evaluate_on_dev(data_loader=dev_dataloaders[0], task=tasks[0])
                        logger.info('Epoch {}, global step {}/{}\ttask_specific forward passes: {}\ttrain loss: {:.5f}\t dev score: {}'.format(epoch,
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
                            self.save(os.path.join(outdir, 'best_model/model_{}.pt'.format(epoch)), optimizer)




            if predict:
                for task in tasks:
                    logger.info('Predicting {} test data at end of epoch {}'.format(task.dataset, epoch))
                    self.eval()
                    test_results = self.evaluate_on_dev(data_loader=test_dataloaders[task.task_id], task=task)
                    test_score, test_report, test_predictions = test_results['score'], test_results['results'], test_results['predictions']
                    # dump to file
                    dump_json(fname=os.path.join(outdir, 'test_preds_{}_{}.json'.format(task.dataset, epoch)),
                              data={'f1': test_score, 'report': test_report, 'predictions': test_predictions})



        if not save_best:
            # save model
            logger.info('Saving model after epoch {} to {}'.format(epoch, os.path.join(outdir, 'model_{}.pt'.format(epoch))))
            self.save(os.path.join(outdir, 'model_{}.pt'.format(epoch)), optimizer)


    def evaluate_on_dev(self, data_loader, task):
        self.eval()

        if task.task_type == 'seqlabeling':
            results = evaluate_seq_labeling(data_loader, self, task)
        elif task.task_type == 'seqclassification':
            results = evaluate_seq_classification(data_loader, self, task)
        return results

    def save(self, outpath, optimizer):
        config = {'encoder': self.encoder.config._name_or_path}
        # add the label maps for the different output layers
        label_map = {}
        for task in self.tasks:
            label_map[task.task_id] = task.label_map

        torch.save({
            'label_map': label_map,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, outpath)

        self.tokenizer.save_pretrained('/'.join(outpath.split('/')[:-1]))


if __name__=="__main__":
    tasks = []

    #task = load_task(os.path.join('../task_specs', '{}.yml'.format('bioconv')))
    #task.num_labels = 5
    #task.task_id = 0
    #tasks.append(task)
    task = load_task(os.path.join('../task_specs', '{}.yml'.format('iula')))
    task.num_labels = 5
    task.task_id = 0
    tasks.append(task)
    import configparser
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read('../preprocessing/config.cfg')
    tokenizer = setup_customized_tokenizer(model='bert-base-cased', do_lower_case=False, config=config,
                                           tokenizer_class=BertTokenizer)
    model = MTLModel(bert_encoder='bert-base-cased', device='cpu', tasks=tasks, padding_label_idx=-1, load_checkpoint=False, tokenizer=tokenizer)
    optimizer = get_optimizer(optimizer_name='adamw', model=model, lr=5e-5, eps=1e-6, decay=0)
    tokenizer.save_pretrained('checkpoints/test')
    model.save('checkpoints/test/model.pt', optimizer)
    #model = MTLModel(bert_encoder=None, device='cpu', tasks=tasks, padding_label_idx=-1, load_checkpoint=True, checkpoint='checkpoints/test/model.pt' )
    #tokenizer = BertTokenizer.from_pretrained('checkpoints/test')
    #print(tokenizer.tokenize('[CUE]'))


