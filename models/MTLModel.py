import torch
import os

from transformers import  BertConfig, BertModel

from models.evaluation import evaluate_seq_labeling, evaluate_seq_classification
from models.task_specific_layers import OutputLayerSeqLabeling, OutputLayerSeqClassification


from torch import nn
import logging
import json
import test_config

from util import dump_json






logger = logging.getLogger(__name__)


def print_steps(ts):
    s = ''
    for key, val in ts.items():
        s += ('task{}:{}|'.format(key, val))
    return s[:-1]


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
        self.to(device)


    def forward(self, task, input_ids, attention_mask,  labels=None):
        # feed input ids through transformer
        logging.info(input_ids.device)
        logging.info(attention_mask.device)
        logging.info(self.encoder.device)
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


                def batch_to_device(batch, device):
                    moved_batch = {}
                    for key, val in batch.items():
                        moved_val = val.to(device)
                        moved_batch[key] = moved_val
                    return moved_batch

                # batch to device
                logging.info('Moving input to {}'.format(self.device))

                batch = batch_to_device(batch, self.device)
                for key, val in batch.items():
                    logging.info('Moved input to {}'.format(val.device))

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

