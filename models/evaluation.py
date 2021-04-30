import numpy as np
import torch
from sklearn.metrics import classification_report

import metrics


def batch_to_device(batch, device):
    moved_batch = {}
    for key, val in batch.items():
        moved_val = val.to(device)
        moved_batch[key] = moved_val
    return moved_batch

def evaluate_seq_labeling(data_loader, model, task):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cropped_preds = []
    cropped_golds = []

    for step, batch in enumerate(data_loader):
        # batch to device
        batch = batch_to_device(batch, device)

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

    pcs = metrics.compute_pcs(cropped_preds, cropped_golds, task.reverse_label_map, dataset=task.dataset)
    f1 = metrics.compute_scope_prf(predicts=cropped_preds,labels=cropped_golds,label_mapper=task.reverse_label_map,dataset=task.dataset, metric='f')
    prec =   metrics.compute_scope_prf(predicts=cropped_preds,labels=cropped_golds,label_mapper=task.reverse_label_map,dataset=task.dataset, metric='p')
    rec =  metrics.compute_scope_prf(predicts=cropped_preds,labels=cropped_golds,label_mapper=task.reverse_label_map,dataset=task.dataset, metric='r')

    report = {'f1': f1, 'p': prec, 'r': rec}
    results = {'score': f1, 'predictions': cropped_preds, 'results': report}
    return results


def evaluate_seq_classification(data_loader, model, task):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    preds = None
    out_label_ids = None

    for step, batch in enumerate(data_loader):
        # batch to device
        batch = batch_to_device(batch, device)

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