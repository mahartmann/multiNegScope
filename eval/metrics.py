# Copyright (c) Microsoft. All rights reserved.
from enum import Enum

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report

from sklearn.metrics import classification_report as classification_report_sklearn


def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts)

def compute_f1mac(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='macro')

def compute_f1mic(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='micro')

def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof

def compute_auc(predicts, labels):
    auc = roc_auc_score(labels, predicts)
    return 100.0 * auc

def compute_seqacc(predicts, labels, label_mapper):
    y_true, y_pred = [], []
    def trim(predict, label):
        temp_1 =  []
        temp_2 = []
        for j, m in enumerate(predict):
            if j == 0:
                continue
            if label_mapper[label[j]] != 'X':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        temp_1.pop()
        temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)
    for predict, label in zip(predicts, labels):
        trim(predict, label)
    report = classification_report(y_true, y_pred,digits=4)
    return report

cue_in_scope = {'iulajoint': False,
                'iulaequaljoint': False,
                'biojoint': True,
                'bioequaljoint': True,
                'sfuesjoint': True,
                'sfuesequaljoint': True,
                'nubesjoint': False,
                'nubesequaljoint': False,
                'frenchjoint': False,
                'frenchequaljoint': False}

def compute_pcs(predicts, labels, label_mapper, dataset):
    """
    compute correctly predicted full spans. If cues and scopes are predicted jointly, convert cue labels to I/O labels depending on the
    annotation scheme for the considered dataset
    :param predicts:
    :param labels:
    :return:
    """
    def trim_and_convert(predict, label, label_mapper, dataset):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(predict):
            if label_mapper[label[j]] != 'X' and label_mapper[label[j]] != 'CLS' and label_mapper[label[j]] != 'SEP':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        if 'joint' in dataset:
            if cue_in_scope[dataset] is True:
                replacement= 'I'
            else: replacement = 'O'
            for j, m in enumerate(temp_1):
                if m == 'C':
                    temp_1[j] = replacement
            for j, m in enumerate(temp_2):
                if m == 'C':
                    temp_2[j] = replacement
        return temp_2, temp_1

    tp = 0.

    for predict, label in zip(predicts, labels):
        predict, label = trim_and_convert(predict, label, label_mapper,dataset)
        if predict == label:
            tp += 1
    return tp/len(predicts)

def compute_scope_p(predicts, labels, label_mapper, dataset):
    return compute_scope_prf(predicts, labels, label_mapper, dataset, metric='p')

def compute_scope_r(predicts, labels, label_mapper, dataset):
    return compute_scope_prf(predicts, labels, label_mapper, dataset, metric='r')

def compute_scope_f(predicts, labels, label_mapper, dataset):
    return compute_scope_prf(predicts, labels, label_mapper, dataset, metric='f')

def compute_scope_prf(predicts, labels, label_mapper, dataset, metric='f'):
    """
    compute correctly predicted full spans
    :param predicts:
    :param labels:
    :return:
    """
    def trim_and_convert(predict, label, label_mapper, dataset):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(predict):
            if label_mapper[label[j]] != 'X' and label_mapper[label[j]] != 'CLS' and label_mapper[label[j]] != 'SEP':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        if 'joint' in dataset:
            if cue_in_scope[dataset] is True:
                replacement = 'I'
            else:
                replacement = 'O'
            for j, m in enumerate(temp_1):
                if m == 'C':
                    temp_1[j] = replacement
            for j, m in enumerate(temp_2):
                if m == 'C':
                    temp_2[j] = replacement
        return temp_2, temp_1


    y_gold = []
    y_pred = []
    for predict, label in zip(predicts, labels):
        predict, label = trim_and_convert(predict, label, label_mapper, dataset)
        y_gold.extend(label)
        y_pred.extend(predict)

    prf = precision_recall_fscore_support(y_gold, y_pred, labels=['I', 'O'])

    p = prf[0][0]
    r = prf[1][0]
    f = prf[2][0]
    if metric == 'f': return f
    elif metric == 'p': return p
    elif metric == 'r': return r


def compute_cue_f_seq(predicts, labels, label_mapper, dataset, metric='f'):
    """
    compute correctly predicted full spans
    :param predicts:
    :param labels:
    :return:
    """

    def trim_and_convert(predict, label, label_mapper, dataset):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(predict):
            if label_mapper[label[j]] != 'X' and label_mapper[label[j]] != 'CLS' and label_mapper[label[j]] != 'SEP':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])

        for j, m in enumerate(temp_1):
            if m != 'C':
                temp_1[j] = '0'
        for j, m in enumerate(temp_2):
            if m != 'C':
                temp_2[j] = '0'
        return temp_2, temp_1

    y_gold = []
    y_pred = []
    for predict, label in zip(predicts, labels):
        predict, label = trim_and_convert(predict, label, label_mapper, dataset)
        y_gold.extend(label)
        y_pred.extend(predict)

    prf = precision_recall_fscore_support(y_gold, y_pred, labels=['C'])

    p = prf[0][0]
    r = prf[1][0]
    f = prf[2][0]
    if metric == 'f':
        return f
    elif metric == 'p':
        return p
    elif metric == 'r':
        return r



def compute_clue_f(predicts, labels, label_mapper):
    """
    compute correctly predicted full spans
    :param predicts:
    :param labels:
    :return:
    """
    def trim(predict, label):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(predict):
            if label_mapper[label[j]] != 'X' and label_mapper[label[j]] != 'CLS' and label_mapper[label[j]] != 'SEP':
                temp_1.append(label_mapper[label[j]])
                temp_2.append(label_mapper[m])
        return temp_2, temp_1

    y_gold = []
    y_pred = []
    for predict, label in zip(predicts, labels):
        predict, label = trim(predict, label)
        y_gold.extend(label)
        y_pred.extend(predict)

    f = precision_recall_fscore_support(y_gold, y_pred, labels=['1','0'])
    return f[2][0]


def compute_p_r_f_multi(predicts, labels, label_mapper, dataset):
    f = compute_p_r_f_multi_report(predicts, labels, label_mapper, dataset)
    if dataset == 'chemprot' or dataset == 'ddirelations' or dataset == 'ddirelationssilverspan1' or  dataset == 'chemprotsilverspan1':
        return f['subset']['micro avg']['f1-score']
    else:
        return f['macro avg']['f1-score']


def compute_p_r_f_multi_report(predicts, labels, label_mapper, dataset):
    label_strings = []
    label_idxs = []
    for key, val in label_mapper.tok2ind.items():
        label_strings.append(key)
        label_idxs.append(val)
    f = classification_report_sklearn(labels, predicts, labels=label_idxs, target_names=label_strings, output_dict=True)
    if dataset == 'chemprot' or dataset == 'ddirelations' or dataset == 'ddirelationssilverspan1' or  dataset == 'chemprotsilverspan1':
        filtered_preds, filtered_gold, filtered_label_idxs, filtered_label_strings = compute_p_r_f_subset(predicts, labels, label_mapper, dataset)
        f_subset = classification_report_sklearn(y_true=filtered_gold, y_pred=filtered_preds, labels=filtered_label_idxs, target_names=filtered_label_strings, output_dict=True)
        f['subset'] = f_subset
    return f




class Metric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    AUC = 5
    SeqEval = 7
    EmF1 = 8
    F1MAC = 9
    F1MIC = 10
    PCS = 11
    CLUEF = 12
    SCOPEP = 13
    SCOPER = 14
    SCOPEF = 15
    PRF = 16
    PRFReport = 17
    CUEF_SEQ = 18



METRIC_FUNC = {

    Metric.ACC: compute_acc,
    Metric.F1: compute_f1,
    Metric.MCC: compute_mcc,
    Metric.Pearson: compute_pearson,
    Metric.Spearman: compute_spearman,
    Metric.AUC: compute_auc,
    Metric.SeqEval: compute_seqacc,
    Metric.F1MAC: compute_f1mac,
    Metric.F1MIC: compute_f1mic,
    Metric.PCS: compute_pcs,
    Metric.CLUEF: compute_clue_f,
    Metric.SCOPEP: compute_scope_p,
    Metric.SCOPER: compute_scope_r,
    Metric.SCOPEF: compute_scope_f,
    Metric.PRF: compute_p_r_f_multi,
    Metric.PRFReport: compute_p_r_f_multi_report,
    Metric.CUEF_SEQ: compute_cue_f_seq
}


def calc_metrics(metric_meta, golds, predictions, scores, dataset, label_mapper=None):
    """Label Mapper is used for NER/POS etc. 
    TODO: a better refactor, by xiaodl
    """
    metrics = {}
    for mm in metric_meta:
        metric_name = mm.name
        metric_func = METRIC_FUNC[mm]
        if mm in (Metric.ACC, Metric.F1, Metric.MCC, Metric.F1MAC, Metric.F1MIC):
            metric = metric_func(predictions, golds)
        elif mm == Metric.SeqEval:
            metric = metric_func(predictions, golds, label_mapper)
        elif mm == Metric.PCS:
            metric = metric_func(predictions, golds, label_mapper, dataset)
        elif mm == Metric.CLUEF:
            metric = metric_func(predictions, golds, label_mapper)
        elif mm == Metric.SCOPEP or mm == Metric.SCOPER or mm == Metric.SCOPEF:
            metric = metric_func(predictions, golds, label_mapper, dataset)
        elif mm == Metric.CUEF_SEQ:
            metric = metric_func(predictions, golds, label_mapper, dataset)
        elif mm == Metric.EmF1:
            metric = metric_func(predictions, golds)
        elif mm == Metric.PRF:
            metric = metric_func(predictions, golds, label_mapper, dataset)
        elif mm == Metric.PRFReport:
            metric = metric_func(predictions, golds, label_mapper, dataset)
        else:
            if mm == Metric.AUC:
                assert len(scores) == 2 * len(golds), "AUC is only valid for binary classification problem"
                scores = scores[1::2]
            metric = metric_func(scores, golds)
        metrics[metric_name] = metric
    return metrics


if __name__=="__main__":
    from data_utils import vocab
    pred = [0,1,2,3,4,5,5,5]
    gold = [0,1,2,3,4,5,3,3]
    dataset= 'chemprot'
    _label_mapper = {0: 'CPR:3', 1: 'CPR:4', 2: 'CPR:5', 3: 'CPR:6', 4: 'CPR:9', 5: 'false'}
    label_mapper = vocab.Vocabulary(neat=True)
    for key in range(6):
        label_mapper.add(_label_mapper[key])
    f = compute_p_r_f_multi(pred, gold, label_mapper, dataset)
    report = compute_p_r_f_multi_report(pred, gold, label_mapper, dataset)
    print(f)
    for key, val in report.items():
        print(key, val)

