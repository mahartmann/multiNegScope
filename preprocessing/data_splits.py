import numpy as np
import os
import json
import itertools
"""
produce train/test/dev splits
"""

import random

def write_lines(fname, lines):
    with open(fname, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
    f.close()

def generate_train_dev_test_splits(num_data):
    np.random.seed(42)
    # generate 70/15/15 splits
    idx = [i for i in range(num_data)]
    np.random.shuffle(idx)
    train_idxs = idx[:int(np.ceil(0.7*len(idx)))]
    dev_idxs = idx[int(np.ceil(0.7*len(idx))):int(np.ceil((0.7+0.15)*len(idx)))]
    test_idxs = idx[int(np.ceil((0.7+0.15)*len(idx))):]
    return train_idxs, dev_idxs, test_idxs

def generate_train_dev_splits(num_data):
    np.random.seed(42)
    # generate 85/15 splits
    idx = [i for i in range(num_data)]
    np.random.shuffle(idx)
    train_idxs = idx[:int(np.ceil(0.85*len(idx)))]
    dev_idxs = idx[int(np.ceil(0.85*len(idx))):]
    return train_idxs, dev_idxs


def write_train_dev_test_data(fstem, data, setting, sample_equal=False):
    """
    if sample_equal is True, we sample identical amounts of sentences with and without negation
    """
    split_idxs = generate_train_dev_test_splits(len(data))
    for i, splt in enumerate(['train', 'dev', 'test']):
        idxs = split_idxs[i]
        out_data = []
        for sid, idx in enumerate(idxs):
            for cid, elm in enumerate(data[idx]):
                uid = len(out_data)
                out_data.append({'uid': uid,
                                 'seq': elm[1],
                                 'labels': elm[0],
                                 'sid': '{}_{}'.format(sid, cid),
                                 'cue_indicator': elm[2]})
        fstem_out = fstem
        if setting == 'embed':
            fstem_out = fstem+'embed'
        elif setting == 'nocond':
            fstem_out = fstem + 'nocond'
        elif setting == 'joint':
            fstem_out = fstem + 'joint'
        write_split('{}_{}.tsv'.format(fstem_out, splt), out_data)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}_{}.tsv'.format(fstem_out, splt) ))
    return  split_idxs


def write_train_dev_test_data_m2c2(fstem, data, sample_equal=False):
    """
    if sample_equal is True, we sample identical amounts of sentences with and without negation
    """
    split_idxs = generate_train_dev_splits(len(data))
    for i, splt in enumerate(['train',  'test']):
        idxs = split_idxs[i]
        out_data = []
        for sid, idx in enumerate(idxs):
            elm = data[idx]
            uid = len(out_data)
            out_data.append([uid,elm['labels'],elm['seq']])
        write_split('{}_{}.tsv'.format(fstem, splt), out_data,json_format=False)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}_{}.tsv'.format(fstem, splt) ))
    return  split_idxs

def write_train_dev_test_data_ddi(fstem, train_data, test_data):
    """
      split train_data into train/dev, test_split is fixed
      :return:
    """
    split_idxs = generate_train_dev_splits(len(train_data))
    for i, splt in enumerate(['train', 'dev']):
        idxs = split_idxs[i]
        out_data = []
        for sid, idx in enumerate(idxs):
            for cid, elm in enumerate(train_data[idx]):
                uid = len(out_data)
                print(elm)
                out_data.append({'uid': uid,
                                 'seq': elm['seq'],
                                 'labels': elm['label'],
                                 'sid': '{}_{}'.format(sid, cid)})
        write_split('{}_{}.tsv'.format(fstem, splt), out_data, json_format=True)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}_{}.tsv'.format(fstem, splt) ))
    out_data = []
    splt = 'test'
    for sid, sent_data in enumerate(test_data):
        for cid, elm in enumerate(sent_data):
            uid = len(out_data)
            out_data.append({'uid': uid,
                             'seq': elm['seq'],
                             'labels': elm['label'],
                             'sid': '{}_{}'.format(sid, cid)})
    write_split('{}_{}.tsv'.format(fstem, splt), out_data, json_format=True)
    print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(test_data), len(out_data),
                                                                           '{}_{}.tsv'.format(fstem, splt)))
    return split_idxs

def write_train_dev_test_cue_data(fstem, data, split_idxs):
    for i, splt in enumerate(['train', 'dev', 'test']):
        out_data = []
        # write another split that can be used for cue prediction, which has no 'CUE tokens'
        out_data_filtered = []
        idxs = split_idxs[i]
        for uid, idx in enumerate(idxs):
            elm = data[idx]
            seq = elm[1]
            labels = ['1' if label.startswith('1') else '0' for label in elm[0] ]
            out_data.append({'uid': uid,
                             'seq': seq,
                             'labels': labels,
                             'sid': '{}_{}'.format(uid, 0)
                             })
            filtered_seq = [seq[i] for i, elm in enumerate(seq) if elm != 'CUE']
            filtered_labels = [labels[i] for i, elm in enumerate(seq) if elm != 'CUE']
            assert len(filtered_seq) == len(filtered_labels)
            out_data_filtered.append({'uid': uid,
                             'seq': filtered_seq,
                             'labels': filtered_labels,
                             'sid': '{}_{}'.format(uid, 0)
                             })


        write_split('{}#cues_{}.tsv'.format(fstem, splt), out_data)
        write_split('{}nocues_{}.tsv'.format(fstem, splt), out_data_filtered)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}#cues_{}.tsv'.format(fstem, splt) ))


    return

def write_train_dev_test_data_drugs(fstem, train_data, test_data):
    """
      split train_data into train/dev, test_split is fixed
      :return:
    """
    split_idxs = generate_train_dev_splits(len(train_data))
    for i, splt in enumerate(['train', 'dev']):
        idxs = split_idxs[i]
        out_data = []
        for idx in idxs:
            out_data.append([len(out_data), train_data[idx]['rating'], train_data[idx]['review']])
        write_split('{}_{}.tsv'.format(fstem, splt), out_data, json_format=False)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),  '{}_{}.tsv'.format(fstem, splt) ))
    out_data = []
    splt = 'test'
    for elm in test_data:
        out_data.append([len(out_data), elm['rating'], elm['review']])
    write_split('{}_{}.tsv'.format(fstem, splt), out_data, json_format=False)
    print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(test_data), len(out_data),
                                                                           '{}_{}.tsv'.format(fstem, splt)))
    return split_idxs

def write_data_gad_format(fstem, train_data, test_data):
    """
    split train_data into train/dev, test_split is fixed
    :return:
    """
    split_idxs = generate_train_dev_splits(len(train_data))
    for i, splt in enumerate(['train', 'dev']):
        idxs = split_idxs[i]
        out_data = []
        for idx in idxs:
            out_data.append([len(out_data), train_data[idx]['label'], train_data[idx]['seq']])
        write_split('{}_{}.tsv'.format(fstem, splt), out_data, json_format=False)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),
                                                                           '{}_{}.tsv'.format(fstem, splt)))
    out_data = []
    splt = 'test'
    for elm in test_data:
        out_data.append([len(out_data), elm['label'], elm['seq']])
    write_split('{}_{}.tsv'.format(fstem, splt), out_data, json_format=False)
    print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(test_data), len(out_data),
                                                                       '{}_{}.tsv'.format(fstem, splt)))
    return split_idxs


def write_train_dev_udep(fstem, train_data):
    split_idxs = generate_train_dev_splits(len(train_data))
    for i, splt in enumerate(['train', 'dev']):
        idxs = split_idxs[i]
        out_data = []
        for idx in idxs:
            out_data.append([len(out_data)] +  train_data[idx])
        write_split('{}_{}.tsv'.format(fstem, splt), out_data, json_format=False)
        print('{} has {} sentences and {} instances. Writing to {}'.format(splt, len(idxs), len(out_data),
                                                                           '{}_{}.tsv'.format(fstem, splt)))
def shuffle_and_prepare(data, shuffle=True):
        np.random.seed(42)
        if shuffle:
            np.random.shuffle(data)
        out_data = []
        for elm in data:
            out_data.append([len(out_data)] + elm)
        return out_data

def write_split(fname, data, json_format=True):
    print('Writing to {}'.format(fname))
    outlines = []
    if not json_format:
        for elm in data:
            s = ''
            for f in elm:
                s += '{}\t'.format(f)
            outlines.append(s.strip('\t') + '\n')
        write_lines(fname, outlines)
    else:
        with open(fname, 'w') as f:
            for elm in data:
                f.write(json.dumps(elm) + '\n')
        f.close()


def sample_scope_annotations(fnamestem, outnamestem):
    """
    sample instances from data such that number of sents with negation is equal to number of sents w/o negation
    """

    def split_data(data):
        notnegated = []
        negated = []
        for sent in data:
            #print(sent)
            elm = sent[0]
            labels = set(elm['labels'])
            if len(labels) == 1 and 'O' in labels:
                notnegated.append(sent)
            else:
                negated.append(sent)
        print('{} negated, {} nonnegated'.format(len(negated), len(notnegated)))
        return negated, notnegated

    np.random.seed(42)

    def load_data(fname):
        d = []
        with open(fname) as f:

            sent = []
            sid = '-1'
            for line in f:
                elm = json.loads(line)
                if elm['sid'].split('_')[0] == sid:
                    sent.append(elm)
                else:
                    d.append(sent)
                    sent = [elm]
                sid = elm['sid'].split('_')[0]
            d.append(sent)

        return d[1:]

    d = load_data(fnamestem + '_train.tsv')
    print(d[0])
    negated, notnegated = split_data(d)
    sampled_data = []
    if len(negated) <= len(notnegated):
        sampled_data.extend(negated)
        np.random.shuffle(notnegated)
        sampled_data.extend(notnegated[:len(negated)])
    else:
        sampled_data.extend(notnegated)
        np.random.shuffle(negated)
        sampled_data.extend(negated[:len(notnegated)])
    np.random.shuffle(sampled_data)
    # rewrite uids
    for i, sents in enumerate(sampled_data):
        for elm in sents:
            elm['uid'] = i
    split_data(sampled_data)

    sampled_data = list(itertools.chain.from_iterable(sampled_data))

    write_split(outnamestem + '_train.tsv', sampled_data, json_format=True)

    for split in ['dev', 'test']:
        d = load_data(fnamestem + '_{}.tsv'.format(split))

        split_data(d)
        d = list(itertools.chain.from_iterable(d))
        write_split(outnamestem +  '_{}.tsv'.format(split), d, json_format=True)





if __name__=="__main__":
    import configparser

    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)
    outpath = config.get('Files', 'preproc_data')

    datasets = [
                'biofullall', 'bioabstractsall', 'bioall',
                'sherlockenall', 'sherlockzhall',
                'iulaall',
                'sfuenall', 'sfuesall',
                'itaall',
                'nubesall',
                'dtnegall', 'soccall','frenchall']
    datasets = [
        'iula all']
    setting = ''
    for ds in datasets:
        stem = '{}{}'.format(ds, setting)
        fnamestem = os.path.join(outpath, stem)
        outnamestem = os.path.join(outpath, stem.replace('all','equal'))
        sample_scope_annotations(fnamestem, outnamestem)
