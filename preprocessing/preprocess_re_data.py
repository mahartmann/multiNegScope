"""
Preprocess relation extraction datasets
replace target entities with special tokens
split into train/dev/test splits
"""
import csv
import codecs
from preprocessing.data_splits import generate_train_dev_test_splits
import json
import itertools
import os

def read_lines(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line.strip())
    return lines

def write_json_lines(data, fname):
    print('Writing {} lines to {}'.format(len(data), fname))
    with open(fname, 'w') as f:
        for elm in data:
            f.write(json.dumps(elm) + '\n')
    f.close()


def replace_mentions(replacements, text):
    # produce a string where mentions of particpants are replaced by their labels
    spans = sorted(list(replacements.keys()), key=lambda x: x[1])
    new_text = ''
    for i, span in enumerate(spans):
        start = span[0]
        end = span[1]

        if i == 0:
            new_text += text[:start] + '@{}$'.format(replacements[span])
        else:
            new_text += text[spans[i - 1][1]:start] + '@{}$'.format(replacements[span])
        if i == len(spans) - 1:
            new_text += text[end:]
    new_text = new_text.replace('  ', ' ')
    return new_text

def preprocess_gad(fnames_in, out_stem):
    sent2data = {}
    # we need to keep track of the original sentences in order to ensure that different versions of the original sentence are not spread across train and test splits
    origsent2replaced = {}
    for fname in fnames_in:
        with codecs.open(fname, 'r', encoding='utf-8') as f:

            reader = csv.DictReader(f, delimiter='\t',
                                    fieldnames=['GAD_ID', 'GAD_ASSOC', 'GAD_GENE_SYMBOL', 'GAD_GENE_NAME', 'GAD_ENTREZ_ID',
                                                'NER_GENE_ENTITY', 'NER_GENE_OFFSET', 'GAD_DISEASE_NAME',
                                                'NER_DISEASE_ENTITY', 'NER_DISEASE_OFFSET', 'GAD_CONCLUSION'])

            i = -1
            for row in reader:
                i += 1
                if i != 0:
                    label = row['GAD_ASSOC']
                    text = row['GAD_CONCLUSION']
                    gene_text = row['NER_GENE_ENTITY']
                    gene_offset = row['NER_GENE_OFFSET']
                    gene_span = (int(gene_offset.split('#')[0]), int(gene_offset.split('#')[1]), 1)
                    disease_text = row['NER_DISEASE_ENTITY']
                    disease_offset = row['NER_DISEASE_OFFSET']
                    disease_span = (int(disease_offset.split('#')[0]), int(disease_offset.split('#')[1]), 1)
                    gid = row['GAD_ID']
                    # this is to catch a weird formatting issue in the original data
                    if gene_text == '09/15/14':
                        gene_text = 'Sep15'
                    assert text[gene_span[0]:gene_span[1]] == gene_text
                    assert text[disease_span[0]:disease_span[1]] == disease_text
                    replacements = {gene_span: 'GENE', disease_span: 'DISEASE'}
                    replaced_text = replace_mentions(replacements, text)

                    origsent2replaced.setdefault(text, []).append(replaced_text)

                    d = {'uid': gid, 'seq': replaced_text, 'label': label}
                    # this is to filter out duplicates later, i.e. identical sentences with identical or different labels
                    sent2data.setdefault(replaced_text, []).append((label, d))
    for key, val in sent2data.items():
        if len(val) > 1 and len(set([elm[0] for elm in val])) > 1:
            print('\n')
            for s in val:
                print(s)
    data = []
    for orig, keys in origsent2replaced.items():
        s_data = []
        for key in keys:
            instances = sent2data[key]
            if len(instances) > 1:
                if len(set([elm[0] for elm in instances])) > 1:
                    print('\nSkipping the following instances due to ambiguous labels:')
                    for s in instances:
                        print(s)
            else:
                s_data.append(instances[0][1])
        data.append(s_data)
    train_idxs, dev_idxs, test_idxs = generate_train_dev_test_splits(len(data))
    train_data = list(itertools.chain.from_iterable([data[i] for i in train_idxs]))
    # there is one instance with label 'P'
    # filter this out
    train_data = [elm for elm in train_data if elm['label'] != 'P']
    dev_data = list(itertools.chain.from_iterable([data[i] for i in dev_idxs]))
    test_data = list(itertools.chain.from_iterable([data[i] for i in test_idxs]))
    for split, d in zip(['train', 'dev', 'test'], [train_data, dev_data, test_data]):

        write_json_lines(d, '{}_{}.jsonl'.format(out_stem, split))
    return data

if __name__=="__main__":
    preprocess_gad(['../data/gad/GAD_Y_N.csv', '../data/gad/GAD_F.csv'], '../data/preprocessed/gad')