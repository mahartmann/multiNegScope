""""
provides methods to augment sequences with information about detected clues, e.g. lexicon based detection
"""
import spacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer

from preprocessing import annotation_reader
import configparser
import os
import json


def read_file(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def load_data(fname):
    data = []
    with open(fname) as f:
        for line in f:
            elm = json.loads(line)
            elm['labels'] = [elm['labels'][i] for i,t in enumerate(elm['seq']) if t != '[CUE]' ]
            elm['seq'] = [t for i,t in enumerate(elm['seq']) if t != '[CUE]' ]
            assert len(elm['seq']) == len(elm['labels'])
            data.append(elm)
    return data

def mark_clue(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    toks =  doc[start: end]
    for tok in toks:
        tok.set_extension("clue", default=None, force=True)
        tok._.clue = 'CUE_{}_{}'.format(start,end)



def setup_matcher(triggers, matcher):
    """
    for each trigger, add corresponding rule to the matcher
    :return:
    """
    spacy.tokens.Token.set_extension("clue", default=None, force=True)
    for elm in triggers:
        variants = [elm, elm[0].upper() + elm[1:], elm.upper()]
        for var in variants:
            pattern = [{'ORTH': tok} for tok in var.split()]
            matcher.add(var, mark_clue, pattern)


def process_doc(nlp, matcher, text, split_sents=False):
    """
    returns a list of of lists of tokens augmented with cues. also returns a list of idxs of tokens, in front of which a cue was inserted
    :param nlp:
    :param matcher:
    :param text:
    :param split_sents:
    :return:
    """
    doc = nlp(text)
    sent_ids = [(sent.start, sent.end) for sent in doc.sents]
    matches = matcher(doc)
    # identify matches over the same span (MWE matches)

    outputs = []
    insertion_idxs = []
    if split_sents:
        segment_ids = sent_ids
    else:
        segment_ids = [(sent_ids[0][0], sent_ids[-1][1])]

    for start, end in segment_ids:
        sent_outputs = []
        sent_insertion_idxs = []
        # get sentence clues
        sent_cues = set([elm._.clue for elm in doc[start:end] if elm._.clue!= None])
        for sent_cue in sent_cues:
            per_match_outputs = []
            per_match_insertion_idxs = []
            for i, elm in enumerate(doc[start:end]):
                if elm._.clue == sent_cue:
                    per_match_outputs.append('[CUE]')
                    per_match_insertion_idxs.append(i)
                per_match_outputs.append(elm.text)
            sent_outputs.append(per_match_outputs)
            sent_insertion_idxs.append(per_match_insertion_idxs)
        outputs.append(sent_outputs)
        insertion_idxs.append(sent_insertion_idxs)
    return outputs, insertion_idxs


def generate_silver_data(nlp, matcher, data):
    # multiple sequences in data can be made from the same original sequence
    observed_seqs = set()
    filtered_data = []
    for elm in data:
        labels = elm['labels']
        seq = elm['seq']
        orig_seq = [seq[i] for i, _ in enumerate(seq) if seq[i] != '[CUE]']
        orig_label = [labels[i] for i, _ in enumerate(seq) if seq[i] != '[CUE]']
        assert len(orig_seq) == len(orig_label)

        if ' '.join(orig_seq) not in observed_seqs:
            observed_seqs.add(' '.join(orig_seq))
            filtered_data.append([orig_label, orig_seq])

    silver_data = []
    for orig_labels, orig_seq in filtered_data:
        augmented_seqs, insertion_idxs = process_doc(nlp, matcher, ' '.join(orig_seq), split_sents=False)

        for augmented_seq, insertion_idx in zip(augmented_seqs[0], insertion_idxs[0]):
            insertion_idx = set(insertion_idx)
            augmented_label = []
            for i, l in enumerate(orig_labels):
                if i - 1 in insertion_idx:
                    augmented_label.append(orig_labels[i])
                augmented_label.append(orig_labels[i])
            assert len(augmented_label) == len(augmented_seq)
            silver_data.append([augmented_label, augmented_seq])
    return silver_data


def generate_clue_annotated_data(data):
    # multiple sequences in data can be made from the same original sequence
    observed_seqs = set()
    filtered_data = []
    for elm in data:
        labels = elm['labels']
        seq = elm['seq']
        orig_seq = [seq[i] for i, _ in enumerate(seq) if seq[i] != '[CUE]']
        orig_label = [labels[i] for i, _ in enumerate(seq) if seq[i] != '[CUE]']
        assert len(orig_seq) == len(orig_label)

        if ' '.join(orig_seq) not in observed_seqs:
            observed_seqs.add(' '.join(orig_seq))
            filtered_data.append([orig_label, orig_seq])

    silver_data = []
    for orig_labels, orig_seq in filtered_data:
        augmented_seqs, insertion_idxs = process_doc(nlp, matcher, ' '.join(orig_seq), split_sents=False)

        for augmented_seq, insertion_idx in zip(augmented_seqs[0], insertion_idxs[0]):
            print(augmented_seq)
            insertion_idx = set(insertion_idx)
            augmented_label = []
            for i, l in enumerate(orig_labels):
                if i - 1 in insertion_idx:
                    augmented_label.append(orig_labels[i])
                augmented_label.append(orig_labels[i])
            assert len(augmented_label) == len(augmented_seq)
            silver_data.append([augmented_label, augmented_seq])
    return silver_data




if __name__=="__main__":
    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    # it doesn't matter which models we load here because we only do white space or rule-based tokenization anyway
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab)
    matcher = Matcher(nlp.vocab)

    lexicon_file = config.get('Files', 'triggers_es')
    triggers = read_file(lexicon_file)
    setup_matcher(triggers, matcher)
    ds = 'iulaall'
    split = 'combined'
    # load data
    data = load_data(os.path.join(config.get('Files', 'preproc_data'), '{}_{}.tsv'.format(ds, split)))

    silver = generate_clue_annotated_data( data)
    for s, l in silver:
        print('{}\n{}\n\n'.format(s,l))
    #silver = generate_silver_data(nlp, matcher, data)

    #for labels, seq in silver:
    #    print('{}\t{}'.format(seq, labels))

    """
    text = 'He went there instead of her and she went there instead of him. She tried the lack of tv and he tried the lack of bla'
    doc = nlp(text)

    outputs, idxs = process_doc(nlp, matcher, text, split_sents=True)
    i = 0
    for sent_output, sent_idx in zip(outputs,idxs):
        i += 1
        print(i)
        for match_out, match_idx in zip(sent_output, sent_idx):
            print(match_out)
            print(match_idx)

    split_sents = False
    data = annotation_reader.read_bioscope(config.get('Files', 'biofull'), setting='')
    # make option for sentence splitting
    text = ' '.join([' '.join(seq) for _,seq in data[:3]])
    output, insertion_idxs = process_doc(nlp, matcher, text, split_sents=True)
    for i, elm in enumerate(output):
        print(text)
        print(elm)
        print(insertion_idxs[i])
    """


