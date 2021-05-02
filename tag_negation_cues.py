"""
loads a negation cue lexicon and tags all negation cues found in a sentence
sentences are preprocessed by: sentence tokenization, tokenization
"""

import spacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer

import json
import argparse

from visualization.heatmap import html_heatmap
from util import create_logger


def read_file(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def load_data(fname, labeled=False):
    data = []
    with open(fname) as f:
        for line in f:
            elm = json.loads(line)
            elm['seq'] = [t for i,t in enumerate(elm['seq']) if t != '[CUE]' ]
            if labeled:
                elm['labels'] = [elm['labels'][i] for i, t in enumerate(elm['seq']) if t != '[CUE]']
                assert len(elm['seq']) == len(elm['labels'])
            data.append(elm)
    return data

def mark_cue(matcher, doc, i, matches):
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
            matcher.add(var, [pattern], on_match=mark_cue)


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


def main(args):
    logger = create_logger('logger')
    # it doesn't matter which models we load here because we only do white space or rule-based tokenization anyway
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab)
    matcher = Matcher(nlp.vocab)

    triggers = read_file(args.cue_list)
    setup_matcher(triggers, matcher)

    # load data
    data = load_data(args.input_file)

    tagged_sentences = []

    for seqid, _ in enumerate(data):
        observed_sequences = set()
        out, idxs = process_doc(nlp, matcher, ' '.join(data[seqid]['seq']), split_sents=False)
        for sid, elm in enumerate(out[0]):
            if ' '.join(elm) not in observed_sequences:
                tagged_sentences.append({'uid': len(tagged_sentences), 'seq': elm, 'sid': '{}_{}'.format(seqid, sid)})
                observed_sequences.add(' '.join(elm))
    logger.info('Writing tagged sequences to {}'.format('.'.join(args.input_file.split('.')[:-1]) + '#cues.jsonl'))
    with open('.'.join(args.input_file.split('.')[:-1]) + '#cues.jsonl', 'w') as fout:
        for elm in tagged_sentences:
            fout.write(json.dumps(elm) + '\n')
    fout.close()

    # produce html with colored negation cues
    logger.info('Writing html for visualization to {}'.format('.'.join(args.input_file.split('.')[:-1]) + '#cues.html'))
    html = []
    for seq in tagged_sentences:
        seq = seq['seq']
        labels = ['O']*len(seq)
        for i, tok in enumerate(seq):
            if tok == '[CUE]':
                labels[i] = 'CUE'
                if i < len(labels)-1:
                    labels[i+1] = 'CUE'
        html.append(html_heatmap(words = [elm for elm in seq if elm != '[CUE]'] + ['<br>'], labels = [elm for i,elm in enumerate(labels) if seq[i] != '[CUE]'] + ['O']))
    with open('.'.join(args.input_file.split('.')[:-1]) + '#cues.html', 'w') as fout:
        for elm in html:
            fout.write(elm + '\n')
    fout.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Train MTL model')
    parser.add_argument('--cue_list',
                        default='./data/cues/cues_danish.txt')
    parser.add_argument('--input_file',
                        default='./examples/Pt.jsonl')

    args = parser.parse_args()

    main(args)
