"""
Methods for reading in text from various formats and bring them into the required input format for the negation scope reolution pipeline
"""
import nltk
import json
import argparse
from nltk import sent_tokenize


def read_data_from_plain_text(fname):
    """
    read in data from plain text file
    sentences are tokenized using the spacy tokenizer
    """
    with open(fname) as f:
        sents = []
        for line in f:
            line = line.strip()
            if line != '':
                for sent in sent_tokenize(line):
                    print(sent)
                    sents.append({'seq': sent.split(' '), 'uid': len(sents)})
    return sents

def write_data(data, outfile):
    with open(outfile, 'w') as f:
        for elm in data:
            f.write(json.dumps(elm) + '\n')
    f.close()

if __name__=="__main__":
    fname = './examples/Pt.txt'
    sents = read_data_from_plain_text(fname)
    write_data(sents, './examples/Pt.jsonl')