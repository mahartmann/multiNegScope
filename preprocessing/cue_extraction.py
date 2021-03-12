
import json
import configparser
from collections import Counter

"""
provides methods to extract cues from existing datasets
"""

def write_data(fname, data):
    with open(fname, 'w') as f:
        for elm in data:
            f.write('{}\n'.format(elm))
    f.close()

def load_data(fname):
    d = []
    with open(fname) as f:
        for line in f:
            d.append(json.loads(line))
    return d

def get_clues(data):
    clues = []
    for labels, seq in data:
        cue = []
        for i, tok in enumerate(seq):
            if i == len(seq) -1:
                clues.append(' '.join(cue))
                break
            if tok == '[CUE]':
                cue.append(seq[i + 1])
                if i + 2 > len(seq)-1 or seq[i + 2] != '[CUE]':
                    clues.append(' '.join(cue))
                    cue = []
    return clues

if __name__=="__main__":
    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)
    outpath = config.get('Files', 'preproc_data')
    datasets = ['bio','ddi','sherlocken','sfuen']
    all_cues = []
    for ds in datasets:
        for splt in ['train', 'dev', 'test']:
            data = load_data('{}/{}_{}.tsv'.format(config.get('Files', 'preproc_data'), ds, splt))
            cues = get_clues([([], elm['seq']) for elm in data])
            all_cues.extend([' '.join(elm.split('_')) for elm in cues if elm != ''])
            print(all_cues)
    all_cues = [elm.lower() for elm in all_cues]

    print(len(all_cues))
    out_data = []
    for key, val in Counter(all_cues).most_common():
        out_data.append('{}\t{}'.format(key, val))
    write_data('{}/triggers/triggers_extracted_{}.txt'.format(config.get('Files', 'data'), ''.join(datasets)), out_data)
    # clean the cues
    # remove underscores
    # remove punctuation
