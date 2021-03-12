import configparser
import json
import numpy as np

def load_data(fname):
    data = []
    with open(fname) as f:
        for line in f:
            data.append(json.loads(line))
    print('Loaded {} instances from {}'.format(len(data), fname))
    return data

def write_data(fname, data):
    print('Writing {} instances to {}'.format(len(data), fname))
    with open(fname, 'w') as f:
        for elm in data:
            f.write(json.dumps(elm) + '\n')
    f.close()

if __name__=="__main__":
    import configparser

    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)
    outpath = config.get('Files', 'preproc_data')

    datasets = ['bio', 'sfuen', 'sfues']
    for splt in ['train', 'dev', 'test']:
        data = []
        for ds in datasets:
            data.extend(load_data('{}/{}_{}.tsv'.format(config.get('Files', 'preproc_data'), ds, splt)))
        np.random.seed(42)
        np.random.shuffle(data)
        out_data = []
        for elm in data:
            elm['uid'] = len(out_data)
            out_data.append(elm)
        write_data('{}/{}_{}.tsv'.format(config.get('Files', 'preproc_data'), ''.join(datasets), splt), out_data)