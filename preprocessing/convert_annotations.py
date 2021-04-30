import json
import configparser




def load_data(fname):
    d = []
    with open(fname) as f:
        for line in f:
            d.append(json.loads(line))
    return d

if __name__=="__main__":

    for ds in ['sfuen', 'sfuenequal', 'bio', 'bioequal']:
        for split in ['train', 'dev', 'test']:
            fname = '/home/mareike/PycharmProjects/multiNegScope/data/preprocessed/{}_{}.json'.format(ds, split)
            fname_out = '/home/mareike/PycharmProjects/multiNegScope/data/preprocessed/{}conv_{}.json'.format(ds, split)
            d = load_data(
                fname)
            punctuation = set([elm for elm in ':;.,?!'])
            print(punctuation)
            cfg = '/home/mareike/PycharmProjects/negscope/code/mt-dnn/preprocessing/config.cfg'
            config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            config.read(cfg)

            converted_data = []
            orig_labels = []
            for elm in d:
                seq = elm['seq']
                labels = elm['labels']
                print(labels)
                orig_labels.append(elm['labels'].copy())
                # convert punctuation
                for i, tok in enumerate(seq):
                    if tok in punctuation and labels[i] == 'I':
                        if i == len(seq)-1 or labels[i+1] != 'I':
                            for s, l in zip(seq, labels):
                                print(s,l)
                            print('\n')
                            labels[i] = 'O'
                for i, tok in enumerate(seq):
                    if tok == '[CUE]':
                        labels[i] = 'I'
                        labels[i+1] = 'I'
                # make all CUE tokens in scope
                elm['label'] = labels
                converted_data.append(elm)
            for elm, oll in zip(converted_data, orig_labels):
                print('\n')
                seq = elm['seq']
                for tok, label, ol in zip(seq,elm['labels'], oll):
                    print('{}\t{} --> {}'.format(tok, ol, label))
            print(fname_out)
            with open(fname_out, 'w') as f:
                for elm in converted_data:
                    f.write(json.dumps(elm) + '\n')
            f.close()
