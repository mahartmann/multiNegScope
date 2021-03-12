import json
import configparser
from prepro_std import setup_customized_tokenizer
from pretrained_models import BertTokenizer





def load_predictions(fname):
    with open(fname) as f:
        d = json.load(f)
    return d['predictions']


def load_data(fname):
    d = []
    with open(fname) as f:
        for line in f:
            d.append(json.loads(line))
    return d

if __name__=="__main__":
    tokenizer_class = BertTokenizer
    model = 'bert-base-multilingual-cased'
    do_lower_case = False
    ds = 'biosfuensfues'
    for split in ['train', 'dev', 'test']:
        fname = '/home/mareike/PycharmProjects/negscope/data/formatted/{}/{}_{}.json'.format(model, ds, split)
        fname_out = '/home/mareike/PycharmProjects/negscope/data/formatted/{}/{}conv_{}.json'.format(model, ds, split)
        d = load_data(
            fname)
        punctuation = set([elm for elm in ':;.,?!'])
        print(punctuation)
        cfg = '/home/mareike/PycharmProjects/negscope/code/mt-dnn/preprocessing/config.cfg'
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(cfg)
        tokenizer = setup_customized_tokenizer(tokenizer_class=tokenizer_class, model=model, do_lower_case=do_lower_case, config=config)

        #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        #tokenizer = setup_customized_tokenizer('bert-base-multilingual-cased', BertTokenizer,do_lower_case=False, config=config)
        converted_data = []
        orig_labels = []
        for elm in d:
            seq = tokenizer.convert_ids_to_tokens(elm['token_id'])
            labels = elm['label']
            orig_labels.append(elm['label'].copy())
            # convert punctuation
            for i, tok in enumerate(seq):
                if tok in punctuation and labels[i] == 1:
                    if i == len(seq) or labels[i+1] != 1:
                        for s, l in zip(seq, labels):
                            print(s,l)
                        print('\n')
                        labels[i] = 0
            for i, tok in enumerate(seq):
                if tok == '[CUE]':
                    labels[i] = 1
                    labels[i+1] = 1
            # make all CUE tokens in scope
            elm['label'] = labels
            converted_data.append(elm)
        for elm, oll in zip(converted_data, orig_labels):
            print('\n')
            seq = tokenizer.convert_ids_to_tokens(elm['token_id'])
            for tok, label, ol in zip(seq,elm['label'], oll):
                print('{}\t{} --> {}'.format(tok, ol, label))
        print(fname_out)
        with open(fname_out, 'w') as f:
            for elm in converted_data:
                f.write(json.dumps(elm) + '\n')
        f.close()
