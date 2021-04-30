import json
import configparser
#from prepro_std import setup_customized_tokenizer
#from pretrained_models import BertTokenizer


from transformers import BertTokenizer



def load_data(fname):
    d = []
    with open(fname) as f:
        for line in f:
            d.append(json.loads(line))
    return d

if __name__=="__main__":

    for ds in ['sfuen', 'sfuenequal', 'bio', 'bioequal']:

        tokenized_data = []
        for split in ['train', 'dev', 'test']:
            fname = '/home/mareike/PycharmProjects/negscope/data/experiments/{}_{}.tsv'.format(ds, split)
            fname_out = '/home/mareike/PycharmProjects/multiNegScope/data/preprocessed/{}_{}.json'.format(ds, split)
            d = load_data(
                fname)

            cfg = '/home/mareike/PycharmProjects/negscope/code/mt-dnn/preprocessing/config.cfg'
            config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            config.read(cfg)

            converted_data = []
            orig_labels = []

            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            for elm in d:
                #print(elm)
                seq = elm['seq']
                labels = elm['labels']
                tokenized_seq = []
                expanded_labels = []
                for label, tok in zip(labels,seq):
                    if tok == "[CUE]":
                        tokenized_seq.append(tok)
                        expanded_labels.append(label)
                    else:
                        tokens = tokenizer.tokenize(tok)
                        if len(tokens) == 1:
                            tokenized_seq.append(tokens[0])
                            expanded_labels.append(label)
                        elif len(tokens)== 0:
                            continue
                        else:
                            # join the subwords, i.e. attach everything that is '##' to it's preceding word
                            joined_tokens = [tokens[0]]
                            i = 1
                            while i < len(tokens):
                                if tokens[i].startswith('##'):
                                    joined_tokens[-1] = joined_tokens[-1] + tokens[i].split('##')[-1]
                                    i += 1
                                else:
                                    joined_tokens.append(tokens[i])
                                    i += 1
                                    if i < len(tokens) and not tokens[i].startswith('##'):
                                        joined_tokens.append(tokens[i])
                                        i += 1

                            for t in joined_tokens:
                                tokenized_seq.append(t)
                                expanded_labels.append(label)
                    assert len(tokenized_seq) == len(expanded_labels)
                elm['seq'] = tokenized_seq
                elm['labels'] = expanded_labels
                tokenized_data.append(elm)
            with open(fname_out, 'w') as f:
                for elm in tokenized_data:
                    f.write(json.dumps(elm) + '\n')
            f.close()