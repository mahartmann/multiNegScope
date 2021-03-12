def get_conll_sents(fname):
    with open(fname) as f:
        sents = []
        sent = []
        for line in f:
            line = line.strip()
            if line.startswith('# sent_id') and len(sent) > 0:
                sents.append(sent)
                sent = []
            elif not line.startswith('#') and line != '':
                sent.append(line)
    return sents


def process_sent_binary_en_gum(sent):
    """
    produce binary data indicating if negation is present in the sentence
    """
    for elm in sent:
        splt = elm.split('\t')
        if splt[5] == 'Polarity=Neg':
            label = 1
            break
        else:
            label = 0
    toks = [elm.split('\t')[1] for elm in sent]
    # if label == 1:
    #    print(' '.join(toks))
    return [label, ' '.join(toks)]


def process_sent_binary_en_lines(sent):
    """
    produce binary data indicating if negation is present in the sentence
    """
    for elm in sent:
        splt = elm.split('\t')
        if splt[4] == 'NEG':
            label = 1
            break
        else:
            label = 0

    toks = [elm.split('\t')[1] for elm in sent]
    # if label == 1:
    #    print(' '.join(toks))
    return [label, ' '.join(toks)]

def process_sent_binary_es_ancora(sent):
    """
    produce binary data indicating if negation is present in the sentence
    """
    for elm in sent:
        splt = elm.split('\t')
        if splt[5] == 'Polarity=Neg' or 'PronType=Neg' in splt[5]:
            label = 1
            break
        else:
            label = 0
    toks = [elm.split('\t')[1] for elm in sent]
    if label == 1:
        print(' '.join(toks))
    return [label, ' '.join(toks)]

def process_sent_binary_zh(sent):
    """
    produce binary data indicating if negation is present in the sentence
    """
    for elm in sent:
        splt = elm.split('\t')
        if splt[5] == 'Polarity=Neg' or 'PronType=Neg' in splt[5]:
            label = 1
            break
        else:
            label = 0
    toks = [elm.split('\t')[1] for elm in sent]
    if label == 1:
        print(' '.join(toks))
    return [label, ' '.join(toks)]

def process_sent_binary_it(sent):
    """
    produce binary data indicating if negation is present in the sentence
    """
    for elm in sent:
        splt = elm.split('\t')
        if splt[5] == 'Polarity=Neg' or 'PronType=Neg' in splt[5]:
            label = 1
            break
        else:
            label = 0
    toks = [elm.split('\t')[1] for elm in sent]
    if label == 1:
        print(' '.join(toks))
    return [label, ' '.join(toks)]

def process_sent_binary_fr(sent):
    """
    produce binary data indicating if negation is present in the sentence
    """
    for elm in sent:
        splt = elm.split('\t')
        if splt[5] == 'Polarity=Neg' or 'PronType=Neg' in splt[5]:
            label = 1
            break
        else:
            label = 0
    toks = [elm.split('\t')[1] for elm in sent]
    if label == 1:
        print(' '.join(toks))
    return [label, ' '.join(toks)]

def read_udep(fname, ds):
    data = []
    sents = get_conll_sents(fname)
    assert ds in set(['udengum', 'udenlines', 'udenpartut',\
                      'udesancora', 'udesgsd',\
                      'udzhgsd', 'udzhgsdsimp',\
                      'uditisdt','uditpostwita', 'uditpartut', 'udittwittiro', 'uditvit',
                      'udfrgsd', 'udfrpartut', 'udfrsequoia'])
    for sent in sents:
        if ds == 'udengum':
            data.append(process_sent_binary_en_gum(sent))
        elif ds == 'udenlines':
            data.append(process_sent_binary_en_lines(sent))
        elif ds == 'udenpartut':
            data.append(process_sent_binary_en_gum(sent))
        elif ds == 'udesancora':
            data.append(process_sent_binary_es_ancora(sent))
        elif ds == 'udesgsd':
            data.append(process_sent_binary_es_ancora(sent))
        elif ds == 'udzhgsd':
            data.append(process_sent_binary_zh(sent))
        elif ds == 'udzhgsdsimp':
            data.append(process_sent_binary_zh(sent))
        elif ds == 'uditisdt':
            data.append(process_sent_binary_it(sent))
        elif ds == 'uditpartut':
            data.append(process_sent_binary_it(sent))
        elif ds == 'uditpostwita':
            data.append(process_sent_binary_it(sent))
        elif ds == 'udittwittiro':
            data.append(process_sent_binary_it(sent))
        elif ds == 'uditvit':
            data.append(process_sent_binary_it(sent))
        elif ds == 'udfrgsd':
            data.append(process_sent_binary_fr(sent))
        elif ds == 'udfrpartut':
            data.append(process_sent_binary_fr(sent))
        elif ds == 'udfrsequoia':
            data.append(process_sent_binary_fr(sent))

    return data