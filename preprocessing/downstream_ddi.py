import json
from xml.etree import ElementTree as ET
from preprocessing.annotation_reader import read_ddi_doc
from predict_and_combine import add_span_labels_within, get_spans
from preprocessing.data_splits import write_train_dev_test_data_ddi
def replace_mentions(replacements, text):
    # produce a string where mentions of particpants are replaced by their labels
    spans = sorted(list(replacements.keys()), key=lambda x: x[1])
    new_text = ''
    idxs = [elm for elm in range(len(text.split()))]
    for i, span in enumerate(spans):
        start = span[0]
        end = span[1]
        pid = span[2]
        if i == 0:
            new_text += text[:start] + '@{}{}$'.format(replacements[span], pid)
        else:
            new_text += text[spans[i - 1][1]:start] + '@{}{}$'.format(replacements[span], pid)
        if i == len(spans) - 1:
            new_text += text[end:]
        # update the idxs in case MWEs are replaced
        num_replace = len(text[start:end].split()) -1

        for i, elm in enumerate(idxs):
            if i > len(text[:start].split()):
                idxs[i] += num_replace

    new_text = new_text.replace('  ', ' ')
    idxs = idxs[:len(new_text.split())]
    return new_text, idxs

def read_ddi_neg_relations(fname, add_offset=True):
    try:
        root = ET.parse(fname).getroot()
        entities = {}
        relations = {}
        outdata = []
        out_sent = []
        for sentence in root.iter('sentence'):
            sent_text = sentence.attrib['text']
            for entity in sentence.iter('entity'):
                offsets = entity.attrib['charOffset'].split(';')
                for ofs in offsets:
                    splt = ofs.split('-')

                    start = int(splt[0])
                    if add_offset:
                        end = int(splt[1])+1
                    else:
                        end = int(splt[1])
                    did = entity.attrib['id']
                    type = entity.attrib['type']
                    text = entity.attrib['text']
                    entities[did] = {'start': start, 'end': end, 'type': type, 'text': text}
                    break
            for pair in sentence.iter('pair'):
                label = pair.attrib['ddi']
                e1 = pair.attrib['e1']
                e2 = pair.attrib['e2']
                replaced_text, adjusted_idxs = replace_mentions(text=sent_text, replacements={(entities[e1]['start'], entities[e1]['end'],''): 'DRUG',
                                 (entities[e2]['start'], entities[e2]['end'],''): 'DRUG'})
                out_sent.append((sent_text, replaced_text, adjusted_idxs, label))
            outdata.append(out_sent)
            out_sent = []
        return outdata
    except ET.ParseError:
        raise ET.ParseError

def load_data(p, add_offset, add_neg=True,  include_all=True):
    reldata = []
    negdata = []
    sent_data = []
    for fi, f in enumerate([p + f for f in os.listdir(p) if f.endswith('_cleaned.xml')]):
        print(f)
        print(fi)
        if f.endswith('Arformoterol_ddi_cleaned.xml') or \
                f.endswith('L-Histidine_ddi_cleaned.xml') or \
                f.endswith('Chlorothiazide_ddi_cleaned.xml') or \
                f.endswith('Amiloride_ddi_cleaned.xml') or \
                f.endswith('Levetiracetam_ddi_cleaned.xml') or \
                f.endswith('Abciximab_ddi_cleaned.xml') or \
                f.endswith('Hydrochlorothiazide_ddi_cleaned.xml') or \
                f.endswith('Aprepitant_ddi_cleaned.xml') or \
                f.endswith('Erlotinib_ddi_cleaned.xml') or \
                f.endswith('Clonazepam_ddi_cleaned.xml') or \
                f.endswith('Methylergonovine_cleaned.xml'):
            continue

        try:
            n, cue_data = read_ddi_doc(f, setting='embed', include_all=True)

            negdata.extend(n)
            reldata.extend(read_ddi_neg_relations(f, add_offset=add_offset))
        except ET.ParseError:
            pass
        outdata= []
        for rel_sent, neg_sents in zip(reldata, negdata):

            if len(rel_sent) > 0 and len(neg_sents) > 0:

                combined_neg_labels = ['O' for elm in neg_sents[0][0]]
                for sent in neg_sents:
                    #print(sent)
                    combined_neg_labels = ['I' if sent[0][i] == 'I' else combined_neg_labels[i] for i, _ in
                                           enumerate(sent[0])]
                #print('1 combined labels {}, {}'.format(combined_neg_labels, len(combined_neg_labels)))
                for elm in rel_sent:

                    adjusted_neg_labels = [combined_neg_labels[i] for i in elm[2]]
                    #print('2 adjusted labels {}, {}'.format(adjusted_neg_labels, len(adjusted_neg_labels)))
                    spans = get_spans(adjusted_neg_labels)

                    if not include_all:
                        if len(spans) > 0:
                            if add_neg:
                                mod_sent = add_span_labels_within(spans, elm[1].split())
                            else:
                                mod_sent = add_span_labels_within({}, elm[1].split())
                    else:
                        if add_neg:
                            mod_sent = add_span_labels_within(spans, elm[1].split())
                        else:
                            mod_sent = add_span_labels_within({}, elm[1].split())
                    sent_data.append({'label':elm[3], 'seq': ' '.join(mod_sent)})
                    #print({'label':elm[3], 'seq': mod_sent})
                outdata.append(sent_data)
                sent_data = []
    return outdata

if __name__=="__main__":
    import configparser
    import os
    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    p_test = '/home/mareike/PycharmProjects/negscope/data/ddi_drug/NegDrugBank2013/TestNegDrugbank2013/'
    p_train = '/home/mareike/PycharmProjects/negscope/data/ddi_drug/NegDrugBank2013/TrainNegDrugbank2013/'

    include_all = True
    add_neg = True
    fstem = 'ddinegrelationsallgold'

    train_data = load_data(p_train, include_all=include_all, add_neg=add_neg, add_offset=True)
    for elm in train_data:
        for e in elm:
            print('elm {}'.format(e))
    test_data = load_data(p_test, include_all=include_all, add_neg=add_neg, add_offset=False)
    write_train_dev_test_data_ddi(fstem=fstem, train_data=train_data, test_data=test_data)

