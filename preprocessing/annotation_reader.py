from preprocessing.downstream_tasks import read_drugs, read_gad, read_biorelex, read_cdr, read_ade_doc, read_ddi_relations, read_m2c2assert, read_chemprot_relations
from preprocessing.nested_xml import dfs, dfs3, build_surface, build_surface_ddi
from preprocessing.data_splits import *
import xml.etree.ElementTree as ET
import itertools
import os

import spacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
import ast

from preprocessing import udep as udep
from preprocessing.data_splits import write_train_dev_udep, shuffle_and_prepare

def read_file(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]



def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def read_bioscope(fname, setting='augment', cue_type='negation', include_all=False):
    root = ET.parse(fname).getroot()
    data = []
    cue_data = []

    for doc in root.iter('Document'):
        for part in doc.iter('DocumentPart'):
            for sent in part.iter('sentence'):


                print('\n{}'.format(sent.attrib['id']))

                children, p2c, c2p = dfs([], {}, {}, sent)
                siblings = dfs3(set(), p2c, c2p, sent,  {}, 0)
                constituents = build_surface({}, p2c, c2p, sent, siblings, 0)
                # print(constituents[sent])
                # collect all active cues


                def get_label(tag):
                    if tag.tag == 'xcope':
                        return '{}-{}'.format(tag.tag, tag.attrib['id'])
                    elif tag.tag == 'cue':
                        return '{}-{}-{}'.format(tag.tag, tag.attrib['type'], tag.attrib['ref'])
                    else:
                        return tag.tag

                cids = set([get_label(elm[1]).split('-')[-1] for elm in constituents[sent] if
                            get_label(elm[1]).startswith('cue-{}'.format(cue_type))])
                print('Labels: {}'.format(cids))
                sent_data = []

                if include_all and len(cids) == 0:
                    print("################## This sentence has no negation. Adding to ds")
                    cids = [None]

                for cid in cids:
                    toks = []
                    labels = []
                    cue_labelseq = []

                    for chunk, tag in constituents[sent]:

                        def get_all_tags(node, c2p):
                            # retrieve tags of the node and all its parents
                            tags = [get_label(node)]
                            while node in c2p:
                                tags.append(get_label(c2p[node]))
                                node = c2p[node]
                            return tags

                        all_tags = set(get_all_tags(tag, c2p))
                        if chunk is not None:
                            for t in chunk.split():
                                is_cue = 0
                                if 'cue-{}-{}'.format(cue_type, cid) in all_tags:
                                    is_cue = 1
                                    if setting == 'augment':
                                        print('{}\t{}\t{}'.format('[CUE]', 'I', ' '.join(get_all_tags(tag, c2p))))
                                        toks.append('[CUE]')
                                        labels.append('I')
                                        label = 'I'
                                        cue_labelseq.append(is_cue)
                                    elif setting == 'replace':
                                        t = '[CUE]'
                                        label = 'I'


                                elif 'xcope-{}'.format(cid) in all_tags:
                                    label = 'I'
                                else:
                                    label = 'O'
                                toks.append(t)
                                labels.append(label)
                                cue_labelseq.append(is_cue)

                                print('{}\t{}\t{}'.format(t, label, ' '.join(get_all_tags(tag, c2p))))

                    sent_data.append([labels, toks, cue_labelseq])
                if len(sent_data) > 0:
                    data.append(sent_data)
                    # get clue annotated data
                    cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data

def get_clue_annotated_data(sent_data):
    seq = [elm for elm in sent_data[0][1] if elm != '[CUE]']
    labels = ['0'] *len(seq)
    for i, sent in enumerate([elm[1] for elm in sent_data]):

        idx = 0
        for tok in sent:

            if tok != '[CUE]':
                idx += 1
            else:
                labels[idx] = '1_{}'.format(i)
    print(labels)
    return [labels, seq]


def read_sherlock(fname, setting='augment', include_all=False):
    lines = read_file(fname)
    sents = {}
    for line in lines:
        splt = line.split('\t')
        if len(splt) > 1:
            cid = splt[0]
            sid = splt[1]
            sents.setdefault(sid + cid, []).append(line)
    data = []
    cue_data = []
    # first cue at index 7
    for key, lines in sents.items():
        cols = {}
        for line in lines:
            splt = line.split('\t')
            if len(splt) >= 9:
                for cid, elm in enumerate(splt):
                    cols.setdefault(cid, []).append(elm)
        negs = {}
        cues = {}
        if len(cols) > 7:
            for nid in range(7, len(cols), 3):
                for tid, elm in enumerate(cols[nid]):
                    if elm != '_':
                        cues.setdefault(nid, []).append(tid)
                for tid, elm in enumerate(cols[nid + 1]):
                    if elm != '_':
                        negs.setdefault(nid, []).append(tid)

        tid2neg = {}
        tid2cues = {}
        for nid, tids in negs.items():
            for tid in tids:
                tid2neg.setdefault(tid, set()).add(str(nid))
        for nid, tids in cues.items():
            for tid in tids:
                tid2cues.setdefault(tid, set()).add(str(nid))
        negations = tid2neg.values()
        negations = list(set(list(itertools.chain.from_iterable([list(elm) for elm in negations]))))
        if include_all and len(negations) == 0:
            print("################## This sentence has no negation. Adding to ds")
            negations = [None]
        if len(negations) > 0:
            print('\n')
            print(negations)
            sent_data = []
            for negation in negations:
                toks = []
                labels = []
                cue_labelseq = []
                for tid, line in enumerate(lines):
                    splt = line.split('\t')
                    is_cue = 0
                    if tid in tid2neg and negation in tid2neg[tid]:
                        neg = 'I'
                    else:
                        neg = 'O'

                    if tid in tid2cues and negation in tid2cues[tid]:
                        cue_print = 'CUE{}'.format(' '.join(tid2cues[tid]))
                        is_cue = 1
                        if setting == 'augment':
                            print('CUE\t{}\t{}'.format(neg, cue_print))
                            print('{}\t{}'.format(splt[3], neg))
                            toks.append('[CUE]')
                            labels.append(neg)
                            cue_labelseq.append(is_cue)
                    toks.append(splt[3])
                    labels.append(neg)
                    cue_labelseq.append(is_cue)



                sent_data.append([labels, toks, cue_labelseq])
            if len(sent_data) > 0:
                data.append(sent_data)
                cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data



def read_IULA(path, setting='augment', include_all=False):
    fs = sorted(list(set([elm for elm in os.listdir(path) if elm.endswith('.txt')])))
    fnames = ['{}/{}'.format(path, f) for f in fs]
    anno_names = ['{}/{}.ann'.format(path, f.split('.txt')[0]) for f in fs]
    data = []
    cue_data = []
    for fname, anno_name in zip(fnames, anno_names):
        print(fname)
        data_ex, cue_data_ex = read_IULA_doc(fname, anno_name, setting, include_all=include_all)
        data.extend(data_ex)
        cue_data.extend(cue_data_ex)
    return data, cue_data

def read_IULA_doc(fname_txt, fname_anno, setting, include_all):
    class Token(object):
        def __init__(self, start, end, surf, tid):
            self.c_start = start
            self.c_end = end
            self.surface = surf
            self.tid = tid

        def set_label(self, label):
            self.label = label

        def __lt__(self, other):
            val1 = self.c_start + self.c_end
            val2 = other.c_start + other.c_end
            if val1 < val2:
                return 1
            else:
                return -1

    data = []
    cue_data = []
    tid2tok = {}
    span2tok = {}

    anno_lines = read_file(fname_anno)

    def read_file_wn(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return ' '.join([line for line in lines])

    s = read_file_wn(fname_txt)
    labels = set()
    annos = {}
    for line in anno_lines:
        splt = line.split('\t')
        labels.add(splt[1].split()[0])
        annos.setdefault(splt[1].split()[0], []).append(line)

    for line in anno_lines:
        splt = line.split('\t')[1]

        if line.split('\t')[0].startswith('T'):
            gold_surf = line.split('\t')[-1].strip()

            start_orig = int(splt.split(' ')[1])
            end_orig = int(splt.split(' ')[2])
            start = start_orig
            end = end_orig
            surf = s[start:end]

            if gold_surf != s[start:end]:

                start = start_orig - 1
                end = end_orig - 1
                surf = s[start:end]
                if surf == gold_surf:
                    pass
                else:
                    start = start_orig - 2
                    end = end_orig - 2
                    surf = s[start:end]
                    if surf == gold_surf:
                        pass
                    else:
                        start = start_orig - 3
                        end = end_orig - 3
                        surf = s[start:end]
                        if surf == gold_surf:
                            pass
                        else:
                            start = start_orig - 4
                            end = end_orig - 4
                            surf = s[start:end]
                            if surf == gold_surf:
                                pass
                            else:
                                print('############### {} {}'.format(line, surf))
            else:
                pass

            tid = line.split('\t')[0]
            label = splt.split()[0]
            tok = Token(start, end, surf, tid)

            tok.set_label(label)
            tid2tok[tid] = tok
            span2tok.setdefault(start, []).append(tok)
            for st in range(start, end):
                span2tok.setdefault(st, []).append(tok)

    span2r = {}
    for line in anno_lines:

        if line.split('\t')[0].startswith('R'):
            splt = line.split('\t')[1].split()
            rid = line.split('\t')[0]
            label = line.split('\t')[1].split()[0]
            for elm in splt:
                if ':' in elm:
                    span = (tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end)
                    span2r.setdefault(tid2tok[elm.split(':')[-1]].c_start, []).append((span, elm, rid, label))
                    for st in range(tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end):
                        span2r.setdefault(st, []).append(((st, tid2tok[elm.split(':')[-1]].c_end), elm, rid, label))

    i = 0

    surf2span = []
    chars = []
    while i < len(s) - 1:

        # add chars up to next whitespace
        c = s[i]
        start = i
        while i < len(s) - 1:
            if c == ' ':
                i += 1
                c = s[i]
                break
            chars.append(c)
            i += 1
            c = s[i]

        surf2span.append((''.join(chars), start))
        chars = []
    sents = []
    sent = []

    for surf, span in surf2span:
        if '\n' in surf:
            sent.append((surf.strip('\n'), span))
            sents.append(sent)
            sent = []
        else:
            sent.append((surf, span))

    for sent in sents:
        tok2labels = []

        for surf, span in sent:

            if surf != '':
                labels = []
                if span in span2r:
                    labels.extend(['{}:{}:{}'.format(elm[3], elm[1], elm[2]) for elm in sorted(span2r[span])])

                if span in span2tok:
                    labels.extend(['{}:_{}'.format(elm.tid, elm.label) for elm in sorted(span2tok[span])])
                else:
                    labels.append('Unlabeled')
                tok2labels.append((surf, set(labels)))
        sent_labels = set()
        for _, labels in tok2labels:
            for label in sorted(labels):
                if label.split(':')[-1].startswith('R'):
                    sent_labels.add(label.split(':')[-1])
        sent_data = []
        if include_all and len(sent_labels) == 0:
            print("################## This sentence has no negation. Adding to ds")
            sent_labels = [None]

        for sent_label in sorted(sent_labels):
            print('\n')
            print(sent_label)
            out_toks = []
            out_labels = []
            cue_labelseq =[]
            for tok, labels in tok2labels:
                is_cue = 0
                def tok2sent_labels(labels):
                    return set([label.split(':')[-1] for label in sorted(labels) if label.split(':')[-1].startswith('R')])


                if sent_label in tok2sent_labels(labels):
                    out_label = 'I'
                    print('{}\t{}'.format(tok, labels))
                    if len([elm for elm in labels if 'NegMarker' in elm or 'NegPredMarker' in elm or 'NegPolItem' in elm]) > 0:
                        is_cue = 1
                        if setting == 'augment':
                            out_toks.append('[CUE]')
                            out_labels.append(out_label)
                            cue_labelseq.append(is_cue)
                else:
                    out_label = 'O'
                    print('{}\tUnlabeled'.format(tok))
                out_toks.append(tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)

            sent_data.append([out_labels, out_toks, cue_labelseq])
            print('cue labelseq {}'.format(cue_labelseq))
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data

def join_cue_and_scope_labels(data):
    joined_data = []
    for sents in data:
        joined_sents = []
        for sent in sents:
            for i, cue_label in enumerate(sent[2]):

                 labels = sent[0]
                 toks = sent[1]
                 if cue_label == 1:
                     labels[i] = 'C'
            joined_sents.append([labels, toks, sent[2]])
        joined_data.append(joined_sents)
    return joined_data

def read_sfu_en(path, setting='augment',include_all=False):
    data = []
    cue_data = []
    for topic in [elm for elm in os.listdir(path) if os.path.isdir(os.path.join(path, elm))]:
        for fname in [os.path.join(path, topic, elm) for elm in os.listdir(os.path.join(path, topic))]:
            data_ex, cue_data_ex = read_sfu_en_doc(fname, setting,include_all=include_all)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
    return data, cue_data

def read_sfu_en_doc(fname, setting,include_all):

    def get_all_parents(elm, c2p):
        parents = set()
        while True:
            if elm in c2p:
                parents.add(c2p[elm])
                elm = c2p[elm]
            else:
                break
        return parents

    def get_tag(elm):
        if elm.tag == 'cue':
            return '{}-{}-{}'.format('cue', elm.attrib['type'], elm.attrib['ID'])
        elif elm.tag == 'xcope':
            return '{}-{}'.format('xcope', elm.find('ref').attrib['SRC'])
        else:
            return elm.tag

    def walk(sent, toks, labels, c2p):
        for elm in list(sent):
            c2p[elm] = sent
            if elm.tag == 'W':
                toks.append(elm.text)
                labels.append(set([get_tag(elm) for elm in get_all_parents(elm, c2p)]))
            elif elm.tag == 'cue':
                walk(elm, toks, labels, c2p)
            elif elm.tag == 'xcope':
                if elm.find('ref') is not None:
                    walk(elm, toks, labels, c2p)
                else:
                    continue
            elif elm.tag == 'C':
                walk(elm, toks, labels, c2p)
        return toks, labels, c2p


    root = ET.parse(fname).getroot()
    data = []
    cue_data = []
    print(fname)
    for p in root.iter('P'):
        for sent in p.iter('SENTENCE'):
            toks, labels, c2p = walk(sent, [], [], {})
            # collect all negation cues
            cues = []
            for l in labels:
                cues.extend([elm for elm in l if 'negation' in elm])
            cues = set(cues)
            if include_all and len(cues) == 0:
                print("################## This sentence has no negation. Adding to ds")
                cues = ['XXX-X']
            if len(cues) > 0:
                sent_data = []
                for cue in cues:
                    outtoks = []
                    outlabels = []
                    cue_labelseq = []
                    for t, l in zip(toks, labels):
                        lsurf = 'O'
                        is_cue = 0
                        if 'xcope-{}'.format(cue.split('-')[-1]) in l:
                            lsurf = 'I'
                            print('{}\t{}'.format(t, lsurf))
                            # print('{}\t{}'.format(t, 'xcope-{}'.format(cue.split('-')[-1])))
                        if cue in l:
                            is_cue = 1
                            if setting == 'augment':
                                print('{}\t{}'.format('[CUE]', lsurf))
                                outtoks.append('[CUE]')
                                outlabels.append(lsurf)
                                cue_labelseq.append(is_cue)

                        print('{}\t{}'.format(t, lsurf))

                        def replace_hexcodes(t):
                            t = t.replace('\x85', ' ')
                            t = t.replace('\x92', "'")
                            t = t.replace('\x93', '"')
                            t = t.replace('\x94', '"')
                            t = t.replace('\x97', ' ')
                            return t

                        t = replace_hexcodes(t).strip()
                        if t != ' ' and t != '':
                            for subtok in t.split(' '):
                                outtoks.append(subtok)
                                outlabels.append(lsurf)
                                cue_labelseq.append(is_cue)

                    sent_data.append([outlabels, outtoks, cue_labelseq])
                if len(sent_data) > 0:
                    data.append(sent_data)
                    cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data


def read_sfu_es(path, setting='augment', include_all=False):
    data = []
    cue_data = []
    for topic in [elm for elm in os.listdir(path) if os.path.isdir(os.path.join(path, elm))]:
        for fname in [os.path.join(path, topic, elm) for elm in os.listdir(os.path.join(path, topic))]:
            data_ex, cue_data_ex = read_sfu_es_doc(fname, setting,include_all=include_all)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
    return data, cue_data


def read_sfu_es_doc(fname, setting,include_all):

    def get_label(elm, cneg):
        return '{}-{}'.format(elm.tag, cneg)

    def parse_scope(init_elm, toks, labels, c2p, cneg):
        if init_elm is None:
            # print('elm is None {}'.format(init_elm))
            return toks, labels
        for child in list(init_elm):
            # print('add parent {} {} {}'.format(child, init_elm, type(init_elm)))
            c2p[child] = init_elm
            if child.tag == 'neg_structure':
                cneg += 1
                scope = child.find('scope')
                if scope is not None:
                    parse_scope(scope, toks, labels, c2p, cneg)
            elif child.tag == 'negexp':
                for elm in list(child):
                    c2p[elm] = child
                    if 'lem' in elm.attrib:
                        toks.append(elm.attrib['lem'])
                        labels.append(
                            [get_label(child, cneg)] + [get_label(p, cneg) for p in get_all_parents(elm, c2p)])
                    else:
                        print(elm)
            elif child.tag == 'event':
                for elm in list(child):
                    c2p[elm] = child
                    if 'lem' in elm.attrib:
                        toks.append(elm.attrib['lem'])
                        labels.append(
                            [get_label(child, cneg)] + [get_label(p, cneg) for p in get_all_parents(elm, c2p)])
            elif child.tag == 'scope':
                parse_scope(child, toks, labels, c2p, cneg)
            else:
                if 'lem' not in child.attrib:
                    print(child, print(c2p[child]))
                else:
                    toks.append(child.attrib['lem'])
                    # print([p for p in get_all_parents(child, c2p)])
                    labels.append([get_label(child, cneg)] + [get_label(p, cneg) for p in get_all_parents(child, c2p)])
        return toks, labels

    def get_all_parents(elm, c2p):
        parents = set()
        while True:
            if elm in c2p:
                parents.add(c2p[elm])
                elm = c2p[elm]
            else:
                break
        return parents


    data = []
    cue_data = []
    print('########################## {}'.format(fname))
    root = ET.parse(fname).getroot()
    for sent in list(root):

        print('\n')
        toks, labels = parse_scope(sent, [], [], {}, 0)
        # get all scopes
        scopes = []
        for l in labels:
            scopes.extend([elm for elm in l if elm.startswith('scope')])
        scopes = set(scopes)
        print(scopes)
        sent_data = []
        if include_all and len(scopes) == 0:
            print("################## This sentence has no negation. Adding to ds")
            scopes = [None]
        for scope in scopes:
            outtoks = []
            outlabels = []
            cue_labelseq = []
            for t, l in zip(toks, labels):
                l = set(l)
                is_cue = 0
                if scope in l:
                    lsurf = 'I'
                    if 'negexp-{}'.format(scope.split('-')[-1]) in l:
                        is_cue = 1
                        if setting == 'augment':
                            print('CUE\t{}'.format(lsurf))
                            outtoks.append('[CUE]')
                            outlabels.append(lsurf)
                            cue_labelseq.append(is_cue)
                else:
                    lsurf = 'O'
                print('{}\t{}'.format(t, lsurf))
                outtoks.append(t)
                outlabels.append(lsurf)
                cue_labelseq.append(is_cue)
            sent_data.append([outlabels, outtoks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data


def read_ddi(path, setting='augment', include_all=False):
    skipped = []
    data = []
    cue_data = []
    for fname in ['{}/{}'.format(path, elm) for elm in os.listdir(path) if
                  elm.endswith('_cleaned.xml') and not ' ' in elm]:
        try:
            data_ex, cue_data_ex = read_ddi_doc(fname, setting,include_all=include_all)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
        except ET.ParseError:
            skipped.append(fname)
    print('Could not parse the following files:')
    for skip in skipped:
        print(skip)
    return data, cue_data


def read_ddi_doc(fname, setting,include_all):

    def get_all_parents(node, c2p):
        # retrieve the node and all its parents
        tags = [node]
        while node in c2p:
            tags.append(c2p[node])
            node = c2p[node]
        return tags

    data = []
    cue_data = []
    #print('\n' + fname)
    try:
        # if True:
        root = ET.parse(fname).getroot()
        for sentence in root.iter('sentence'):
            negtagss = [elm for elm in sentence.iter('negationtags')]
            sent_tags = []
            constituents = []
            if not include_all and len(negtagss) == 0:
                continue
            else:

                for negtags in negtagss:
                    negtags.set('id', 'X')

                    children, p2c, c2p = dfs([], {}, {}, negtags)
                    siblings = dfs3(set(), p2c, c2p, negtags, {}, 0)
                    constituents = build_surface_ddi({}, p2c, c2p, negtags, siblings, 0, 0, 0)

                    # get sent_tags
                    sent_tags = []
                    for k, v in constituents[negtags]:
                        sent_tags.extend(
                            [elm.attrib['id'] for elm in get_all_parents(v, c2p) if elm.attrib['id'] != 'X'])
                    sent_tags = set(sent_tags)
            sent_data = []
            if include_all and len(sent_tags) == 0:
                #print("################## This sentence has no negation. Adding to ds")
                #print(sentence.attrib['text'])
                out_toks = []
                out_labels = []
                cue_labelseq = []
                for tok in sentence.attrib['text'].split():
                    out_toks.append(tok)
                    out_labels.append('O')
                    cue_labelseq.append(0)
                sent_data.append([out_labels, out_toks, cue_labelseq])
            if len(sent_tags) > 0:
                for sent_tag in sent_tags:
                    #print(sentence.attrib['text'])
                    #print('\n{}'.format(sent_tag))
                    out_toks = []
                    out_labels = []
                    cue_labelseq = []
                    for k, v in constituents[negtags]:
                        is_cue = 0
                        if k != None:
                            #print('constituent: "{}"'.format(k))
                            k_labels = set(
                                        ['{}_{}'.format(elm.attrib['id'], elm.tag) for elm in get_all_parents(v, c2p)])

                            # check if scope
                            if '{}_xcope'.format(sent_tag) in k_labels:
                                out_label = 'I'
                            else:
                                out_label = 'O'

                            # check if cue
                            if '{}_cue'.format(sent_tag) in k_labels:
                                is_cue = 1
                                if setting == 'augment':
                                    #print('CUE\t{}'.format(out_label))
                                    out_toks.append('[CUE]')
                                    out_labels.append(out_label)
                                    cue_labelseq.append(is_cue)

                            def split_keep_delim(text, delim):
                                splt =  [e + delim for i, e in enumerate(text.split(delim)) if e]
                                if not text.endswith(delim):
                                    splt[-1] = splt[-1].split(delim)[0]
                                return splt

                            for c, tok in enumerate(split_keep_delim(k, ' ')):
                                if not k.startswith(' ') and len(out_toks) > 0 and c == 0 and tok.strip() in set([elm for elm in ',.[]{}()?!']):
                                    out_toks[-1] += tok


                                else:
                                    out_toks.append(tok)
                                    out_labels.append(out_label)
                                    cue_labelseq.append(is_cue)
                                #print(out_toks)

                    assert len(out_labels) == len(out_toks)
                    sent_data.append([out_labels, out_toks, cue_labelseq])
            if len(sent_data) > 0:
                data.append(sent_data)
                cue_data.append(get_clue_annotated_data(sent_data))
        if setting == 'joint':
            data = join_cue_and_scope_labels(data)
    except ET.ParseError:
        raise ET.ParseError
    return data, cue_data


def read_ita(pname, setting='augment',include_all=False):
    data = []
    cue_data = []
    for f in os.listdir(pname):
        fname = os.path.join(pname, f)
        data_ex, cue_data_ex = read_ita_doc(fname, setting,include_all=include_all)
        data.extend(data_ex)
        cue_data.extend(cue_data_ex)
    return data, cue_data


def read_ita_doc(fname, setting, include_all):
    data = []
    cue_data = []
    root = ET.parse(fname).getroot()
    toks = {}
    tid2sid = {}
    sents = {}
    for tok in root.iter('token'):
        tid = tok.attrib['t_id']
        sid = tok.attrib['sentence']
        toks[tid] = tok.text
        tid2sid[tid] = sid
        sents.setdefault(sid, []).append(tid)

    tid2anno = {}

    for elm in root.iter('Markables'):
        mark = elm
    for clue in mark.iter('CUE-NEG'):
        for t in clue.iter('token_anchor'):
            tid = t.attrib['t_id']
            tid2anno.setdefault(tid, set()).add('{}_scope{}'.format('[CUE]', clue.attrib['scope']))
    for clue in mark.iter('SCOPE-NEG'):
        sids = set()
        scope_toks = []

        for t in clue.iter('token_anchor'):
            tid = t.attrib['t_id']
            tid2anno.setdefault(tid, set()).add('{}_{}'.format('SCOPE', clue.attrib['m_id']))
            sids.add(tid2sid[tid])
            scope_toks.append(toks[tid])

    for sid, sent in sents.items():
        # get sent labels
        sent_labels = []
        for tid in sent:
            if tid in tid2anno:
                sent_labels.extend([elm.split('_')[-1] for elm in tid2anno[tid] if elm.startswith('SCOPE')])
        sent_labels = set(sent_labels)
        sent_data = []
        if include_all and len(sent_labels) == 0:
            print("################## This sentence has no negation. Adding to ds")
            sent_labels = [None]
        for scope in sent_labels:
            out_labels = []
            out_toks = []
            all_labelss = []
            cue_labelseq = []
            for tid in sent:

                tok = toks[tid]
                out_label = 'O'
                is_cue = 0
                if tid in tid2anno:
                    labels = tid2anno[tid]
                    print(labels)
                    if 'SCOPE_{}'.format(scope) in labels:
                        out_label = 'I'
                    if 'CUE_scope{}'.format(scope) in labels:
                        is_cue = 1
                        if setting == 'augment':

                            out_toks.append('[CUE]')
                            out_labels.append(out_label)
                            all_labelss.append(labels)
                            cue_labelseq.append(is_cue)

                else:
                    labels = set()
                all_labelss.append(labels)
                out_toks.append(tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)
            sent_data.append([out_labels, out_toks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data

def read_nubes(pname, setting='augment',include_all=False):
    data = []
    cue_data = []
    for i in range(1, 10):
        ddir = os.path.join(pname, 'SAMPLE-00{}'.format(i))
        for f in os.listdir(ddir):
            if f.endswith('.ann'):
                fstem  = os.path.join(ddir, f.strip('.ann'))
                data_ex, cue_data_ex = read_nubes_doc(fstem+ '.txt', fstem + '.ann', setting=setting, include_all=include_all)
                data.extend(data_ex)
                cue_data.extend(cue_data_ex)
    return data, cue_data

def read_nubes_doc(fname_txt, fname_anno, setting, include_all):
    syn_markers = 0
    class Token(object):
        def __init__(self, start, end, surf, tid):
            self.c_start = start
            self.c_end = end
            self.surface = surf
            self.tid = tid

        def set_label(self, label):
            self.label = label

    data = []
    cue_data = []
    tid2tok = {}
    span2tok = {}

    anno_lines = read_file(fname_anno)

    def read_anno_file(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sentence_markers = []
        outlines = ''
        for line in lines:
            outlines += line.strip() + ' '
            sentence_markers.append(len(outlines) - 1)
        return outlines[:-1], sentence_markers

    s, sentence_markers = read_anno_file(fname_txt)

    labels = set()
    annos = {}
    for line in anno_lines:
        splt = line.split('\t')
        labels.add(splt[1].split()[0])
        annos.setdefault(splt[1].split()[0], []).append(line)

    for line in anno_lines:
        splt = line.split('\t')[1]

        if line.split('\t')[0].startswith('T'):
            gold_surf = line.split('\t')[-1].strip()

            start_orig = int(splt.split(' ')[1])
            end_orig = int(splt.split(' ')[2])
            start = start_orig
            end = end_orig
            surf = s[start:end]

            assert gold_surf == s[start:end]
            tid = line.split('\t')[0]
            label = splt.split()[0]
            tok = Token(start, end, surf, tid)

            tok.set_label(label)
            tid2tok[tid] = tok
            span2tok.setdefault(start, []).append(tok)
            for st in range(start, end):
                span2tok.setdefault(st, []).append(tok)

    span2r = {}
    for line in anno_lines:

        if line.split('\t')[0].startswith('R'):
            splt = line.split('\t')[1].split()
            rid = line.split('\t')[0]
            label = line.split('\t')[1].split()[0]
            for elm in splt:
                if ':' in elm:
                    span = (tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end)
                    span2r.setdefault(tid2tok[elm.split(':')[-1]].c_start, []).append((span, elm, rid, label))
                    for st in range(tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end):
                        span2r.setdefault(st, []).append(((st, tid2tok[elm.split(':')[-1]].c_end), elm, rid, label))

    i = 0
    surf2span = []
    chars = []
    while i < len(s) - 1:

        # add chars up to next whitespace
        c = s[i]
        start = i
        while i < len(s) - 1:
            if c == ' ':
                i += 1
                c = s[i]
                break
            chars.append(c)
            i += 1
            c = s[i]

        surf2span.append((''.join(chars), start))
        chars = []
    sents = []
    sent = []
    sentence_markers = set(sentence_markers)
    for surf, span in surf2span:
        if span + 1 in sentence_markers:
            sent.append((surf.strip('\n'), span))
            sents.append(sent)
            sent = []
        else:
            sent.append((surf, span))

    for sent in sents:

        tok2labels = []

        for surf, span in sent:

            if surf != '':
                labels = []
                if span in span2r:
                    labels.extend(['{}:{}:{}'.format(elm[3], elm[1], elm[2]) for elm in span2r[span]])

                if span in span2tok:
                    labels.extend(['{}:_{}'.format(elm.tid, elm.label) for elm in span2tok[span]])
                else:
                    labels.append('Unlabeled')
                tok2labels.append((surf, set(labels)))
        print(tok2labels)
        sent_labels = set()
        for _, labels in tok2labels:
            for label in labels:
                if label.split(':')[-1].startswith('R'):
                    sent_labels.add(label.split(':')[-1])
        sent_data = []
        if include_all and len(sent_labels) == 0:
            print("################## This sentence has no negation. Adding to ds")
            sent_labels = [None]
        for sent_label in sent_labels:
            print('\n')
            print(sent_label)
            out_toks = []
            out_labels = []
            cue_labelseq = []
            for tok, labels in tok2labels:
                is_cue = 0

                def tok2sent_labels(labels):
                    return set([label.split(':')[-1] for label in labels if label.split(':')[-1].startswith('R')])

                if sent_label in tok2sent_labels(labels):
                    out_label = 'I'
                    print('{}\t{}'.format(tok, labels))
                    #if len([elm for elm in labels if 'NegSynMarker' in elm or 'NegLexMarker' in elm or 'NegMorMarker' in elm]) > 0:
                    if len([elm for elm in labels if
                            'NegSynMarker' in elm]) > 0:
                        is_cue = 1
                        if setting == 'augment':
                            out_toks.append('[CUE]')
                            out_labels.append(out_label)
                            cue_labelseq.append(is_cue)
                else:
                    out_label = 'O'
                    print('{}\tUnlabeled'.format(tok))
                out_toks.append(tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)
            # check if there is at least one CUE in the data, otherwise don;t add. This is to filter out scopes of uncertainty cues
            if len(set(cue_labelseq)) > 1:
                sent_data.append([out_labels, out_toks, cue_labelseq])
            elif len(set(cue_labelseq)) == 1 and include_all:
                sent_data.append([out_labels, out_toks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data


def read_socc(pname, setting='augment', include_all=False):
    data = []
    cue_data = []
    for f in os.listdir(pname):
        dir_name = os.path.join(pname, f)
        for fname in [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.startswith('CURATION')]:
            data_ex, cue_data_ex = read_socc_doc(fname, setting, include_all=include_all)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
    return data, cue_data


def read_socc_doc(fname, setting, include_all):
    data = []
    cue_data = []
    lines = read_file(fname)
    current_anno_id = 0
    sents = {}
    for line in lines:
        if line != '' and not line.startswith('#'):
            splt = line.split('\t')
            if len(splt) < 4:
                break
            sid = splt[0].split('-')[0]
            tok = splt[2]
            annos = set(splt[3].split('|'))
            annos_with_id = set()
            for anno in annos:
                if '[' in anno:
                    aid = int(anno.split('[')[-1].strip(']'))
                    if aid > current_anno_id:
                        current_anno_id = aid

                    annos_with_id.add('{}_{}'.format(anno.split('[')[0], aid))
                else:
                    if anno == 'NEG':
                        current_anno_id += 1
                        annos_with_id.add('NEG_{}'.format(current_anno_id))

            sents.setdefault(sid, []).append((tok, annos_with_id))
    for sid, sent in sents.items():
        sent_cues = set()
        for tok, annos in sent:
            for anno in annos:
                if 'NEG' in anno:
                    sent_cues.add(anno)
        sent_data = []
        if include_all and len(sent_cues) == 0:
            print("################## This sentence has no negation. Adding to ds")
            sent_cues = ['-1']
        for sent_cue in sent_cues:
            out_toks = []
            out_labels = []
            cue_labelseq = []
            print('\n')
            print(fname)
            print(sent_cue)
            aid = int(sent_cue.split('_')[-1])
            for tok, annos in sent:
                is_cue = 0
                if 'SCOPE_{}'.format(aid + 1) in annos:
                    display_label = 'SCOPE_{}'.format(aid + 1)
                    out_label = 'I'
                else:
                    display_label = 'O'
                    out_label = 'O'
                if sent_cue in annos:
                    is_cue = 1
                    if setting == 'augment':
                        out_tok = '[CUE]'
                        out_toks.append(out_tok)
                        cue_labelseq.append(is_cue)
                        print(out_tok, display_label)
                out_tok = tok
                out_toks.append(out_tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)
                print(out_tok, display_label)
            sent_data.append([out_labels, out_toks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data


def read_cas(fname, setting='augment', include_all=False):
    sents = {}
    with open(fname) as f:

        for line in f:
            line = line.strip()
            if line != '':
                sid = line.split('\t')[0]
                sents.setdefault(sid, []).append(line.strip())
    data = []
    cue_data = []
    for sid, sent in sents.items():
        cols = {}
        for line in sent:
            splt = line.split('\t')
            if len(splt) >= 5:
                for cid, elm in enumerate(splt):
                    cols.setdefault(cid, []).append(elm)

        negations = []
        tid2cue = {}
        tid2scope = {}

        if len(cols) > 6:
            tid2cue = {i: cols[5][i] for i in range(len(sent)) if 'neg' in cols[5][i]}
            tid2scope = {i: cols[6][i] for i in range(len(sent)) if i < len(cols[6]) and 'scope' in cols[6][i]}
            negations = tid2cue.keys()
        sent_data = []
        if include_all and len(negations) == 0:
            print("################## This sentence has no negation. Adding to ds")
            negations = [None]

        if len(negations) > 0:
            toks = []
            labels = []
            cue_labelseq = []
            for tid, tok in enumerate(cols[2]):
                is_cue = 0
                if tid in tid2scope:
                    neg = 'I'
                else:
                    neg = 'O'
                if tid in tid2cue:
                    is_cue = 1
                    if setting == 'augment':
                        toks.append('[CUE]')
                        labels.append(neg)
                        cue_labelseq.append(is_cue)
                toks.append(tok)
                labels.append(neg)
                cue_labelseq.append(is_cue)
            sent_data.append([labels, toks, cue_labelseq])
            for t, l, s in zip(toks, labels, cue_labelseq):
                print('{}\t{}\t{}'.format(t, l, s))
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data



def read_dtneg(fname, setting='augment'):
    lines = read_file(fname)
    data = []
    cue_data = []
    answers = set()
    for line in lines:
        if line.startswith('ANNOTATEDANSWER'):
            if line not in answers:
                sent_data = []
                answers.add(line)
                text = line.strip('ANNOTATEDANSWER:\t')
                text = text.replace('>>', '>')
                text = text.replace('<<', '<')
                text = text.replace('>', ' >')
                text = text.replace('<', '< ')
                text = text.replace('[', '[ ')
                text = text.replace(']', ' ]')

                label = 'O'
                is_clue = False
                print('\n')
                out_labels = []
                out_toks = []
                cue_labelseq = []
                for tok in text.split():
                    if tok == '[':
                        label = 'I'
                    elif tok == ']':
                        label = 'O'
                    elif tok == '<':
                        is_clue = True
                    elif tok == '>':
                        is_clue = False
                    else:
                        if is_clue and setting =='augment':
                            out_toks.append('[CUE]')
                            out_labels.append(label)
                            cue_labelseq.append(is_clue)
                        out_toks.append(tok.strip('{}'))
                        out_labels.append(label)
                        cue_labelseq.append(is_clue)
                if len(out_labels) > 3:
                    for t, l in zip(out_toks, out_labels):
                        print(t,l)
                    cue_labelseq = [1 if elm == True else 0 for elm in cue_labelseq]
                    sent_data.append([out_labels, out_toks, cue_labelseq])
                if len(sent_data) > 0:
                    data.append(sent_data)
                    cue_data.append(get_clue_annotated_data(sent_data))
    if setting == 'joint':
        data = join_cue_and_scope_labels(data)
    return data, cue_data

def get_clues(data):
    clues = set()
    for labels, seq in data:
        cue = []
        for i, tok in enumerate(seq):
            if i == len(seq) -1:
                clues.add(' '.join(cue))
                break
            if tok == '[CUE]':
                cue.append(seq[i + 1])
                if i + 2 > len(seq)-1 or seq[i + 2] != '[CUE]':
                    clues.add(' '.join(cue))
                    cue = []
    return clues


'''
tid = line.split('\t')[0]
            label = splt.split()[0]
            tok = Token(start, end, surf, tid)

            tok.set_label(label)
            tid2tok[tid] = tok
            span2tok.setdefault(start, []).append(tok)
            for st in range(start, end):
                span2tok.setdefault(st, []).append(tok)

    span2r = {}
    for line in anno_lines:

        if line.split('\t')[0].startswith('R'):
            splt = line.split('\t')[1].split()
            rid = line.split('\t')[0]
            label = line.split('\t')[1].split()[0]
            for elm in splt:
                if ':' in elm:
                    span = (tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end)
                    span2r.setdefault(tid2tok[elm.split(':')[-1]].c_start, []).append((span, elm, rid, label))
                    for st in range(tid2tok[elm.split(':')[-1]].c_start, tid2tok[elm.split(':')[-1]].c_end):
                        span2r.setdefault(st, []).append(((st, tid2tok[elm.split(':')[-1]].c_end), elm, rid, label))

    i = 0

    surf2span = []
    chars = []
    while i < len(s) - 1:

        # add chars up to next whitespace
        c = s[i]
        start = i
        while i < len(s) - 1:
            if c == ' ':
                i += 1
                c = s[i]
                break
            chars.append(c)
            i += 1
            c = s[i]

        surf2span.append((''.join(chars), start))
        chars = []
    sents = []
    sent = []

    for surf, span in surf2span:

        if '\n' in surf:
            sent.append((surf.strip('\n'), span))
            sents.append(sent)
            sent = []
        else:
            sent.append((surf, span))

    for sent in sents:

        tok2labels = []

        for surf, span in sent:

            if surf != '':
                labels = []
                if span in span2r:
                    labels.extend(['{}:{}:{}'.format(elm[3], elm[1], elm[2]) for elm in span2r[span]])

                if span in span2tok:
                    labels.extend(['{}:_{}'.format(elm.tid, elm.label) for elm in span2tok[span]])
                else:
                    labels.append('Unlabeled')
                tok2labels.append((surf, set(labels)))
        sent_labels = set()
        for _, labels in tok2labels:
            for label in labels:
                if label.split(':')[-1].startswith('R'):
                    sent_labels.add(label.split(':')[-1])
        sent_data = []
        for sent_label in sent_labels:
            print('\n')
            print(sent_label)
            out_toks = []
            out_labels = []
            cue_labelseq =[]
            for tok, labels in tok2labels:
                is_cue = 0
                def tok2sent_labels(labels):
                    return set([label.split(':')[-1] for label in labels if label.split(':')[-1].startswith('R')])


                if sent_label in tok2sent_labels(labels):
                    out_label = 'I'
                    print('{}\t{}'.format(tok, labels))
                    if len([elm for elm in labels if 'NegMarker' in elm]) > 0:
                        is_cue = 1
                        if setting == 'augment':
                            out_toks.append('[CUE]')
                            out_labels.append(out_label)
                            cue_labelseq.append(is_cue)
                else:
                    out_label = 'O'
                    print('{}\tUnlabeled'.format(tok))
                out_toks.append(tok)
                out_labels.append(out_label)
                cue_labelseq.append(is_cue)

            sent_data.append([out_labels, out_toks, cue_labelseq])
        if len(sent_data) > 0:
            data.append(sent_data)
            cue_data.append(get_clue_annotated_data(sent_data))
    return data, cue_data

'''





def load_data_from_tsv(fname):
    data = []
    with open(fname) as f:
        for line in f.readlines():
            splt = line.split('\t')
            labels = ast.literal_eval(splt[1].strip())
            seq = ast.literal_eval(splt[2].strip())
            data.append([labels, seq])
    return data

def count_cues(cue_data):
    cues = []
    counter= 0
    for elm in cue_data:
        toks = elm[1]

        labels = elm[0]
        print(labels)
        cues.append(' '.join([toks[i].lower() for i, l in enumerate(labels) if l != '0']))
        if len(set(labels)) > 1:
            counter += 1
    from collections import Counter
    total = float(len(cues))
    c = Counter(cues)
    differents = len(c)
    print('{} different cues'.format(total))
    cum_sum = 0
    cum_sum_vals = []
    for key, val in c.most_common():
        print('{}\t|{} ( {} perc.)'.format(key, val, (val/total)*100))
        cum_sum += (val/total)*100
        cum_sum_vals.append(cum_sum)
    print(cum_sum_vals[:1000])
    print('Found {} of {} sentences containinge negation'.format(counter, len(cue_data)))

if __name__=="__main__":

    datasets = ['biofull', 'bioabstracts', 'bio',
                'sherlocken', 'sherlockzh',
                'iula', 'sfuen', 'sfues',
                 'ita', 'socc', 'dtneg', 'nubes',
                'biofullall', 'bioabstractsall', 'bioall',
                'sherlockenall', 'sherlockzhall',
                'iulaall',
                'sfuenall','sfuesall',
                'itaall',
                'nubesall',
                'dtnegall', 'soccall',
                'french', 'frenchall']

    datasets = ['sherlocken']
    """
    datasets = ['biofullall', 'bioabstractsall', 'bioall',
                'sherlockenall', 'sherlockzhall',
                'iulaall',
                'sfuenall','sfuesall',
                'itaall',
                'nubesall',
                'dtnegall', 'soccall',
                'french', 'frenchall']
    """
    # parse bioscope abstracts
    import configparser

    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)
    print(config)
    outpath = config.get('Files', 'preproc_data')
    make_directory(outpath)

    # load lexicon for silver clue detection

    # it doesn't matter which models we load here because we only do white space or rule-based tokenization anyway
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = Tokenizer(nlp.vocab)
    matcher = Matcher(nlp.vocab)

    lexicon_file = config.get('Files', 'triggers_en')
    #triggers = read_file(lexicon_file)
    #clue_detection.setup_matcher(triggers, matcher)

    def join_sentence_annos(data):
        """
        join scope annotations for one sentence for the setting with no conditioning on cues
        """
        joined_data = []
        for sent_data in data:

            joined_labels =  ['O']*len(sent_data[0][0])
            joined_cue_seq = ['0']*len(sent_data[0][2])
            seq = sent_data[0][1]

            for label, _, cue_seq in sent_data:
                for i, elm in enumerate(label):
                    if elm == 'C':
                        joined_labels[i] = 'C'
                    elif elm == 'I' and joined_labels[i] != 'C':
                        joined_labels[i] = 'I'
                for i, elm in enumerate(cue_seq):
                    if elm == '1':
                        joined_cue_seq[i] = '1'
            if len(sent_data) > 1:
                print('Joining {} to {}'.format([elm[0] for elm in sent_data], joined_labels))
            joined_data.append([[joined_labels, seq, joined_cue_seq]])

        return joined_data



    setting = 'augment'

    for ds in datasets:
        if ds == 'biofull':
            data, cue_data = read_bioscope(config.get('Files', ds), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'biofullall':
            data, cue_data = read_bioscope(config.get('Files', ds.split('all')[0]), setting=setting, include_all=True)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)

        elif ds == 'bioabstracts':
            data, cue_data = read_bioscope(config.get('Files', ds), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'bioabstractsall':
            data, cue_data = read_bioscope(config.get('Files', ds.split('all')[0]), setting=setting,  include_all=True)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'bio':
            data, cue_data = read_bioscope(config.get('Files', 'biofull'), setting=setting)
            data_extension, cue_data_extension = read_bioscope(config.get('Files', 'bioabstracts'), setting=setting)
            data.extend(data_extension)
            cue_data.extend(cue_data_extension)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'bioall':
            data, cue_data = read_bioscope(config.get('Files', 'biofull'), setting=setting, include_all=True)
            data_extension, cue_data_extension = read_bioscope(config.get('Files', 'bioabstracts'), setting=setting, include_all=True)
            data.extend(data_extension)
            cue_data.extend(cue_data_extension)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sherlocken':
            data, cue_data = read_sherlock(config.get('Files', ds), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sherlockenall':
            data, cue_data = read_sherlock(config.get('Files', ds.split('all')[0]), setting=setting, include_all=True)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sherlockzh':
            data, cue_data = read_sherlock(config.get('Files', ds), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sherlockzhall':
            data, cue_data = read_sherlock(config.get('Files', ds.split('all')[0]), setting=setting, include_all=True)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'iula':
            data, cue_data = read_IULA(config.get('Files', ds), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
            count_cues(cue_data)
        elif ds == 'iulaall':
            data, cue_data = read_IULA(config.get('Files', ds.split('all')[0]), setting=setting, include_all=True)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sfuen':
            data, cue_data = read_sfu_en(config.get('Files', ds), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sfuenall':
            data, cue_data = read_sfu_en(config.get('Files', ds.split('all')[0]), setting=setting, include_all=True)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sfues':
            data, cue_data = read_sfu_es(config.get('Files', ds), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'sfuesall':
            data, cue_data = read_sfu_es(config.get('Files', ds.split('all')[0]), setting=setting, include_all=True)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'ddi':
            data_train, cue_data_train = read_ddi(config.get('Files', 'dditrain'), setting=setting)
            data_test, cue_data_test = read_ddi(config.get('Files', 'dditest'), setting=setting)
            data = data_train + data_test
            cue_data = cue_data_train + cue_data_test
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'ddiall':
            data_train, cue_data_train = read_ddi(config.get('Files', 'dditrain'), setting=setting, include_all=True)
            data_test, cue_data_test = read_ddi(config.get('Files', 'dditest'), setting=setting,include_all=True)
            data = data_train + data_test
            cue_data = cue_data_train + cue_data_test
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'ita':
            data, cue_data = read_ita(config.get('Files', 'ita1'), setting=setting)
            data_ex, cue_data_ex = read_ita(config.get('Files', 'ita2'), setting=setting)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs =  write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'french':
            data, cue_data = read_cas(config.get('Files', 'cas'), setting=setting)
            data_ex, cue_data_ex = read_cas(config.get('Files', 'essai'), setting=setting)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
            count_cues(cue_data)
        elif ds == 'frenchall':
            data, cue_data = read_cas(config.get('Files', 'cas'), setting=setting, include_all=True)
            data_ex, cue_data_ex = read_cas(config.get('Files', 'essai'), setting=setting, include_all=True)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'itaall':
            data, cue_data = read_ita(config.get('Files', 'ita1'), setting=setting, include_all=True)
            data_ex, cue_data_ex = read_ita(config.get('Files', 'ita2'), setting=setting,include_all=True)
            data.extend(data_ex)
            cue_data.extend(cue_data_ex)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'socc':
            data, cue_data = read_socc(config.get('Files', 'socc'), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'soccall':
            data, cue_data = read_socc(config.get('Files', 'socc'), setting=setting, include_all=True)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'dtneg':
            data, cue_data = read_dtneg(config.get('Files', 'dtneg'), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'dtnegall':
            #dtneg only has sentences with negation
            data, cue_data = read_dtneg(config.get('Files', 'dtneg'), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
        elif ds == 'nubes':
            data, cue_data = read_nubes(config.get('Files', 'nubes'), setting=setting)
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
            count_cues(cue_data)
        elif ds == 'nubesall':
            data, cue_data = read_nubes(config.get('Files', 'nubes'), setting=setting, include_all=True)
            print(len(data))
            if setting == 'nocond' or setting == 'joint':
                data = join_sentence_annos(data)
            idxs = write_train_dev_test_data(os.path.join(outpath, ds), data, setting=setting)
            if setting == 'augment':
                write_train_dev_test_cue_data(os.path.join(outpath, ds), cue_data, idxs)
            count_cues(cue_data)


        elif ds == 'drugs':
            train_data = read_drugs(config.get('Files', 'drugstrain'), setting=setting)
            test_data = read_drugs(config.get('Files', 'drugstest'), setting=setting)
            idxs = write_train_dev_test_data_drugs(os.path.join(outpath, ds), train_data, test_data)
        elif ds == 'ddirelations':
            for split in ['train', 'dev', 'test']:
                data = read_ddi_relations(os.path.join(config.get('Files', 'ddi_relations_path'), '{}.tsv'.format(split)))
                write_split(os.path.join(outpath, ds) + '_{}.tsv'.format(split), data, json_format=True)
        elif ds == 'chemprot':
            for split in ['train', 'dev', 'test']:
                data = read_chemprot_relations(
                    os.path.join(config.get('Files', 'chemprot_path'), '{}.tsv'.format(split)))
                write_split(os.path.join(outpath, ds) + '_{}.tsv'.format(split), data, json_format=True)
        elif ds == 'gad':
            # gad is the preprocessed version provided by biobert
            # already has 10 train/test splits but no dev splits
            gad_path = config.get('Files', 'gad')
            for fold in [1,2,3,4,5,6,7,8,9,10]:
                train_data = read_gad(os.path.join(gad_path, str(fold), 'train.tsv'), split='train')
                test_data = read_gad(os.path.join(gad_path, str(fold), 'test.tsv'), split='test')
                write_data_gad_format(os.path.join(outpath, '{}{}'.format(ds, fold)), train_data,  test_data)
        elif ds == 'adr':
            # adr is the preprocessed version provided by biobert
            # already has 10 train/test splits but no dev splits, same format as gad
            adr_path = config.get('Files', 'adr')
            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                train_data = read_gad(os.path.join(adr_path, str(fold), 'train.tsv'), split='train')
                test_data = read_gad(os.path.join(adr_path, str(fold), 'test.tsv'), split='test')
                write_data_gad_format(os.path.join(outpath, '{}{}'.format(ds, fold)), train_data, test_data)
        elif ds == 'biorelex':
            # use official dev split for testing
            train_data = read_biorelex(config.get('Files', 'biorelex_train'))
            test_data = read_biorelex(config.get('Files', 'biorelex_dev'))
            write_data_gad_format(os.path.join(outpath, ds), train_data, test_data)
        elif ds == 'm2c2assert':
            data = read_m2c2assert('{}/beth'.format(config.get('Files', 'm2c2assert')))
            data.extend(read_m2c2assert('{}/partners'.format(config.get('Files', 'm2c2assert'))))
            idxs = write_train_dev_test_data_m2c2(os.path.join(outpath, ds), data)
        elif ds == 'cdr':
            train_data = read_cdr(config.get('Files', 'cdr_train'))
            #test_data = read_biorelex(config.get('Files', 'biorelex_dev'))
            #write_data_gad_format(os.path.join(outpath, ds), train_data, test_data)
        elif ds == 'ade':
            fnames = list(set([elm.split('.')[0] for elm in os.listdir(config.get('Files', 'ade_train'))]))
            fnames = sorted(fnames)
            for f in fnames:
                print(f)
                read_ade_doc(os.path.join(config.get('Files', 'ade_train'), '{}.txt'.format(f)),
                             os.path.join(config.get('Files', 'ade_train'), '{}.ann'.format(f)))
                break
        elif ds == 'uden':
            train_data = []
            dev_data = []
            test_data = []
            for d in ['udengum', 'udenlines', 'udenpartut']:
                train_data.extend(udep.read_udep(fname=config.get('Files', d).format('train'), ds=d))
                dev_data.extend(udep.read_udep(fname=config.get('Files', d).format('dev'), ds=d))
                test_data.extend(udep.read_udep(fname=config.get('Files', d).format('test'), ds=d))


            train_data_out = shuffle_and_prepare(train_data)
            dev_data_out = shuffle_and_prepare(dev_data)
            test_data_out = shuffle_and_prepare(test_data)

            write_split(os.path.join(outpath, ds) + '_train.tsv', train_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_dev.tsv', dev_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_test.tsv', test_data_out, json_format=False)
        elif ds == 'udes':
            train_data = []
            test_data = []
            for d in ['udesgsd', 'udesancora']:
                train_data.extend(udep.read_udep(fname=config.get('Files', d).format('train'), ds=d))
                test_data.extend(udep.read_udep(fname=config.get('Files', d).format('test'), ds=d))

            test_data_out = shuffle_and_prepare(test_data, shuffle=False)
            write_train_dev_udep(os.path.join(outpath, ds), train_data)
            write_split(os.path.join(outpath, ds) + '_test.tsv', test_data_out, json_format=False)
        elif ds == 'udzh':
            train_data = []
            dev_data = []
            test_data = []
            for d in ['udzhgsd', 'udzhgsdsimp']:
                train_data.extend(udep.read_udep(fname=config.get('Files', d).format('train'), ds=d))
                dev_data.extend(udep.read_udep(fname=config.get('Files', d).format('dev'), ds=d))
                test_data.extend(udep.read_udep(fname=config.get('Files', d).format('test'), ds=d))

            train_data_out = shuffle_and_prepare(train_data)
            dev_data_out = shuffle_and_prepare(dev_data)
            test_data_out = shuffle_and_prepare(test_data)

            write_split(os.path.join(outpath, ds) + '_train.tsv', train_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_dev.tsv', dev_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_test.tsv', test_data_out, json_format=False)
        elif ds == 'udit':
            train_data = []
            dev_data = []
            test_data = []
            for d in ['uditisdt', 'uditpartut', 'uditpostwita', 'udittwittiro', 'uditvit']:
                train_data.extend(udep.read_udep(fname=config.get('Files', d).format('train'), ds=d))
                dev_data.extend(udep.read_udep(fname=config.get('Files', d).format('dev'), ds=d))
                test_data.extend(udep.read_udep(fname=config.get('Files', d).format('test'), ds=d))

            train_data_out = shuffle_and_prepare(train_data)
            dev_data_out = shuffle_and_prepare(dev_data)
            test_data_out = shuffle_and_prepare(test_data)

            write_split(os.path.join(outpath, ds) + '_train.tsv', train_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_dev.tsv', dev_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_test.tsv', test_data_out, json_format=False)

        elif ds == 'udfr':
            train_data = []
            dev_data = []
            test_data = []
            for d in ['udfrgsd', 'udfrpartut', 'udfrsequoia']:
                train_data.extend(udep.read_udep(fname=config.get('Files', d).format('train'), ds=d))
                dev_data.extend(udep.read_udep(fname=config.get('Files', d).format('dev'), ds=d))
                test_data.extend(udep.read_udep(fname=config.get('Files', d).format('test'), ds=d))

            train_data_out = shuffle_and_prepare(train_data)
            dev_data_out = shuffle_and_prepare(dev_data)
            test_data_out = shuffle_and_prepare(test_data)

            write_split(os.path.join(outpath, ds) + '_train.tsv', train_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_dev.tsv', dev_data_out, json_format=False)
            write_split(os.path.join(outpath, ds) + '_test.tsv', test_data_out, json_format=False)
        if cue_data:
            print(count_cues(cue_data))

