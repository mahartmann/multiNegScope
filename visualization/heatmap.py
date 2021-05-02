'''
This code is based on code from a repository of Leila Arras
'''

import matplotlib.pyplot as plt

def getRGB(c_tuple):
    return "#%02x%02x%02x" % (int(c_tuple[0]), int(c_tuple[1]), int(c_tuple[2]))


def span_word(word, color):
    return "<span style=\"background-color:" + getRGB(color) + "\">" + word + "</span>"


def html_heatmap(words, labels):
    cue_color = (255, 153, 153)
    span_color = (153, 204, 255)
    neutral_color = (255, 255, 255)
    output_text = ""

    for idx, w in enumerate(words):
        if labels[idx] == 'I':
            color = span_color
        elif labels[idx] == 'CUE':
            color = cue_color
        else:
            color = neutral_color

        output_text = output_text + span_word(w, color) + " "

    return output_text + "\n"