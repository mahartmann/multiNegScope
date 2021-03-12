from typing import Union, List

class SeqLabelingInputExample:
    """
    Structure for one sequence labeling input example with list of tokens, list of labels and a unique id
    """
    def __init__(self, guid: str, text: List[str], label: List[str]):
        """
        Creates one InputExample with the given texts, guid and label

        :param guid
            id for the example
        :param texts
            the text for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.seq = text
        self.label = label


class SeqClassificationInputExample:
    """
    Structure for one sequence classification input example with list of tokens, label and a unique id
    """
    def __init__(self, guid: str, text: List[str], label: str):
        """
        Creates one InputExample with the given texts, guid and label

        :param guid
            id for the example
        :param texts
            the text for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.seq = text
        self.label = label


class SeqPairClassificationInputExample:
    """
    Structure for one sequence classification input example with list of tokens, label and a unique id
    """
    def __init__(self, guid: str, seq1: List[str], seq2: List[str], label: str):
        """
        Creates one InputExample with the given texts, guid and label

        """
        self.guid = guid
        self.seq1 = seq1
        self.seq2 = seq2
        self.label = label