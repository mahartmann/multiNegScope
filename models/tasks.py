import yaml

class Task():
    def __init__(self, type, dataset,  dropout_prob, splits, max_seq_len, split_seqs=False):
        self.dataset=dataset
        self.splits = splits
        self.task_type = type
        self.hidden_dropout_prob = dropout_prob
        self.label_map = None
        self.reverse_label_map = None
        self.max_seq_len = max_seq_len
        self.split_seqs = split_seqs
        self.task_id = None

    def set_task_id(self, tid):
        self.task_id = tid

    def set_label_map(self, label_map):
        self.label_map = label_map
        self.reverse_label_map = {val: key for key, val in label_map.items()}
        self.num_labels = len(label_map)

def load_task(fname):
    task_def = yaml.safe_load(open(fname))
    task = Task(type=task_def['task_type'], dropout_prob=task_def['dropout_prob'], splits=task_def['splits'], dataset=task_def['dataset'],
         split_seqs=task_def['split_seqs'], max_seq_len=task_def['max_seq_len'])
    return task
