from mydatasets.NegScopeDataset import NegScopeDataset, read_examples
from mydatasets.RelClassificationDataset import RelClassificationDataset, read_relation_classification_examples

def get_data(task, split, config, tokenizer):
    if task.dataset_type == 'relclassification':
        data = RelClassificationDataset(read_relation_classification_examples(
            os.path.join(config.get('Files', 'preproc_data'), '{}_{}.jsonl'.format(task.dataset, split))),
            tokenizer=tokenizer, max_seq_len=task.max_seq_len)
    elif task.dataset_type == 'negscope':
        data = NegScopeDataset(
            read_examples(
                os.path.join(config.get('Files', 'preproc_data'), '{}_{}.jsonl'.format(task.dataset, split))),
            tokenizer=tokenizer, max_seq_len=task.max_seq_len, split_seqs=task.split_seqs)
    return data
