from torch import nn
from torch.nn import CrossEntropyLoss


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class OutputLayerSeqLabeling(nn.Module):
    def __init__(self, task, hidden_size, padding_idx):
        super(OutputLayerSeqLabeling, self).__init__()
        self.task_type = task.task_type
        self.dropout = nn.Dropout(task.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, task.num_labels)
        self.criterion = CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, last_hidden_states, labels=None):
        # get a classifcation for each token in the sequence
        last_hidden_states = self.dropout(last_hidden_states)
        logits = self.classifier(last_hidden_states)

        # logits: bs x seq_len x num_labels -> (bs x seq_len) x num_labels
        flattened_logits = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
        if labels:
            # labels: bs x seq_len -> (bs x seq_len)
            flattened_labels = labels.view(-1)

            loss = self.criterion(flattened_logits, flattened_labels)
            return loss, logits
        else:
            return None, logits


class OutputLayerSeqClassification(nn.Module):
    def __init__(self, task, hidden_size, padding_idx):
        super(OutputLayerSeqClassification, self).__init__()
        self.task_type = task.task_type
        self.pooler = BertPooler(hidden_size=hidden_size)

        self.dropout = nn.Dropout(task.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, task.num_labels)
        self.criterion = CrossEntropyLoss(ignore_index=padding_idx)


    def forward(self, input, labels=None):
        pooled_input = self.pooler(input)
        pooled_input = self.dropout(pooled_input)
        logits = self.classifier(pooled_input)
        if labels:
            loss = self.criterion(logits, labels)
            return loss, logits
        else:
            return None, logits