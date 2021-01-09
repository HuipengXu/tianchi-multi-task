import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import BertPreTrainedModel, BertModel, BertConfig


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor):
        q = self.fc(hidden_state).squeeze(dim=-1)
        q = q.masked_fill(mask, -np.inf)
        w = F.softmax(q, dim=-1).unsqueeze(dim=1)
        h = w @ hidden_state
        return h.squeeze(dim=1)


class AttentionClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super(AttentionClassifier, self).__init__()
        self.attn = Attention(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        h = self.attn(hidden_states, mask)
        out = self.fc(h)
        return out


class MultiDropout(nn.Module):

    def __init__(self, hidden_size: int, num_classes: int):
        super(MultiDropout, self).__init__()
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, hidden_states: torch.Tensor):
        max_pool, _ = hidden_states.max(dim=1)
        avg_pool = hidden_states.mean(dim=1)
        pool = torch.cat([max_pool, avg_pool], dim=-1)
        logits = []
        for dropout in self.dropout:
            out = dropout(pool)
            out = self.fc(out)
            logits.append(out)
        logits = torch.stack(logits, dim=2).mean(dim=2)
        return logits


class BertMultiTaskModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig, task_num_classes: dict, model_path: str):
        super(BertMultiTaskModel, self).__init__(config)

        self.bert = BertModel.from_pretrained(model_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.task_classifiers = nn.ModuleDict({task_id: AttentionClassifier(config.hidden_size, num_classes)
                                               for task_id, num_classes in task_num_classes.items()})
        self.task_num_classes = task_num_classes

    def forward(self,
                task_id: str,
                input_ids: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        mask = input_ids == 0
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = self.dropout(outputs[0])

        logits = self.task_classifiers[task_id](hidden_states, mask)

        outputs = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.task_num_classes[task_id]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
