#!/usr/bin/env python
# coding:utf8

import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface.modeling import BertConfig
from huggingface.modeling import BERTLayerNorm

class PretrainedBertModel(nn.Module):
    def __init__(self, config, *unused_arg, **unused_kwargs):
        super(PretrainedBertModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, BERTLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def load_pretrained_bert_model(cls, bert_config_path, pretrained_model_path,
                                   *inputs, **kwargs):
        config = BertConfig.from_json_file(bert_config_path)
        model = cls(config, *inputs, **kwargs)
        pretrained_model_weights = torch.load(pretrained_model_path,
                                              map_location='cpu')
        model.bert.load_state_dict(pretrained_model_weights)
        return model


class ClassificationBase(nn.Module):
    def __init__(self, drop_prob, hidden_size, num_labels):
        super(ClassificationBase, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output, labels=None):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits


class MultiLabelClassificationBase(nn.Module):
    def __init__(self, drop_prob, hidden_size, num_labels, label_weight=None):
        super(MultiLabelClassificationBase, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.label_weight = label_weight

    def forward(self, pooled_output, labels=None):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=None)
            loss = loss_fct(logits, labels)
            pos_loss = torch.masked_select(loss, labels.ge(1))
            neg_loss = torch.masked_select(loss, labels.le(0))
            return loss.mean(dim=-1), pos_loss.mean(dim=-1), neg_loss.mean(dim=-1), logits 
        else:
            return torch.sigmoid(logits) 
            
            

class MatchingBase(ClassificationBase):
    def _pointwise_forward(self, pooled_output, labels=None):
        return super(MatchingBase, self).forward(pooled_output, labels)

    def _pairwise_forward(self, first_pooled_output, second_pooled_output,
                          labels, margin):
        first_logits = self._pointwise_forward(first_pooled_output, labels=None)
        second_logits = self._pointwise_forward(second_pooled_output, labels=None)
        first_score = torch.sigmoid(first_logits[:, 1])
        second_score = torch.sigmoid(second_logits[:, 1])
        if labels is not None:
            loss_fct = nn.MarginRankingLoss(margin)
            labels = labels * 2 - 1.0
            loss = loss_fct(first_score, second_score, labels.float())
            return loss, first_score, second_score
        return first_score, second_score

    def forward(self, first_pooled_output, labels=None, second_pooled_output=None,
                margin=None):
        if second_pooled_output is not None:
            return self._pairwise_forward(first_pooled_output, second_pooled_output,
                                          labels, margin)
        return self._pointwise_forward(first_pooled_output, labels)


class SequenceLabelingBase(nn.Module):
    def __init__(self, drop_prob, hidden_size, num_labels):
        super(SequenceLabelingBase, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, last_encoder_layer, attention_mask=None, labels=None):
        sequence_outputs = self.dropout(last_encoder_layer[:, 1:])
        logits = self.classifier(sequence_outputs)

        if labels is not None:
            label_mask = attention_mask[:, 1:].float()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            seq_losses = loss_fct(logits.transpose(1, 2), labels)
            loss = ((seq_losses * label_mask).sum()) / label_mask.sum()
            return loss, logits
        return logits


class SpanRankingBase(nn.Module):
    def __init__(self, drop_prob, hidden_size, num_labels=2):
        super(SpanRankingBase, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True,
                          bidirectional=True)
        self.projector = nn.Linear(hidden_size * 8, num_labels)

    def _get_span_logits(self, fw_rep, bw_rep, span_idx):
        fw_start_rep = fw_rep[torch.arange(fw_rep.size(0)), span_idx[:, 0]]
        fw_end_rep = fw_rep[torch.arange(fw_rep.size(0)), span_idx[:, 1]]
        bw_start_rep = bw_rep[torch.arange(bw_rep.size(0)), span_idx[:, 0]]
        bw_end_rep = bw_rep[torch.arange(bw_rep.size(0)), span_idx[:, 1]]
        fw_span_rep = torch.cat([fw_start_rep, fw_end_rep, fw_start_rep + \
            fw_end_rep, fw_start_rep - fw_end_rep], dim=-1)
        bw_span_rep = torch.cat([bw_start_rep, bw_end_rep, bw_start_rep + \
            bw_end_rep, bw_start_rep - bw_end_rep], dim=-1)
        span_rep = torch.cat([fw_span_rep, bw_span_rep], dim=-1)
        span_logits = self.projector(self.dropout(span_rep))
        return span_logits

    def _pointwise_forward(self, fw_rep, bw_rep, span_idx, labels=None):
        span_logits = self._get_span_logits(fw_rep, bw_rep, span_idx)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(span_logits, labels)
            return loss, span_logits
        return span_logits

    def _pairwise_forward(self, fw_rep, bw_rep, first_span_idx,
                          second_span_idx, labels=None, margin=None):
        first_span_logits = self._get_span_logits(fw_rep, bw_rep, first_span_idx)
        second_span_logits = self._get_span_logits(fw_rep, bw_rep, second_span_idx)
        first_score = torch.sigmoid(first_span_logits[:, 1])
        second_score = torch.sigmoid(second_span_logits[:, 1])
        if labels is not None:
            loss_fct = nn.MarginRankingLoss(margin)
            labels = labels * 2 - 1.0
            loss = loss_fct(first_score, second_score, labels.float())
            return loss, first_score, second_score
        return first_score, second_score

    def forward(self, last_encoder_layer, span_idx, labels=None,
                second_span_idx=None, margin=None):
        rnn_output, _ = self.rnn(last_encoder_layer, None)
        fw_rep, bw_rep = torch.split(rnn_output, rnn_output.size(2) // 2, dim=2)
        if second_span_idx is not None:
            return self._pairwise_forward(fw_rep, bw_rep, span_idx,
                                          second_span_idx, labels, margin)
        return self._pointwise_forward(fw_rep, bw_rep, span_idx, labels)
