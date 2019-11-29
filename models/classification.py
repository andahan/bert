#!/usr/bin/env python
# coding:utf8

# Customized models for sequence classification task using BERT pretrained model.


from huggingface.modeling import BertModel
from models.base import ClassificationBase
from models.base import PretrainedBertModel
from models.base import MultiLabelClassificationBase
from utils import load_label_weight
import torch

class BertForSequenceClassification(PretrainedBertModel):
    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = ClassificationBase(config.hidden_dropout_prob,
                                             config.hidden_size,
                                             num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(pooled_output, labels)


class BertForMultiLabelClassification(PretrainedBertModel):
    def __init__(self, config, num_labels=2, label_weight=None):
        super(BertForMultiLabelClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.label_weight = label_weight
        self.classifier = MultiLabelClassificationBase(config.hidden_dropout_prob,
                                             config.hidden_size,
                                             num_labels,
                                             self.label_weight)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(pooled_output, labels)

class BertDForMultiLabelClassification(PretrainedBertModel):
    def __init__(self, config, num_labels=2, label_weight=None):
        super(BertDForMultiLabelClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.label_weight = label_weight
        self.classifier = MultiLabelClassificationBase(config.hidden_dropout_prob,
                                             config.hidden_size * 2,
                                             num_labels,
                                             self.label_weight)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]

        mean_pooled_output = sequence_output.mean(dim=1)
        max_pooled_output = sequence_output.max(dim=1)[0]
        pooled_output = torch.cat([mean_pooled_output, max_pooled_output], 1)

        #self.activation(pooled_output)

        return self.classifier(pooled_output, labels)
