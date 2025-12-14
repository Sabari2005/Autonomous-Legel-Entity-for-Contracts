import torch
from transformers import BertForSequenceClassification

class BertForMultiTaskClassification(BertForSequenceClassification):
    def __init__(self, config, num_labels_level, num_labels_category):
        super().__init__(config)
        self.num_labels_level = num_labels_level
        self.num_labels_category = num_labels_category
        self.classifier_level = torch.nn.Linear(config.hidden_size, num_labels_level)
        self.classifier_category = torch.nn.Linear(config.hidden_size, num_labels_category)

    def forward(self, input_ids, attention_mask=None, labels_level=None, labels_category=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        logits_level = self.classifier_level(pooled_output)
        logits_category = self.classifier_category(pooled_output)

        loss = None
        if labels_level is not None and labels_category is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss_level = loss_fct(logits_level.view(-1, self.num_labels_level), labels_level.view(-1))
            loss_category = loss_fct(logits_category.view(-1, self.num_labels_category), labels_category.view(-1))
            loss = loss_level + loss_category

        return {
            "logits_level": logits_level,
            "logits_category": logits_category,
            "loss": loss
        }