import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Final_Full_Dataset.csv")
df = df.dropna(subset=["Risk Level"])

label_encoder_level = LabelEncoder()
label_encoder_category = LabelEncoder()
df['Risk Level'] = label_encoder_level.fit_transform(df['Risk Level'])
df['Risk Category'] = label_encoder_category.fit_transform(df['Risk Category'])

risk_level_mapping = {i: label for i, label in enumerate(label_encoder_level.classes_)}
risk_category_mapping = {i: label for i, label in enumerate(label_encoder_category.classes_)}

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BertForMultiTaskClassification(BertForSequenceClassification):
    def __init__(self, config, num_labels_level, num_labels_category):
        super().__init__(config)
        self.num_labels_level = num_labels_level
        self.num_labels_category = num_labels_category
        self.classifier_level = torch.nn.Linear(config.hidden_size, num_labels_level)
        self.classifier_category = torch.nn.Linear(config.hidden_size, num_labels_category)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits_level = self.classifier_level(pooled_output)
        logits_category = self.classifier_category(pooled_output)
        return {"logits_level": logits_level, "logits_category": logits_category}

model = BertForMultiTaskClassification.from_pretrained(
    "bert-large-uncased",
    num_labels_level=len(label_encoder_level.classes_),
    num_labels_category=len(label_encoder_category.classes_)
)
model.load_state_dict(torch.load('Risk_analysis_BERT.pth', map_location=device))
model = model.to(device)
model.eval()

def predict_risk(text_chunk):
    cleaned_text = clean_text(text_chunk)
    encoded_dict = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
  
    predicted_level = np.argmax(outputs["logits_level"].cpu().numpy(), axis=1).flatten()[0]
    predicted_category = np.argmax(outputs["logits_category"].cpu().numpy(), axis=1).flatten()[0]

    return (
        risk_level_mapping.get(predicted_level, "Unknown"),
        risk_category_mapping.get(predicted_category, "Unknown")
    )