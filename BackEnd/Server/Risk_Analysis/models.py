import pandas as pd
import torch
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from BackEnd.Server.Risk_Analysis.config import DATASET_PATH, MODEL_PATH

nltk.download('stopwords')
from nltk.corpus import stopwords


df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["Risk Level"])

label_encoder_level = LabelEncoder()
label_encoder_category = LabelEncoder()

df['Risk Level'] = label_encoder_level.fit_transform(df['Risk Level'])
df['Risk Category'] = label_encoder_category.fit_transform(df['Risk Category'])

risk_level_mapping = {i: label for i, label in enumerate(label_encoder_level.classes_)}
risk_category_mapping = {i: label for i, label in enumerate(label_encoder_category.classes_)}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(r"http\S+", "", text)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')
    sw = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True, model_max_length=512)

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForMultiTaskClassification.from_pretrained(
    "bert-large-uncased",
    num_labels_level=len(label_encoder_level.classes_),
    num_labels_category=len(label_encoder_category.classes_)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def predict_risk(text):
    cleaned_text = clean_text(text)
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