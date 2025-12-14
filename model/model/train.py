import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler, random_split
import numpy as np
from config import config, MODEL_SAVE_PATH
from data_preprocessing import load_and_preprocess_data
from model import BertForMultiTaskClassification
import wandb

def prepare_dataloaders(df, label_encoder_level, label_encoder_category):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True, model_max_length=512)

    input_ids = []
    attention_masks = []
    labels_level = df['Risk Level'].values
    labels_category = df['Risk Category'].values

    for text in df['clause Text']:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,  
            padding='max_length', 
            truncation=True,  
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_level = torch.tensor(labels_level)
    labels_category = torch.tensor(labels_category)

    dataset = TensorDataset(input_ids, attention_masks, labels_level, labels_category)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config['batch_size'])
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=config['batch_size'])
    
    return train_dataloader, validation_dataloader

def train_model():
    df, label_encoder_level, label_encoder_category = load_and_preprocess_data()
    train_dataloader, validation_dataloader = prepare_dataloaders(df, label_encoder_level, label_encoder_category)
    
    model = BertForMultiTaskClassification.from_pretrained(
        "bert-large-uncased",
        num_labels_level=len(label_encoder_level.classes_),
        num_labels_category=len(label_encoder_category.classes_)
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], eps=1e-8)
    epochs = config['epochs']
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels_level = batch[2].to(device)
            b_labels_category = batch[3].to(device)

            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, 
                          labels_level=b_labels_level, labels_category=b_labels_category)
            loss = outputs["loss"]
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully.")

    artifact = wandb.Artifact("Risk-Analysis-Bert", type="model")
    artifact.add_file(MODEL_SAVE_PATH)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    train_model()