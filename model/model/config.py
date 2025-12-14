import wandb


config = {
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "max_length": 512,
    "model_name": "bert-large-uncased"
}


wandb.init(project="Risk-Analysis-Bert", config=config)


DATA_PATH = "/teamspace/studios/this_studio/Final_Full_Dataset.csv"
MODEL_SAVE_PATH = "/teamspace/studios/this_studio/Risk_analysis_bert.pth"