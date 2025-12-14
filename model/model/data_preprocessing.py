import pandas as pd
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from config import DATA_PATH

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(r"http\S+", "", text)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')
    sw = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = " ".join(text)
    return text

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df = df.dropna(subset=["Risk Level"])
    
    label_encoder_level = LabelEncoder()
    label_encoder_category = LabelEncoder()

    df['Risk Level'] = label_encoder_level.fit_transform(df['Risk Level'])
    df['Risk Category'] = label_encoder_category.fit_transform(df['Risk Category'])
    
    df['clause Text'] = df['clause Text'].apply(lambda x: clean_text(x))
    
    return df, label_encoder_level, label_encoder_category