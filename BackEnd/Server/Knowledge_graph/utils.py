import re
from nltk.corpus import stopwords

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

def split_into_chunks(text, tokenizer, max_length=512):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        temp_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        tokens = tokenizer.tokenize(temp_chunk)
        
        if len(tokens) <= max_length:
            current_chunk = temp_chunk
        else:
            if current_chunk == "":
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_sentence_chunk = ""
                for sent in sentences:
                    temp_sent_chunk = f"{current_sentence_chunk} {sent}".strip()
                    sent_tokens = tokenizer.tokenize(temp_sent_chunk)
                    
                    if len(sent_tokens) <= max_length:
                        current_sentence_chunk = temp_sent_chunk
                    else:
                        chunks.append(current_sentence_chunk)
                        current_sentence_chunk = sent
                
                if current_sentence_chunk:
                    chunks.append(current_sentence_chunk)
            else:
                chunks.append(current_chunk)
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks