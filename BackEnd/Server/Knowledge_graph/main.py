from models import predict_risk, tokenizer
from utils import split_into_chunks
from similarity import find_similar_clauses
from visualization import create_combined_graph

def process_text(input_text):
    chunks = split_into_chunks(input_text, tokenizer)
    chunks_data = []
    
    for i, chunk in enumerate(chunks):
        risk_level, risk_category = predict_risk(chunk)
        similar_clauses, _ = find_similar_clauses(chunk, risk_level, risk_category)
        chunks_data.append((chunk, risk_level, risk_category, similar_clauses))
    
    return chunks_data

if __name__ == "__main__":
    input_text = input("Enter contract clause text (can be long): ").strip()

    if len(input_text) < 50:
        print("Text seems too short. Please provide a more substantial clause.")
    else:
        chunks_data = process_text(input_text)
        create_combined_graph(chunks_data)