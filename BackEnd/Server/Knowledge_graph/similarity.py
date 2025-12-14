from sentence_transformers import SentenceTransformer, util
from database import fetch_all_clauses

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar_clauses(clause_text, risk_level, risk_category, top_k=5):
    input_embedding = similarity_model.encode(clause_text, convert_to_tensor=True)
    clauses = fetch_all_clauses()

    risk_level = risk_level.lower().strip()
    risk_category = risk_category.lower().strip()

    filtered_clauses = [
        c for c in clauses
        if c["risk_level"].lower().strip() == risk_level and
           c["risk_category"].lower().strip() == risk_category
    ]

    if not filtered_clauses:
        return [], clause_text

    clause_texts = [clause['text'] for clause in filtered_clauses]
    clause_embeddings = similarity_model.encode(clause_texts, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(input_embedding, clause_embeddings)[0]
    top_results = scores.topk(k=min(top_k, len(filtered_clauses)))

    matched = []
    for score, idx in zip(top_results.values, top_results.indices):
        clause = filtered_clauses[idx]
        clause["similarity"] = float(score)
        matched.append(clause)

    return matched, clause_text