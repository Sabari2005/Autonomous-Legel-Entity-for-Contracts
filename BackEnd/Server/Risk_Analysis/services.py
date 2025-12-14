from groq import Groq
import shap
import torch
from transformers import BertTokenizer
from BackEnd.Server.Risk_Analysis.config import GROQ_API_KEY
from BackEnd.Server.Risk_Analysis.models import predict_risk, risk_level_mapping, risk_category_mapping, label_encoder_level, device, tokenizer

client = Groq(api_key=GROQ_API_KEY)

def explain_chunk(chunk, risk_level, risk_category):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                Analyze this contract chunk classified as {risk_level} risk ({risk_category}):
                {chunk}
                Provide HTML-formatted explanation without markdown.
                """
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip()

def modifier_model(text, risk_level, risk_category):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                Modify this {risk_level} risk text ({risk_category}):
                {text}
                Provide modified text only.
                """
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip()

def format_legal_text(modified_text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""
                Format this text with HTML:
                {modified_text}
                """
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip()

def process_chunk(chunk):
    try:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        risk_level, risk_category = predict_risk(chunk)
        
        explainer = shap.Explainer(predict, tokenizer, output_names=label_encoder_level.classes_.tolist())
        shap_values = explainer([chunk])
        shap_plot = shap.plots.text(shap_values, display=False)
        
        explanation = explain_chunk(chunk, risk_level, risk_category)
        if risk_level in ["High", "Medium", "Low"]:
            modified_text = modifier_model(chunk, risk_level, risk_category)
            formatted_text = format_legal_text(modified_text)
        else:
            modified_text = chunk
            formatted_text = format_legal_text(chunk)
        
        return {
            "original": chunk,
            "risk_level": risk_level,
            "risk_category": risk_category,
            "shap_plot": shap_plot,
            "explanation": explanation,
            "formatted": formatted_text
        }
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return {
            "original": chunk,
            "risk_level": "Error",
            "formatted": format_legal_text(chunk)
        }

def save_results_as_html(results, filename="report.html"):
    pass