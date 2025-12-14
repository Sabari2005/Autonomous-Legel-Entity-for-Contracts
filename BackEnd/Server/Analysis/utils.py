from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text, max_length=150):
    if not text or len(text.split()) < 20:
        return text
    try:
        summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing text: {e}"