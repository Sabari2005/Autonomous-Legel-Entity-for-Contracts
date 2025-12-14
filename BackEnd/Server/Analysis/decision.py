def final_contract_decision(legal_cases, web_reviews):
    """Analyze data and provide contract recommendation."""
    if "fraud" in legal_cases.lower() or "lawsuit" in legal_cases.lower():
        return "High Risk: Not Recommended for contract."
    elif "negative" in web_reviews.lower():
        return "Medium Risk: Proceed with caution."
    else:
        return "Low Risk: Suitable for contract."