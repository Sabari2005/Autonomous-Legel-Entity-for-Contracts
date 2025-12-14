import requests
from config import OPENCORPORATES_API_TOKEN, SERP_API_KEY
from utils import summarize_text

def fetch_company_details(company_name):
    try:
        base_url = "https://api.opencorporates.com/v0.4/companies/search"
        params = {"q": company_name, "api_token": OPENCORPORATES_API_TOKEN}
        response = requests.get(base_url, params=params)
        data = response.json()

        if data["results"]["companies"]:
            company = data["results"]["companies"][0]["company"]
            print("\nCompany Details:")
            print(f"    Name: {company.get('name', 'N/A')}")
            print(f"    Incorporation Date: {company.get('incorporation_date', 'N/A')}")
            print(f"    Address: {company.get('registered_address', 'No address available')}")
            # Add remaining print statements from original code
        else:
            print("\nNo company found with the given name.")
    except Exception as e:
        print(f"\nError fetching company details: {e}")

def fetch_web_reviews(company_name):
    query = f"{company_name} company reviews site:trustpilot.com OR site:glassdoor.com"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        reviews = [item.get("snippet", "No description available") for item in data.get("organic_results", [])]
        return summarize_text(" ".join(reviews)) if reviews else "No reviews found."
    except Exception as e:
        return f"Error fetching web reviews: {e}"

def fetch_legal_cases(company_name):
    query = f"{company_name} lawsuits OR legal disputes site:justice.gov OR site:sec.gov"
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        cases = [item.get("snippet", "No description available") for item in data.get("organic_results", [])]
        return summarize_text(" ".join(cases)) if cases else "No legal cases found."
    except Exception as e:
        return f"Error fetching legal cases: {e}"