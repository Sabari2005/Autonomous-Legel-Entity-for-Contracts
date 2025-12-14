from company_info import fetch_company_details, fetch_web_reviews, fetch_legal_cases
from stock_analysis import get_stock_symbol, fetch_stock_data, visualize_stock_trends
from decision import final_contract_decision

if __name__ == "__main__":
    company_name = input("Enter the company name: ")
    fetch_company_details(company_name)
    stock_symbol = get_stock_symbol(company_name)
    if stock_symbol:
        print(f"\nStock Symbol: {stock_symbol}")
        stock_data = fetch_stock_data(stock_symbol)
        if stock_data is not None:
            visualize_stock_trends(stock_data, stock_symbol)
        else:
            print("\nNo stock data available.")
    else:
        print("\nNo stock symbol found.")
    print("\nFetching Web Reviews...")
    web_reviews = fetch_web_reviews(company_name)
    print(f"Web Reviews Summary:\n{web_reviews}")
    print("\nFetching Legal Cases...")
    legal_cases = fetch_legal_cases(company_name)
    print(f"Legal Cases Summary:\n{legal_cases}")
    print("\nContract Decision:")
    decision = final_contract_decision(legal_cases, web_reviews)
    print(decision)