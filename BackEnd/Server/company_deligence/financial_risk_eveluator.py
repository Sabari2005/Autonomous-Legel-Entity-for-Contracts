import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from langchain_ollama import OllamaLLM
import json
from typing import Dict, List
import textwrap
import os
from datetime import datetime


class FinancialRiskEvaluator:
    def __init__(self, finance_api_key: str, news_api_key: str):
        self.finance_api = finance_api_key
        self.news_api = news_api_key
        self.reports_folder = Path("Stock_analyses_output")
        self.reports_folder.mkdir(exist_ok=True)
        self.assessment_parameters = {
            'liquidity_ratio': {'caution': 1.5, 'critical': 1.0},
            'leverage_ratio': {'caution': 1.0, 'critical': 2.0},
            'profitability': {'caution': 0.05, 'critical': 0.0},
            'financial_health_score': {'caution': 3, 'critical': 1}
        }

        self.fraud_detection_weights = {
            'receivables_index': 0.5,
            'margin_index': 1.0,
            'asset_quality': 0.5,
            'growth_index': 1.0,
            'depreciation_rate': 0.5,
            'expense_index': 1.0,
            'leverage_change': 1.0,
            'accruals_ratio': 0.5
        }

    def retrieve_financial_statistics(self, ticker: str) -> dict:
        endpoint = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.finance_api}"
        try:
            api_response = requests.get(endpoint, timeout=10)
            statistics = api_response.json()
            return {
                'valuation_ratio': float(statistics.get('PERatio', 0)),
                'net_margin': float(statistics.get('ProfitMargin', 0)),
                'liquidity_ratio': float(statistics.get('CurrentRatio', 0)),
                'leverage_ratio': float(statistics.get('DebtToEquity', 0)),
                'asset_returns': float(statistics.get('ReturnOnAssetsTTM', 0)),
                'equity_returns': float(statistics.get('ReturnOnEquityTTM', 0)),
                'company_valuation': float(statistics.get('MarketCapitalization', 0)),
                'annual_peak': float(statistics.get('52WeekHigh', 0)),
                'annual_trough': float(statistics.get('52WeekLow', 0))
            }
        except Exception as error:
            logging.error("Failed to retrieve financial data: %s", error)
            return {}

    def obtain_price_history(self, ticker: str) -> dict:

        api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={self.finance_api}&outputsize=full"
        try:
            price_response = requests.get(api_url, timeout=10)
            market_data = price_response.json()
            daily_prices = market_data.get('Time Series (Daily)', {})
            
            price_frame = pd.DataFrame.from_dict(daily_prices, orient='index')
            price_frame = price_frame.apply(pd.to_numeric)
            price_frame.index = pd.to_datetime(price_frame.index)
            price_frame = price_frame.sort_index()
            
            closing_prices = price_frame['4. close']
            recent_close = closing_prices.iloc[-1]
            medium_term_avg = closing_prices.rolling(50).mean()[-1]
            long_term_avg = closing_prices.rolling(200).mean()[-1]
            price_variability = closing_prices.pct_change().std() * np.sqrt(252)
            
            historical_data = {str(date): value for date, value in closing_prices.items()}
            
            return {
                'current_price': recent_close,
                'medium_term_average': medium_term_avg,
                'long_term_average': long_term_avg,
                'price_volatility': price_variability,
                'historical_prices': historical_data
            }
        except Exception as error:
            logging.error("Failed to obtain price data: %s", error)
            return {}

    def compute_financial_health(self, financial_data: dict) -> int:
        score = 0
 
        score += 1 if financial_data.get('asset_returns', 0) > 0 else 0
        score += 1 if financial_data.get('equity_returns', 0) > 0 else 0
        score += 1 if financial_data.get('net_margin', 0) > 0 else 0
        score += 1 if financial_data.get('liquidity_ratio', 0) > self.assessment_parameters['liquidity_ratio']['caution'] else 0
        score += 1 if financial_data.get('leverage_ratio', 0) < self.assessment_parameters['leverage_ratio']['caution'] else 0
        score += 1 if financial_data.get('asset_returns', 0) > financial_data.get('equity_returns', 0) * 0.5 else 0
        
        return score

    def estimate_fraud_probability(self, financial_data: dict) -> float:
        fraud_score = (
            -4.84 + 0.92 * self.fraud_detection_weights['receivables_index'] 
            + 0.528 * self.fraud_detection_weights['margin_index'] 
            + 0.404 * self.fraud_detection_weights['asset_quality'] 
            + 0.892 * self.fraud_detection_weights['growth_index'] 
            + 0.115 * self.fraud_detection_weights['depreciation_rate'] 
            - 0.172 * self.fraud_detection_weights['expense_index'] 
            + 4.679 * self.fraud_detection_weights['accruals_ratio'] 
            - 0.327 * self.fraud_detection_weights['leverage_change']
        )
        
        return fraud_score

    def analyze_media_sentiment(self, organization: str) -> dict:
        news_endpoint = f"https://newsapi.org/v2/everything?q={organization}&apiKey={self.news_api}&language=en&sortBy=publishedAt"
        try:
            news_response = requests.get(news_endpoint, timeout=10)
            articles_data = news_response.json()
            
            favorable = 0
            unfavorable = 0
            risk_terms = ['fraud', 'misconduct', 'probe', 'regulatory', 'irregularity']
            
            for story in articles_data.get('articles', [])[:50]:
                content = f"{story.get('title', '')} {story.get('description', '')}".lower()
                if any(term in content for term in ['positive', 'growth', 'bullish']):
                    favorable += 1
                if any(term in content for term in ['negative', 'decline', 'bearish']):
                    unfavorable += 1
            
            risk_mentions = sum(
                1 for story in articles_data.get('articles', []) 
                if any(term in f"{story.get('title', '')} {story.get('description', '')}".lower() 
                      for term in risk_terms)
            )
            
            return {
                'favorable_news': favorable,
                'unfavorable_news': unfavorable,
                'risk_references': risk_mentions,
                'articles_analyzed': len(articles_data.get('articles', []))
            }
        except Exception as error:
            logging.error("News analysis failed: %s", error)
            return {}

    def evaluate_company(self, ticker: str, company_name: str) -> dict:
        financial_metrics = self.retrieve_financial_statistics(ticker)
        market_data = self.obtain_price_history(ticker)
        media_analysis = self.analyze_media_sentiment(company_name)
        
        health_score = self.compute_financial_health(financial_metrics)
        fraud_score = self.estimate_fraud_probability(financial_metrics)
        
        current_value = market_data.get('current_price', 0)
        medium_avg = market_data.get('medium_term_average', 0)
        long_avg = market_data.get('long_term_average', 0)
        
        market_trend = "Neutral"
        if current_value > medium_avg > long_avg:
            market_trend = "Bullish"
        elif current_value < medium_avg < long_avg:
            market_trend = "Bearish"
        
        risk_category = "Low"
        if fraud_score > -1.78:
            risk_category = "Medium"
        if fraud_score > -1.0 or media_analysis.get('risk_references', 0) > 5:
            risk_category = "High"
        
        analysis_report = {
            'metadata': {
                'ticker_symbol': ticker,
                'organization': company_name,
                'evaluation_date': datetime.now().isoformat(),
                'data_providers': ['AlphaVantage', 'NewsAPI']
            },
            'financial_indicators': financial_metrics,
            'market_information': market_data,
            'media_evaluation': media_analysis,
            'risk_assessment': {
                'financial_stability': health_score,
                'fraud_probability': fraud_score,
                'risk_classification': risk_category,
                'market_trend': market_trend
            },
            'risk_notifications': self.generate_risk_alerts(financial_metrics, health_score, fraud_score, media_analysis)
        }
        
        self.store_analysis(analysis_report)
        
        return analysis_report

    def generate_risk_alerts(self, financials: dict, health_score: int, fraud_score: float, media: dict) -> list:
        notifications = []
        
        if health_score <= self.assessment_parameters['financial_health_score']['critical']:
            notifications.append("Critical financial weakness detected")
        elif health_score <= self.assessment_parameters['financial_health_score']['caution']:
            notifications.append("Financial health concerns identified")
            
        if financials.get('liquidity_ratio', 0) < self.assessment_parameters['liquidity_ratio']['critical']:
            notifications.append("Severe liquidity risk present")
            
        if financials.get('leverage_ratio', 0) > self.assessment_parameters['leverage_ratio']['critical']:
            notifications.append("Excessive debt burden identified")
            
        if fraud_score > -1.78:
            notifications.append("Potential financial irregularities suggested")
        if fraud_score > -1.0:
            notifications.append("High probability of financial misrepresentation")
            
        if media.get('risk_references', 0) > 3:
            notifications.append(f"Multiple risk references in media ({media.get('risk_references')} mentions)")
        if media.get('unfavorable_news', 0) > media.get('favorable_news', 0) * 2:
            notifications.append("Predominantly negative media coverage")
            
        return notifications

    def store_analysis(self, report: dict) -> str:
        def serialize_dates(data):
            if isinstance(data, (datetime, pd.Timestamp)):
                return data.isoformat()
            if isinstance(data, dict):
                return {key: serialize_dates(value) for key, value in data.items()}
            if isinstance(data, (list, tuple)):
                return [serialize_dates(item) for item in data]
            return data
        
        serialized_report = serialize_dates(report)
        output_file = self.reports_folder / "stock_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as output:
            json.dump(serialized_report, output, indent=2)
            
        return str(output_file)
    