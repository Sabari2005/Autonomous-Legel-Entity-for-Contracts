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
class CompanyDataVisualizer:
    def __init__(self, file_path):
        self.data = self.load_company_data(file_path)
        self.company_name = self.data['metadata']['organization']
        self.ticker_symbol = self.data['metadata']['ticker_symbol']
        
    def load_company_data(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def create_financial_health_chart(self):
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        figure.suptitle(f'{self.company_name} ({self.ticker_symbol}) Financial Health', y=1.02)
        
        profit_metrics = {
            'Net Margin': self.data['financial_indicators']['net_margin'] * 100,
            'ROA': self.data['financial_indicators']['asset_returns'] * 100,
            'ROE': self.data['financial_indicators']['equity_returns'] * 100
        }
        axes[0,0].bar(profit_metrics.keys(), profit_metrics.values(), color=['#4CAF50', '#2196F3', '#009688'])
        axes[0,0].set_title('Profitability Metrics (%)')
        axes[0,0].grid(axis='y', linestyle='--', alpha=0.7)
        
        company_valuation = {
            'P/E Ratio': self.data['financial_indicators']['valuation_ratio'],
            'Market Cap ($B)': self.data['financial_indicators']['company_valuation'] / 1e9
        }
        axes[0,1].bar(company_valuation.keys(), company_valuation.values(), color=['#FF9800', '#E91E63'])
        axes[0,1].set_title('Valuation Metrics')
        axes[0,1].grid(axis='y', linestyle='--', alpha=0.7)
        
        financial_ratios = {
            'Current Ratio': self.data['financial_indicators']['liquidity_ratio'],
            'Debt/Equity': self.data['financial_indicators']['leverage_ratio']
        }
        axes[1,0].bar(financial_ratios.keys(), financial_ratios.values(), color=['#9C27B0', '#607D8B'])
        axes[1,0].set_title('Financial Structure')
        axes[1,0].grid(axis='y', linestyle='--', alpha=0.7)
        
        stock_prices = {
            'Current': self.data['market_information']['current_price'],
            '52W High': self.data['financial_indicators']['annual_peak'],
            '52W Low': self.data['financial_indicators']['annual_trough']
        }
        axes[1,1].bar(stock_prices.keys(), stock_prices.values(), color=['#795548', '#F44336', '#3F51B5'])
        axes[1,1].set_title('Price Levels ($)')
        axes[1,1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/financial_health_dashboard.png', bbox_inches='tight')
        plt.close()
        
    def create_price_history_chart(self):
        plt.figure(figsize=(14, 7))
        
        price_history = pd.DataFrame.from_dict(
            self.data['market_information']['historical_prices'], 
            orient='index', 
            columns=['Price']
        )
        price_history.index = pd.to_datetime(price_history.index)
        price_history = price_history.sort_index()
        
        plt.plot(price_history.index, price_history['Price'], color='#1f77b4', linewidth=2, label='Daily Close')
        
        moving_average_50 = price_history.rolling(50).mean()
        moving_average_200 = price_history.rolling(200).mean()
        plt.plot(moving_average_50.index, moving_average_50['Price'], '--', color='#ff7f0e', linewidth=1.5, label='50-Day MA')
        plt.plot(moving_average_200.index, moving_average_200['Price'], '--', color='#2ca02c', linewidth=1.5, label='200-Day MA')
        
        current_stock_price = self.data['market_information']['current_price']
        plt.axhline(y=current_stock_price, color='r', linestyle='-', linewidth=1, label=f'Current: ${current_stock_price:.2f}')
        
        plt.title(f'{self.company_name} ({self.ticker_symbol}) Price History with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/price_analysis.png')
        plt.close()
        
    def create_risk_assessment_chart(self):
        figure, axes = plt.subplots(1, 2, figsize=(14, 6))
        figure.suptitle(f'{self.company_name} Risk Analysis', y=1.05)
        
        risk_metrics = {
            'Financial Stability': self.data['risk_assessment']['financial_stability'],
            'Fraud Probability': self.data['risk_assessment']['fraud_probability']
        }
        risk_colors = ['#4CAF50' if score > 5 else '#FF9800' if score > 3 else '#F44336' for score in [risk_metrics['Financial Stability'], 0]]
        risk_colors[1] = '#F44336' if risk_metrics['Fraud Probability'] > -1.78 else '#FF9800' if risk_metrics['Fraud Probability'] > -2.5 else '#4CAF50'
        
        axes[0].barh(list(risk_metrics.keys()), list(risk_metrics.values()), color=risk_colors)
        axes[0].set_title('Risk Scores')
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].grid(True, axis='x', linestyle='--', alpha=0.7)
        
        news_sentiment = {
            'Favorable': self.data['media_evaluation']['favorable_news'],
            'Neutral': self.data['media_evaluation']['articles_analyzed'] - self.data['media_evaluation']['favorable_news'] - self.data['media_evaluation']['unfavorable_news'],
            'Unfavorable': self.data['media_evaluation']['unfavorable_news'],
            'Risk Mentions': self.data['media_evaluation']['risk_references']
        }
        axes[1].bar(news_sentiment.keys(), news_sentiment.values(), 
                  color=['#4CAF50', '#FFC107', '#F44336', '#9E9E9E'])
        axes[1].set_title('Media Sentiment Analysis')
        axes[1].set_ylabel('Number of Articles')
        axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/risk_analysis.png')
        plt.close()
        
    def create_summary_report(self):
        report = {
            "Company Overview": {
                "Name": self.company_name,
                "Ticker": self.ticker_symbol,
                "Analysis Date": self.data['metadata']['evaluation_date']
            },
            "Financial Summary": {
                "Market Cap ($B)": round(self.data['financial_indicators']['company_valuation'] / 1e9, 2),
                "P/E Ratio": round(self.data['financial_indicators']['valuation_ratio'], 2),
                "Net Margin (%)": round(self.data['financial_indicators']['net_margin'] * 100, 2),
                "Current Ratio": round(self.data['financial_indicators']['liquidity_ratio'], 2),
                "Debt/Equity": round(self.data['financial_indicators']['leverage_ratio'], 2)
            },
            "Market Performance": {
                "Current Price": round(self.data['market_information']['current_price'], 2),
                "50-Day MA": round(self.data['market_information']['medium_term_average'], 2),
                "200-Day MA": round(self.data['market_information']['long_term_average'], 2),
                "52-Week Range": f"{self.data['financial_indicators']['annual_trough']} - {self.data['financial_indicators']['annual_peak']}",
                "Volatility": round(self.data['market_information']['price_volatility'], 4)
            },
            "Risk Assessment": {
                "Financial Stability Score": self.data['risk_assessment']['financial_stability'],
                "Fraud Probability Score": round(self.data['risk_assessment']['fraud_probability'], 2),
                "Overall Risk Classification": self.data['risk_assessment']['risk_classification'],
                "Market Trend": self.data['risk_assessment']['market_trend'],
                "Risk Notifications": self.data['risk_notifications']
            },
            "Media Coverage": {
                "Total Articles Analyzed": self.data['media_evaluation']['articles_analyzed'],
                "Favorable Sentiment": self.data['media_evaluation']['favorable_news'],
                "Unfavorable Sentiment": self.data['media_evaluation']['unfavorable_news'],
                "Risk Mentions": self.data['media_evaluation']['risk_references']
            }
        }
        
        with open('/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/company_summary_report.json', 'w') as output_file:
            json.dump(report, output_file, indent=2)
            
        return report
    
    def generate_all_reports(self):
        self.create_financial_health_chart()
        self.create_price_history_chart()
        self.create_risk_assessment_chart()
        final_report = self.create_summary_report()
        
        print("Generated the following files:")
        print("- financial_health_dashboard.png")
        print("- price_analysis.png")
        print("- risk_analysis.png")
        print("- company_summary_report.json")
        
        return final_report
