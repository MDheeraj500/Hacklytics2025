import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

class MarketInsightsAgent:
    def __init__(self, data_path: str = None, metadata_paths: Dict[str, str] = None):
        """Initialize the market insights agent with data and metadata"""
        load_dotenv()
        
        # Initialize LLM
        # prompt_template = """You are a market analysis expert.
        # Please analyze the following market data and provide insights:
        
        # Data: {query}
        
        # Analysis:"""

        prompt_template = """You are a market analysis expert with deep expertise in customer sentiment analysis and pricing dynamics. 
        Given the dataset, evaluate short-term and long-term sentiment trends, compare weekday versus weekend sentiment differences, 
        and analyze the correlation between price changes and sentiment shifts. Provide a structured, evidence-based analysis that 
        identifies any anomalies and discusses the potential market implications in a professional tone.
        
        data: {query}
        
        """
        
        prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_template
        )
        
        llm = ChatGroq(temperature=0.7, model_name="deepseek-r1-distill-qwen-32b")
        self.chain = llm | prompt

        # Load data
        if data_path:
            self.df = pd.read_csv(data_path)
            self.models = [col.replace('_Sentiment', '') 
                         for col in self.df.columns 
                         if col.endswith('_Sentiment')]
        else:
            self.df = None
            self.models = []

        # Load metadata
        self.metadata = {}
        if metadata_paths:
            for model, path in metadata_paths.items():
                with open(path, 'r') as f:
                    self.metadata[model] = json.load(f)

    def analyze_sentiment_trends(self, model_name: str) -> dict:
        """Analyze sentiment trends for a specific model"""
        sentiment_col = f"{model_name}_Sentiment"
        sentiments = self.df[sentiment_col]
        
        return {
            'current_sentiment': float(sentiments.iloc[-1]),
            'week_ago_sentiment': float(sentiments.iloc[-7]),
            'sentiment_change': float(sentiments.iloc[-1] - sentiments.iloc[-7]),
            'avg_sentiment_30d': float(sentiments.tail(30).mean()),
            'sentiment_volatility': float(sentiments.tail(30).std())
        }

    def analyze_weekend_effect(self, model_name: str) -> dict:
        """Analyze sentiment differences between weekdays and weekends"""
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['is_weekend'] = self.df['Date'].dt.dayofweek.isin([5, 6])
        
        sentiment_col = f"{model_name}_Sentiment"
        weekend_sentiment = self.df[self.df['is_weekend']][sentiment_col].mean()
        weekday_sentiment = self.df[~self.df['is_weekend']][sentiment_col].mean()
        
        return {
            'weekend_avg_sentiment': float(weekend_sentiment),
            'weekday_avg_sentiment': float(weekday_sentiment),
            'weekend_effect': float(weekend_sentiment - weekday_sentiment)
        }

    def analyze_price_sentiment_relationship(self, model_name: str) -> dict:
        """Analyze relationship between price and sentiment"""
        price_col = f"{model_name}_Price"
        sentiment_col = f"{model_name}_Sentiment"
        
        # Calculate lagged correlations
        sentiment_lead = self.df[sentiment_col].shift(-1)  # Next day's sentiment
        price_lead = self.df[price_col].shift(-1)  # Next day's price
        
        return {
            'immediate_correlation': float(self.df[price_col].corr(self.df[sentiment_col])),
            'sentiment_leads_price': float(self.df[price_col].corr(sentiment_lead)),
            'price_leads_sentiment': float(self.df[sentiment_col].corr(price_lead))
        }

    def identify_sentiment_anomalies(self, model_name: str, threshold: float = 2.0) -> List[Dict]:
        """Identify unusual sentiment patterns"""
        sentiment_col = f"{model_name}_Sentiment"
        sentiments = self.df[sentiment_col]
        
        # Calculate rolling statistics
        rolling_mean = sentiments.rolling(window=7).mean()
        rolling_std = sentiments.rolling(window=7).std()
        
        # Identify anomalies
        z_scores = (sentiments - rolling_mean) / rolling_std
        anomalies = self.df[abs(z_scores) > threshold]
        
        return [{
            'date': str(date),
            'sentiment': float(sentiment),
            'z_score': float(z_scores[idx])
        } for date, sentiment, idx in zip(
            anomalies['Date'],
            anomalies[sentiment_col],
            anomalies.index
        )]

    def analyze(self, model_name: str = None) -> dict:
        """Generate complete market insights analysis"""
        if model_name and model_name in self.models:
            models_to_analyze = [model_name]
        else:
            models_to_analyze = self.models

        analysis = {}
        for model in models_to_analyze:
            analysis[model] = {
                'sentiment_trends': self.analyze_sentiment_trends(model),
                'weekend_effect': self.analyze_weekend_effect(model),
                'price_sentiment_relationship': self.analyze_price_sentiment_relationship(model),
                'sentiment_anomalies': self.identify_sentiment_anomalies(model),
                'metadata': self.metadata.get(model, {})
            }
        
        return analysis

    def get_market_summary(self, model_name: str) -> str:
        """Generate a natural language summary of market insights"""
        analysis = self.analyze(model_name)[model_name]
        
        query = f"""
        Based on the following analysis for {model_name}:
        - Current sentiment: {analysis['sentiment_trends']['current_sentiment']}
        - Sentiment change (7d): {analysis['sentiment_trends']['sentiment_change']}
        - Price-Sentiment correlation: {analysis['price_sentiment_relationship']['immediate_correlation']}
        - Weekend vs Weekday sentiment difference: {analysis['weekend_effect']['weekend_effect']}
        
        Provide a brief market summary.
        """
        result = self.chain.invoke(query)

        return result