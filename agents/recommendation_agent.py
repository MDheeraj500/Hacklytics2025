import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import Dict, List, Optional
from dotenv import load_dotenv
import json
import os
from langchain_core.output_parsers import StrOutputParser  # NEW IMPORT

class RecommendationAgent:
    def __init__(self, data_path: str = None, metadata_paths: Dict[str, str] = None):
        """Initialize the recommendation agent with data and metadata"""
        load_dotenv()
        prompt_template = """You are a pricing strategy expert with extensive experience in competitive market analysis.
Based on the provided market insights and mathematical forecasting data, determine the optimal price adjustment for the product.
Clearly articulate the underlying factors—including sentiment trends, forecasted price changes, and historical pricing data—by
providing specific numerical justification. Recommend an optimal price, a detailed price range, and explain the product's competitive positioning.
data: {query}
"""
        prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_template
        )
        llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768")
        self.chain = prompt | llm | StrOutputParser()

        if data_path:
            self.df = pd.read_csv(data_path)
            self.models = [col.replace('_Price', '')
                           for col in self.df.columns if col.endswith('_Price')]
        else:
            self.df = None
            self.models = []

        self.metadata = {}
        if metadata_paths:
            for model, path in metadata_paths.items():
                with open(path, 'r') as f:
                    self.metadata[model] = json.load(f)

    def calculate_optimal_price(self, model_name: str,
                                market_analysis: dict,
                                math_analysis: dict) -> dict:
        price_col = f"{model_name}_Price"
        current_price = float(self.df[price_col].iloc[-1])
        sentiment_trend = market_analysis[model_name]['sentiment_trends']['sentiment_change']
        price_forecast = math_analysis[model_name]['forecast']['forecast'][0]
        price_sentiment_corr = market_analysis[model_name]['price_sentiment_relationship']['immediate_correlation']
        sentiment_factor = 0.01 * sentiment_trend * price_sentiment_corr
        forecast_factor = (price_forecast - current_price) / current_price
        adjustment = current_price * (sentiment_factor + forecast_factor) / 2
        optimal_price = current_price + adjustment
        return {
            'current_price': current_price,
            'optimal_price': float(optimal_price),
            'adjustment': float(adjustment),
            'confidence_score': float(abs(price_sentiment_corr))
        }

    def generate_price_ranges(self, model_name: str,
                              optimal_price: float,
                              confidence: float) -> dict:
        margin = optimal_price * (1 - confidence) * 0.1
        return {
            'minimum': float(optimal_price - margin),
            'maximum': float(optimal_price + margin),
            'ideal': float(optimal_price)
        }

    def assess_competitive_position(self, model_name: str,
                                    optimal_price: float) -> dict:
        price_col = f"{model_name}_Price"
        historical_prices = self.df[price_col]
        return {
            'price_position': 'premium' if optimal_price > historical_prices.mean() else 'competitive',
            'historical_avg': float(historical_prices.mean()),
            'historical_min': float(historical_prices.min()),
            'historical_max': float(historical_prices.max())
        }

    def generate_recommendations(self, model_name: str,
                                 market_analysis: dict,
                                 math_analysis: dict) -> dict:
        price_calc = self.calculate_optimal_price(model_name, market_analysis, math_analysis)
        ranges = self.generate_price_ranges(model_name, price_calc['optimal_price'], price_calc['confidence_score'])
        position = self.assess_competitive_position(model_name, price_calc['optimal_price'])
        return {
            'price_optimization': price_calc,
            'recommended_ranges': ranges,
            'market_position': position,
            'metadata': self.metadata.get(model_name, {})
        }

    def analyze(self, market_analysis: dict, math_analysis: dict,
                model_name: str = None) -> dict:
        if model_name and model_name in self.models:
            models_to_analyze = [model_name]
        else:
            models_to_analyze = self.models
        recommendations = {}
        for model in models_to_analyze:
            recommendations[model] = self.generate_recommendations(model, market_analysis, math_analysis)
        return recommendations

    def get_recommendation_summary(self, model_name: str,
                                   recommendations: dict) -> str:
        model_rec = recommendations[model_name]
        query = f"""
Based on the following recommendations for {model_name}:
- Current price: ${model_rec['price_optimization']['current_price']}
- Recommended price: ${model_rec['price_optimization']['optimal_price']}
- Price adjustment: ${model_rec['price_optimization']['adjustment']}
- Confidence score: {model_rec['price_optimization']['confidence_score']}
- Recommended range: ${model_rec['recommended_ranges']['minimum']} - ${model_rec['recommended_ranges']['maximum']}
- Market position: {model_rec['market_position']['price_position']}
Provide a brief recommendation summary.
"""
        try:
            result = self.chain.invoke({"query": query})
            return result
        except Exception as e:
            print(f"[RecommendationAgent] Error in get_recommendation_summary with query: {query}\nError: {e}")
            raise
