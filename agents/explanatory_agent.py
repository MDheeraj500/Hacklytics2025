import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import Dict, List, Optional
from dotenv import load_dotenv
import json
import os
from langchain_core.output_parsers import StrOutputParser  # NEW IMPORT

class ExplanatoryAgent:
    def __init__(self, data_path: str = None, metadata_paths: Dict[str, str] = None):
        """Initialize the explanatory agent with data and metadata"""
        load_dotenv()
        # Define the prompt template
        prompt_template = """
You are an expert communicator with a strong background in market analytics and pricing strategies.
Your task is to translate complex quantitative analyses into clear, actionable business insights.
Explain the pricing strategy in detail by discussing market positioning, trend analysis, and forecasting insights.
Use a structured approach to highlight key metrics, compare historical benchmarks, and clearly describe the rationale behind the
recommended pricing adjustments in accessible, professional language.
data: {query}
Explanation:
"""
        prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_template
        )
        llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768")
        # Updated LCEL pipeline: prompt first, then LLM, and finally the output parser
        self.chain = prompt | llm | StrOutputParser()

        # Load data
        if data_path:
            self.df = pd.read_csv(data_path)
            self.models = [col.replace('_Price', '')
                           for col in self.df.columns if col.endswith('_Price')]
        else:
            self.df = None
            self.models = []

        # Load metadata
        self.metadata = {}
        if metadata_paths:
            for model, path in metadata_paths.items():
                with open(path, 'r') as f:
                    self.metadata[model] = json.load(f)

    def create_price_trend_visualization(self, model_name: str) -> go.Figure:
        # (Visualization code remains unchanged)
        price_col = f"{model_name}_Price"
        sentiment_col = f"{model_name}_Sentiment"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df['Date'],
            y=self.df[price_col],
            name='Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=self.df['Date'],
            y=self.df[sentiment_col],
            name='Sentiment',
            line=dict(color='red'),
            yaxis='y2'
        ))
        fig.update_layout(
            title=f'{model_name} Price and Sentiment Trends',
            yaxis=dict(title='Price ($)'),
            yaxis2=dict(title='Sentiment', overlaying='y', side='right'),
            hovermode='x unified'
        )
        return fig

    def create_feature_importance_chart(self, model_name: str,
                                        market_analysis: dict,
                                        math_analysis: dict) -> go.Figure:
        features = {
            'Sentiment Trend': abs(market_analysis[model_name]['sentiment_trends']['sentiment_change']),
            'Price Volatility': math_analysis[model_name]['trends']['volatility'],
            'Price-Sentiment Correlation': abs(market_analysis[model_name]['price_sentiment_relationship']['immediate_correlation']),
            'Weekend Effect': abs(market_analysis[model_name]['weekend_effect']['weekend_effect'])
        }
        fig = go.Figure([go.Bar(
            x=list(features.keys()),
            y=list(features.values()),
            text=[f'{v:.2f}' for v in features.values()],
            textposition='auto'
        )])
        fig.update_layout(
            title=f'Feature Importance Analysis - {model_name}',
            yaxis_title='Importance Score',
            xaxis_title='Features'
        )
        return fig

    def generate_comparative_analysis(self, recommendations: dict, model_name: str) -> str:
        model_rec = recommendations[model_name]
        price_col = f"{model_name}_Price"
        historical_prices = self.df[price_col]
        query = f"""
Analyze the pricing strategy for {model_name}:
- Current price: ${model_rec['price_optimization']['current_price']:.2f}
- Recommended price: ${model_rec['price_optimization']['optimal_price']:.2f}
- Market position: {model_rec['market_position']['price_position']}
- Historical context:
  * Average price: ${historical_prices.mean():.2f}
  * Price range: ${historical_prices.min():.2f} - ${historical_prices.max():.2f}
Provide insights on the model's pricing strategy and market positioning.
"""
        try:
            return self.chain.invoke({"query": query})
        except Exception as e:
            print(f"[ExplanatoryAgent] Error in generate_comparative_analysis with query: {query}\nError: {e}")
            raise

    def explain_recommendation_rationale(self, model_name: str,
                                         market_analysis: dict,
                                         math_analysis: dict,
                                         recommendations: dict) -> str:
        model_rec = recommendations[model_name]
        model_market = market_analysis[model_name]
        model_math = math_analysis[model_name]
        query = f"""
Explain the rationale behind the pricing recommendation for {model_name}:
Market Conditions:
- Sentiment trend: {model_market['sentiment_trends']['sentiment_change']}
- Market position: {model_rec['market_position']['price_position']}
Mathematical Analysis:
- Price volatility: {model_math['trends']['volatility']}
- Forecast direction: {'up' if model_math['forecast']['forecast'][0] > model_math['forecast']['last_price'] else 'down'}
Recommendation:
- Current price: ${model_rec['price_optimization']['current_price']}
- Recommended price: ${model_rec['price_optimization']['optimal_price']}
- Confidence: {model_rec['price_optimization']['confidence_score']}
Please provide a detailed explanation of this recommendation.
"""
        try:
            return self.chain.invoke({"query": query})
        except Exception as e:
            print(f"[ExplanatoryAgent] Error in explain_recommendation_rationale with query: {query}\nError: {e}")
            raise

    def generate_business_insights(self, model_name: str,
                                   market_analysis: dict,
                                   math_analysis: dict,
                                   recommendations: dict) -> dict:
        return {
            'price_analysis': self.explain_recommendation_rationale(
                model_name, market_analysis, math_analysis, recommendations
            ),
            'market_position': self.generate_comparative_analysis(recommendations, model_name),
            'visualizations': {
                'price_trends': self.create_price_trend_visualization(model_name),
                'feature_importance': self.create_feature_importance_chart(model_name, market_analysis, math_analysis)
            }
        }

    def analyze(self, market_analysis: dict,
                math_analysis: dict,
                recommendations: dict,
                model_name: str) -> dict:
        if not model_name or model_name not in self.models:
            raise ValueError(f"Invalid model name: {model_name}")
        return {
            model_name: self.generate_business_insights(model_name, market_analysis, math_analysis, recommendations)
        }
