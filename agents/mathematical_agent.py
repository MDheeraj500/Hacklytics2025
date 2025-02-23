import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser  # NEW IMPORT
from prophet import Prophet

class MathematicalAgent:
    def __init__(self, data_path=None):
        """Initialize the mathematical agent with LLM chain and data"""
        load_dotenv()
        prompt_template = """You are a seasoned mathematical and statistical analyst specializing in financial time series.
Analyze the provided pricing data by calculating current values, recent changes, 30-day averages, and volatility.
Use facebook's prophet or other relevant forecasting techniques to project future prices, and provide a detailed, step-by-step explanation
that includes numerical evidence, statistical metrics, and a final summary of the pricing trend.
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
            self.models = [col.replace('_Price', '') for col in self.df.columns if col.endswith('_Price')]
        else:
            self.df = None
            self.models = []

    def run(self, query):
        """Process a mathematical query and return the response"""
        try:
            response = self.chain.invoke({"query": query})
            return response
        except Exception as e:
            print(f"[MathematicalAgent] Error in run with query: {query}\nError: {e}")
            raise Exception(f"Error processing query: {str(e)}")

    def analyze_price_trends(self, model_name: str) -> dict:
        price_col = f"{model_name}_Price"
        prices = self.df[price_col]
        return {
            'current_price': float(prices.iloc[-1]),
            'week_ago_price': float(prices.iloc[-7]),
            'price_change': float(prices.iloc[-1] - prices.iloc[-7]),
            'avg_price_30d': float(prices.tail(30).mean()),
            'volatility': float(prices.tail(30).std())
        }

    def analyze_correlations(self, model_name: str) -> dict:
        price_col = f"{model_name}_Price"
        sentiment_col = f"{model_name}_Sentiment"
        return {
            'price_sentiment_corr': float(self.df[price_col].corr(self.df[sentiment_col])),
            'price_gdp_corr': float(self.df[price_col].corr(self.df['GDP']))
        }

   
    def forecast_prices_prophet(self, model_name: str, days: int = 7) -> dict:
        """
        Forecast pricing using Prophet while incorporating macroeconomic data (GDP) and the mobile device's sentiment score.
        For the future period, this minimalistic approach assumes that GDP and sentiment remain constant at their latest available values.
        """
        price_col = f"{model_name}_Price"
        sentiment_col = f"{model_name}_Sentiment"
        date_col = "Date"

        # Prepare data for Prophet: include Date, Price, Sentiment, and GDP.
        df_prophet = self.df[[date_col, price_col, sentiment_col, "GDP"]].copy()
        df_prophet.rename(columns={
            date_col: "ds",
            price_col: "y",
            sentiment_col: "sentiment"
        }, inplace=True)
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

        # Initialize Prophet and add regressors for GDP and sentiment.
        model = Prophet()
        model.add_regressor("GDP")
        model.add_regressor("sentiment")
        model.fit(df_prophet)

        # Create a future dataframe.
        future = model.make_future_dataframe(periods=days)

        # Minimalistic approach: assume GDP and sentiment remain constant at their last observed values.
        last_gdp = df_prophet["GDP"].iloc[-1]
        last_sentiment = df_prophet["sentiment"].iloc[-1]
        future["GDP"] = last_gdp
        future["sentiment"] = last_sentiment

        forecast = model.predict(future)
        forecast_list = forecast.tail(days)["yhat"].tolist()
        last_price = df_prophet["y"].iloc[-1]
        return {
            "forecast": forecast_list,
            "last_price": float(last_price)
        }

    # NEW METHOD: Wrapper to predict the next 30 days using Prophet
    def predict_next_month_prophet(self, model_name: str) -> dict:
        """Predict the price for the next 180 days using Prophet"""
        return self.forecast_prices_prophet(model_name, days=180)

    def analyze(self, model_name: str = None) -> dict:
        """Generate complete analysis for one or all models"""
        if model_name and model_name in self.models:
            models_to_analyze = [model_name]
        else:
            models_to_analyze = self.models
        analysis = {}
        for model in models_to_analyze:
            analysis[model] = {
                'trends': self.analyze_price_trends(model),
                'correlations': self.analyze_correlations(model),
                'forecast': self.forecast_prices_prophet(model) 
            }
    

    def analyze(self, model_name: str = None) -> dict:
        if model_name and model_name in self.models:
            models_to_analyze = [model_name]
        else:
            models_to_analyze = self.models
        analysis = {}
        for model in models_to_analyze:
            analysis[model] = {
                'trends': self.analyze_price_trends(model),
                'correlations': self.analyze_correlations(model),
                'forecast': self.forecast_prices_prophet(model)
            }
        return analysis
