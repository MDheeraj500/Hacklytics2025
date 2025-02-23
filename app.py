import os
import streamlit as st
from typing import Dict, Optional
from dotenv import load_dotenv
from agents.mathematical_agent import MathematicalAgent
from agents.market_insights_agent import MarketInsightsAgent
from agents.recommendation_agent import RecommendationAgent
from agents.explanatory_agent import ExplanatoryAgent
import plotly.graph_objects as go
import json

class AgentOrchestrator:
    def __init__(self, data_path: str, metadata_paths: Dict[str, str]):
        """Initialize the orchestrator with all required agents"""
        load_dotenv()
        
        # Initialize all agents with the same data sources
        self.mathematical_agent = MathematicalAgent(data_path=data_path)
        self.market_insights_agent = MarketInsightsAgent(
            data_path=data_path,
            metadata_paths=metadata_paths
        )
        self.recommendation_agent = RecommendationAgent(
            data_path=data_path,
            metadata_paths=metadata_paths
        )
        self.explanatory_agent = ExplanatoryAgent(
            data_path=data_path,
            metadata_paths=metadata_paths
        )

    def analyze_model(self, model_name: str) -> Dict:
        """Run complete analysis pipeline for a specific model"""
        try:
            # Step 1: Get mathematical analysis
            math_analysis = self.mathematical_agent.analyze(model_name)
            
            # Step 2: Get market insights
            market_analysis = self.market_insights_agent.analyze(model_name)
            
            # Step 3: Generate recommendations
            recommendations = self.recommendation_agent.analyze(
                market_analysis=market_analysis,
                math_analysis=math_analysis,
                model_name=model_name
            )
            
            # Step 4: Generate explanatory insights
            explanations = self.explanatory_agent.analyze(
                market_analysis=market_analysis,
                math_analysis=math_analysis,
                recommendations=recommendations,
                model_name=model_name
            )
            
            return {
                'mathematical_analysis': math_analysis,
                'market_analysis': market_analysis,
                'recommendations': recommendations,
                'explanations': explanations
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing model {model_name}: {str(e)}")

def display_price_trends(model_results: Dict, model_name: str):
    """Display price trends visualization"""
    if 'explanations' in model_results:
        fig = model_results['explanations'][model_name]['visualizations']['price_trends']
        st.plotly_chart(fig, key="price_trends_main")

def display_feature_importance(model_results: Dict, model_name: str):
    """Display feature importance visualization"""
    if 'explanations' in model_results:
        fig = model_results['explanations'][model_name]['visualizations']['feature_importance']
        st.plotly_chart(fig, key="feature_importance_main")

def display_recommendations(model_results: Dict, model_name: str):
    """Display pricing recommendations"""
    if 'recommendations' in model_results:
        rec = model_results['recommendations'][model_name]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Current Price",
                f"${rec['price_optimization']['current_price']:.2f}",
                f"{rec['price_optimization']['adjustment']:.2f}"
            )
        with col2:
            st.metric(
                "Recommended Price",
                f"${rec['price_optimization']['optimal_price']:.2f}",
                f"Confidence: {rec['price_optimization']['confidence_score']:.2f}"
            )
        
        st.write("### Recommended Price Ranges")
        ranges = rec['recommended_ranges']
        st.write(f"Minimum: ${ranges['minimum']:.2f}")
        st.write(f"Maximum: ${ranges['maximum']:.2f}")
        st.write(f"Ideal: ${ranges['ideal']:.2f}")

def display_business_insights(model_results: Dict, model_name: str):
    """Display business insights and explanations"""
    if 'explanations' in model_results:
        insights = model_results['explanations'][model_name]
        
        # Display price analysis
        st.write("### Price Analysis")
        st.write(insights['price_analysis'])
        
        # Display market position analysis
        st.write("### Market Position Analysis")
        st.write(insights['market_position'])
        
        # commented the graphs cause they were redundant.
        # Display visualizations in columns
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.write("### Price and Sentiment Trends")
        #     st.plotly_chart(
        #         insights['visualizations']['price_trends'], 
        #         use_container_width=True,
        #         key="price_trends_insights"
        #     )
        
        # with col2:
        #     st.write("### Feature Importance Analysis")
        #     st.plotly_chart(
        #         insights['visualizations']['feature_importance'], 
        #         use_container_width=True,
        #         key="feature_importance_insights"
        #     )

def main():
    st.set_page_config(page_title="iPhone Price Analysis", layout="wide")
    
    st.title("iPhone Price Analysis Dashboard")
    
    # Define data paths
    data_path = "data/final_merged_data.csv"
    metadata_paths = {
        "iPhone_12": "data/metadata_iphone_12.json",
        "iPhone_12_Mini": "data/metadata_iphone_12_mini.json",
        "iPhone_12_Pro": "data/metadata_iphone_12_pro.json"
    }
    
    # Initialize orchestrator
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = AgentOrchestrator(data_path, metadata_paths)
    
    # Model selection
    models = ["iPhone_12", "iPhone_12_Mini", "iPhone_12_Pro"]
    selected_model = st.selectbox("Select iPhone Model", models)
    
    # Analysis button
    if st.button("Analyze"):
        with st.spinner(f"Analyzing {selected_model}..."):
            try:
                # Run analysis
                results = st.session_state.orchestrator.analyze_model(selected_model)
                st.session_state.results = results
                
                # Display results
                st.success("Analysis complete!")
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Business Insights",
                    "Price Trends", 
                    "Recommendations", 
                    "Market Analysis"
                ])
                
                with tab1:
                    st.header("Business Insights and Explanations")
                    display_business_insights(results, selected_model)
                
                with tab2:
                    st.header("Price and Sentiment Trends")
                    display_price_trends(results, selected_model)
                    display_feature_importance(results, selected_model)
                
                with tab3:
                    st.header("Pricing Recommendations")
                    display_recommendations(results, selected_model)
                    
                    if 'recommendations' in results:
                        summary = st.session_state.orchestrator.recommendation_agent.get_recommendation_summary(
                            selected_model, results['recommendations']
                        )
                        st.write("### Recommendation Summary")
                        st.write(summary)
                
                with tab4:
                    st.header("Market Insights")
                    if 'market_analysis' in results:
                        # Display market summary
                        summary = st.session_state.orchestrator.market_insights_agent.get_market_summary(
                            selected_model
                        )
                        st.write("### Market Summary")
                        st.write(summary)
                        
                        # Display sentiment anomalies
                        anomalies = results['market_analysis'][selected_model]['sentiment_anomalies']
                        if anomalies:
                            st.write("### Sentiment Anomalies")
                            st.dataframe(anomalies)
                        
                        # Display additional market metrics
                        st.write("### Market Metrics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_trends = results['market_analysis'][selected_model]['sentiment_trends']
                            st.metric(
                                "Current Sentiment",
                                f"{sentiment_trends['current_sentiment']:.2f}",
                                f"{sentiment_trends['sentiment_change']:.2f}"
                            )
                        
                        with col2:
                            price_sentiment = results['market_analysis'][selected_model]['price_sentiment_relationship']
                            st.metric(
                                "Price-Sentiment Correlation",
                                f"{price_sentiment['immediate_correlation']:.2f}"
                            )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Add sidebar with metadata and help
    with st.sidebar:
        st.header("Model Specifications")
        if selected_model:
            with open(metadata_paths[selected_model], 'r') as f:
                metadata = json.load(f)
                
                # Display Display Specifications
                st.subheader("📱 Display")
                st.write(f"**Size:** {metadata['display']['size']}")
                st.write(f"**Resolution:** {metadata['display']['resolution']}")

                # Display Camera Specifications
                st.subheader("📸 Camera System")
                st.write("**Rear Camera:**")
                for cam_type, spec in metadata['camera']['rear'].items():
                    if isinstance(spec, bool):  # For LiDAR
                        st.write(f"- {cam_type.title()}: Yes")
                    else:
                        st.write(f"- {cam_type.title()}: {spec}")
                st.write(f"**Front Camera:** {metadata['camera']['front']}")

                # Display Core Specifications
                st.subheader("⚡ Core Specs")
                st.write(f"**Processor:** {metadata['processor']}")
                st.write(f"**RAM:** {metadata['memory']['ram']}")
                st.write(f"**Storage:** {metadata['memory']['storage']}")
                st.write(f"**OS:** {metadata['operating_system']}")

                # Display Connectivity
                st.subheader("📡 Connectivity")
                for conn_type, spec in metadata['connectivity'].items():
                    st.write(f"**{conn_type.title()}:** {spec}")

                # Display Security Features
                st.subheader("🔒 Security")
                st.write(f"**Type:** {metadata['security']['type']}")
                st.write(f"**Feature:** {metadata['security']['feature']}")

                # Display Battery and Physical Specs
                st.subheader("🔋 Battery & Physical")
                st.write(f"**Battery Capacity:** {metadata['battery']['capacity']}")
                st.write(f"**Weight:** {metadata['physical']['weight']['grams']}g ({metadata['physical']['weight']['ounces']}oz)")

                # Display Release Date
                st.subheader("📅 Release Info")
                st.write(f"**Release Date:** {metadata['release_date']}")
        
        st.markdown("---")
        st.header("About")
        st.write("""
        This dashboard provides comprehensive analysis of iPhone pricing:
        - Business Insights: Key findings and explanations
        - Price Trends: Historical price and sentiment analysis
        - Recommendations: Pricing optimization suggestions
        - Market Analysis: Detailed market metrics and anomalies
        """)

if __name__ == "__main__":
    main() 