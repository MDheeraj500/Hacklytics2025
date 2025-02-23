# Multi-Agent AI System for Dynamic Predictive Insights at Scale

## Overview
This project is a Multi-Agent AI system developed to accelerate predictive insights‐at‐scale through the integration of dynamic market data, macroeconomic indicators, and real‑time sentiment analysis. Our solution addresses the limitations of traditional 2D historical models by leveraging multiple specialized agents that collectively generate AI-powered recommendations and explainable business insights. The system is designed to support strategic decision-making in pricing optimization, demand forecasting, inventory management, risk assessment, and customer engagement.

## Problem Statement & Use Cases
Traditional predictive models often struggle with adapting to real‑time market volatility, sudden macroeconomic shifts, and unstructured last‑minute information. Our challenge was to build a solution that:
- **Case 1 – Pricing Strategy:** Predicts the monthly price for mobile devices (e.g., iPhone 11 Plus 256GB, iPhone 12 Mini 128GB) over the next 6 months (180 days) using advanced forecasting techniques.
- **Case 2 – Insights Reporting (Free Style):** Provides an analytical dashboard that extracts actionable insights on asset protection or social responsibility in disaster recovery, including anomaly detection, predictive trends, and value‑driven recommendations.

## Architecture & Design
Our solution is built with a modular, multi‐agent architecture that consists of the following key components:

- **Mathematical Reasoning Agent:**  
  - Uses both ARIMA and Prophet models, with dynamic integration of external regressors (GDP and device sentiment) to accurately forecast future prices.
  - Recently enhanced to dynamically forecast regressors themselves, thereby improving forecast responsiveness.
  
- **Market Insights Agent:**  
  - Analyzes real‑time market sentiment, historical pricing trends, and factors such as weekend effects.
  - Identifies anomalies and computes correlations to provide context for forecast outputs.
  
- **Recommendation Agent:**  
  - Utilizes insights from both the forecasting and market analysis agents to generate optimal pricing strategies.
  - Outputs recommendations that include pricing ranges, adjustments, and competitive positioning.
  
- **Explanatory Agent:**  
  - Leverages LLM-powered natural language generation (via LangChain and ChatGroq) to produce clear, interpretable explanations of model outputs.
  - Provides actionable, transparent insights to support rapid business decision‑making.

The agents are orchestrated through a central `AgentOrchestrator` and integrated using LCEL (LangChain Expression Language) pipelines to ensure consistency and rapid responsiveness. The multi-agent system is exposed via a Streamlit dashboard, enabling interactive and dynamic exploration of insights.

## Features
- **Extended Forecasting:**  
  Predict mobile device prices for the next 6 months (180 days) using dynamic external regressors.
- **Dynamic Regressor Integration:**  
  Forecast macroeconomic indicators (GDP) and sentiment scores using Prophet, then integrate these predictions into price forecasting.
- **Anomaly Detection & Trend Analysis:**  
  Identify statistical anomalies and extract market trends from historical data.
- **Explainable AI:**  
  Generate human‑readable explanations and visualizations to enhance interpretability and facilitate decision‑making.
- **Multi-Agent Architecture:**  
  Modular design enabling independent scaling and integration of mathematical, market insights, and recommendation agents.
- **Interactive Dashboard:**  
  A Streamlit-based user interface for real‑time data visualization, segmentation of analysis (business insights, price trends, recommendations, and market analysis), and scenario simulation.

## Technologies & Dependencies
- **Programming Language:** Python  
- **Core Libraries:**  
  - [pandas](https://pandas.pydata.org/)  
  - [numpy](https://numpy.org/)  
  - [statsmodels](https://www.statsmodels.org/)  
  - [Prophet](https://facebook.github.io/prophet/)  
  - [python-dotenv](https://pypi.org/project/python-dotenv/)  
  - [streamlit](https://streamlit.io/)  
  - [LangChain](https://langchain.readthedocs.io/) (and related packages: langchain-openai, langchaincore)  
- **LLM Integration:** OpenAI GPT (via ChatGroq and LangChain)

_For a complete list of Python dependencies, refer to [requirements.txt](requirements.txt)._

## Data Sources
- **Primary Data:**  
  The core dataset (`data/final_merged_data.csv`) contains historical pricing data, device-specific sentiment scores, and macroeconomic indicators (e.g., GDP).
  
- **Metadata:**  
  Supplemental JSON files (e.g., `data/metadata_iphone_12.json`) provide additional context and characteristics for each device model.

## Installation

1. **Clone the Repository:**

```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Create and Activate a Virtual Environment:**
```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
```
pip install -r requirements.txt
```


4. **Configure Environment Variables:**
Create a `.env` file in the root directory and add any necessary API keys or configurations (e.g., OpenAI API key).

## Usage

- **Launching the Dashboard:**
Run the following command to start the interactive Streamlit dashboard:

```
streamlit run app.py
```

The dashboard offers tabs for business insights, price trends, recommendations, and market analysis. Navigate through the tabs to explore forecasts, interactive visualizations, and detailed explanations.


## Contributing
We welcome contributions to enhance this project. Please fork the repository, create a feature branch, and submit pull requests with detailed descriptions of changes. For major modifications, please open an issue first to discuss proposed changes.


## Acknowledgments
- Thank you to the GTDS - the Hacklytics 2025 team at Georgia Tech and hackathon sponsors for inspiring innovative solutions.
- Special thanks to the open-source community for contributing powerful libraries (e.g., Prophet, LangChain, Streamlit) that make this project possible.
