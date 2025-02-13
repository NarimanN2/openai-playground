import streamlit as st
import numpy as np
from typing import TypedDict, List

from pycoingecko import CoinGeckoAPI
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

class CoinInfo(TypedDict):
    description: str
    market_cap: int
    market_cap_rank: int
    twitter_followers: int
    recent_commits_count: int

class TechnicalIndicators(TypedDict):
    ma_20: List[float]
    ma_50: List[float]
    ma_200: List[float]

def fetch_coin_info(coin_id):
    cg = CoinGeckoAPI()
    coin_info = cg.get_coin_by_id(coin_id)

    return CoinInfo(
        description=coin_info['description']['en'],
        market_cap=coin_info['market_data']['market_cap']['usd'],
        market_cap_rank=coin_info['market_cap_rank'],
        twitter_followers=coin_info['community_data']['twitter_followers'],
        recent_commits_count=coin_info['developer_data']['commit_count_4_weeks']
    )


def fetch_price_history(coin_id):
    cg = CoinGeckoAPI()
    return cg.get_coin_market_chart_by_id(coin_id, vs_currency='usd', days=365)

def calculate_moving_averages(coin_id):
    cg = CoinGeckoAPI()
    price_history = cg.get_coin_market_chart_by_id(coin_id, vs_currency='usd', days=365)
    prices = [price[1] for price in price_history['prices']]

    return TechnicalIndicators(
        ma_20=np.convolve(prices, np.ones(20)/20, mode='valid'),
        ma_50=np.convolve(prices, np.ones(50)/50, mode='valid'),
        ma_200=np.convolve(prices, np.ones(200)/200, mode='valid'),
    )

llm = OpenAI(model="gpt-4o", temperature=0)

coin_info_tool = FunctionTool.from_defaults(fn=fetch_coin_info)
price_history_tool = FunctionTool.from_defaults(fn=fetch_price_history)
moving_averages_tool = FunctionTool.from_defaults(fn=calculate_moving_averages)

agent = OpenAIAgent.from_tools(
    [coin_info_tool, price_history_tool, moving_averages_tool],
    llm=llm,
    system_prompt="""
    You are an advanced AI crypto analysis agent designed to analyze project fundamentals, historical price data, and key technical indicators to evaluate cryptocurrencies. Your goal is to assign a rating to each cryptocurrency based on a scale from Strong Buy (A) to Strong Sell (E) and provide a clear explanation for your rating.
    Consider the following factors:
    
    - Usability and real-world application
    - Adoption and recognition within the crypto community
    - Developer activity on the project
    - Historical price trends and technical indicators
    
    After the analysis, assign one of the following ratings and provide a detailed explanation for the rating:
    
    A - Strong Buy: The cryptocurrency has strong real-world applications, solid community support, strong market momentum, and bullish technical indicators.
    B - Buy: The project has good fundamentals, growing adoption, and positive technical signals, though it may have some risks or uncertainties.
    C - Hold: The cryptocurrency is fairly valued with mixed signals from fundamental and technical analysis. Holding is advised unless major developments arise.
    D - Sell: The project shows weak adoption, declining trends, or negative market sentiment, suggesting downside risk.
    E - Strong Sell: The cryptocurrency has serious usability issues, weak community support, significant downside risk, or bearish trends indicating potential losses.
    """,
    verbose=True
)

coin = st.selectbox(
    "Select a cryptocurrency",
    ["bitcoin", "ethereum", "cardano", "solana", "dogecoin"]
)

if coin:
    st.markdown(
        agent.chat(f'Should I invest in {coin}?')
    )




