import asyncio
import streamlit as st
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
import yfinance as yf

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,
    model_settings=ModelSettings(temperature=0),
    system_prompt="""
    You are an advanced AI stock rating agent designed to analyze financial reports, 
    historical price data, and key technical indicators to evaluate stocks. 
    Your goal is to assign a rating to each stock based on a scale from Strong Buy (A) 
    to Strong Sell (E) and provide a clear explanation for your rating. 
    
    Consider the following factors:
    - Revenue growth 
    - Profitability 
    - Price history trends
    - Technical indicators 
       
    After the analysis, assign one of the following ratings and provide a detailed explanation for the rating:
    A - Strong Buy: The stock is undervalued with strong growth potential, solid financials, and positive market momentum.
    B - Buy: The stock has good fundamentals and technical indicators but may have some risks or uncertainties.
    C - Hold: The stock is fairly valued with mixed signals from fundamental and technical analysis. Holding is advised unless major catalysts emerge.
    D - Sell: The stock shows weak fundamentals, declining trends, or negative market sentiment, suggesting downside risk.
    E - Strong Sell: The stock has serious financial or structural problems, significant downside risk, or bearish trends indicating potential losses.
    """
)

@agent.tool
def fetch_stock_info(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    return {key: stock.info[key] for key in ['longName', 'marketCap', 'sector']}

@agent.tool
def fetch_quarterly_financials(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    return stock.quarterly_financials.T[['Total Revenue', 'Net Income']].to_csv()

@agent.tool
def fetch_annual_financials(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    return stock.financials.T[['Total Revenue', 'Net Income']].to_csv()

@agent.tool
def fetch_weekly_price_history(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    return stock.history(period='1y', interval='1wk').to_csv()

@agent.tool
def calculate_rsi_weekly(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    data = stock.history(period='1y', interval='1wk')
    delta = data['Close'].diff()

    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

symbol = st.selectbox('Please select a stock symbol', ['AAPL', 'TSLA', 'OXY'])
result = agent.run_sync("Analyze this stock", deps=symbol)
st.markdown(result.data)



