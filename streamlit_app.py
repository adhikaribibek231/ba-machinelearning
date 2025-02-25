import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from scraper import scrape_stock_data  # Import the scraping function

def get_available_stocks():
    stock_dir = "Stock/"
    return [f.split("_price_history.csv")[0] for f in os.listdir(stock_dir) if f.endswith("_price_history.csv")]

def ensure_stock_data(stock_symbol):
    file_path = f"Stock/{stock_symbol}_price_history.csv"
    if stock_symbol not in ["Select a stock...", "Search for a new stock..."] and not os.path.exists(file_path):
        st.warning(f"File {file_path} not found. Attempting to scrape data...")
        success = scrape_stock_data(stock_symbol)
        if not success or not os.path.exists(file_path):
            st.error("Failed to retrieve stock data. Please check the stock symbol.")
            return None
    return file_path

st.title("Stock Price Analysis & Prediction App")

available_stocks = get_available_stocks()
selected_stock = st.selectbox("Select a stock:", ["Select a stock..."] + available_stocks + ["Search for a new stock..."])

if selected_stock == "Search for a new stock...":
    stock_symbol = st.text_input("Enter new stock symbol:").upper()
else:
    stock_symbol = selected_stock

if stock_symbol and stock_symbol not in ["Select a stock...", "Search for a new stock..."]:
    file_path = ensure_stock_data(stock_symbol)
    if file_path:
        last_modified = os.path.getmtime(file_path)
        last_fetched_date = pd.to_datetime(last_modified, unit='s').tz_localize('UTC').tz_convert('Asia/Kathmandu').strftime('%Y-%m-%d %H:%M:%S')
        st.write(f"Last fetched: {last_fetched_date}")
        
        if st.button("Update to latest data"):
            st.write("Fetching latest data...")
            scrape_stock_data(stock_symbol)
            file_path = ensure_stock_data(stock_symbol)
            st.rerun()
        
        data = pd.read_csv(file_path, index_col="published_date", parse_dates=True)
        required_columns = {"open", "high", "low", "close", "traded_quantity"}
        
        if not required_columns.issubset(data.columns):
            st.error("CSV file is missing required columns.")
        else:
            data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "traded_quantity": "Volume"}, inplace=True)
            data = data.sort_index()
            predictors = ["Open", "High", "Low", "Close", "Volume"]
            
            train = data.iloc[:-3]
            test = data.iloc[-3:]
            
            models = {}
            predictions = {}
            for target in ["Open", "Close"]:
                model = RandomForestRegressor(n_estimators=200, random_state=1)
                model.fit(train[predictors], train[target])
                models[target] = model
                predictions[target] = model.predict(test[predictors])
            
            predictions_df = pd.DataFrame(predictions, index=test.index)
            
            # Main stock price graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Open"], mode='lines', name='Open Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
            fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
            st.plotly_chart(fig)
            
            # Biggest trend change
            st.subheader("Biggest Trend Change")
            data['Daily Change'] = data['Close'].diff()
            max_trend_change = data['Daily Change'].abs().idxmax()
            max_trend_value = data.loc[max_trend_change, 'Daily Change']
            st.write(f"Biggest Trend Change: {max_trend_change.date()} with a change of {max_trend_value:.2f}")
            
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            trend_fig.add_vline(x=max_trend_change, line=dict(color='red', width=2))
            trend_fig.update_layout(title="Biggest Trend Change Highlight", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(trend_fig)
            
            # Highest and lowest stock values
            highest_value_date = data['Close'].idxmax()
            highest_value = data.loc[highest_value_date, 'Close']
            lowest_value_date = data['Close'].idxmin()
            lowest_value = data.loc[lowest_value_date, 'Close']
            st.write(f"Highest Stock Value Recorded: {highest_value:.2f} on {highest_value_date.date()}")
            st.write(f"Lowest Stock Value Recorded: {lowest_value:.2f} on {lowest_value_date.date()}")
            
            # Volatility Chart
            st.subheader("Daily Volatility")
            data['Daily Volatility'] = (data['Close'] - data['Open']) / data['Open'] * 100
            volatility_fig = go.Figure()
            volatility_fig.add_trace(go.Bar(x=data.index, y=data['Daily Volatility'], name='Volatility %'))
            volatility_fig.update_layout(title="Daily Volatility Chart", xaxis_title="Date", yaxis_title="% Change")
            st.plotly_chart(volatility_fig)
            
            # Moving Averages
            st.subheader("Moving Averages")
            data['MA_7'] = data['Close'].rolling(window=7).mean()
            data['MA_30'] = data['Close'].rolling(window=30).mean()
            ma_fig = go.Figure()
            ma_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            ma_fig.add_trace(go.Scatter(x=data.index, y=data['MA_7'], mode='lines', name='7-Day MA'))
            ma_fig.add_trace(go.Scatter(x=data.index, y=data['MA_30'], mode='lines', name='30-Day MA'))
            ma_fig.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(ma_fig)
            
            # Volume Spike Detection
            st.subheader("Volume Spike Detection")
            volume_threshold = data['Volume'].quantile(0.95)
            spikes = data[data['Volume'] > volume_threshold]
            st.write(f"Detected {len(spikes)} volume spikes.")
            st.dataframe(spikes[['Volume']])
