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
    if not os.path.exists(file_path):
        st.warning(f"File {file_path} not found. Attempting to scrape data...")
        success = scrape_stock_data(stock_symbol)
        if not success or not os.path.exists(file_path):
            st.error("Failed to retrieve stock data. Please check the stock symbol.")
            return None
    return file_path

st.title("Stock Price Prediction App")

available_stocks = get_available_stocks()
selected_stock = st.selectbox("Select a stock:", available_stocks + ["Search for a new stock..."])

if selected_stock == "Search for a new stock...":
    stock_symbol = st.text_input("Enter new stock symbol:").upper()
else:
    stock_symbol = selected_stock

if stock_symbol:
    file_path = ensure_stock_data(stock_symbol)
    if file_path:
        last_modified = os.path.getmtime(file_path)
        last_fetched_date = pd.to_datetime(last_modified, unit='s').strftime('%Y-%m-%d %H:%M:%S')
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
            
            train = data.iloc[:-3]  # Use all data except the last 3 days for training
            test = data.iloc[-3:]   # Use the last 3 days for testing
            
            models = {}
            predictions = {}
            for target in ["Open", "Close"]:
                model = RandomForestRegressor(n_estimators=200, random_state=1)
                model.fit(train[predictors], train[target])
                models[target] = model
                predictions[target] = model.predict(test[predictors])
            
            predictions_df = pd.DataFrame(predictions, index=test.index)
            
            # Plotly interactive chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Open"], mode='lines', name='Open Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
            fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
            st.plotly_chart(fig)
            
            # Comparison DataFrame
            comparison = test[["Open", "Close"].copy()]
            comparison["Predicted_Open"] = predictions_df["Open"]
            comparison["Predicted_Close"] = predictions_df["Close"]
            comparison = comparison[["Open", "Predicted_Open", "Close", "Predicted_Close"]]
            
            st.subheader("Predicted vs Actual Prices for the Last 3 Days")
            st.dataframe(comparison)
            
            # Calculate accuracy
            accuracy_open = 100 - mean_absolute_percentage_error(test["Open"], predictions_df["Open"]) * 100
            accuracy_close = 100 - mean_absolute_percentage_error(test["Close"], predictions_df["Close"]) * 100
            overall_mape = mean_absolute_percentage_error(test[["Open", "Close"]], predictions_df[["Open", "Close"]])
            overall_accuracy = 100 - (overall_mape * 100)
            
            st.subheader("Accuracy Percentage")
            st.write(f"Open Price Prediction: {accuracy_open:.2f}%")
            st.write(f"Close Price Prediction: {accuracy_close:.2f}%")
            st.write(f"Overall Model Accuracy: {overall_accuracy:.2f}%")
            
            # Determine price trend
            st.subheader("Price Trend Predictions")
            previous_close = data.iloc[-4]["Close"] if len(data) > 3 else None
            trend_results = []
            
            for date, row in predictions_df.iterrows():
                predicted_close = row["Close"]
                actual_close = test.loc[date, "Close"]
                if previous_close is not None:
                    trend = "Increase" if predicted_close > previous_close else "Decrease"
                    correct_prediction = (trend == "Increase" and actual_close > previous_close) or (trend == "Decrease" and actual_close < previous_close)
                    correctness = "Correct" if correct_prediction else "Incorrect"
                    trend_results.append((date.date(), predicted_close, trend, actual_close, correctness))
                previous_close = actual_close
            
            trend_df = pd.DataFrame(trend_results, columns=["Date", "Predicted Close", "Trend", "Actual Close", "Prediction Accuracy"])
            st.dataframe(trend_df)
