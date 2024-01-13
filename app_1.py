

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:20:34 2024

@author:Group 06
"""
import subprocess
subprocess.run(["pip", "install", "plotly"])
import pandas as pd
import streamlit as st 
#import plotly.graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_stockprice(df):
    
    df = technical_dimensions(df)
    df['Date'] = (df['Date'] - pd.to_datetime("1970-01-01")).dt.days.astype(float)
    
    df.index = range(len(df))

    test_size  = 0.15
    valid_size = 0.15
    
    test_split_idx  = int(df.shape[0] * (1-test_size))
    valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))
    
    train_df  = df.loc[:valid_split_idx].copy()
    valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
    test_df   = df.loc[test_split_idx+1:].copy()

    drop_cols = ['Volume', 'Open', 'Low', 'High', 'OpenInt']

    train_df = train_df.drop([col for col in drop_cols if col in train_df.columns], axis=1)
    valid_df = valid_df.drop([col for col in drop_cols if col in valid_df.columns], axis=1)
    test_df = test_df.drop([col for col in drop_cols if col in test_df.columns], axis=1)
    
    y_train = train_df['Close'].copy()
    X_train = train_df.drop(['Close'], 1)
    
    y_valid = valid_df['Close'].copy()
    X_valid = valid_df.drop(['Close'], 1)
    
    y_test  = test_df['Close'].copy()
    X_test  = test_df.drop(['Close'], 1)

    # Drop NaN values from the datasets
    X_train.dropna(inplace=True)
    y_train = y_train.loc[X_train.index]  # Ensure y_train aligns with X_train after dropping NaNs
    
    X_valid.dropna(inplace=True)
    y_valid = y_valid.loc[X_valid.index]  # Ensure y_valid aligns with X_valid after dropping NaNs
    
    X_test.dropna(inplace=True)
    y_test = y_test.loc[X_test.index]     # Ensure y_test aligns with X_test after dropping NaNs
    
    # Ensure dates are in Pandas datetime format
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    valid_df['Date'] = pd.to_datetime(valid_df['Date'])
    
    # Assuming X_test also has a Date column
    X_test['Date'] = pd.to_datetime(X_test['Date'])
    
    X_train = X_train.drop(columns=['Date'])
    X_valid = X_valid.drop(columns=['Date'])
    X_test = X_test.drop(columns=['Date'])
    
    model = LinearRegression()
    model.fit(X_train, y_train)

   # Validate the model
    y_valid_pred = model.predict(X_valid)
    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    # Predict the stock prices using the test data
    y_pred = model.predict(X_test)
    # Optionally, compare the predictions to the actual values
    test_mse = mean_squared_error(y_test, y_pred)
    
    # Replace 'original_df' with the DataFrame that contains your original dates
    # Ensure the dates align with your test data
    test_dates = df['Date'].tail(len(y_test))  # Adjust this to match your test dataset's dates
    
    # Creating a subplot figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Adding the real closing prices
    fig.add_trace(go.Scatter(x=test_dates, y=y_test, name='Real Closing Price', mode='lines'), secondary_y=False)
    
    # Adding the predicted closing prices
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred, name='Predicted Closing Price', mode='lines'), secondary_y=False)
    
    # Adding title and labels
    fig.update_layout(title_text="Real vs Predicted Closing Prices")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Closing Price", secondary_y=False)
    
    st.plotly_chart(fig)
    return y_pred[-1]
    
def technical_dimensions(df):
    
    ## Download historical data for Reliance Industries for the last 5 years
    df.reset_index(inplace=True)
    
    df.drop(columns=['Adj Close'], inplace=True)
    
    df_close = df[['Date', 'Close']].copy()
    df_close['Date'] = pd.to_datetime(df_close['Date'])
    df_close = df_close.set_index('Date')
    # Calculating 50 DMA
    df['50_DMA'] = df['Close'].rolling(window=50).mean()
    # Calculated 200 DMA
    df['200_DMA'] = df['Close'].rolling(window=200).mean()
    # Calculating MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())
    # Calculate the Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Create a subplot figure with 2 rows and 1 column
    fig = make_subplots(rows=2, cols=1)
    
    fig.update_layout(title_text="Reliance Industries Stock Analysis")
    
    
    # Adding traces to the subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['50_DMA'], name='50 DMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['200_DMA'], name='200 DMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'], name='Signal Line'), row=2, col=1)
    
    # Show plot
    st.plotly_chart(fig)
    
    return df
    
 # Function to get sentiment for the current date
def get_sentiment_score(df, current_date):
     current_date = pd.to_datetime(current_date)
     row = df[df['Date'] == current_date]
     if not row.empty:
         return row['compound_score'].values[0]
     else:
         return '0.00'

def main():
    st.title("Stock Price Prediction")
    html_temp = """
    <div style="background-color:red;padding:10px">
    <h2 style="color:white;text-align:center;">Group 06 Stock Model App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    result=""
    ticker = 'RELIANCE.NS'
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=5)
    
    df = yf.download(ticker, start=start_date, end=end_date)
    if st.button("Predict Reliance stock price"):
        result=predict_stockprice(df)
        st.success(result)
    if st.button("Fetch Sentiment for Today"):
        current_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        df_t = pd.read_csv("https://github.com/mmiswarya/Group06_FP1/raw/main/twitter_df.csv", sep=',', encoding='utf-8')
        sentiment_score = get_sentiment_score(df_t, current_date)  
        if sentiment_score is not None:
            sentiment_score = float(sentiment_score)
            if -0.5 < sentiment_score < 0.5:
                st.warning("Neutral Sentiment!")
            elif sentiment_score >= 0.5:
                st.success("Positive Sentiment!")
            else:
                st.error("Negative Sentiment!")
        #else:
            #st.warning(f"No sentiment data available for {current_date}")
        
    #if st.button("About"):
     #   st.text("Lets LEarn")
     #  st.text("Built with Streamlit")

if __name__=='__main__':
    main()
