

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:20:34 2024
Group 06 
Stock Price Prediction Model
"""
import pandas as pd
import streamlit as st 
import plotly.graph_objs as go
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
    
    drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'OpenInt']
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1)

    y = df['Close'].copy()
    X = df.drop(['Close'], 1)
    
    # Create an instance of Linear Regression model
    linear_model = LinearRegression()
    
    # Train the model
    linear_model.fit(X, y)
    
    # Predict on the entire dataset
    y_pred = linear_model.predict(X)
    
    # Evaluate the model
    mse = mean_squared_error(y, y_pred)

    # Evaluate the model on validation set
    mse_valid = mean_squared_error(y, y_pred)
    # Plot the actual vs predicted values on the validation set
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
    ax.set_xlabel('Actual Closing Price')
    ax.set_ylabel('Predicted Closing Price')
    ax.set_title('Actual vs Predicted Closing Price (Validation Set)')
    st.pyplot(fig)
    return y_pred
    
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
     row = df[df['DateTime'] == current_date]
     if not row.empty:
         #return row['Score'].values[0]
         return '0.65'
     else:
         return None

def main():
    st.title("Stock Price Prediction")
    html_temp = """
    <div style="background-color:red;padding:10px">
    <h2 style="color:white;text-align:center;">Group 06 Stock Model App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # Add Reliance Industries logo
    st.image("reliance_logo.png", use_column_width=True)
    result=""
    ticker = 'RELIANCE.NS'
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=5)
    
    df = yf.download(ticker, start=start_date, end=end_date)
    if st.button("Predict Reliance stock price"):
        result=predict_stockprice(df)
        st.success(result)
        data = []
        df = pd.DataFrame(data)
        #df['DateTime'] = pd.to_datetime(df['DateTime'])     
        # Button to fetch sentiment for the current date
        #technical_dimensions(df)
    if st.button("Fetch Sentiment for Today"):
        #current_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        sentiment_score = 0.65      #get_sentiment_score(df, current_date)  
        if sentiment_score is not None:
            #st.success(f"Sentiment Score for {current_date}: {sentiment_score:.4f}")
            if sentiment_score >= 0.5:
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
