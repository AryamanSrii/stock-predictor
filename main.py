import streamlit as st
import pandas as pd
import numpy as np 
from keras.models import load_model
import pandas_datareader as data 
import matplotlib.pyplot as plt 

start = '2010-01-01' #10 year date starting from
end = '2020-12-31'   #10 year date end

st.title('AmrevX Stock Predictor') #title
user_input = st.text_input("Enter Ticker", 'AAPL') #place holder text
df = data.DataReader(user_input, 'yahoo', start, end) #read the user input and search yahoo finance

st.subheader('Data from 2010 - 2020') #last 10 year data
st.write(df.describe())

st.subheader('Closing Price vs Time chat') #heading of graph
fig = plt.figure(figsize = (12,6)) #size of graph
plt.plot(df.Close) 
st.pyplot(fig) #final plot


st.subheader('21 Days Exponential Moving Average ')
ma21 = df.Close.rolling(21).mean()
fig = plt.figure(figsize = (12,6)) #size of graph
plt.plot(ma21)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('50 Days Exponential Moving Average ')
ma50 = df.Close.rolling(50).mean()
fig = plt.figure(figsize = (12,6)) #size of graph
plt.plot(ma50)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('100 Days Exponential Moving Average ')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6)) #size of graph
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('200 Days Exponential Moving Average ')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6)) #size of graph
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

