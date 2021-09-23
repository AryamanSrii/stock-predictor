import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model 
import streamlit as st
from datetime import date



start = '2010-01-01' 
end = date.today()

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

# spliting data into training and testing

data_training =pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# scaling down data in 0,1 scale
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)



model = load_model('keras_model.h5')



past_100_days= data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)
input_data =scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted =y_predicted*scale_factor
y_test =y_test*scale_factor



# final graph
st.subheader('Predictions vs Orignal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label ='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
