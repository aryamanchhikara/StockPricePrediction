
import numpy as np
import time
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


st.title("STOCK PRICE PREDICTION SYSTEM")
with st.sidebar:
    st.write("Trending Stocks")

    with st.spinner("Loading..."):
        time.sleep(2)
        st.success("META")
        st.success("TSLA")
        st.success("APPL")
        st.success("BRK-B")


# Load data from a CSV file
def load_data(file_path):#file_path):
   
    data = yf.download(file_path,start='2012-01-1',end='2016-12-31')
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaler, scaled_data

def prepare_training_data(scaled_data, prediction_days):
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def plot(actual_prices,predicted_prices,file_path):
    fig, ax = plt.subplots()
    ax.plot(actual_prices, color="black", label=f"Actual {file_path} price")
    ax.plot(predicted_prices, color="green", label=f"Predicted {file_path} price")
    ax.set_title(f"{file_path} Share Price ")
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{file_path} Share Price ')
    ax.legend()

    # Display the plot using streamlit
    st.pyplot(fig)


    
def prediction(model_inputs,prediction_days):
    real_data = [model_inputs[len(model_inputs)+1 - prediction_days:len(model_inputs+1),0]]

    real_data = np.array(real_data)
    real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    return prediction
    #st.text_area(label="Price of the next day at OPEN:", value=prediction, height=250)
    #print(f"Prediction: {prediction}")
    #st.write(prediction_values)




file_path = st.text_input("Enter the company name")

if st.button("Run Prediction"):
    with st.status("Gathering data from API"):
        time.sleep(5)
        st.write("compiling data")
        
    data = load_data(file_path) 
    scaler, scaled_data = preprocess_data(data)
    
    prediction_days=60
    x_train, y_train = prepare_training_data(scaled_data, prediction_days)
    model = Sequential()
    

    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    with st.status("Running epoc cycles"):    
        st.write("epoch started")
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.fit(x_train,y_train,epochs=25,batch_size=32)

        


    

    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now()

    test_data= yf.download(file_path,start=test_start,end=test_end)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)

    model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)


    # Make predictions on Test Data

    x_test=[]
    for x in range (prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x,0])

    x_test = np.array(x_test)
    x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    predicted_prices = model.predict(x_test)
    predicted_prices=scaler.inverse_transform(predicted_prices)

    # Plot the test predictions
    with st.status("Plotting Graph...."):
        time.sleep(2)
    plot(actual_prices,predicted_prices,file_path)
    prediction_price = prediction(model_inputs,prediction_days)

    '''
    prediction_output= st.text_area(
        f"Company Name -{file_path} ",
        f"Share Price [before close] - {actual_prices[:0]} ",
        f"Share Price [Open the next day ]- {prediction_price}"
    )
    '''
    st.text_area(label="Comopany Name :", value=file_path, height=150)
    st.text_area(label="Predicted price at OPEN and other details :", value=prediction_price, height=150)
    st.text_area(label="Comopany price at close :", value=actual_prices, height=150)

    


