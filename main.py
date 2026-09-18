import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LSTM
from sklearn.metrics import mean_absolute_error , mean_squared_error
import numpy as np

def lstm_stock(stock : str , seq_len : int , start : str = '2015-01-01' , end : str = '2023-01-01' , epochs : int = 50 , batch_size : int = 32 ):
  data = yf.download(stock, start = start , end = end)

  #Normalize Data
  scaler = MinMaxScaler(feature_range = (0 , 1))
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1 , 1))

  #Init Sequence len (how much past data we will look at)
  sequence_length = seq_len

  #X and y as empty data lists
  X = []
  y = []

  #1 - 60 is appened to X whilst 61st value is appended to y(since its the target value)
  for i in range(sequence_length, len(scaled_data)):
      X.append(scaled_data[i-sequence_length : i])
      y.append(scaled_data[i])

  #Convert to Numpy arr
  X , y = np.array(X) , np.array(y)

  #Train Test Split
  train_size = int(len(X) * 0.8)

  X_train , X_test = X[:train_size] , X[train_size:]
  y_train , y_test = y[:train_size] , y[train_size:]


  #Create Model
  model = Sequential()

  model.add(LSTM(units = 50 , return_sequences = True , input_shape = (X.shape[1] , 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units = 50 , return_sequences = False))
  model.add(Dropout(0.2))
  model.add(Dense(units = 1))

  model.compile(optimizer = 'adam' , loss = 'mean_squared_error')

  #Fitting to Model
  print(f"Started for\nStock : {stock} || Sequence Length : {seq_len}")
  model.fit(X_train , y_train , epochs = epochs , batch_size = batch_size , validation_data = (X_test , y_test) , verbose = 0)

  #Getting our prediction
  prediction = model.predict(X_test)

  #Finding derivative
  prediction = scaler.inverse_transform(prediction)
  actual_price = scaler.inverse_transform(y_test.reshape(-1 , 1))

  #Finding Loss
  ame = mean_absolute_error(actual_price , prediction)
  mse = mean_squared_error(actual_price , prediction)

  print(f"{stock} , {sequence_length}:\nAbsloute Mean Error : {ame}\nMean Squared Error : {mse}")

  #Plotting results
  plt.figure(figsize = (10 , 6))
  plt.plot(actual_price , color = 'red' , label = 'Actual Price')
  plt.plot(prediction , color = 'blue' , label = 'Model Prediction')

  plt.title(f'Stock Price Prediction LSTM || Stock : {stock} , Seq len : {seq_len}')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend()
  plt.show()



try:
  stocks = ['AMZN' , 'TSLA']
  seq_len = [30 , 60 , 90]
  for s in stocks:
    for seq in seq_len:
      lstm_stock(s , seq)
except KeyboardInterrupt:
  print("!!! User Inturepted stopping Program !!!")
