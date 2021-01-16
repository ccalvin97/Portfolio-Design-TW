
# coding: utf-8

# In[ ]:


# pip install keras==2.0.9


# In[ ]:


# pip install tensorflow==1.4.0


# In[ ]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt

### valid data should be bigger than seq_len ###
train_set_size_percentage = 98 # percentage for the training over the whole original dataset
stock_name='EQIX' # Choose whick stock
seq_len = 20 # choose sequence length for a time slot
window = seq_len-1
feature_num_pre=1 # dimension for the prediction (label)
feature_num_nn_input=5  # dimension for NN input features
checkpoint_path = "parameter_v5.ckpt" # check point name


# 1. One dimension output
# 
# 
# Problem:  
# create a checkpoint
# LR or other parameters need to be tuned  
# 預測後滯性  
# feature selection method
# 
# 心得:   
# 其他loss 不穩, pearson 是一個穩定指標  
# 

# In[ ]:


url = 'https://raw.githubusercontent.com/ccalvin97/terminal_used/master/data/prices-split-adjusted.csv?token=AKAMPRQTFLSRIZFK7RJ6XXK6TWBQK'
df = pd.read_csv(url)
# Dataset is now stored in a Pandas Dataframe


# In[ ]:


# import all stock prices 
# df = pd.read_csv("/content/sample_data/prices-split-adjusted.csv", index_col = 0)
df.info()
df.head()

# number of different stocks
print('\nnumber of different stocks: ', len(list(set(df.symbol))))
print(list(set(df.symbol))[:10])


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.iloc[:,2:].corr()
df.corr(method='pearson')[(df.corr() > 0.8) & (df.corr() != 1) | (df.corr() < -0.8) & (df.corr() != -1)]


# In[ ]:


df.iloc[:,2:].corr()
df.corr(method='spearman')[(df.corr() > 0.8) & (df.corr() != 1) | (df.corr() < -0.8) & (df.corr() != -1)]


# In[ ]:


plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(df[df.symbol == stock_name].open.values, color='red', label='open')
plt.plot(df[df.symbol == stock_name].close.values, color='green', label='close')
plt.plot(df[df.symbol == stock_name].low.values, color='blue', label='low')
plt.plot(df[df.symbol == stock_name].high.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
#plt.show()

plt.subplot(1,2,2);
plt.plot(df[df.symbol == stock_name].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');


# In[ ]:


# choose one stock
df_stock = df[df.symbol == stock_name].copy()
df_stock.drop(['symbol'],1,inplace=True)
# df_stock.drop(['volume'],1,inplace=True)

df_stock.shape


# In[ ]:


# split data in train/validation sets

valid_set_size = int(np.round(train_set_size_percentage/100*df_stock.shape[0]))
train_data, test_data = df_stock[:valid_set_size][:], df_stock[valid_set_size:][:]
print(test_data.shape)
train_data.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler(feature_range=(0, 1))
train_data['open'] = scaler1.fit_transform(np.array(train_data['open']).reshape(-1,1))
test_data['open']= scaler1.transform(np.array(test_data['open']).reshape(-1,1))

scaler2 = MinMaxScaler(feature_range=(0, 1))
train_data['close'] = scaler2.fit_transform(np.array(train_data['close']).reshape(-1,1))
test_data['close']=scaler2.transform(np.array(test_data['close']).reshape(-1,1))

scaler3 = MinMaxScaler(feature_range=(0, 1))
train_data['low'] = scaler3.fit_transform(np.array(train_data['low']).reshape(-1,1))
test_data['low']=scaler3.transform(np.array(test_data['low']).reshape(-1,1))

scaler4 = MinMaxScaler(feature_range=(0, 1))
train_data['high'] = scaler4.fit_transform(np.array(train_data['high']).reshape(-1,1))
test_data['high']=scaler4.transform(np.array(test_data['high']).reshape(-1,1))

scaler5 = MinMaxScaler(feature_range=(0, 1))
train_data['volume'] = scaler5.fit_transform(np.array(train_data['volume']).reshape(-1,1))
test_data['volume']=scaler5.transform(np.array(test_data['volume']).reshape(-1,1))

# print(train_data)
print(test_data.shape)


# In[ ]:


plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(train_data.open.values, color='red', label='open')
plt.plot(train_data.close.values, color='green', label='low')
plt.plot(train_data.low.values, color='blue', label='low')
plt.plot(train_data.high.values, color='black', label='high')
plt.plot(train_data.volume.values, color='gray', label='volume')
plt.title('train_data')
plt.xlabel('time [days]')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')

plt.subplot(1,2,2);
plt.plot(test_data.open.values, color='red', label='open')
plt.plot(test_data.close.values, color='green', label='low')
plt.plot(test_data.low.values, color='blue', label='low')
plt.plot(test_data.high.values, color='black', label='high')
plt.plot(test_data.volume.values, color='gray', label='volume')
plt.title('test_data')
plt.xlabel('time [days]')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')
# plt.show()


# In[ ]:


# function to create train, validation, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.iloc[:,:].values
    # print(data_raw)
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    # 輸入data_raw 維度: (1762, 5), 第一維度為資料筆數, 第二維度為五個feature
    # 輸出data 維度: (1742, 20, 5), (20, 5)為我們一次的考慮長度, 分別是20個時間點和5個features,  1742為NN考慮的時間段
    # 看筆記

    data = np.array(data); 
    # 切分時間段個數->分到training, validation, testing set
    valid_set_size = int(np.ceil(100-train_set_size_percentage)/100*data.shape[0])
    train_set_size = data.shape[0] - (valid_set_size);
    
    # 檢查 test data 長度
    try: 
      x_train = data[:train_set_size,:-1,1:]
      y_train = data[:train_set_size,-1,1:]
    except:
      print("檢查問題: test data should be bigger than seq_len ### ")
    else:
      print('test data is ok for the seq_len setting')

    
    return [x_train, y_train]

# create train, test data
x_train, y_train = load_data(train_data, seq_len)
x_test, y_test = load_data(test_data, seq_len)



print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ', y_test.shape)


# In[ ]:


y_train=y_train[:,:feature_num_pre]
y_test=y_test[:,:feature_num_pre]


# In[ ]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from tensorflow.contrib.metrics import streaming_pearson_correlation
import keras
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from keras import backend as K
from keras.layers.normalization import BatchNormalization

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]


def build_model(layers):
    d = 0.5
    model = Sequential()
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    # model.add(BatchNormalization())
    model.add(Dropout(d))
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    # model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    # model.add(Dropout(d))
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))    
    model.add(Dense(feature_num_pre,kernel_initializer="uniform",activation='linear'))
    start = time.time()
    adam = keras.optimizers.Adam(lr=0.0005)
    model.compile(loss='mse',optimizer=adam, metrics=[tf_pearson])
    print("Compilation Time : ", time.time() - start)
    K.get_session().run(tf.local_variables_initializer())
    return model


# In[ ]:


# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

model = build_model([feature_num_nn_input,window])
print(model.summary())


# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
history=model.fit(x_train,y_train,batch_size=32,epochs=300,validation_split=0.1,verbose=1, callbacks=[callback])
# history=model.fit(x_train,y_train,batch_size=32,epochs=400,validation_split=0.1,verbose=1, callbacks=[cp_callback])


# In[ ]:


# print(history.history.keys())


# In[ ]:


plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['tf_pearson'])
plt.plot(history.history['val_tf_pearson'])
plt.title('model Pearson')
plt.ylabel('Pearson')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[ ]:


df_stock1 = df[df.symbol == stock_name].copy()
df_stock1.drop(['symbol'],1,inplace=True)
df_stock1


# In[ ]:


def model_score(model, x_train, y_train, x_test, y_test):
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.5f MSE ' % (trainScore[0]))

    testScore = model.evaluate(x_test, y_test, verbose=0)
    print('Test Score: %.5f MSE ' % (testScore[0]))
    return trainScore, testScore

## GitHub 參考 result
# Train Score: 0.00019 MSE (0.01 RMSE)
# Test Score: 0.00033 MSE (0.02 RMSE)

model_score(model, x_train, y_train, x_test, y_test)


# In[ ]:


# df_stock1 = df[df.symbol == stock_name].copy()
# df_stock1.drop(['symbol'],1,inplace=True)
pred = model.predict(x_test)
pred[:,0]= list(scaler1.inverse_transform(np.array(pred[:,0]).reshape(-1,1)))
# pred[:,1]= list(scaler2.inverse_transform(np.array(pred[:,1]).reshape(-1,1)))
# pred[:,2]= list(scaler3.inverse_transform(np.array(pred[:,2]).reshape(-1,1)))
# pred[:,3]= list(scaler4.inverse_transform(np.array(pred[:,3]).reshape(-1,1)))
# pred[:,4]= list(scaler5.inverse_transform(np.array(pred[:,4]).reshape(-1,1)))
pred.shape


# In[ ]:


y_test = y_test.reshape(y_test.shape[0] , y_test.shape[-1])
y_test[:,0]= list(scaler1.inverse_transform(np.array(y_test[:,0]).reshape(-1,1)))
# y_test[:,1]= list(scaler2.inverse_transform(np.array(y_test[:,1]).reshape(-1,1)))
# y_test[:,2]= list(scaler3.inverse_transform(np.array(y_test[:,2]).reshape(-1,1)))
# y_test[:,3]= list(scaler4.inverse_transform(np.array(y_test[:,3]).reshape(-1,1)))
# y_test[:,4]= list(scaler5.inverse_transform(np.array(y_test[:,4]).reshape(-1,1)))
y_test.shape


# In[ ]:


y_test[:,0].shape


# In[ ]:


# baseline_y=[]

# # initialisation using y_valid at time 0 in valid
# for i in range(feature_num_pre):
#   baseline_y.append(list(y_test[i,i]))

# for i in range(len(y_test[:,1])-1): #140
#   for j in range(feature_num_pre): #4
#     baseline_y[j].extend(y_test[i+1,j])
# baseline_y=np.array(baseline_y).T

# np.array(baseline_y).shape


# In[ ]:


print('prediction marker is \'o\'')
plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
plt.plot(np.mean(y_test[:,:feature_num_pre],axis=1) , 'b', label='label')
plt.plot(np.mean(pred[:,:feature_num_pre],axis=1) , 'r',label='prediction')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend(loc='best')
plt.title('prediction and label - average 4 features')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(y_test[:,0], color='red', label='open_valid')
plt.plot(pred[:,0], color='red', label='close_pred',marker='o')
# plt.plot(y_test[:,1], color='blue', label='close_valid')
# plt.plot(pred[:,1], color='blue', label='low_pred',marker='o')
# plt.plot(y_test[:,2], color='green', label='low_valid')
# plt.plot(pred[:,2], color='green', label='low_pred',marker='o')
# plt.plot(y_test[:,3], color='gray', label='high_valid')
# plt.plot(pred[:,3], color='gray', label='high_pred',marker='o')
# plt.title('prediction and label - 4 features')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend(loc='best')
plt.grid(True)
plt.show()

