
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import copy

### valid data should be bigger than seq_len ###
train_set_size_percentage = 98 # percentage for the training over the whole original dataset
stock_name='EQIX' # Choose whick stock
seq_len = 20 # choose sequence length for a time slot
window = seq_len-1
feature_num_pre=1 # dimension for the prediction (label)
feature_num_nn_input=5  # dimension for NN input features
checkpoint_path = "parameter_v5.ckpt" # check point name


# 1. seq2seq
# 
# Failed due to the seq2seq cannot beused in this data structure
# 

# In[ ]:


url = 'https://raw.githubusercontent.com/ccalvin97/terminal_used/master/data/prices-split-adjusted.csv?token=AKAMPRWMXS7YQFM7GTYCG7K6VJNFU'
df = pd.read_csv(url)
# Dataset is now stored in a Pandas Dataframe


# In[198]:


# import all stock prices 
# df = pd.read_csv("/content/sample_data/prices-split-adjusted.csv", index_col = 0)
df.info()
df.head()

# number of different stocks
print('\nnumber of different stocks: ', len(list(set(df.symbol))))
print(list(set(df.symbol))[:10])


# In[199]:


df.tail()


# In[200]:


df.describe()


# In[201]:


df.info()


# In[202]:


df.iloc[:,2:].corr()
df.corr(method='pearson')[(df.corr() > 0.8) & (df.corr() != 1) | (df.corr() < -0.8) & (df.corr() != -1)]


# In[203]:


df.iloc[:,2:].corr()
df.corr(method='spearman')[(df.corr() > 0.8) & (df.corr() != 1) | (df.corr() < -0.8) & (df.corr() != -1)]


# In[204]:


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


# In[205]:


# choose one stock
df_stock = df[df.symbol == stock_name].copy()
df_stock.drop(['symbol'],1,inplace=True)
# df_stock.drop(['volume'],1,inplace=True)

df_stock.shape


# In[206]:


# split data in train/validation sets

valid_set_size = int(np.round(train_set_size_percentage/100*df_stock.shape[0]))
train_data, test_data = df_stock[:valid_set_size][:], df_stock[valid_set_size:][:]
print(test_data.shape)
train_data.shape


# In[207]:


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


# In[208]:


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


# In[209]:


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
      x_train = data[:train_set_size,:-6,1:]
      y_train = data[:train_set_size,-6:,1:]
    except:
      print("檢查問題: test data should be bigger than seq_len ### ")
    else:
      print('test data is ok for the seq_len setting')

    
    return [x_train, y_train]

# create train, test data
encoder_train, decoder_label_train = load_data(train_data, seq_len)
encoder_test, decoder_label_test = load_data(test_data, seq_len)



print('encoder_train.shape = ',encoder_train.shape)
print('decoder_label_train.shape = ', decoder_label_train.shape)
print('encoder_test.shape = ',encoder_test.shape)
print('decoder_label_test.shape = ', decoder_label_test.shape)


# (2175, 522, 1)
# (2175, 14, 1)
# (2175, 14, 1)


# In[210]:


decoder_label_train=np.expand_dims(decoder_label_train[:,:,0], axis=-1)
decoder_label_test=np.expand_dims(decoder_label_test[:,:,0], axis=-1)
decoder_label_train.shape


# In[211]:


decoder_input_train=copy.deepcopy(decoder_label_train)
decoder_input_test=copy.deepcopy(decoder_label_test)
decoder_input_train[:,-3:,:].shape


# In[212]:


decoder_input_train[:,-3:,:]=np.expand_dims(decoder_input_train[:,:3,0], axis=-1)
decoder_input_train[:,:3,:]=np.expand_dims(encoder_train[:,-3:,0],axis=-1)

decoder_input_test[:,-3:,:]=np.expand_dims(decoder_input_test[:,:3,0], axis=-1)
decoder_input_test[:,:3,:]=np.expand_dims(encoder_test[:,-3:,0],axis=-1)

decoder_input_train.shape
# (1673, 6, 1)


# In[ ]:


import tensorflow_probability as tfp
def tf_pearson(y_true, y_pred):
  return tfp.stats.correlation(y_true, y_pred)


# In[ ]:


from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import keras

latent_dim = 128 # LSTM hidden units
dropout = .20 

encoder_inputs = Input(shape=(None, 5)) 

encoder_outpus2 = LSTM(1,return_sequences=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, dropout=dropout, return_state=True)(encoder_outpus2)
encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None, 1)) 
decoder_lstm = LSTM(latent_dim, dropout=dropout, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm([decoder_inputs,state_h, state_c])
decoder_dense = Dense(1) # 1 continuous output at each timestep
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# In[231]:


adam = keras.optimizers.Adam(lr=0.1)
model.compile(loss='mse',optimizer=adam, metrics=[tf_pearson])

print(model.summary())


# In[232]:


# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
history=model.fit([encoder_train , decoder_input_train], decoder_label_train, batch_size=16,epochs=5,validation_split=0.1,verbose=1)


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


# In[219]:


def model_score(model, encoder_train, decoder_input_train, decoder_label_train, encoder_test, decoder_label_test, decoder_input_test):
    trainScore = model.evaluate([encoder_train , decoder_input_train], decoder_label_train, verbose=0)
    print('Train Score: %.5f MSE ' % (trainScore[0]))

    testScore = model.evaluate([encoder_test , decoder_input_test], decoder_label_test, verbose=0)
    print('Test Score: %.5f MSE ' % (testScore[0]))
    return trainScore, testScore

## GitHub 參考 result
# Train Score: 0.00019 MSE (0.01 RMSE)
# Test Score: 0.00033 MSE (0.02 RMSE)

model_score(model, encoder_train, decoder_input_train, decoder_label_train, encoder_test, decoder_label_test, decoder_input_test)


# In[ ]:


# from our previous model - mapping encoder sequence to state vectors
encoder_model = Model(encoder_inputs, encoder_states)

# A modified version of the decoding stage that takes in predicted target inputs
# and encoded state vectors, returning predicted target outputs and decoder state vectors.
# We need to hang onto these state vectors to run the next step of the inference loop.
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 1))
    
    # Populate the first target sequence with end of encoding series pageviews
    target_seq[0, 0, 0] = input_seq[0, -1, 0]

    # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
    # (to simplify, here we assume a batch of size 1).

    decoded_seq = np.zeros((1,pred_steps,1))
    
    for i in range(pred_steps):
        
        output, h, c = decoder_model.predict([target_seq] + states_value)
        
        decoded_seq[0,i,0] = output[0,0,0]

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = output[0,0,0]

        # Update states
        states_value = [h, c]

    return decoded_seq


# In[ ]:


def predict_and_plot(encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):

    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:] 
    print('encode_series.shape')
    print(encode_series.shape)
    pred_series = decode_sequence(encode_series)
    print('pred_series.shape')
    print(pred_series.shape)
    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)   
    target_series = decoder_target_data[sample_ind,:,:1].reshape(-1,1) 
    print('target_series.shape')
    print(target_series.shape) 
    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
    x_encode = encode_series_tail.shape[0]


    target_series= list(scaler1.inverse_transform(np.array(target_series[:,0].reshape(-1,1))))
    pred_series= list(scaler1.inverse_transform(np.array(pred_series[:,0].reshape(-1,1))))
    encode_series_tail= list(scaler1.inverse_transform(np.array(encode_series_tail[:,0].reshape(-1,1))))

    plt.figure(figsize=(10,6))   
    plt.plot(range(1,x_encode+1),encode_series_tail)
    plt.plot(range(x_encode,x_encode+pred_steps),target_series,color='orange')
    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')
    plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
    plt.legend(['Encoding Series','Target Series','Predictions'])


# In[ ]:


print(decoder_input_train.shape)
print(decoder_label_train.shape)


# In[ ]:


pred_steps = 6
predict_and_plot(encoder_test, decoder_label_test, 6)

