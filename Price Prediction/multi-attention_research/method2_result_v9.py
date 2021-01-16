
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


# 1. Multihead_attention  
# 
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


url = 'https://raw.githubusercontent.com/ccalvin97/terminal_used/master/data/prices-split-adjusted.csv?token=AKAMPRT2OZ7ZSWA2WQ6BLVS6UAXIY'
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
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.contrib.metrics import streaming_pearson_correlation
import keras
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.engine.topology import Layer


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.qs_layer = Dense(n_head*d_k, use_bias=False)
        self.ks_layer = Dense(n_head*d_k, use_bias=False)
        self.vs_layer = Dense(n_head*d_v, use_bias=False)
        self.attention = ScaledDotProductAttention(d_model)
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
        ks = self.ks_layer(k)
        vs = self.vs_layer(v)

        def reshape1(x):
            s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], n_head, d_k])
            x = tf.transpose(x, [2, 0, 1, 3])  
            x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
            return x
        qs = Lambda(reshape1)(qs)
        ks = Lambda(reshape1)(ks)
        vs = Lambda(reshape1)(vs)
        head, attn = self.attention(qs, ks, vs, mask=mask)  
            
        def reshape2(x):
            s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
            x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
            return x

        head = Lambda(reshape2)(head)
        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs, attn # return outputs, attn


# In[ ]:


def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]


# In[ ]:


### START Model HERE ###
d = 0.7
input_x =  Input(shape=([window,feature_num_nn_input]), dtype='float32') ## Dim: [batch, timestep, features]
X = LSTM(128, return_sequences=True)(input_x)
X = Dropout(d)(X)
# X = LSTM(32, return_sequences=False)(X)
# X = Dropout(d)(X)
X, slf_attn = MultiHeadAttention(n_head=3, d_model=292, d_k=128, d_v=128, dropout=d)(X, X, X)
avg_pool = GlobalAveragePooling1D()(X)
max_pool = GlobalMaxPooling1D()(X)
X = concatenate([avg_pool, max_pool])
X = Dense(16,kernel_initializer="uniform",activation='relu')(X)
X = Dense(feature_num_pre,kernel_initializer="uniform",activation='linear')(X)
model = Model(inputs=[input_x], outputs=X)

adam = keras.optimizers.Adam(lr=0.00005)
model.compile(loss='mse',optimizer=adam, metrics=[tf_pearson])
start = time.time()
print("Compilation Time : ", time.time() - start)
K.get_session().run(tf.local_variables_initializer())


# In[ ]:



# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

print(model.summary())


# In[ ]:



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
history=model.fit(x_train,y_train,batch_size=32,epochs=2,validation_split=0.1,verbose=1, callbacks=[callback])

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


# In[ ]:





# embed_size = 60

# class LayerNormalization(Layer):
#     def __init__(self, eps=1e-6, **kwargs):
#         self.eps = eps
#         super(LayerNormalization, self).__init__(**kwargs)
#     def build(self, input_shape):
#         self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
#                                      initializer=Ones(), trainable=True)
#         self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
#                                     initializer=Zeros(), trainable=True)
#         super(LayerNormalization, self).build(input_shape)
#     def call(self, x):
#         mean = K.mean(x, axis=-1, keepdims=True)
#         std = K.std(x, axis=-1, keepdims=True)
#         return self.gamma * (x - mean) / (std + self.eps) + self.beta
#     def compute_output_shape(self, input_shape):
#         return input_shape

# class ScaledDotProductAttention():
#     def __init__(self, d_model, attn_dropout=0.1):
#         self.temper = np.sqrt(d_model)
#         self.dropout = Dropout(attn_dropout)
#     def __call__(self, q, k, v, mask):
#         attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
#         if mask is not None:
#             mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
#             attn = Add()([attn, mmask])
#         attn = Activation('softmax')(attn)
#         attn = self.dropout(attn)
#         output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
#         return output, attn

# class MultiHeadAttention():
#     # mode 0 - big martixes, faster; mode 1 - more clear implementation
#     def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
#         self.mode = mode
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#         self.dropout = dropout
#         if mode == 0:
#             self.qs_layer = Dense(n_head*d_k, use_bias=False)
#             self.ks_layer = Dense(n_head*d_k, use_bias=False)
#             self.vs_layer = Dense(n_head*d_v, use_bias=False)
#         elif mode == 1:
#             self.qs_layers = []
#             self.ks_layers = []
#             self.vs_layers = []
#             for _ in range(n_head):
#                 self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
#                 self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
#                 self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
#         self.attention = ScaledDotProductAttention(d_model)
#         self.layer_norm = LayerNormalization() if use_norm else None
#         self.w_o = TimeDistributed(Dense(d_model))

#     def __call__(self, q, k, v, mask=None):
#         d_k, d_v = self.d_k, self.d_v
#         n_head = self.n_head

#         if self.mode == 0:
#             qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
#             ks = self.ks_layer(k)
#             vs = self.vs_layer(v)

#             def reshape1(x):
#                 s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
#                 x = tf.reshape(x, [s[0], s[1], n_head, d_k])
#                 x = tf.transpose(x, [2, 0, 1, 3])  
#                 x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
#                 return x
#             qs = Lambda(reshape1)(qs)
#             ks = Lambda(reshape1)(ks)
#             vs = Lambda(reshape1)(vs)

#             if mask is not None:
#                 mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
#             head, attn = self.attention(qs, ks, vs, mask=mask)  
                
#             def reshape2(x):
#                 s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
#                 x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
#                 x = tf.transpose(x, [1, 2, 0, 3])
#                 x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
#                 return x
#             head = Lambda(reshape2)(head)
#         elif self.mode == 1:
#             heads = []; attns = []
#             for i in range(n_head):
#                 qs = self.qs_layers[i](q)   
#                 ks = self.ks_layers[i](k) 
#                 vs = self.vs_layers[i](v) 
#                 head, attn = self.attention(qs, ks, vs, mask)
#                 heads.append(head); attns.append(attn)
#             head = Concatenate()(heads) if n_head > 1 else heads[0]
#             attn = Concatenate()(attns) if n_head > 1 else attns[0]

#         outputs = self.w_o(head)
#         outputs = Dropout(self.dropout)(outputs)
#         if not self.layer_norm: return outputs, attn
#         # outputs = Add()([outputs, q]) # sl: fix
#         return self.layer_norm(outputs), attn

# class PositionwiseFeedForward():
#     def __init__(self, d_hid, d_inner_hid, dropout=0.1):
#         self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
#         self.w_2 = Conv1D(d_hid, 1)
#         self.layer_norm = LayerNormalization()
#         self.dropout = Dropout(dropout)
#     def __call__(self, x):
#         output = self.w_1(x) 
#         output = self.w_2(output)
#         output = self.dropout(output)
#         output = Add()([output, x])
#         return self.layer_norm(output)

# class EncoderLayer():
#     def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
#         self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
#         self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
#     def __call__(self, enc_input, mask=None):
#         output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
#         output = self.pos_ffn_layer(output)
#         return output, slf_attn


# def GetPosEncodingMatrix(max_len, d_emb):
#     pos_enc = np.array([
#         [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
#         if pos != 0 else np.zeros(d_emb) 
#             for pos in range(max_len)
#             ])
#     pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
#     pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
#     return pos_enc

# def GetPadMask(q, k):
#     ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
#     mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
#     mask = K.batch_dot(ones, mask, axes=[2,1])
#     return mask

# def GetSubMask(s):
#     len_s = tf.shape(s)[1]
#     bs = tf.shape(s)[:1]
#     mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
#     return mask

# class Transformer():
#     def __init__(self, len_limit, embedding_matrix, d_model=embed_size, \
#               d_inner_hid=512, n_head=10, d_k=64, d_v=64, layers=2, dropout=0.1, \
#               share_word_emb=False, **kwargs):
#         self.name = 'Transformer'
#         self.len_limit = len_limit
#         self.src_loc_info = False # True # sl: fix later
#         self.d_model = d_model
#         self.decode_model = None
#         d_emb = d_model

#         pos_emb = Embedding(len_limit, d_emb, trainable=False, \
#                             weights=[GetPosEncodingMatrix(len_limit, d_emb)])

#         i_word_emb = Embedding(max_features, d_emb, weights=[embedding_matrix]) # Add Kaggle provided embedding here

#         self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
#                                word_emb=i_word_emb, pos_emb=pos_emb)

        
#     def get_pos_seq(self, x):
#         mask = K.cast(K.not_equal(x, 0), 'int32')
#         pos = K.cumsum(K.ones_like(x, 'int32'), 1)
#         return pos * mask

#     def compile(self, active_layers=999):
#         src_seq_input = Input(shape=(None, ))
#         x = Embedding(max_features, embed_size, weights=[embedding_matrix])(src_seq_input)
        
#         # LSTM before attention layers
#         x = Bidirectional(LSTM(128, return_sequences=True))(x)
#         x = Bidirectional(LSTM(64, return_sequences=True))(x) 
        
#         x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
        
#         avg_pool = GlobalAveragePooling1D()(x)
#         max_pool = GlobalMaxPooling1D()(x)
#         conc = concatenate([avg_pool, max_pool])
#         conc = Dense(64, activation="relu")(conc)
#         x = Dense(2, activation="softmax")(conc)   
        
        
#         self.model = Model(inputs=src_seq_input, outputs=x)
#         self.model.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics=['accuracy'])

