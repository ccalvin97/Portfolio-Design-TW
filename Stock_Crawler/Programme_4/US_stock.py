
# coding: utf-8

# In[1]:


import numpy as np
import yfinance as yf
import requests
import pandas as pd
import time


start = '2010-01-03'
end='2020-11-14'
out_dir = '/Users/calvin/GitHub/portfolio-design/Stock_Crawler/Programme_4'


# In[2]:


# 最近一日交易量最大的前 100 檔熱門美股
url = 'https://finance.yahoo.com/screener/predefined/most_actives?count=100&offset=0'
data = pd.read_html(url)[0]
# 欄位『Symbol』就是股票代碼
stk_list_100 = data.Symbol


# In[3]:


# 貼上連結
url = 'https://www.slickcharts.com/sp500'
headers = {"User-Agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}
request = requests.get(url, headers = headers)
data = pd.read_html(request.text)[0]
# 欄位『Symbol』就是股票代碼
stk_list = data.Symbol
# 用 replace 將符號進行替換
stk_list_sp500 = data.Symbol.apply(lambda x: x.replace('.', '-'))


# In[4]:


# 貼上連結
url = 'https://www.slickcharts.com/nasdaq100'
headers = {"User-Agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}
request = requests.get(url, headers = headers)
data = pd.read_html(request.text)[0]
# 欄位『Symbol』就是股票代碼
stk_list = data.Symbol
# 用 replace 將符號進行替換
stk_list_nasdaq100 = data.Symbol.apply(lambda x: x.replace('.', '-'))


# In[5]:


stk_list=[]
stk_list.extend(stk_list_100)
stk_list.extend(stk_list_sp500)
stk_list.extend(stk_list_nasdaq100)


# In[6]:


stk_list = set(stk_list)


# In[7]:


for name in stk_list:
    df = yf.download(name, start=start, end=end, progress=False)
    df.to_csv(out_dir+'/'+name+'.csv')
    time.sleep(5)
    # times needs to be before 1 day of the start date & after 1 day of the end date

