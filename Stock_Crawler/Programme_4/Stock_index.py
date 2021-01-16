
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import requests
import json
import pdb
import feature_selection as feature_selection
import time

start_time='2010/01/01'
out_dir = '/Users/calvin/GitHub/portfolio-design/Stock_Crawler/Programme_4'


# In[ ]:


def crawl_world_index():
    url = "https://finance.yahoo.com/world-indices/"
    response = requests.get(url)

    import io
    f = io.StringIO(response.text)
    dfs = pd.read_html(f)
    return dfs[0]

def crawl_price(stock_id):

    d = datetime.datetime.now()
    url = "https://query1.finance.yahoo.com/v8/finance/chart/"+stock_id+"?period1=0&period2="+str(int(d.timestamp()))+"&interval=1d&events=history&=hP2rOschxO0"

    res = requests.get(url)
    data = json.loads(res.text)
    df = pd.DataFrame(data['chart']['result'][0]['indicators']['quote'][0], index=pd.to_datetime(np.array(data['chart']['result'][0]['timestamp'])*1000*1000*1000))
    return df


world_index = crawl_world_index()


# In[2]:


world_index_history = {}
for symbol, name in zip(world_index['Symbol'], world_index['Name']):
    
    print(name)
    
    world_index_history[name] = crawl_price(symbol)
    time.sleep(5)


# In[4]:


for id, x in world_index_history.items():
    x.index = x.index.strftime("%Y/%m/%d")


# In[6]:


res=pd.DataFrame()
for id, x in world_index_history.items():
    right = x[x.index>=start_time]
    right.columns = [id+'_'+i for i in  right.columns]
    res = pd.concat([res, right], axis=1)


# In[8]:


res=res.dropna(how='all')


# In[9]:


res=res.fillna(method='ffill')


# In[10]:


res=res.fillna(method='bfill')


# In[11]:


res


# In[12]:


res.to_csv(out_dir+'/'+'stock_index.csv', index=True, header=True )

