
# coding: utf-8

# In[10]:


# -*- coding: utf-8 -*-
import csv
import os
import re
import sys
import pandas as pd
from os import mkdir
from os.path import isdir
import requests
from bs4 import BeautifulSoup
import shutil


# ## 下載最新Stock List

# In[ ]:


def getList():
    url = "http://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    res = requests.get(url, verify = False)
    soup = BeautifulSoup(res.text, 'html.parser')

    table = soup.find("table", {"class" : "h4"})
    c = 0
    
    for row in table.find_all("tr"):
        data=[]
        i=0
        for col in row.find_all('td'):
            if i == 1 and c == 0:
                data.extend(' ')
                c=c+1
            if i == 0:
                reg=col.text.strip()
                pattern = re.compile('[^\u4e00-\u9fa50-9]')
                reg=pattern.split(reg,1)
                data.extend(reg)
                i=i+1
            else:
                data.append(col.text.strip().replace('\u3000', ''))    
        
        if len(data) == 1:
            continue # title 股票, 上市認購(售)權證, ...
        else:
            print(data)
        name='個股list'
        if not isdir(name):
                mkdir(name)
        f = open('{}/{}.csv'.format(name,'個股list'),'a',encoding='utf-8-sig')
        cw = csv.writer(f, lineterminator='\n')
        cw.writerow(data)
    f.close()
getList()


# ## 重新整理Stock List File  
# 
# update_stock_file - input: file list dir  
# 注意須將上面的 "個股list" file 放到 “base_dir” 下面

# In[12]:


class update_stock_file():
    def __init__(self,base_dir,prefix="update_stock_file"):
        ''' Make directory if not exist when initialize '''
        if not isdir(base_dir+'/'+prefix):
            mkdir(base_dir+'/'+prefix)
        self.prefix = prefix
        self.base_dir=base_dir    
        
    def current_stock_file(self):
        current_stock_file=[]
        for (root,dirs,files) in os.walk(self.base_dir + '/'+'data', topdown=True): 
            current_stock_file.extend(files) 
        return current_stock_file
    
    def new_stock_file(self):
        data=pd.read_csv(self.base_dir + '/個股list/個股list.csv',encoding='utf-8-sig') 
        current_stock_file=data['有價證券代號及名稱']
        current_stock_file=current_stock_file.apply(lambda x: str(x)+'.csv')
        return list(current_stock_file)
    
    def mapping(self):
        '''
        new: up to date file list
        old: old file list
        '''
        old=self.current_stock_file()
        new=self.new_stock_file()
        
        res=[]
        for i in new:
            if i in old:
    #             res.append(i)
                shutil.copy(self.base_dir+'/'+'data'+'/'+i, self.base_dir+'/'+self.prefix)


# In[13]:


update_stock_file=update_stock_file('/Users/calvin/GitHub')
update_stock_file.mapping()

