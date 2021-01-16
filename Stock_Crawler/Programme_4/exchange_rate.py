
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import pdb
import time 
import re
import numpy as np

start = '2020-01'
out_dir = '/Users/calvin/GitHub/test/a3c+dqn_env/exchange_rate'
url = "https://rate.bot.com.tw/xrt?Lang=zh-TW"



# This programme can only support the current year and previous 1 year data
# If you need historical data, please go this website - 
# https://www.bot.com.tw/Services/DownLoad/Pages/default.aspx

# 現金匯率: 是指你手上持有現鈔時，跟銀行交易時使用的匯率。 由於銀行持有現鈔會有一定的成本，所以現金匯率的價格一般會比即期匯率來的差。
# 
# 即期匯率:是指你的外幣存款帳戶要轉存到新台幣存款， 或是你收到的外幣匯款要轉存成新台幣存款， 或是新台幣存款要轉存到你的外幣存款時，所使用的匯率。
# 
# 銀行買入價:是指銀行用新台幣跟你買外幣的價格。
# 
# 銀行賣出價:是指銀行賣給你外幣的價格。

# In[2]:


def get_history_rate_link(url):
    resp = requests.get(url)
    resp.encoding = 'utf-8'
    html = BeautifulSoup(resp.text, "lxml")
    rate_table = html.find(name='table', attrs={'title':'牌告匯率'}).find(name='tbody').find_all(name='tr')

    history_rate_link_list = []


    for rate in rate_table:
        # 擷取匯率表格，把美金(也就是匯率表的第一個元素)擷取出來，查詢其歷史匯率
        currency = rate.find(name='div', attrs={'class':'visible-phone print_hide'})
        #print(currency.get_text().replace(" ", ""))  # 貨幣種類
        #print(currency)

        # 針對美金，找到其「歷史匯率」的首頁
        history_link = rate.find(name='td', attrs={'data-table':'歷史匯率'})
        #print(history_link)
        history_rate_link = "https://rate.bot.com.tw" + history_link.a["href"]  # 該貨幣的歷史資料首頁
        #print(history_rate_link)
        history_rate_link_list.append(history_rate_link)

    return history_rate_link_list

#%%

#
# 到貨幣歷史匯率網頁，選則該貨幣的「歷史區間」，送出查詢後，觀察其網址變化情形，再試著抓取其歷史匯率資料
#
# 用「quote/年-月」去取代網址內容，就可以連到該貨幣的歷史資料

def get_historical_plt(history_rate_link,timeframe):
    headers={
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36 Edg/83.0.478.56',

    }

    quote_history_url = history_rate_link.replace("history", "quote/"+timeframe)
    resp = requests.get(quote_history_url,headers = headers)
    resp.encoding = 'utf-8'
    history = BeautifulSoup(resp.text, "lxml")
    history_table = history.find(name='table', attrs={'title':'歷史本行營業時間牌告匯率'}).find(name='tbody').find_all(name='tr')

    currency_name = history_rate_link.split('/')[-1]

    date_history = []
    history_buyin = []
    history_sellout = []

    
    string = "~!@#$%^&*()_+-*/<>,.[]\/"
    
    for history_rate in history_table:
        # 擷取日期資料
#         try:
        date_string = history_rate.a.get_text()
        date = datetime.datetime.strptime(date_string, '%Y/%M/%d').strftime('%Y%M%d')  # 轉換日期格式
        date_history.append(date)  # 日期歷史資料

        history_ex_rate = history_rate.find_all('td', attrs={'class':'rate-content-sight text-right print_table-cell'})

        if history_ex_rate[0].get_text() in string:
            history_buyin.append(np.nan)
        else:
            history_buyin.append(float(history_ex_rate[0].get_text()))  # 歷史買入匯率
            
        if history_ex_rate[1].get_text() in string:
            history_sellout.append(np.nan)
        else:            
            history_sellout.append(float(history_ex_rate[1].get_text()))  # 歷史賣出匯率


    # 將匯率價格資料存成dataframe形式
    History_Ex_Rate = pd.DataFrame({'日期': date_history,
                                        '即期買入收盤匯率'+ currency_name:history_buyin,
                                        '即期賣出收盤匯率'+ currency_name:history_sellout})

    History_Ex_Rate = History_Ex_Rate.set_index('日期')  # 指定'日期'欄位為datafram的index
    History_Ex_Rate = History_Ex_Rate.sort_index(ascending=True)
    
    return History_Ex_Rate


# In[3]:


date=pd.date_range(start='1/1/2020', periods=12, freq='M')
date=pd.Series(date.format())
date = date.apply(lambda x: x.split('-')[0] + '-' +x.split('-')[1])


# In[4]:


url_list = get_history_rate_link(url)
for ul in url_list:
    res = pd.DataFrame()
    currency_name = ul.split('/')[-1]
    for i in date:
        df=get_historical_plt(ul,i)
        res = pd.concat([res, df])
    res = res[::-1]
    res.to_csv(out_dir + '/' + currency_name+ '.csv',encoding = 'utf_8_sig' )
    print('Success - {}'.format(currency_name))

