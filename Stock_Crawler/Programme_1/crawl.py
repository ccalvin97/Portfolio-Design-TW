#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import re
import sys
import csv
import time
import string
import logging
import requests
import argparse
from datetime import datetime, timedelta
from os import mkdir
from os.path import isdir
import pandas as pd
from io import StringIO
import numpy as np
import os

class Crawler():
    def __init__(self,data_source,prefix="data"):
        ''' Make directory if not exist when initialize '''
        if not isdir(prefix):
            mkdir(prefix)
        self.prefix = prefix
        self.data_source=data_source    
    
    def _header(self):
        header=['Time','Trading Volume','Trading Price','Start Price',
                'Max Price','Min Price','End Price','Gross Spread','Trading Count']
        for dirpath,dirnames,filenames in os.walk(self.prefix):
            for i in filenames:
                f = open('{}/{}'.format(self.prefix, i),'r+')
                old = f.read()
                f.seek(0,0)
                cw = csv.writer(f, lineterminator='\n')
                cw.writerow(header)
                f.write(old)
                f.close()  
            break
        

    def _clean_row(self, row):
        ''' Clean comma and spaces '''
        for index, content in enumerate(row):
            row[index] = re.sub(",", "", content.strip())
        return row

    def _record(self, stock_id, row):
        ''' Save row to csv file '''
        f = open('{}/{}.csv'.format(self.prefix, stock_id), 'a')
        cw = csv.writer(f, lineterminator='\n')
        cw.writerow(row)
        f.close()
        
##### 台灣證交所 ##### 
    def _get_tse_data(self, date_tuple):  
        date_str = '{0}{1:02d}{2:02d}'.format(date_tuple[0], date_tuple[1], date_tuple[2])
        # EX: date_str 20200708 
        r = requests.post('http://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date=' + str(date_str)+ '&type=ALLBUT0999')
        print('request success on {}'.format(date_str))

        if not r.text:
            print('Error: no colum to write')
        
        ret = pd.read_csv(StringIO("\n".join([i.translate({ord(c): None for c in ' '}) 
                                                for i in r.text.split('\n') 
                                                if len(i.split('",')) == 17 and i[0] != '='])))
        ret=ret.fillna('null')
        ret = ret.set_index('證券代號')
        ret['成交金額'] = ret['成交金額'].str.replace(',','')
        ret['成交股數'] = ret['成交股數'].str.replace(',','')
        res=ret
        date_str_mingguo = '{0}/{1:02d}/{2:02d}'.format(date_tuple[0]-1911, date_tuple[1], date_tuple[2])
        for row,data in res.iterrows():
            row=([
                date_str_mingguo,#日期
                data[1], # 成交股數
                data[3], # 成交金額
                data[4], # 開盤價
                data[5], # 最高價
                data[6], # 最低價
                data[7], # 收盤價
                str(data[8]) + str(data[9]), # 漲跌價差
                data[2], # 成交筆數
            ])
            self._record(data.name.strip(), row)
        time.sleep(10)
        
        
###### 台北證券買賣中心資料#########
    def _get_otc_data(self, date_tuple):
        date_str = '{0}/{1:02d}/{2:02d}'.format(date_tuple[0]-1911, date_tuple[1], date_tuple[2])
        ttime = str(int(time.time()*100))
        url = 'http://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d={}&_={}'.format(date_str, ttime)
        page = requests.get(url)
            
        if not page.ok:
            logging.error("Can not get OTC data at {}".format(date_str))
            return

        result = page.json()
        if not result['mmData'] and not result['aaData']:
            print('Error: no colum to write')
            
        if result['reportDate'] != date_str:
            logging.error("Get error date OTC data at {}".format(date_str))
            return
        
        for table in [result['mmData'], result['aaData']]:
            for tr in table:
                row = self._clean_row([
                    date_str,
                    tr[8], # 成交股數
                    tr[9], # 成交金額
                    tr[4], # 開盤價
                    tr[5], # 最高價
                    tr[6], # 最低價
                    tr[2], # 收盤價
                    tr[3], # 漲跌價差
                    tr[10] # 成交筆數
                ])
                self._record(tr[0], row)
        time.sleep(10)

    def get_data(self, date_tuple):
        print('Crawling {}'.format(date_tuple))
        if self.data_source==1:
            self._get_tse_data(date_tuple)
        else:
            self._get_otc_data(date_tuple)

def main():
    # Set logging
    if not os.path.isdir('log'):
        os.makedirs('log')
    logging.basicConfig(filename='log/crawl-error.log',
        level=logging.ERROR,
        format='%(asctime)s\t[%(levelname)s]\t%(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    # Get arguments
    parser = argparse.ArgumentParser(description='Crawl data at assigned day')
    parser.add_argument('day', type=int, nargs='*',
        help='assigned day (format: YYYY MM DD), default is today')
    parser.add_argument('-b', '--back', action='store_true',
        help='crawl back from assigned day until 2004/2/11')
    parser.add_argument('-c', '--check_cumber', type=int, default=1,
                help='crawl back n days for check data')
    parser.add_argument('-w', '--data_source', type=int, default=1,
                help='choose data scource, 台灣證券交易所=1, 證券櫃檯買賣中心=2')
    args = parser.parse_args()


    # Day only accept 0 or 3 arguments
    if len(args.day) == 0:
        first_day = datetime.today()
    elif len(args.day) == 3:
        first_day = datetime(args.day[0], args.day[1], args.day[2])
    else:
        parser.error('Date should be assigned with (YYYY MM DD) or none')
        return

    crawler = Crawler(args.data_source)
    

    # If back flag is on, crawl till 2004/2/11, else crawl one day
    if args.back or args.check_cumber:
        # otc first day is 2007/04/20
        # tse first day is 2004/02/11

        last_day = datetime(2004, 2, 11) if args.back else first_day - timedelta(args.check_cumber-1)
        max_error = 15
        error_times = 0

        while error_times < max_error and first_day >= last_day:
            try:
                crawler.get_data((first_day.year, first_day.month, first_day.day))
                error_times = 0
            except:
                date_str = first_day.strftime('%Y/%m/%d')
                print('Crawl raise error {}'.format(date_str))
                logging.error('Crawl raise error {}'.format(date_str))
                error_times += 1
                continue
            finally:
                first_day -= timedelta(1)
    else:
        crawler.get_data((first_day.year, first_day.month, first_day.day))
    
    crawler._header()

if __name__ == '__main__':
    main()
