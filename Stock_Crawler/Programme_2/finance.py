#!/usr/bin/python
# -*- coding: utf-8 -*-

from io import StringIO
import requests
import pandas as pd
import numpy as np
import pdb
from os import mkdir
from os.path import isdir
import argparse
from datetime import datetime

class finance():
    def __init__(self,year,season,prefix="finance"):
        if not isdir(prefix):
            mkdir(prefix)
        self.year = year
        self.season = season
        self.prefix = prefix
        
    def financial_statement(self,table):

        if self.year >= 1000:
            self.year -= 1911

        if table == '綜合損益彙總表':
            url = 'https://mops.twse.com.tw/mops/web/t163sb04'
        elif table == '資產負債彙總表':
            url = 'https://mops.twse.com.tw/mops/web/t163sb05'
        elif table == '營益分析彙總表':
            url = 'https://mops.twse.com.tw/mops/web/t163sb06'
        else:
            print('table does not match')

        r = requests.post(url, {
            'encodeURIComponent':1,
            'step':1,
            'firstin':1,
            'off':1,
            'TYPEK':'sii',
            'year':str(self.year),
            'season':str(self.season),
        })
        r.encoding = 'utf_8_sig'

        dfs = pd.read_html(StringIO(r.text), header=None, encoding='utf_8_sig')
        
        if table == '營益分析彙總表':
            res=pd.DataFrame(dfs[9])
            res=res.drop_duplicates(keep='first')  
            res.columns=res.loc[0]
            res=res.drop([0])
            res=res.set_index(['公司代號']).astype(str)
        else:
            res=pd.concat(dfs[10:16], axis=0, sort=False).set_index(['公司代號']).astype(str)

        return res



    def merge_statement(self):
        table_1=self.financial_statement('資產負債彙總表')
        table_2=self.financial_statement('綜合損益彙總表')
        table_3=self.financial_statement('營益分析彙總表')
        table_1=table_1.join(table_2, lsuffix='_left', rsuffix='_right')
        table_1=table_1.join(table_3, lsuffix='_left', rsuffix='_right')
        table_1.to_csv('{}/{}_{}.csv'.format(self.prefix, self.year, self.season), encoding='utf_8_sig' )

        
def get_data(data_source, year, season):
    if data_source==1:
        finance(year,season).merge_statement()
        print('success on year {}. season {}'.format(year+1911,season))
    else:
        print('TBD')
        
        
def main():

    # Get arguments
    parser = argparse.ArgumentParser(description='Crawl financial data at assigned day')
    parser.add_argument('year_season', type=int, nargs='*',
        help='assigned seasonal data (format: YYYY S), default is this season')
    parser.add_argument('-c', '--check_number', type=int,
            help='crawl back n days for check data')
    parser.add_argument('-a', '--all', action='store_true',
        help='crawl back from assigned day until 2015 2 season')
    parser.add_argument('-w', '--data_source', type=int, default=1,
        help='choose data scource, 公開資訊觀測站 - 資產負債彙總表,綜合損益彙總表,營益分析彙總表 =1, TBD=2')
    args = parser.parse_args()
    
    if len(args.year_season) == 2:
        season=int(args.year_season[1])
        year=args.year_season[0]-1911
    else: 
        parser.error('Date should be assigned with (YYYY S) or none')
        return
    
    # Day only accept 0 or 3 arguments
    if year == 104 and season == 1:
        print('season is wrong due to out of range')
        return
    elif season > 4 or season <= 0 or year > datetime.today().year or year  < 104 :
        print('season is wrong due to out of range')
        return
    else:
        first_day = [year, season]



    if args.check_number :
        # first data : year 104 season 2
        last_day = [104,2]
        max_error = 2
        error_times = 0
        c=0
        while error_times < max_error and first_day[0] >= last_day[0] and c < args.check_number:
            if first_day[0] == last_day[0] and first_day[1] > last_day[1]:
                break
            try:
                print('crawling')
                get_data(args.data_source, first_day[0], first_day[1])
                error_times = 0
                
            except:
                print('Crawl raise error year {}. season {}'.format(year+1911,season))
                error_times += 1
                continue
            finally:
                c=c+1
                if first_day[1]==1:
                    first_day[0]=first_day[0]-1
                    first_day[1]=4
                else:
                    first_day[1]=first_day[1]-1
                    
    elif args.all:
        # first data : year 104 season 2
        last_day = [104,2]
        max_error = 2
        error_times = 0
        while error_times < max_error and first_day[0] >= last_day[0] :
            print('crawling')
            if first_day[0] == last_day[0] and first_day[1] < last_day[1]:
                break
            try:
                get_data(args.data_source, first_day[0], first_day[1])
                error_times = 0
                
            except:
                print('Crawl raise error year {}. season {}'.format(first_day[0]+1911,first_day[1]))
                error_times += 1
                continue
            finally:
                if first_day[1]==1:
                    first_day[0]=first_day[0]-1
                    first_day[1]=4
                else:
                    first_day[1]=first_day[1]-1
                    
    else:
        today = datetime.today()
        season = today.month//3+1
        year=today.year-1911
        get_data(args.data_source, first_day[0], first_day[1])
    
    print('programme finished')
    

if __name__ == '__main__':
    main()
