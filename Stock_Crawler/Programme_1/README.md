# Stock Crawler - Day based

## Programme 1:  
crawl.py - Stock Price Crawler, date based stock crawler  
post_process.py - post process for the crawler, cleaning duplicated data  


## Setup

```
$ cd tsec

$ pip install -r requirements.txt
```

### Command

爬當日

```
$ python crawl.py
```

爬指定日期

```
$ python crawl.py YYYY MM DD

e.g.

$ python crawl.py 2016 02 15
```

### Flag

```
-b, --back`: 往回爬直到 `2004/2/11

-c, --check_number`: 回爬多少天  

-w --check_cumber: choose dara scource, 台灣證券交易所=1, 證券櫃檯買賣中心=2  

ex: $ python crawl.py 2016 03 01 -c 4 -w 2 -> 從2016 03 01回爬4天從 證券櫃檯買賣中心
```


## 資料格式

- 每個檔案的檔名 `XXX.csv`，`XXX` 是股票編號
- 每個檔案中有數列，每列為一天交易的資訊
- 每列包含：交易日期、成交股數、成交金額、開盤價、最高價、最低價、收盤價、漲跌價差、成交筆數，共 9 欄。
- 符號說明: +表示漲、- 表示跌、X表示不比價
- 當日統計資訊含一般、零股、盤後定價、鉅額交易，不含拍賣、標購。

範例：`104/02/13,7599922.0,528270219.0,69.35,69.65,69.35,69.45,0.45,1771.0`
