import tushare as ts
import akshare as ak
import pandas as pd
import glob
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
pro = ts.pro_api('cbf6a83fa87c20245096ecc565faa7d6a3487dfb3f46ddd9333e7dfc')



data_L = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
print(data_L.shape)
#print(data_L)
data_D = pro.stock_basic(exchange='', list_status='D', fields='ts_code,symbol,name,area,industry,list_date,delist_date')
print(data_D.shape)
print(data_D)

all_csv = glob.glob(r'.\data\stock_origin\*.csv')
all_stock = []
for csv in all_csv:
    csv = (csv.split('\\')[-1]).split('.')[0][2:]
    if csv not in list(data_L['symbol']) and csv not in list(data_D['symbol']):
        print(csv)

df = pro.daily(ts_code='600074.SH', start_date='20210222', end_date='20210222')
print(df)

df = pro.daily(trade_date='20210222')
print(df)

df = pro.hk_hold(trade_date='20190625')
print(df)

df = pro.moneyflow(ts_code='600000.SH', start_date='20070101', end_date='20100101')
print(df)