from urllib.request import urlopen
import json
from random import randint
import pandas as pd

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

def _random(n=16):
    start = 10**(n-1)
    end = (10**n)-1
    return str(randint(start, end))


stock_code = r'sh000001'
k_type = 'day'
num = 30000


url = r'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_%sqfq&param=%s,%s,,,%s,qfq&r=0.%s'
url = url % (k_type, stock_code, k_type, num, _random())
content = urlopen(url).read().decode()


content = content.split('=', maxsplit=1)[-1]
content = json.loads(content)  # 自己去仔细看下这里面有什么数据


k_data = content['data'][stock_code]
if k_type in k_data:
    k_data = k_data[k_type]
elif 'qfq' + k_type in k_data:  # qfq是前复权的缩写
    k_data = k_data['qfq' + k_type]
else:
    raise ValueError('已知的key在dict中均不存在，请检查数据')
df = pd.DataFrame(k_data)


rename_dict = {0: 'candle_end_time', 1: 'open', 2: 'close', 3: 'high', 4: 'low', 5: 'amount', 6: 'info'}
df.rename(columns=rename_dict, inplace=True)
df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])
if 'info' not in df:
    df['info'] = None
df = df[['candle_end_time', 'open', 'high', 'low', 'close', 'amount', 'info']]


print(df)
df.to_csv(r'..\data\sh000001.csv', index=False)

