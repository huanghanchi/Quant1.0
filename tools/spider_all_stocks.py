from urllib.request import urlopen
import pandas as pd
from datetime import datetime
import time
import re
import os
import json
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)


def get_content_from_internet(url, max_try_num=10, sleep_time=5):
    get_success = False
    for try_num in range(max_try_num):
        try:
            content = urlopen(url=url, timeout=10).read()
            get_success = True
            break
        except Exception as e:
            print('抓取数据报错，次数：', try_num+1, '报错内容：', e)
            time.sleep(sleep_time)

    # 判断是否成功抓取内容
    if get_success:
        return content
    else:
        raise ValueError('使用urlopen抓取网页数据不断报错，达到尝试上限，停止程序，请尽快检查问题所在')


def get_today_data_from_sinajs(code_list):

    url = r"http://hq.sinajs.cn/list=" + ",".join(code_list)
    content = get_content_from_internet(url)
    content = content.decode('gbk')

    content = content.strip()
    data_line = content.split('\n')
    data_line = [i.replace('var hq_str_', '').split(',') for i in data_line]
    df = pd.DataFrame(data_line, dtype='float')


    df[0] = df[0].str.split('="')
    df['stock_code'] = df[0].str[0].str.strip()
    df['stock_name'] = df[0].str[-1].str.strip()
    df['candle_end_time'] = df[30] + ' ' + df[31]
    df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])

    rename_dict = {1: 'open', 2: 'pre_close', 3: 'close', 4: 'high', 5: 'low', 6: 'buy1', 7: 'sell1',
                   8: 'amount', 9: 'volume', 32: 'status'}

    df.rename(columns=rename_dict, inplace=True)

    df['status'] = df['status']
    df = df[['stock_code', 'stock_name', 'candle_end_time', 'open', 'high', 'low', 'close', 'pre_close', 'amount',
             'volume', 'buy1', 'sell1', 'status']]

    return df


def is_today_trading_day():
    df = get_today_data_from_sinajs(code_list=['sh000001'])
    sh_date = df.iloc[0]['candle_end_time']  # 上证指数最近交易日

    return datetime.now().date() == sh_date.date()


def get_all_today_stock_data_from_sina_marketcenter():
    raw_url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=%s' \
              '&num=80&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=sort'
    page_num = 1

    all_df = pd.DataFrame()

    df = get_today_data_from_sinajs(code_list=['sh000001'])
    sh_date = df.iloc[0]['candle_end_time'].date()  # 上证指数最近交易日

    while True:
        # 构建url
        url = raw_url % (page_num)
        print('开始抓取页数：', page_num)

        # 抓取数据
        content = get_content_from_internet(url)
        content = content.decode('gbk')
        # print(content)
        # 判断页数是否为空
        if ('null' in content) or (content == '[]'):
            print('抓取到页数的尽头，退出循环')
            break

        content = re.sub(r'(?<={|,)([a-zA-Z][a-zA-Z0-9]*)(?=:)', r'"\1"', content)

        content = json.loads(content)

        df = pd.DataFrame(content, dtype='float')
        rename_dict = {'symbol': '股票代码', 'name': '股票名称', 'open': '开盘价', 'high': '最高价', 'low': '最低价',
                       'trade': '收盘价', 'settlement': '前收盘价', 'volume': '成交量', 'amount': '成交额',
                       'mktcap': '总市值', 'nmc': '流通市值'}
        df.rename(columns=rename_dict, inplace=True)
        df['交易日期'] = pd.to_datetime(sh_date)
        df['总市值'] = df['总市值'] * 10000
        df['流通市值'] = df['流通市值'] * 10000
        df = df[['股票代码', '股票名称', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '成交量', '成交额', '流通市值', '总市值']]

        all_df = all_df.append(df, ignore_index=True)

        page_num += 1
        # time.sleep(1)


    all_df = all_df[all_df['开盘价'] - 0 > 0.00001]
    all_df.reset_index(drop=True, inplace=True)

    return all_df


df = get_all_today_stock_data_from_sina_marketcenter()


for i in df.index:
    t = df.iloc[i:i+1, :]
    stock_code = t.iloc[0]['股票代码']

    path = r'C:\Users\SpiceeYJ\Desktop\量化投资\project_demo\data\stock_origin\\' + stock_code + '.csv'

    if os.path.exists(path):
        t.to_csv(path, header=None, index=False, mode='a', encoding='gbk')
    else:
        pd.DataFrame(columns=['数据由byh整理']).to_csv(path, index=False, encoding='gbk')
        t.to_csv(path, index=False, mode='a', encoding='gbk')
    print(stock_code)
