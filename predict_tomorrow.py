import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data.dataset import Dataset
import glob
import torch
import random
import pandas as pd
import numpy as np
from torch.autograd import Variable
from data_preprocessing import *
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

def predict_fix_time(stock_path, date, alphanet):
    # input code, date:yyyy-mm-dd
    df = pd.read_csv(stock_path)
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    try:
        df = df[df['交易日期'] <= pd.to_datetime(date)]
        df = df[df['交易日期'] >= pd.to_datetime(date) - pd.Timedelta(days=40)]

        df = df[['股票代码', '开盘价_复权', '最高价_复权', '最低价_复权', '收盘价_复权', '5天线', '涨跌幅',
                 '换手率', '大盘开盘价', '大盘最高价', '大盘最低价', '大盘收盘价', '大盘涨跌幅',
                 '五天后的收益率']]
        df = df[-20:]

        df['开盘价_复权'] = df['开盘价_复权'] / 100
        df['最高价_复权'] = df['最高价_复权'] / 100
        df['最低价_复权'] = df['最低价_复权'] / 100
        df['收盘价_复权'] = df['收盘价_复权'] / 100
        df['5天线'] = df['5天线'] / 100
        df['涨跌幅'] = df['涨跌幅'] * 10
        df['大盘涨跌幅'] = df['大盘涨跌幅'] * 10
        df['大盘开盘价'] = df['大盘开盘价'] / 5000
        df['大盘最高价'] = df['大盘最高价'] / 5000
        df['大盘最低价'] = df['大盘最低价'] / 5000
        df['大盘收盘价'] = df['大盘收盘价'] / 5000

        df = np.array(df)

        label = df[-1, -1]
        if label > 0:
            label = 1
        else:
            label = 0

        code = df[0, 0]

        df = df[:, 1:-1]

        df = np.array(df, dtype=np.float32)
        df[np.isinf(df)] = 0
        # print(df.shape)
        df = torch.FloatTensor(df.T)
        label = torch.LongTensor([label])
        # code = torch.LongTensor([code])

        df = Variable(df.cuda())
        df = df.unsqueeze(0)
        # print(df.shape)
        label = Variable(label.cuda())
        # code = Variable(code.cuda())
        with torch.no_grad():
            prob = alphanet(df, 0)
            down_prob = float(prob[0].data[0])
            up_prob = float(prob[0].data[1])
            return down_prob, up_prob, code

    except Exception as e:
        return RuntimeError("pass predict fix time".format(stock_path))

def predict_all_stock():
    # 得到大盘交易日数据
    # dates = get_week_dates()

    all_csv = glob.glob(r'.\data\stock_preprocessed_with_nan\*.csv')

    all_predict = []
    dates = ['2021/3/5']
    for date in dates:
        for csv in all_csv:
            print(date, csv)
            try:
                down_prob, up_prob, code = predict_fix_time(csv, date, alphanet)
                decalage = up_prob - down_prob
                all_predict.append([date, code, decalage])
            except Exception as e:
                print('pass predict all stock')


    all_predict = pd.DataFrame(all_predict)
    rename_dict = {0: '交易日期', 1: '股票代码', 2: '置信度'}
    all_predict.rename(columns=rename_dict, inplace=True)
    all_predict.to_csv(r'.\result\predict_tomorrow.csv', index=False)


def back():

    df = pd.read_csv(r'.\result\predict_tomorrow.csv')
    dates = ['2021/3/5']
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[df['交易日期'] >= pd.to_datetime('20201030')]
    select_num = 10

    for date in dates:
        try:
            df_week = df.groupby('交易日期').get_group(date)
            df_week['置信度排名'] = df_week['置信度'].rank(ascending=False, method='first')
            df_week = df_week[df_week['置信度排名'] <= select_num]
            print(df_week['置信度'])
            print(df_week['股票代码'])
        except Exception as e:
            print('pass')

    return

if __name__ == '__main__':
    ckpt_load_path = r'.\weights\16200.ckpt'
    alphanet = torch.load(ckpt_load_path)
    print('load', ckpt_load_path)
    alphanet = alphanet.cuda()
    alphanet.eval()
    predict_all_stock()
    back()
