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
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数\

def get_week_dates():
    # 得到大盘交易日数据
    sh000001_data = import_sh000001_data(r'.\data\sh000001.csv')
    sh000001_data['交易日期'] = pd.to_datetime(sh000001_data['交易日期'])
    sh000001_data['交易日期2'] = sh000001_data['交易日期'].dt.strftime('%Y-%m-%d')
    sh000001_data.set_index('交易日期', inplace=True)
    rule_type = '1W'
    sh000001_data_week = sh000001_data.resample(rule=rule_type).agg(
        {
            '交易日期2': 'last',
        }
    )
    dates = np.array(sh000001_data_week['交易日期2'])
    print(dates[-11:-2])
    return dates[-11:-2]

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

        code = df.iloc[0, 0]
        df = np.array(df)

        label = df[-1, -1]
        if label > 0:
            label = 1
        else:
            label = 0

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
            return code, down_prob, up_prob

    except Exception as e:
        return print("pass predict fix time {}".format(stock_path))

def predict_all_stock():
    # 得到大盘交易日数据
    dates = get_week_dates()

    all_csv = glob.glob(r'.\data\stock_preprocessed_with_nan\*.csv')

    all_predict = []
    for date in dates[-1:]:
        for csv in all_csv:
            print(date, csv)
            df = pd.read_csv(csv)
            try:
                code = df.iloc[0, 0]
                gain = df[df['交易日期'] == date]['五天后的收益率']
                code2, down_prob, up_prob = predict_fix_time(csv, date, alphanet)
                decalage = up_prob - down_prob
                all_predict.append([date, code, decalage, gain.values[0]])
            except Exception as e:
                print('pass predict all stock')


    all_predict = pd.DataFrame(all_predict)
    rename_dict = {0: '交易日期', 1: '股票代码', 2: '置信度', 3: '五天后的涨跌幅'}
    all_predict.rename(columns=rename_dict, inplace=True)
    all_predict.to_csv(r'.\result\predict.csv', index=False)


if __name__ == '__main__':
    ckpt_load_path = r'.\weights\16200.ckpt'
    alphanet = torch.load(ckpt_load_path)
    print('load', ckpt_load_path)
    alphanet = alphanet.cuda()
    alphanet.eval()
    predict_all_stock()
    # print(predict_fix_time(stock_path, '20200629', alphanet))
    # os.system('python ./train_AlphaNet.py')
