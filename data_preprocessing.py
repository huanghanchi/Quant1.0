import pandas as pd
import glob
import tqdm
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
import os

def import_sh000001_data(path):
    df_index = pd.read_csv(path, parse_dates=['candle_end_time'])
    df_index['指数涨跌幅'] = df_index['close'].pct_change()
    # df_index = df_index[['candle_end_time', '指数涨跌幅']]
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)

    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)
    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index

def merge_stock_with_sh000001(df, index_data):
    df = pd.merge(left=df, right=index_data, on='交易日期', how='right', sort=True, indicator=True)

    # df['收盘价'].fillna(method='ffill', inplace=True)
    # df['开盘价'].fillna(value=df['收盘价'], inplace=True)
    # df['最高价'].fillna(value=df['收盘价'], inplace=True)
    # df['最低价'].fillna(value=df['收盘价'], inplace=True)
    # df['前收盘价'].fillna(value=df['收盘价'].shift(), inplace=True)

    # fill_0_list = ['成交量', '成交额', '涨跌幅']
    # df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    # df.fillna(method='ffill', inplace=True)
    # df = df[df['股票代码'].notnull()]

    #cdf['当天是否交易'] = 1
    # df.loc[df['_merge'] == 'right_only', '当天是否交易'] = 0
    # del df['_merge']

    # df.reset_index(drop=True, inplace=True)
    rename_dict = {'open': '大盘开盘价', 'high': '大盘最高价', 'low': '大盘最低价',
                   'close': '大盘收盘价', 'amount': '大盘成交量', '指数涨跌幅': '大盘涨跌幅'}
    df.rename(columns=rename_dict, inplace=True)
    df = df.drop(['info', '_merge'], axis=1)

    return df

def handle_by_stock(path, if_drop=True):
    df = pd.read_csv(path, encoding='gbk', skiprows=1)
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[df['交易日期'] > pd.to_datetime('20061215')]

    if df.shape[0] != 0:
        df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
        df['复权因子'] = (1 + df['涨跌幅']).cumprod()
        df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
        df['开盘价_复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_复权']
        df['最高价_复权'] = df['最高价'] / df['收盘价'] * df['收盘价_复权']
        df['最低价_复权'] = df['最低价'] / df['收盘价'] * df['收盘价_复权']
        df['5天线'] = df['收盘价_复权'].rolling(5, min_periods=1).mean()
        df['10天线'] = df['收盘价_复权'].rolling(10, min_periods=1).mean()
        df['20天线'] = df['收盘价_复权'].rolling(20, min_periods=1).mean()
        df['30天线'] = df['收盘价_复权'].rolling(30, min_periods=1).mean()
        df['60天线'] = df['收盘价_复权'].rolling(60, min_periods=1).mean()
        df['90天线'] = df['收盘价_复权'].rolling(90, min_periods=1).mean()
        df['120天线'] = df['收盘价_复权'].rolling(120, min_periods=1).mean()
        df['换手率'] = df['成交额'] / df['流通市值']
        df['五天后的收益率'] = df['收盘价_复权'].shift(-5) / df['收盘价_复权'] - 1
        df['五天后的收盘价'] = df['收盘价_复权'].shift(-5)

        sh000001_data = import_sh000001_data(r'.\data\sh000001.csv')
        df = merge_stock_with_sh000001(df, sh000001_data)
        # print(df)
        df = df[['股票代码', '交易日期', '开盘价_复权', '最高价_复权', '最低价_复权', '收盘价_复权', '涨跌幅',
                 '成交量', '成交额', '流通市值', '总市值', '换手率',
                 '5天线', '10天线', '20天线', '30天线', '60天线', '90天线', '120天线',
                 '大盘开盘价', '大盘最高价', '大盘最低价', '大盘收盘价', '大盘成交量', '大盘涨跌幅',
                 '五天后的收益率', '五天后的收盘价']]
        if if_drop:
            df = df.dropna(how='any')
            save_path = path.replace('stock_origin', 'stock_preprocessed')
        else:
            save_path = path.replace('stock_origin', 'stock_preprocessed_with_nan')
        df.reset_index(inplace=True, drop=True)
        df.to_csv(save_path, index=False)
    else:
        pass


if __name__ == '__main__':
    all_csv = glob.glob(r'.\data\stock_origin\*.csv')
    for csv in all_csv:
        print(csv)
        # handle_by_stock(csv, True)
        handle_by_stock(csv, False)
    # os.system('python ./train_AlphaNet.py')
    os.system('python ./predict_tomorrow.py')
