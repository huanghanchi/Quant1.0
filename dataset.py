import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data.dataset import Dataset
import glob
import torch
import random
import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数


class SEDataset(Dataset):
    """ Speech enhancement dataset """
    def __init__(self, data_path, rolling=20, test_num=75, if_train=False):
        super(SEDataset, self).__init__()
        self.data_path = data_path
        self.if_train = if_train
        self.rolling = rolling
        self.test_num = test_num
        self.all_stocks = glob.glob(self.data_path + r'*.csv')

        self.stock_dict = {}

        with open(self.data_path + r'stock_name.txt', "w") as f:
            index = 0
            for stock in self.all_stocks:
                stock_name = (stock.split('\\')[-1]).split('.')[0]
                self.stock_dict[stock_name] = index
                to_write = stock_name + ' ' + str(index) + '\n'
                f.write(to_write)
                index += 1
        f.close()

    def __getitem__(self, index):
        if len(self.all_stocks) == 0:
            self.all_stocks = glob.glob(self.data_path + r'*.csv')
        choice = random.choice(self.all_stocks)
        self.all_stocks.remove(choice)
        df = pd.read_csv(choice)
        df = df[['股票代码', '开盘价_复权', '最高价_复权', '最低价_复权', '收盘价_复权', '5天线', '涨跌幅',
                 '换手率', '大盘开盘价', '大盘最高价', '大盘最低价', '大盘收盘价', '大盘涨跌幅',
                 '五天后的收益率']]

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


        df = df.dropna(how='any')
        if self.if_train is True:
            if df.shape[0] > self.rolling + self.test_num:
                begin = random.randint(0, df.shape[0] - self.rolling - self.test_num)
                df = df[begin:begin + self.rolling]
            elif df.shape[0] < self.rolling:
                return self.__getitem__(index)
            else:
                begin = random.randint(0, df.shape[0] - self.rolling)
                df = df[begin:begin + self.rolling]
        else:
            if df.shape[0] > self.rolling + self.test_num:
                end = random.randint(df.shape[0] - self.test_num, df.shape[0])
                df = df[end - self.rolling:end]
            else:
                return self.__getitem__(index)

        assert df.shape[0] == self.rolling
        df = np.array(df)

        df[:, 0] = self.stock_dict[df[0, 0]]
        label = df[-1, -1]
        if label > 0:
            label = 1
        else:
            label = 0

        code = df[0, 0]
        df = df[:, 1:-1]

        df = np.array(df, dtype=np.float32)
        df[np.isinf(df)] = 0
        if np.any(np.isnan(df)):
            raise
        return torch.FloatTensor(df.T), torch.LongTensor([label]),  torch.LongTensor([code])

    def __len__(self):
        return len(glob.glob(self.data_path + r'*.csv'))


if __name__ == '__main__':
    a = SEDataset(r'C:\Users\SpiceeYJ\Desktop\量化投资\project_demo\data\stock_preprocessed\\', if_train=False)
    for i in range(1):
        aa, ss, dd = a['a']
        print(aa)
        print(ss)
        print(dd)
        print(dd.shape)

