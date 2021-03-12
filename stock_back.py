from stock_predict import *
from data_preprocessing import *
import pandas as pd
import glob
import matplotlib.pyplot as plt

def calculate_acc():
    all_predict = glob.glob(r'.\result\predict.csv')
    total_num = 0
    total_correct_num = 0
    for predict in all_predict:
        df = pd.read_csv(predict)
        num = df.shape[0]
        total_num += num
        df = df[((df['置信度'] < 0) & (df['五天后的涨跌幅'] < 0)) | ((df['置信度'] > 0) & (df['五天后的涨跌幅'] > 0))]
        correct_num = df.shape[0]
        total_correct_num += correct_num
        print('{}/{}'.format(correct_num, num))
    print(total_correct_num/total_num)

def calculate_ic():
    # 得到大盘交易日数据
    dates = get_week_dates()
    df = pd.read_csv(r'.\result\predict.csv')

    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[df['交易日期'] >= pd.to_datetime('20200130')]

    corr = 0
    count = 0
    for date in dates[:]:
        try:
            df_week = df.groupby('交易日期').get_group(date)
            corr += df_week['置信度'].corr(df_week['五天后的涨跌幅'])
            count += 1
        except Exception as e:
            pass
    return print(corr/count)

def calculate_rank_ic():
    dates = get_week_dates()
    df = pd.read_csv(r'.\result\predict.csv')

    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[df['交易日期'] >= pd.to_datetime('20200130')]

    corr = 0
    count = 0
    for date in dates[:]:
        try:
            df_week = df.groupby('交易日期').get_group(date)
            df_week['置信度排名'] = df_week['置信度'].rank(ascending=True, method='first')
            df_week['五天后的涨跌幅排名'] = df_week['五天后的涨跌幅'].rank(ascending=True, method='first')

            corr += df_week['置信度排名'].corr(df_week['五天后的涨跌幅排名'])
            # print(df_week['置信度排名'].corr(df_week['五天后的涨跌幅排名']))
            count += 1
        except Exception as e:
             print('pass')
    return print(corr/count)

def back():
    dates = get_week_dates()
    df = pd.read_csv(r'.\result\predict.csv')
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[df['交易日期'] >= pd.to_datetime('20200130')]

    rank_num = 0
    select_num = 3
    gain_ratio = []
    gain_ratio.append(1)
    gain_ratio_ = 1
    for date in dates[:]:
        try:
            df_week = df.groupby('交易日期').get_group(date)
            df_week['置信度排名'] = df_week['置信度'].rank(ascending=False, method='first')
            df_week = df_week[(df_week['置信度排名'] < select_num + rank_num) & (df_week['置信度排名'] >= rank_num)]
            print(df_week['交易日期'])
            print(df_week['股票代码'])
            print(df_week['五天后的涨跌幅'])
            gain_ratio_ *= 1 + df_week['五天后的涨跌幅'].mean()
            gain_ratio.append(gain_ratio_)
        except Exception as e:
            print('pass')

    return gain_ratio

def robustness_test():
    dates = get_week_dates()
    df = pd.read_csv(r'.\result\predict.csv')

    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[df['交易日期'] >= pd.to_datetime('20200130')]

    gain_ratio = np.zeros(10)
    for date in dates[:]:
        try:
            df_week = df.groupby('交易日期').get_group(date)
            df_week['置信度排名'] = df_week['置信度'].rank(ascending=False, pct=True)
            for i in range(10):
                df_week_ = df_week[(df_week['置信度排名'] < (i+1)/10) & (df_week['置信度排名'] > i/10)]
                gain_ratio_ = df_week_['五天后的涨跌幅'].mean()
                gain_ratio[i] += gain_ratio_
        except Exception as e:
            print('pass robustness')

    return gain_ratio

if __name__ == '__main__':
    # df2 = pd.read_csv(r'.\result\predict.csv')
    # df = pd.read_csv(r'.\result\predict2.csv')
    # df = df.append(df2)
    # df.to_csv(r'.\result\predict3.csv', index=False)
    calculate_acc()
    calculate_ic()

    calculate_rank_ic()

    gain_ratio = back()

    plt.plot(np.arange(len(gain_ratio)), gain_ratio)
    plt.show()
    robustness = robustness_test()
    plt.bar(np.arange(len(robustness)), robustness)
    plt.show()


