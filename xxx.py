import statsmodels.api as sm  #最小二乘
from statsmodels.formula.api import ols #加载ols模型
import pandas as pd
from stock_predict import *
# df = pd.read_csv(r'.\result\predict.csv')
#
# dates = get_week_dates()
#
#
# corr = 0
# count = 0
# for date in dates:
#         df_week = df.groupby('交易日期').get_group(date)
#         df_week['置信度排名'] = df_week['置信度'].rank(ascending=True, method='first')
#         df_week['五天后的涨跌幅排名'] = df_week['五天后的涨跌幅'].rank(ascending=True, method='first')
#         traindata = df_week.loc[:, ['五天后的涨跌幅排名', '置信度排名']]
#
#         lm = ols('五天后的涨跌幅排名~ 置信度排名', data=traindata).fit()
#         print(lm.summary())
#         exit()

print(1.04**200)