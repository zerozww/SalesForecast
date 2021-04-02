from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import matplotlib.dates as mdates
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
import dao
import dao2
from Holiday import is_holidays,is_festival,is_weekday
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
import joblib


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



# 获取需要预测的商品编号
goods_sale = dao.get_goods_sale_in_category_with_type('A')
goods_sale = goods_sale.iloc[0:10, 0]
goods_code_values = goods_sale.values

# 获取该种类门店的所有编号
stores_code = dao.get_store_with_type('A')
stores_code_values = stores_code.iloc[:, 0].values
# 获取门店的需要预测的商品的销量 商品码/门店 矩阵
goods_count_stores_code = pd.DataFrame(columns=goods_code_values)
for store_code in stores_code_values:
    goods_count = dao.get_goods_sale_with_store_and_goods(store_code, goods_code_values)
    # 将数组转换成goods_count_stores_code的形式
    goods_count2 = pd.DataFrame(goods_count.iloc[:, 1:].values.T, index=[store_code],
                                columns=goods_count.iloc[:, 0].values)
    goods_count_stores_code = pd.concat([goods_count_stores_code, goods_count2])

count_zero = (goods_count_stores_code.fillna(0) == 0).astype(int).sum(axis=1)
# 取数据最全的门店作为标准门店
standard_store_code = count_zero.idxmin()
# 获得所有门店相对标准门店的销量比例
a = goods_count_stores_code.loc[standard_store_code]
goods_rate_stores_code = goods_count_stores_code.div(goods_count_stores_code.loc[standard_store_code], axis=1)
goods_rate_stores_code['mean'] = goods_rate_stores_code.mean(axis=1)
rate_values = goods_rate_stores_code['mean']


# 数据集
all_data_set = pd.DataFrame()

for good_code in goods_code_values:
    for store_code in stores_code_values:
        # store_data_set = dao.get_good_sale_with_store_and_good(store_code, good_code)
        store_data_set = dao.get_good_sale_with_store_and_good(store_code, good_code)
        if (store_data_set.empty):
            continue
        # 去null去0
        store_data_set.dropna(inplace=True)
        store_data_set = store_data_set[~store_data_set['sales_amount'].isin([0])]
        store_data_set = store_data_set[~store_data_set['sales_income'].isin([0])]
        store_data_set['price'] = store_data_set.apply(lambda x: x['sales_income'] / x['sales_amount'], axis=1)
        store_data_set = store_data_set.drop(['sales_income'], axis=1)
        store_data_set['sales_amount'] = store_data_set.apply(lambda x: x['sales_amount'] / rate_values[store_code],
                                                              axis=1)
        store_data_set.sort_values('sales_date', inplace=True)
        # 对日期进行填充
        store_data_set['sales_date'] = pd.to_datetime(store_data_set['sales_date'])
        helper = pd.DataFrame({'sales_date': pd.date_range(store_data_set['sales_date'].min(), store_data_set['sales_date'].max())})
        # 增加节假日信息
        #helper['weekday'] = helper['sales_date'].apply(lambda x:is_weekday(x))
        helper['holiday'] = helper['sales_date'].apply(lambda x:is_holidays(x))
        #helper['festival'] = helper['sales_date'].apply(lambda x:is_festival(x)[0])
        helper['festival_day'] = helper['sales_date'].apply(lambda x:is_festival(x)[1])
        helper['festival_day'].fillna('none', inplace=True)
        store_data_set = helper.merge(store_data_set, how='outer')
        store_data_set['sales_date'] = store_data_set['sales_date'].apply(lambda x:x.strftime('%Y-%m-%d'))

        # 对线性插值
        store_data_set['sales_amount'] = store_data_set['sales_amount'].interpolate(method='linear')
        store_data_set['price'] = store_data_set['price'].interpolate(method='linear')

        # 获取该门店的天气信息
        weather = dao2.get_weather(store_code)
        #weather['ymd'] = weather['ymd'].apply(lambda x:x.replace('-',''))
        weather['bWendu'] = weather['bWendu'].apply(lambda x:x.replace('℃',''))
        weather['yWendu'] = weather['yWendu'].apply(lambda x: x.replace('℃', ''))

        weather.drop(['id','aqiInfo','city'],axis=1,inplace=True)
        weather.rename(columns={'ymd':'sales_date'},inplace=True)
        store_data_set = store_data_set.merge(weather,how='left')
        store_data_set.set_index(['sales_date'], inplace=True)
        # 将销量放到最后一列，方便划分输入和输出
        store_data_set.insert(store_data_set.shape[1]-1,'sales_amount',store_data_set.pop('sales_amount'))

        # 画图，横坐标单位为月份
        store_data_set_2019 = store_data_set.loc['2019-01-01':'2019-12-31']
        """
        store_data_set_2019.plot()
        pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        pyplot.gca().xaxis.set_major_locator(mdates.MonthLocator())
        pyplot.xticks(rotation=70)
        pyplot.show()
        """

        store_data_set_2020 = store_data_set.loc['2020-01-01':'2020-12-31']
        """
        store_data_set_2020.plot()
        pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        pyplot.gca().xaxis.set_major_locator(mdates.MonthLocator())
        pyplot.xticks(rotation=70)
        pyplot.show()
        """

        if(store_data_set_2019.shape[0]==365):
            all_data_set = all_data_set.append(store_data_set_2019)
            store_data_set_2019.plot()
            pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            pyplot.gca().xaxis.set_major_locator(mdates.MonthLocator())
            pyplot.xticks(rotation=70)
            pyplot.show()

        if(store_data_set_2020.shape[0]==366):
            all_data_set = all_data_set.append(store_data_set_2020)
            store_data_set_2020.plot()
            pyplot.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            pyplot.gca().xaxis.set_major_locator(mdates.MonthLocator())
            pyplot.xticks(rotation=70)
            pyplot.show()

    # 将字符串数据（节假日信息）处理成数字
    # integer encode direction
    encoder = LabelEncoder()
    values = all_data_set.values
    # holiday
    values[:, 0] = encoder.fit_transform(values[:, 0])
    # festival_day
    values[:, 1] = encoder.fit_transform(values[:, 1])
    # tianqi
    values[:, 5] = encoder.fit_transform(values[:, 5])
    # fengxiang
    values[:, 6] = encoder.fit_transform(values[:, 6])
    # fengli
    values[:, 7] = encoder.fit_transform(values[:, 7])

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[11,12, 13, 14, 15,16,17,18,19,20]], axis=1, inplace=True)
    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    #train_store_num = 5 * (365+366)
    train = values[:-731, :]
    test = values[-731:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(train_X, train_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # 保存模型
    model.save(good_code + 'model.h5')


    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    #inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = concatenate((test_X[:, :-1],yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    #inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = concatenate((test_X[:, :-1],test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    predict_data = pd.DataFrame()
    predict_data['sale'] = inv_y
    predict_data['predict'] = inv_yhat
    predict_data.plot()
    pyplot.show()

    print('Test RMSE: %.3f' % rmse)
