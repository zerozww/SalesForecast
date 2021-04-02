import pyodbc
import pandas as pd

host = '127.0.0.1'  # 'dd.whusoft.cn'
port = '1433'
user = 'sa'
password = 'zww123456'  # '15212xXX'
database = 'self_order'

connection = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+host+';DATABASE='+database+';UID='+user+';PWD='+ password)


# 从数据库获取dataframe
def get_df_from_db(sql):
    return pd.read_sql(sql, connection)

# 获取门店2018~2020的天气
def get_weather(store_code):
    sql = 'select * from [tianqi] where [city] = (select [town] from [store] where [store_code] = \'' + store_code +'\')'
    return get_df_from_db(sql)