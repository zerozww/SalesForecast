import pymssql
import pandas as pd

host = '127.0.0.1'  # 'dd.whusoft.cn'
port = '1433'
user = 'sa'
password = 'zww123456'  # '15212xXX'
database = 'auto_order'

connection = pymssql.connect(host=host, port=port, user=user, password=password, database=database, charset='utf8')


# 从数据库获取dataframe
def get_df_from_db(sql):
    return pd.read_sql(sql, connection)


# 组装sql
def get_with_store(store_code):
    sql = 'select * from [store_sales] where [store_code] = \'' + store_code + '\''
    return get_df_from_db(sql)


# 获取商店store_code数组
def get_store_with_type(store_type):
    sql = 'select [store_code] from [store] where [store_type] = \'' + store_type + '\''
    return get_df_from_db(sql)


# 获取各种类商店的商品各类销售量
def get_goods_sale_in_category_with_type(store_type):
    sql = 'SELECT [goods_code],COUNT([id]) as [count] FROM [dbo].[store_sales] where [store_code] in (SELECT [' \
          'store_code] from [dbo].[store] where [store_type] =\'' + store_type + '\') GROUP BY [goods_code] ORDER BY ' \
                                                                                 '[count] DESC '
    return get_df_from_db(sql)


# 获取某个商店指定商品的销量
def get_goods_sale_with_store_and_goods(store_code, goods_code):
    goods_code_str = '\',\''.join(goods_code)
    goods_code_str = '\'' + goods_code_str + '\''
    sql = 'SELECT [goods_code],COUNT([id]) as [count] FROM [dbo].[store_sales] WHERE [store_code] = \'' + store_code + \
          '\' and [goods_code] in (' + goods_code_str + ') GROUP BY [goods_code] ORDER BY [count] DESC '
    return get_df_from_db(sql)


# 获取某个门店某个商品的销量
def get_good_sale_with_store_and_good(store_code, good_code):
    sql = 'SELECT [sales_date],[sales_amount],[sales_income] FROM [dbo].[store_sales] where [store_code] = \'' + store_code + '\' and [goods_code] = \'' + good_code + '\''
    return get_df_from_db(sql)
