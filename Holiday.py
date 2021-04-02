from chinese_calendar import is_workday, is_holiday
import chinese_calendar as calendar
import time, datetime


def is_weekday(date):
    '''
    判断是否为工作日
    '''
    Y = date.year
    M = date.month
    D = date.day
    april_last = datetime.date(Y, M, D)
    return is_workday(april_last)


def is_holidays(date):
    '''
    判断是否为节假日
    '''
    Y = date.year
    M = date.month
    D = date.day
    april_last = datetime.date(Y, M, D)
    return is_holiday(april_last)


def is_festival(date):
    """
    判断是否为节日
    注意：有些时间属于相关节日的调休日也会显示出节日名称，可参考源码https://pypi.org/project/chinesecalendar/
    """
    Y = date.year
    M = date.month
    D = date.day
    april_last = datetime.date(Y, M, D)
    on_holiday, holiday_name = calendar.get_holiday_detail(april_last)
    return on_holiday, holiday_name


if __name__ == "__main__":
    today = datetime.datetime(2020,1,19)
    week = is_weekday(today)
    holiday = is_holidays(today)
    festival = is_festival(today)
    print("week-{}".format(week))
    print("holiday-{}".format(holiday))
    print("festival-{}-{}".format(festival[0], festival[1]))