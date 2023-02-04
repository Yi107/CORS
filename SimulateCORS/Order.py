import pandas as pd
from typing import *
from config import config


class Order(object):
    def __init__(self, trail, pr, id=None, df_row=None, start_time=None, end_time=None, start_lng=None,
                 end_lng=None, start_lat=None, end_lat=None, person_num=None, length=None):
        if df_row is not None:
            names = ['order_id', 'start_time', 'end_time', 'start_lng',
                     'start_lat', 'end_lng', 'end_lat', 'length', 'person_num']
            self.id, self.start_time, self.end_time, self.start_lng, self.start_lat, self.end_lng, \
            self.end_lat, self.length, self.person_num = [getattr(df_row, name) for name in names]
        else:
            self.id = id
            self.start_time = start_time
            self.end_time = end_time
            self.start_lng = start_lng
            self.end_lng = end_lng
            self.start_lat = start_lat
            self.end_lat = end_lat
            self.length = length
            self.person_num = person_num
            self.index=0
        self.ddl = config.ddl + (self.length / config.driver_speed) * config.beta + self.start_time
        self.take_time = self.length / config.driver_speed

        if self.length < 2000:
            self.price = 8
        elif self.length < 10000:
            self.price = (int(self.length - 8) + 1) * 2 + 8
        else:
            self.price = ((int(self.length - 8) + 1) * 2 + 8) * 1.5

        if isinstance(pr, str) and pr.lower() == 'price':
            self.pr = self.price
        else:
            self.pr = pr
        self.trail: Union[None, pd.DataFrame] = trail
