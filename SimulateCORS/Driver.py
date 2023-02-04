import pickle
import numpy as np
import pandas as pd
from typing import *
from Order import Order
from config import config
from utils import get_bearing, lng_lat_travel_time_span, get_picked_order_position
decision_tree = pickle.load(open(config.decision_tree_path, 'rb'))


class Driver(object):
    def __init__(self, longitude, latitude, driver_id):
        self.time = -1
        self.longitude = longitude
        self.latitude = latitude
        self.capacity = 3
        self.vector = -1
        self.id = driver_id
        self.order_n = 0
        self.speed = config.driver_speed

        self.current_num = 0
        self.picked_order = []
        self.slack_time = []
        self.reach_time = []
        self.picked_num = []

    def predict_direction(self) -> float:
        x = [[self.time, self.longitude, self.latitude, self.vector]]
        y = decision_tree.predict(x)
        return y[0]

    def update_driver_arr(self, platform):
        time = self.time
        self.reach_time = []
        for i in range(len(self.picked_order)):
            if i == 0:
                time += lng_lat_travel_time_span((self.latitude, self.longitude),
                                           get_picked_order_position(i,platform.orderlist,True))
            else:
                time += lng_lat_travel_time_span(get_picked_order_position(i,platform.orderlist,True),
                                            get_picked_order_position(i - 1,platform.orderlist,True))
            self.reach_time.append(time)

        self.picked_num = []
        self.slack_time = []
        cur_num = self.current_num
        for i in range(len(self.picked_order)):
            if self.picked_order[i] & 1:
                cur_num -= platform.orderlist[int(get_picked_order_position(i,platform.orderlist,False) / 2)].person_num
                self.slack_time.append(
                    platform.orderlist[int(get_picked_order_position(i,platform.orderlist,False) / 2)].ddl - self.reach_time[i])
            else:
                cur_num += platform.orderlist[int(get_picked_order_position(i,platform.orderlist,False) / 2)].person_num
                self.slack_time.append(np.inf)
            self.picked_num.append(cur_num)
        for i in range(len(self.slack_time) - 2, -1, -1):
            self.slack_time[i] = self.slack_time[i] if self.slack_time[i + 1] > \
                                                       self.slack_time[i] else self.slack_time[i + 1]

    def update_driver(self, time: float, platform: str, orderlist: List[Order] = None):
        if platform == "Mainplatform":
            while self.picked_order != [] and self.time < time:
                self.time = self.reach_time[0]
                order = orderlist[int(self.picked_order[0]/2)]
                if self.picked_order[0] % 2 == 0:
                    self.current_num += order.person_num
                    self.longitude = order.start_lng
                    self.latitude = order.start_lat
                else:
                    self.current_num -= order.person_num
                    self.longitude = order.end_lng
                    self.latitude = order.end_lat
                if self.picked_order != []:
                    self.picked_order.pop(False)
                    self.reach_time.pop(False)
                    self.slack_time.pop(False)
                    self.picked_num.pop(False)
            if self.time < time:
                self.time = time
        elif platform.lower() == 'friend':
            time_ =self.time
            while self.picked_order != [] and time_ < time:
                self.time = time_
                order = orderlist[int(self.picked_order[0]/2)]
                if self.picked_order[0] % 2 == 0:
                    self.current_num += order.person_num
                    self.longitude = order.start_lng
                    self.latitude = order.start_lat
                else:
                    self.current_num -= order.person_num
                    self.longitude = order.end_lng
                    self.latitude = order.end_lat
                    self.order_n -= 1

                trail: pd.DataFrame = order.trail
                start_lng = trail.iloc[0]['lng']
                start_lat = trail.iloc[0]['lat']
                end_lng = start_lng
                end_lat = start_lat

                for i in range(len(trail)):
                    end_lng = trail.iloc[i]['lng']
                    end_lat = trail.iloc[i]['lat']
                    if end_lng != start_lng or end_lat != start_lat:
                        break
                self.vector = get_bearing(start_lng, start_lat, end_lng, end_lat)
                if self.picked_order != []:
                    self.picked_order.pop(False)
                    self.reach_time.pop(False)
                    self.slack_time.pop(False)
                    self.picked_num.pop(False)
                time_ = self.reach_time[0] if self.reach_time else np.inf
            if self.time < time:
                self.time = time

    def pickup_time(self, order):
        return lng_lat_travel_time_span((self.latitude, self.longitude),(order.start_lat, order.start_lng))
