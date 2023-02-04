import numpy as np
from typing import *
from Order import Order
from config import config


def coordination_transform(order: Order, driver, direction) -> Tuple[tuple, tuple]:
    order_start = np.array([order.start_lng, order.start_lat])
    order_end = np.array([order.end_lng, order.end_lat])
    driver_pos = np.array([driver.longitude, driver.latitude])

    # translation
    o1 = order_start - driver_pos
    o2 = order_end - driver_pos

    # rotation
    cos_rot = np.cos(direction)
    sin_rot = np.sin(direction)
    rotation_mat = [[cos_rot, sin_rot], [-sin_rot, cos_rot]]
    o1 = tuple(np.matmul(o1, rotation_mat))
    o2 = tuple(np.matmul(o2, rotation_mat))
    return o1, o2


def euclidean_distance(coord_1: Tuple[float, float], coord_2: Tuple[float, float] = (0, 0)):
    diff_x = coord_1[0] - coord_2[0]
    diff_y = coord_1[1] - coord_2[1]
    return np.sqrt(np.sum(np.square([diff_x, diff_y])))


def lng_lat_distance(coord_1: Tuple[Any, ...], coord_2: Tuple[Any, ...] = (0, 0)):
    if coord_1 == coord_2:
        return 0
    MLat1 = 90 - coord_1[0]
    Mlng1 = coord_1[1]
    MLat2 = 90 - coord_2[0]
    Mlng2 = coord_2[1]

    C = np.sin(MLat1) * np.sin(MLat2) * np.cos(Mlng1 - Mlng2) + np.cos(MLat1) * np.cos(MLat2)
    return config.R * np.arccos(C) * np.pi / 180


def lng_lat_travel_time_span(coord_1: Tuple[Any, ...], coord_2: Tuple[Any, ...] = (0, 0)):
    return lng_lat_distance(coord_1, coord_2) * config.gamma / config.driver_speed


def get_bearing(start_lng, start_lat, end_lng, end_lat):
    start_lng = np.radians(start_lng)
    start_lat = np.radians(start_lat)
    end_lng = np.radians(end_lng)
    end_lat = np.radians(end_lat)
    diff_lng = np.radians(end_lng - start_lng)
    x = np.sin(diff_lng) * np.cos(end_lat)
    y = np.cos(start_lat) * np.sin(end_lat) \
        - np.sin(start_lat) * np.cos((end_lat) * np.cos(diff_lng))
    init_bearing = np.arctan2(x, y)
    init_bearing = np.degrees(init_bearing)
    vector = (init_bearing + 360) % 360
    return vector


def get_picked_order_position(i: int, orderlist: List,is_tuple):
    order_i = i >> 1
    if is_tuple==False:
        return order_i
    else:
        if i % 2 == 0:  # judge whther it is the orgin or end
            return tuple((orderlist[order_i].start_lat, orderlist[order_i].start_lng))
        else:
            return tuple((orderlist[order_i].end_lat, orderlist[order_i].end_lng))


if __name__ == '__main__':
    ...
