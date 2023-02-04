import math
from typing import *
import numpy as np

from Driver import Driver
from Order import Order
from config import config
from utils import *
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.evaluation import WolframCloudSession, SecuredAuthenticationKey

class Platform(object):
    def __init__(self, driver_file):
        self.driver_file = driver_file
        self.UC = 0
        self.orderlist = []

    @property
    def drivers(self) -> List[Driver]:
        try:
            return getattr(self, '_drivers')
        except AttributeError:
            lines = [line.split(',') for line in open(self.driver_file, 'r').readlines()]
            drivers = []
            for line in lines:
                drivers.append(Driver(float(line[1]), float(line[2]), int(line[0])))
            self._drivers = drivers
            return self._drivers

    @property
    def available_drivers(self) -> List[Driver]:
        try:
            return getattr(self, '_available_driver')
        except AttributeError:
            raise RuntimeError("Should call Platform.update_available_drivers before using this attribute")

    def update_available_drivers(self, order, platform):
        for driver in self.drivers:
            driver.update_driver(order.start_time, platform, self.orderlist)
        available_dri = []
        radius = 2500
        max_lat = radius / config.R * 180 / np.pi + order.start_lat
        min_lat = -radius / config.R * 180 / np.pi + order.start_lat
        max_lng = radius / (config.R * math.cos(order.start_lat)) * 180 / np.pi + order.start_lng
        min_lng = -radius / (config.R * math.cos(order.start_lat)) * 180 / np.pi + order.start_lng
        for driver in self.drivers:
            if min_lat < driver.latitude < max_lat and min_lng < driver.longitude < max_lng:
                if driver.capacity >= order.person_num + driver.current_num and driver.order_n <= 1:
                    available_dri.append(driver)
        self._available_driver = available_dri

    def insertion(self, driver, order):
        """
        return: the lower_bound is the extra time when the driver receive the new order
        """
        lower_bound = np.inf
        insert_path = [-1, -1]
        if driver.picked_order == []:
            tmp = driver.time + lng_lat_travel_time_span(
                (driver.latitude, driver.longitude), (order.start_lat, order.start_lng)) + order.take_time
            if tmp < order.ddl and driver.capacity >= order.person_num + driver.current_num:
                lower_bound = tmp - driver.time
                driver.picked_order.append(order.index * 2)
                driver.picked_order.append(order.index * 2 + 1)
                driver.update_driver_arr(self)
            if lower_bound < np.inf:
                return lower_bound
            else:
                return -1
        for i in range(len(driver.picked_order) + 1):
            if i == 0:
                part = lng_lat_travel_time_span(
                    (driver.latitude, driver.longitude), (order.start_lat, order.start_lng))

                detour = part + \
                         lng_lat_travel_time_span((order.end_lat, order.end_lng),
                                                  get_picked_order_position(driver.picked_order[0], self.orderlist,
                                                                            True)) - \
                         (driver.reach_time[0] - driver.time) + order.take_time
                if driver.current_num + order.person_num <= driver.capacity \
                        and detour <= driver.slack_time[i] \
                        and part < order.ddl - driver.time + config.EPS:
                    if lower_bound > detour:
                        lower_bound = detour
                        insert_path = [i, i]
            elif i == len(driver.picked_order):
                detour = lng_lat_travel_time_span(
                    get_picked_order_position(driver.picked_order[i - 1], self.orderlist, True)
                    , (order.start_lat, order.start_lng)) + order.take_time
                if driver.picked_num[i - 1] + order.person_num <= driver.capacity and \
                        detour < order.ddl - driver.reach_time[i - 1] + config.EPS:
                    if lower_bound > detour:
                        lower_bound = detour
                        insert_path = [i, i]
            else:

                part = lng_lat_travel_time_span(
                    get_picked_order_position(driver.picked_order[i - 1], self.orderlist, True),
                    (order.start_lat, order.start_lng))

                detour = part + lng_lat_travel_time_span((order.end_lat, order.end_lng)
                                                         , get_picked_order_position(driver.picked_order[i],
                                                                                     self.orderlist, True)) \
                         - (driver.reach_time[i] - driver.reach_time[i - 1]) + order.take_time
                if driver.picked_order[i - 1] + order.person_num <= driver.capacity and \
                        detour < driver.slack_time[i] and \
                        part < order.ddl - driver.reach_time[i - 1] + config.EPS:
                    if lower_bound > detour:
                        lower_bound = detour
                        insert_path = [i, i]
        Det = []
        for i in range(len(driver.picked_order)):
            tmp = [np.Inf, -1]
            if i == 0:
                if driver.current_num + order.person_num <= driver.capacity:
                    detour = lng_lat_travel_time_span((driver.latitude, driver.longitude),
                                                      (order.start_lat, order.start_lng)) \
                             + lng_lat_travel_time_span((order.start_lat, order.start_lng),
                                                        get_picked_order_position(driver.picked_order[0],
                                                                                  self.orderlist, True)) \
                             - (driver.reach_time[0] - driver.time)
                    if detour < driver.slack_time[i] + config.EPS:
                        tmp = [detour, i]
            else:
                if driver.picked_num[i - 1] + order.person_num <= driver.capacity:
                    tmp = Det[-1]
                    detour = lng_lat_travel_time_span(
                        get_picked_order_position(driver.picked_order[i - 1], self.orderlist, True),
                        (order.start_lat, order.start_lng)) \
                             + lng_lat_travel_time_span((order.start_lat, order.start_lng),
                                                        get_picked_order_position(driver.picked_order[i],
                                                                                  self.orderlist, True)) \
                             - (driver.reach_time[i] - driver.reach_time[i - 1])

                    if detour < driver.slack_time[i] + config.EPS:
                        tmp = [detour, i] if tmp[0] > detour else tmp
            Det.append(tmp)

        for i in range(len(driver.picked_order)):
            if i < len(driver.picked_order) - 1:
                if driver.picked_order[i] > driver.capacity - order.person_num:
                    continue
                part = lng_lat_travel_time_span(get_picked_order_position(driver.picked_order[i], self.orderlist, True),
                                                (order.end_lat, order.end_lng))
                detour = part + lng_lat_travel_time_span((order.end_lat, order.end_lng),
                                                         get_picked_order_position(driver.picked_order[i + 1],
                                                                                   self.orderlist, True)) \
                         - (driver.reach_time[i + 1] - driver.reach_time[i])
                if Det[i][0] + detour < driver.slack_time[i] + config.EPS and \
                        Det[i][0] + part < order.ddl - driver.reach_time[i] + config.EPS:
                    if lower_bound > Det[i][0] + detour:
                        lower_bound = Det[i][0] + detour
                        insert_path = [Det[i][1], i + 1]
            else:
                detour = lng_lat_travel_time_span(
                    get_picked_order_position(driver.picked_order[i], self.orderlist, True),
                    (order.end_lat, order.end_lng))
                if Det[i][0] + detour < order.ddl - driver.reach_time[i] + config.EPS:
                    if lower_bound > Det[i][0] + detour:
                        lower_bound = Det[i][0] + detour
                        insert_path = [Det[i][1], i + 1]

        if (insert_path[0] > -1):
            ret = []
            for j in range(insert_path[0]):
                ret.append(driver.picked_order[j])
            ret.append(order.index * 2)
            for j in range(insert_path[0], insert_path[1]):
                ret.append(driver.picked_order[j])
            ret.append(order.index * 2 + 1)
            for j in range(insert_path[1], len(driver.picked_order)):
                ret.append(driver.picked_order[j])
            driver.picked_order = ret
            driver.update_driver_arr(self)
        else:
            return -1

        return lower_bound


class MainPlatform(Platform):
    def __init__(self, driver_file):
        super().__init__(driver_file)

    def try_insertion_euclid(self, driver, order, lower_bound):
        if driver.picked_order == []:
            tmp = driver.time + lng_lat_travel_time_span \
                ((driver.latitude, driver.longitude), (order.start_lat, order.start_lng)) + order.take_time
            if tmp < order.ddl and driver.capacity >= order.person_num + driver.current_num:
                lower_bound = tmp - driver.time
            return lower_bound

        for i in range(len(driver.picked_order) + 1):
            if i == 0:
                part = lng_lat_travel_time_span \
                    ((driver.latitude, driver.longitude), (order.start_lat, order.start_lng))

                detour = part + \
                         lng_lat_travel_time_span((order.end_lat, order.end_lng),
                                                  get_picked_order_position(driver.picked_order[0], self.orderlist,
                                                                            True)) - \
                         (driver.reach_time[0] - driver.time) + order.take_time
                if driver.current_num + order.person_num <= driver.capacity \
                        and detour <= driver.slack_time[i] \
                        and part < order.ddl - driver.time + config.EPS:
                    if lower_bound > detour:
                        lower_bound = detour
            elif i == len(driver.picked_order):
                detour = lng_lat_travel_time_span(
                    get_picked_order_position(driver.picked_order[i - 1], self.orderlist, True)
                    , (order.start_lat, order.start_lng)) + order.take_time
                if driver.picked_num[i - 1] + order.person_num <= driver.capacity and \
                        detour < order.ddl - driver.reach_time[i - 1] + config.EPS:
                    if lower_bound > detour:
                        lower_bound = detour
            else:

                part = lng_lat_travel_time_span(
                    get_picked_order_position(driver.picked_order[i - 1], self.orderlist, True),
                    (order.start_lat, order.start_lng))

                detour = part + lng_lat_travel_time_span((order.end_lat, order.end_lng)
                                                         , get_picked_order_position(driver.picked_order[i],
                                                                                     self.orderlist, True)) \
                         - (driver.reach_time[i] - driver.reach_time[i - 1]) + order.take_time
                if driver.picked_order[i - 1] + order.person_num <= driver.capacity and \
                        detour < driver.slack_time[i] and \
                        part < order.ddl - driver.reach_time[i - 1] + config.EPS:
                    if lower_bound > detour:
                        lower_bound = detour
        Det = []
        for i in range(len(driver.picked_order)):
            tmp = [np.Inf, -1]
            if i == 0:
                if driver.current_num + order.person_num <= driver.capacity:
                    detour = lng_lat_travel_time_span((driver.latitude, driver.longitude),
                                                      (order.start_lat, order.start_lng)) \
                             + lng_lat_travel_time_span((order.start_lat, order.start_lng),
                                                        get_picked_order_position(driver.picked_order[0],
                                                                                  self.orderlist, True)) \
                             - (driver.reach_time[0] - driver.time)
                    if detour < driver.slack_time[i]:
                        tmp = [detour, i]
            else:
                if driver.picked_num[i - 1] + order.person_num <= driver.capacity:
                    detour = lng_lat_travel_time_span(
                        get_picked_order_position(driver.picked_order[i - 1], self.orderlist, True),
                        (order.start_lat, order.start_lng)) \
                             + lng_lat_travel_time_span((order.start_lat, order.start_lng),
                                                        get_picked_order_position(driver.picked_order[i],
                                                                                  self.orderlist, True)) \
                             - (driver.reach_time[i] - driver.reach_time[i - 1])

                    if detour < driver.slack_time[i] + config.EPS:
                        tmp = [detour, i]
            Det.append(tmp)

        for i in range(len(driver.picked_order)):
            if i < len(driver.picked_order) - 1:
                if driver.picked_order[i] > driver.capacity - order.person_num:
                    continue
                part = lng_lat_travel_time_span(get_picked_order_position(driver.picked_order[i], self.orderlist, True),
                                                (order.end_lat, order.end_lng))
                detour = part + lng_lat_travel_time_span((order.end_lat, order.end_lng),
                                                         get_picked_order_position(driver.picked_order[i + 1],
                                                                                   self.orderlist, True)) \
                         - (driver.reach_time[i + 1] - driver.reach_time[i])
                if Det[i][0] + detour < driver.slack_time[i] + config.EPS and \
                        Det[i][0] + part < order.ddl - driver.reach_time[i] + config.EPS:
                    if lower_bound > Det[i][0] + detour:
                        lower_bound = Det[i][0] + detour
            else:
                detour = lng_lat_travel_time_span(
                    get_picked_order_position(driver.picked_order[i], self.orderlist, True),
                    (order.end_lat, order.end_lng))
                if Det[i][0] + detour < order.ddl - driver.reach_time[i] + config.EPS:
                    if lower_bound > Det[i][0] + detour:
                        lower_bound = Det[i][0] + detour

        return lower_bound

    def assignTaxi(self, worker, order):
        lower_bound = self.insertion(worker, order)
        if lower_bound == -1:
            return -1
        else:
            # calculate the cost of the order
            self.UC += lower_bound * config.alpha * worker.speed * config.OIL_PRICE
            return True

    def pruneGDP(self, order):
        valued_cars = []
        self.update_available_drivers(order, "Mainplatform")
        lowerb = np.Inf
        for driver in self.available_drivers:
            tmp = self.try_insertion_euclid(driver, order, lowerb)
            if tmp < lowerb:
                lowerb = tmp
                pair = [driver, lowerb]
                valued_cars.append(pair)
        if valued_cars == []:
            return False  # it means no cars available
        else:
            worker = valued_cars[-1][0]
        return self.assignTaxi(worker, order)

    def process_order(self, order):
        order.index = len(self.orderlist)
        self.orderlist.append(order)
        return self.pruneGDP(order)


class FriendPlatform(Platform):
    def __init__(self, driver_file):
        super().__init__(driver_file)
        self.driver_evaluator = DriverEvaluator()
        self.sum_pr = 0

    def _accept_order(self, driver: Driver, order: Order):
        pickup_time = driver.pickup_time(order)
        driver.order_n += 1
        cost = (pickup_time + order.take_time) * config.driver_speed * config.OIL_PRICE
        self.UC += config.alpha * cost
        self.sum_pr += order.pr

    def _reject_order(self, order: Order):
        self.UC += order.pr
        self.sum_pr += order.pr

    def choose_driver(self, order: Order) -> Driver:
        min_cost = np.inf
        best_driver = None
        for driver in self.available_drivers:
            # predict the direction of driver's destination
            direction = driver.predict_direction()
            curr_cost = self.driver_evaluator.evaluate(driver, order, direction)
            if curr_cost < min_cost:
                min_cost = curr_cost
                best_driver = driver
        return best_driver

    def random_choose_driver(self) -> Driver:
        driver = np.random.choice(self.available_drivers, (1))[0] if self.available_drivers else None
        return driver

    def process_order(self, order: Order, mode: str):
        order.index = len(self.orderlist)
        self.orderlist.append(order)
        self.update_available_drivers(order, 'friend')

        if mode and mode.lower() == 'random':
            driver = self.random_choose_driver()
        else:
            driver = self.choose_driver(order)

        if driver:
            insert_result = self.insertion(driver, order)
            if insert_result == -1:
                self._reject_order(order)
                return -1
            else:
                self._accept_order(driver, order)
            return True
        else:
            self._reject_order(order)
            return False


class DriverEvaluator():
    def __init__(self):

        # key = SecuredAuthenticationKey(
        #     'mcJrQvLY8hVnrayhvp8yn81K7H6vjSkYxyvkQsVNZ6Q=',
        #     '3AdAYiiS9pZaviP/excJqI6F0/+JSdsFWgd2ToTP4qI=')
        # self.sess = WolframCloudSession(credentials=key)
        self.sess = WolframLanguageSession('/Applications/Wolfram Desktop.app/Contents/MacOS/WolframKernel')

        self.sess.evaluate('root12 = Solve[(y1^2 + (x1 - x)^2)^(1/2) == r12, x]')
        self.sess.evaluate('root13 = Solve[r1 + (y2^2 + (x2 - x)^2)^(1/2) == x + r12, x]')
        self.sess.evaluate('root23 = Solve[r1 + (y2^2 + (x2 - x)^2)^(1/2) == x + (y1^2 + (x1 - x)^2)^(1/2), x]')
        self.sess.evaluate('root1 = Solve[r1 + r12 + (y2^2 + (x2 - x)^2)^(1/2) == p, x]')
        self.sess.evaluate('root2 = Solve[r1 + (y1^2 + (x1 - x)^2)^(1/2) + (y2^2 + (x2 - x)^2)^(1/2) == p, x]')
        self.sess.evaluate('root3 = Solve[x + r12 + (y1^2 + (x1 - x)^2)^(1/2) == p, x]')

    def D1(self, r1, r12, x2, y2, r):
        return r1 + r12 + math.sqrt((x2 - r) ** 2 + y2 ** 2)

    def D2(self, r1, x1, y1, x2, y2, r):
        return r1 + math.sqrt((x1 - r) ** 2 + y1 ** 2) + math.sqrt((x2 - r) ** 2 + y2 ** 2)

    def D3(self, r1, r12, x1, y1, r):
        return r1 + r12 + math.sqrt((x1 - r) ** 2 + y1 ** 2)

    def integral_D1(self, r1, r12, x2, y2, bounds=None, x=None):
        if not bounds:
            return 0
        if x:
            y2_sqr = y2 * y2
            d = np.sqrt(x * x + y2_sqr)
            return (x * d + y2_sqr * np.log(x + d)) / 2

        integral_r1_r12 = np.sum([high - low for low, high in bounds]) * (r1 + r12)
        integral_r23 = np.sum([self.integral_D1(r1, r12, x2, y2, x=high - x2)
                               - self.integral_D1(r1, r12, x2, y2, x=low - x2) for low, high in bounds])
        return integral_r1_r12 + integral_r23

    def integral_D2(self, r1, x1, y1, x2, y2, bounds=None, x=None):
        if not bounds:
            return 0
        if x:
            x_1 = x - x1
            y1_sqr = y1 * y1
            d1 = np.sqrt(x_1 * x_1 + y1_sqr)
            x_2 = x - x2
            y2_sqr = y2 * y2
            d2 = np.sqrt(x_2 * x_2 + y2_sqr)
            return (x_2 * d2 + y2_sqr * np.log(x_2 + d2) + x_1 * d1 + y1_sqr * np.log(x_1 + d1)) / 2

        integral_r1 = np.sum([high - low for low, high in bounds]) * r1
        integral_r13_r23 = np.sum([self.integral_D2(r1, x1, y1, x2, y2, x=high)
                                   - self.integral_D2(r1, x1, y1, x2, y2, x=low)
                                   for low, high in bounds])
        return integral_r1 + integral_r13_r23

    def integral_D3(self, r12, x1, y1, bounds=None, x=None):
        if not bounds:
            return 0
        if x:
            x_1 = x - x1
            y1_sqr = y1 * y1
            d = np.sqrt(x_1 * x_1 + y1_sqr)
            return (x_1 * d + y1_sqr * np.log(x_1 + d) + x * x) / 2

        integral_r12 = np.sum([high - low for low, high in bounds]) * r12
        integral_r_r13 = np.sum([self.integral_D3(r12, x1, y1, x=high)
                                 - self.integral_D3(r12, x1, y1, x=low) for low, high in bounds])
        return integral_r12 + integral_r_r13

    def get_root12(self, x1, y1, r12):
        # names = ['x1', 'y1', 'r12']
        # values = [x1, y1, r12]
        # exprs = ['{}={:.10f}'.format(name, val) for name, val in zip(names, values)]
        # self.sess.evaluate_many(exprs)
        # test = self.sess.evaluate('root12')
        strsss = 'root12 = Solve[(' + str(y1) + '^2 + (' + str(x1) + ' - x)^2)^(1/2) == ' + str(r12) + ', x]'
        roots = [root[0][1] for root in self.sess.evaluate(strsss) if isinstance(root[0][1], (float, int))]
        return sorted(set(root for root in roots if root > 0))

    def get_root13(self, x2, y2, r1, r12):
        # names = ['x2', 'y2', 'r1', 'r12']
        # values = [x2, y2, r1, r12]
        # exprs = ['{}={:.10f}'.format(name, val) for name, val in zip(names, values)]
        # self.sess.evaluate_many(exprs)
        strsss = 'root13 = Solve['+str(r1)+'('+str(y2)+'^2 + ('+str(x2)+'- x)^2)^(1/2) == x + '+str(r12)+', x]'
        roots = [root[0][1] for root in self.sess.evaluate(strsss) if isinstance(root[0][1], (float, int))]
        return sorted(set(root for root in roots if root > 0))

    def get_root23(self, r1, x1, y1, x2, y2):
        # names = ['r1', 'y2', 'x2', 'y1', 'x1']
        # values = [r1, y2, x2, y1, x1]
        # exprs = ['{}={:.10f}'.format(name, val) for name, val in zip(names, values)]
        # self.sess.evaluate_many(exprs)
        strsss = 'root23 = Solve['+ str(r1)+' + ('+str(y2)+'^2 + ('+ str(x2) +' - x)^2)^(1/2) == x + ('+str(y1)+'^2 + ('+str(x1)+' - x)^2)^(1/2), x]'
        roots = [root[0][1] for root in self.sess.evaluate(strsss) if isinstance(root[0][1], (float, int))]
        return sorted(set(root for root in roots if root > 0))

    def get_root1(self, r1, r12, x2, y2, p):
        # names = ['r1', 'r12', 'x2', 'y2', 'p']
        # values = [r1, r12, x2, y2, p]
        # exprs = ['{}={:.10f}'.format(name, val) for name, val in zip(names, values)]
        # self.sess.evaluate_many(exprs)
        strsss = 'root1 = Solve[' + str(r1) + ' + ' + str(r12) + ' + (' + str(y2) + '^2 + (' + str(
            x2) + ' - x)^2)^(1/2) == ' + str(p) + ', x]'
        roots = [root[0][1] for root in self.sess.evaluate(strsss) if isinstance(root[0][1], (float, int))]
        return sorted(set(root for root in roots if root > 0))

    def get_root2(self, r1, x1, y1, x2, y2, p):
        # names = ['r1', 'x1', 'y1', 'x2', 'y2', 'p']
        # values = [r1, x1, y1, x2, y2, p]
        # exprs = ['{}={:.10f}'.format(name, val) for name, val in zip(names, values)]
        # self.sess.evaluate_many(exprs)
        ss5 = 'root2 = Solve[' + str(r1) + ' + (' + str(y1) + '^2 + (' + str(x1) + ' - x)^2)^(1/2) + (' + str(
            y2) + '^2 + (' + str(x2) + ' - x)^2)^(1/2) == ' + str(p) + ', x]'
        roots = [root[0][1] for root in self.sess.evaluate(ss5) if isinstance(root[0][1], (float, int))]
        return sorted(set(root for root in roots if root > 0))

    def get_root3(self, r12, x1, y1, p):
        # names = ['r12', 'x1', 'y1', 'p']
        # values = [r12, x1, y1, p]
        # exprs = ['{}={:.10f}'.format(name, val) for name, val in zip(names, values)]
        # self.sess.evaluate_many(exprs)
        ss6 = 'root3 = Solve[x + ' + str(r12) + ' + (' + str(y1) + '^2 + (' + str(x1) + ' - x)^2)^(1/2) == ' + str(
            p) + ', x]'
        roots = [root[0][1] for root in self.sess.evaluate(ss6) if isinstance(root[0][1], (float, int))]
        return sorted(set(root for root in roots if root > 0))

    def _evaluate_dir(self, o1, o2, p):
        x1, y1 = o1
        x2, y2 = o2
        r1 = euclidean_distance(o1)
        r12 = euclidean_distance(o1, o2)

        root12 = self.get_root12(x1, y1, r12)
        root13 = self.get_root13(x2, y2, r1, r12)
        root23 = self.get_root23(r1, x1, y1, x2, y2)
        root1 = self.get_root1(r1, r12, x2, y2, p)
        root2 = self.get_root2(r1, x1, y1, x2, y2, p)
        root3 = self.get_root3(r12, x1, y1, p)
        try:
            max_root = max(root1 + root2 + root3)
        except Exception as e:
            print(o1, o2, r1, r12)
            print(root12)
            print(root13)
            print(root23)
            print(root1)
            print(root2)
            print(root3)
            raise e
        roots = sorted([0] + root12 + root13 + root23 + root1 + root2 + root3)
        min_Did: List[int] = []
        for i in range(len(roots) - 1):
            x = (roots[i] + roots[i + 1]) / 2
            k = np.argmin([self.D1(r1, r12, x2, y2, x),
                           self.D2(r1, x1, y1, x2, y2, x),
                           self.D3(r1, r12, x1, y1, x)])
            min_Did.append(k)

        curr_Did: int = min_Did[0]
        low = 0
        high = None
        D1_integral_bounds = []
        D2_integral_bounds = []
        D3_integral_bounds = []
        integral_bounds = [D1_integral_bounds, D2_integral_bounds, D3_integral_bounds]
        for i in range(len(roots) - 1):
            if min_Did[i] != curr_Did:
                integral_bounds[curr_Did].append((low, high))
                low = high
                curr_Did = min_Did[i]
            high = roots[i + 1]
        integral_bounds[curr_Did].append((low, high))
        return np.sum([self.integral_D1(r1, r12, x2, y2, bounds=D1_integral_bounds),
                       self.integral_D2(r1, x1, y1, x2, y2, bounds=D2_integral_bounds),
                       self.integral_D3(r12, x1, y1, bounds=D3_integral_bounds)]) / max_root

    def evaluate(self, driver, order, direction):
        # evaluate the original direction
        o1, o2 = coordination_transform(order, driver, direction)
        if driver.order_n:
            original_cost = self._evaluate_dir(o1, o2, order.price)
        else:
            original_cost = euclidean_distance(o1) + euclidean_distance(o1, o2)


        # evaluate the opposite direction
        o1, o2 = coordination_transform(order, driver, 180 - direction)
        if driver.order_n:
            opposite_cost = self._evaluate_dir(o1, o2, order.price)
        else:
            opposite_cost = euclidean_distance(o1) + euclidean_distance(o1, o2)
        return original_cost * .75 + opposite_cost * .25
