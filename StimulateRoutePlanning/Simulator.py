import os
import pickle
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from matplotlib import pyplot as plt
from Order import Order
from config import config
from utils import lng_lat_distance
import time
from Platform import MainPlatform, FriendPlatform
import psutil
import os


class Simulator:
    def __init__(self, order_file, gps_file,
                 main_platform_driver_file, friend_platform_driver_file):
        self.order_file = order_file

        gps_names = ['driver_id', 'order_id', 'time', 'lng', 'lat']
        print('Reading GPS...')
        self.gps = pd.concat(tqdm(pd.read_csv(gps_file, names=gps_names, chunksize=100000)))

        # calculate trail of order
        self.order2trail = dict()
        print('Calculating trail...')
        for order_id, group in tqdm(self.gps.groupby(["order_id"])):
            self.order2trail[order_id] = group
        self.main_platform = MainPlatform(main_platform_driver_file)
        self.friend_platform = FriendPlatform(friend_platform_driver_file)

        self.main_acc_n = 0
        self.main_rej_n = 0
        self.friend_acc_n = 0
        self.friend_rej_n = 0

        self.reject_orders_for_empty = []
        self.reject_orders_for_error = []

        self.main_acc_ratios = []
        self.friend_acc_ratios = []
        self.reject_for_empty_ratios = []

    @property
    def orders(self) -> List[Order]:
        try:
            return getattr(self, '_orders')
        except:
            order_names = ['idx', 'order_id', 'start_time', 'end_time', 'start_lng',
                           'start_lat', 'end_lng', 'end_lat', 'length', 'person_num']
            order_csv = pd.read_csv(self.order_file, names=order_names, header=0)
            order_csv.sort_values(['start_time'], inplace=True)
            print('Reading orders...')
            order_csv = [Order(df_row=row, pr=config.pr, trail=self.order2trail[getattr(row, 'order_id')])
                         for idx, row in tqdm(order_csv.iterrows())]

            self._orders = order_csv
        return self._orders

    def run(self, mode):
        modes = ['random', 'direction']
        if mode not in modes:
            raise ValueError("Error mode: {}, mode should be one of {}".format(mode, modes))

        print("Simulating in mode:", mode)
        bar = tqdm(total=len(self.orders), ncols=140)
        n = 0
        start_time =time.time()
        start = show_info('Begin')
        for order in self.orders:
            if n < 5000:
                self._process_order(order, mode)
                bar.update()
                bar.set_postfix_str('Main: {}, {}; Friend: {}, {}, RejReason: {}, {}'.format(
                    self.main_acc_n, self.main_rej_n,
                    self.friend_acc_n, self.friend_rej_n,
                    len(self.reject_orders_for_error), len(self.reject_orders_for_empty)))
                if n % 1500 == 0:
                    if n != 0:
                        self.main_acc_ratios.append(self.main_acc_n / (self.main_acc_n + self.main_rej_n + 1))
                        self.friend_acc_ratios.append(self.friend_acc_n / (self.friend_acc_n + self.friend_rej_n + 1))
                        self.reject_for_empty_ratios.append(len(self.reject_orders_for_empty) / (self.friend_rej_n + 1))

                        x = np.arange(1, len(self.main_acc_ratios) + 1) * 1500
                        ln_main_acc, = plt.plot(x, self.main_acc_ratios, color='r')
                        ln_friend_acc, = plt.plot(x, self.friend_acc_ratios, color='b')
                        ln_rej_emp_ratio, = plt.plot(x, self.reject_for_empty_ratios, color='g')
                        plt.title('mode:{}, processed order: {}, beta: {}'.format(mode, n, config.beta))
                        plt.legend(handles=[ln_main_acc, ln_friend_acc, ln_rej_emp_ratio],
                                   labels=['main accept', 'friend accept', 'reject for empty'])
                        plt.savefig(os.path.join(config.simulate_fig_folder, 'ratios_' + mode + "_" + str(n) + '.png'))
                        plt.show()

                    self.plot_drivers(n, mode)
                n += 1
        end_time =time.time()
        end = show_info('End')
        run_time = end_time-start_time
        memory = str(end - start) + ' KB'
        print("In mode:", mode)
        self._summary(mode,run_time,memory)
        print()

    def _process_order(self, order, mode):
        main_result = self.main_platform.process_order(order)
        if main_result == -1 or not main_result:
            self.main_rej_n += 1
            use_friend = True
        else:
            self.main_acc_n += 1
            use_friend = False

        if use_friend:
            friend_result = self.friend_platform.process_order(order, mode)
            if friend_result == -1:
                self.friend_rej_n += 1
                self.reject_orders_for_error.append(order)
            elif friend_result:
                self.friend_acc_n += 1
            else:
                self.friend_rej_n += 1
                self.reject_orders_for_empty.append(order)

    def _summary(self, mode,time,memory):
        msgs = [
            'Driver number: {}, MixRatio:{}, beta:{}, Mode: {}\n'.format(config.driver_num, config.driver_mix_ratio,
                                                                         config.beta, mode),
            "Main acc/rej: {} / {}\n".format(self.main_acc_n, self.main_rej_n),
            "Friend acc/rej: {} / {}\n".format(self.friend_acc_n, self.friend_rej_n),
            "Original UC: {}\n".format(self.main_platform.UC + self.friend_platform.sum_pr),
            "New UC: {}\n".format(self.main_platform.UC + self.friend_platform.UC),
            "Improve Rate: {}\n".format((self.main_platform.UC + self.friend_platform.UC)
                                        / (self.main_platform.UC + self.friend_platform.sum_pr)),
            'Reject For Error: {}, {}\n'.format(len(self.reject_orders_for_error),
                                                len(self.reject_orders_for_error) / self.friend_rej_n + 1),
            'Reject For Empty: {}, {}\n'.format(len(self.reject_orders_for_empty),
                                                len(self.reject_orders_for_empty) / self.friend_rej_n + 1),
            'Response time: {}\n'.format(time),
            memory

        ]

        log_file = os.path.join(config.simulate_log_folder, mode + '.txt')
        open(log_file, 'w').writelines(msgs)
        for msg in msgs:
            print(msg, end='')

    def plot_drivers(self, n, mode):
        plt.figure()
        main_drivers = self.main_platform.drivers
        friend_drivers = self.friend_platform.drivers

        main_lng = [driver.longitude for driver in main_drivers]
        main_lat = [driver.latitude for driver in main_drivers]
        main_color = ['g' for _ in main_drivers]

        friend_lng = [driver.longitude for driver in friend_drivers]
        friend_lat = [driver.latitude for driver in friend_drivers]
        friend_color = ['r' for _ in friend_drivers]

        rej_order_lng = [order.start_lng for order in self.reject_orders_for_empty]
        rej_order_lat = [order.start_lat for order in self.reject_orders_for_empty]
        rej_order_color = ['b' for _ in self.reject_orders_for_empty]

        rej_order2_lng = [order.start_lng for order in self.reject_orders_for_error]
        rej_order2_lat = [order.start_lat for order in self.reject_orders_for_error]
        rej_order2_color = ['purple' for _ in self.reject_orders_for_error]

        x = main_lat + friend_lat + rej_order_lat + rej_order2_lat
        y = main_lng + friend_lng + rej_order_lng + rej_order2_lng
        c = main_color + friend_color + rej_order_color + rej_order2_color
        # if len(x) > 5000:
        #     points = np.array(list(zip(x, y, c)))
        #     points = np.random.choice(points, 5000)
        #     x = points[:, 0]
        #     y = points[:, 1]
        #     c = points[:, 2]

        plt.scatter(x, y, c=c, s=1)
        plt.title('mode:{}, processed order: {}, beta: {}'.format(mode, n, config.beta))
        plt.savefig(os.path.join(config.simulate_fig_folder, mode + "_" + str(n) + '.png'))
        plt.show()


class OrderPreprocess():
    def __init__(self, order_file, gps_file, order2len_file, processed_order_file):
        self.order_file = order_file
        self.gps_file = gps_file
        self.order2len_file = order2len_file
        self.processed_order_file = processed_order_file
        self.order2len = dict()

        order_names = ['order_id', 'start_time', 'end_time', 'start_lng', 'start_lat', 'end_lng', 'end_lat']
        self.orders = pd.read_csv(self.order_file, names=order_names)

        # gps_names = ['driver_id', 'order_id', 'time', 'lng', 'lat']
        # print('Reading GPS...')
        # self.gps = pd.concat(tqdm(pd.read_csv(self.gps_file, names=gps_names, chunksize=100000)))

    def run(self):
        # calculate length of order
        for idx, order in tqdm(self.orders.iterrows(), postfix='Calculating order length'):
            order_length = lng_lat_distance((getattr(order, 'start_lat'), getattr(order, 'start_lng')),
                                            (getattr(order, 'end_lat'), getattr(order, 'end_lng')))
            # last_coord = tuple(group.iloc[0][["lat", "lng"]])
            # for coord in zip(group['lat'], group['lng']):
            #     order_length += lng_lat_distance(coord, last_coord)
            #     last_coord = coord
            self.order2len[getattr(order, 'order_id')] = order_length
        pickle.dump(self.order2len, open(self.order2len_file, 'wb'))

        order_lengths = []
        for idx, order in tqdm(self.orders.iterrows()):
            order_id = getattr(order, 'order_id')
            try:
                order_lengths.append(self.order2len[order_id])
            except KeyError:
                order_lengths.append(np.nan)
        self.orders['length'] = order_lengths
        self.orders['person_num'] = np.random.choice([1, 2, 3], len(self.orders),
                                                     replace=True, p=[0.17, 0.45, 0.38])
        self.orders.dropna(inplace=True)
        self.orders.drop_duplicates(['order_id'], inplace=True)
        self.orders.to_csv(self.processed_order_file)

def show_info(start):
     pid = os.getpid()
     p = psutil.Process(pid)
     info = p.memory_full_info()
     memory = info.uss/1024
     return memory

if __name__ == '__main__':
    OrderPreprocess(config.order_file, config.gps_file, config.order2len_file,
                    config.processed_order_file).run()

    mode = 'direction'
    # mode = 'random'
    print("Running in {} mode, driver_n: {}".format(mode, config.driver_num))
    s = Simulator(config.processed_order_file, config.gps_file,
                  config.main_platform_driver_file, config.friend_platform_driver_file)

    s.run(mode)


