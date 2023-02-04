import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from OrderMatching.config import config
import matplotlib.pyplot as plt

class DriverGenerator(object):
    def __init__(self):
        self.driver_file = config.driver_file
        self.main_driver_file = config.main_platform_driver_file
        self.friend_driver_file = config.friend_platform_driver_file

    def generate_from_gps(self, sample_time, eps):
        col_names = ['driver_id', 'order_id', 'time', 'longitude', 'latitude']
        csv = pd.concat([x for x in tqdm(pd.read_csv(config.gps_file, names=col_names, chunksize=100000),
                                         postfix='Reading gps file')])
        true_label = np.abs(csv['time'] - sample_time) < eps
        csv = csv[true_label]

        main_drivers = []
        friend_drivers = []

        lngs = []
        lats = []
        for driver, group in csv.groupby(['driver_id']):
            idx = random.randint(0, len(group) - 1)
            lngs.append(group.iloc[idx]['longitude'])
            lats.append(group.iloc[idx]['latitude'])
        coords = list(zip(lngs, lats))

        platforms = self.devide_driver(coords, lats, lngs)

        self.write_driver(coords, friend_drivers, lats, lngs, main_drivers, platforms)

    def write_driver(self, coords, friend_drivers, lats, lngs, main_drivers, platforms):
        driver_ids = range(len(coords))
        print('Writing drivers...', end='')
        with open(self.driver_file, 'w') as f:
            for driver_id, lng, lat, platform in zip(driver_ids, lngs, lats, platforms):
                line = '{} {} {} {}\n'.format(driver_id, lng, lat, platform)
                if platform == 0:
                    main_drivers.append(line)
                else:
                    friend_drivers.append(line)
                f.write(line)
                driver_id += 1
        print('end')
        print("Writing main drivers...", end='')
        open(self.main_driver_file, 'w').writelines(main_drivers)
        print("end")
        print("Writing friend drivers...", end='')
        open(self.friend_driver_file, 'w').writelines(friend_drivers)
        print("end")

    def devide_driver(self, coords, lats, lngs):
        # get platform with k-means
        model = KMeans(n_clusters=2)
        platforms = np.array(model.fit_predict(coords))
        change = np.random.choice([0, -1], size=platforms.size,
                                  p=[1 - config.driver_mix_ratio, config.driver_mix_ratio])
        platforms = np.abs(platforms + change)
        plt.figure()
        color = ["r", "g"]
        colors = [color[platform] for platform in platforms]
        plt.scatter(lats, lngs, s=1,  c=colors)
        plt.show()
        return platforms

    def generate_from_order(self, sample_n):
        order_names = ['order_id', 'start_time', 'end_time', 'start_lng', 'start_lat', 'end_lng', 'end_lat']
        orders = pd.read_csv(config.order_file, names=order_names)
        true_idxs = np.random.randint(0, len(orders), size=sample_n)
        # true_idxs = np.arange(len(orders))

        main_drivers = []
        friend_drivers = []

        lngs = []
        lats = []
        for idx in tqdm(true_idxs, postfix="Getting drivers from order file..."):
            lng = orders.iloc[idx]['end_lng']
            lat = orders.iloc[idx]['end_lat']
            if lng < 10 or lat < 10:
                continue
            lngs.append(lng)
            lats.append(lat)
        coords = list(zip(lngs, lats))

        platforms = self.devide_driver(coords, lats, lngs)

        self.write_driver(coords, friend_drivers, lats, lngs, main_drivers, platforms)


if __name__ == '__main__':
    # DriverGenerator().generate_from_gps(sample_time=1478165684, eps=10)
    DriverGenerator().generate_from_order(config.driver_num)
