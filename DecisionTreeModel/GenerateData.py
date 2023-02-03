from tqdm import tqdm
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from config import config


class GenerateData:
    def __init__(self, trust_files: bool = True):

        self.order_data_files = config.order_data_files
        self.gps_data_files = config.gps_data_files
        self.order2vector_files = config.order2vector_data_files
        self.data_path = config.data_path
        self.train_data_ratio = config.train_data_ratio
        self.train_x_path = config.train_x_path
        self.train_y_path = config.train_y_path
        self.test_x_path = config.test_x_path
        self.test_y_path = config.test_y_path

        self.trust_files = trust_files

  
    def generate_order(self):
        print('Generate Order Set')
        gps_labels = ['driver_id', 'order_id', 'time', 'lng', 'lat']
        order_labels = ['order_id', 'start_t', 'end_t', 'start_lng', 'start_lat', 'end_lng', 'end_lat']

        dataloader = pd.read_csv(self.gps_file, chunksize=1000000, header=0, names=gps_labels)
        gps_data = pd.concat([chunk for chunk in dataloader])
        orders = []
        vectors_info = []
        for order_id, group in gps_data.groupby(['order_id']):
            vector_info = [104.07513, 30.72724, 104.08513, 30.72724]
            if len(group) == 1:
                vector_info = [group['order_id'].array[0], group['time'].array[0], group['time'].array[0],
                               group['lng'].array[0], group['lat'].array[0],
                               group['lng'].array[0], group['lat'].array[0]]
            if len(group) >= 2:
                vector_info = [group['order_id'].array[0], group['time'].array[0], group['time'].array[len(group) - 1],
                               group['lng'].array[0], group['lat'].array[0],
                               group['lng'].array[len(group) - 1], group['lat'].array[len(group) - 1]]
            vectors_info.append(vector_info)
        vectors_info = pd.DataFrame(vectors_info,
                                    columns=order_labels)
        vectors_info.to_csv(self.order_file)

    def __get_bearing(self, order_csv: pd.DataFrame):
        """
        Calculate vectors of orders in order_csv using the algorithm of ox.bearing.get_bearing.
        Args:
            order_csv (pd.DataFrame): The csv file that stores order info.
        """
        start_lng = np.radians(order_csv["start_lng"])
        start_lat = np.radians(order_csv["start_lat"])
        end_lng = np.radians(order_csv["end_lng"])
        end_lat = np.radians(order_csv["end_lat"])
        diff_lng = np.radians(end_lng - start_lng)
        x = np.sin(diff_lng) * np.cos(end_lat)
        y = np.cos(start_lat) * np.sin(end_lat) \
            - np.sin(start_lat) * np.cos((end_lat) * np.cos(diff_lng))
        init_bearing = np.arctan2(x, y)
        init_bearing = np.degrees(init_bearing)
        vector = (init_bearing + 360) % 360
        return vector

    def calculate_order_start_vector(self):
        print('Calculating start_vector of orders')
        gps_labels = ['driver_id', 'order_id', 'time', 'lng', 'lat']
        for order2vector_file, gps_file in \
                tqdm(zip(self.order2vector_files, self.gps_data_files)):
            dataloader = pd.read_csv(gps_file, chunksize=1000000, header=0, names=gps_labels)
            gps_data = pd.concat([chunk for chunk in dataloader])
            orders = []
            vectors_info = []
            for order_id, group in gps_data.groupby(['order_id']):
                vector_info = [104.07513, 30.72724, 104.08513, 30.72724]
                if len(group) >= 2:
                    for i in range(len(group) - 1):
                        if group['lat'].array[i] == group['lat'].array[i + 1] and \
                                group['lng'].array[i] == group['lng'].array[i + 1]:
                            continue
                        else:
                            vector_info = [group['lng'].array[i], group['lat'].array[i],
                                           group['lng'].array[i + 1], group['lat'].array[i + 1]]
                            break
                vectors_info.append(vector_info)
                orders.append(order_id)
            vectors_info = pd.DataFrame(vectors_info,
                                        columns=['start_lng', 'start_lat', 'end_lng', 'end_lat'])
            vectors = self.__get_bearing(vectors_info)
            order2vector = pd.DataFrame(zip(orders, vectors),
                                        columns=['order_id', 'start_vector'])
            order2vector.to_csv(order2vector_file)

    @property
    def data(self):
        try:
            return getattr(self, '_data')
        except AttributeError:
            if self.trust_files and os.path.exists(self.data_path):
                self._data = pd.read_csv(self.data_path)
            else:

                print('Failed to load data at {}, calculating it instead...'
                      .format(self.data_path))
                order_names = ['order_id', 'start_t', 'end_t', 'start_lng',
                               'start_lat', 'end_lng', 'end_lat']
                o2v_names = ['order_id', 'start_vector']
                all_csv = []
                for order2vector_file, order_file in \
                        tqdm(zip(self.order2vector_files, self.order_data_files)):
                    order_csv = pd.read_csv(order_file, names=order_names)
                    if not os.path.exists(order2vector_file):
                        print('Failed to load order2vector file at {}. Calculating it instead...'.format(
                            order2vector_file))
                        self.calculate_order_start_vector()
                    order2vector = pd.read_csv(order2vector_file)
                    order_csv['vector'] = self.__get_bearing(order_csv)
                    f = pd.merge(left=order_csv, right=order2vector, how='inner',
                                 left_on='order_id', right_on='order_id')
                    all_csv.append(f)
                print('end.')
                all_csv = pd.concat(all_csv)
                all_csv.to_csv(self.data_path)
                self._data = all_csv
            return self._data

    def _split_train_test_data(self):
        feature_columns = ['start_t', 'start_lng', 'start_lat', 'start_vector']
        # feature_columns = ['start_lng', 'start_lat', 'start_vector']
        label_column = 'vector'
        x = self.data[feature_columns]
        y = self.data[label_column]
        self._train_x, self._test_x, self._train_y, self._test_y = \
            train_test_split(x, y, test_size=1 - self.train_data_ratio)
        pickle.dump(self._train_x, open(self.train_x_path, 'wb'))
        pickle.dump(self._train_y, open(self.train_y_path, 'wb'))
        pickle.dump(self._test_x, open(self.test_x_path, 'wb'))
        pickle.dump(self._test_y, open(self.test_y_path, 'wb'))

    @property
    def train_x(self):
        try:
            return getattr(self, '_train_x')
        except AttributeError:
            if self.trust_files and os.path.exists(self.train_x_path):
                self._train_x: pd.DataFrame = pickle.load(open(self.train_x_path, 'rb'))
            else:
                # self._train_x, self._train_y, self._test_y, self._test_y are set here
                self._split_train_test_data()
            return self._train_x

    @property
    def train_y(self):
        try:
            return getattr(self, '_train_y')
        except AttributeError:
            if self.trust_files and os.path.exists(self.train_y_path):
                self._train_y: pd.DataFrame = pickle.load(open(self.train_y_path, 'rb'))
            else:
                # self._train_x, self._train_y, self._test_y, self._test_y are set here
                self._split_train_test_data()
            return self._train_y

    @property
    def test_x(self):
        try:
            return getattr(self, '_test_x')
        except AttributeError:
            if self.trust_files and os.path.exists(self.test_x_path):
                self._test_x: pd.DataFrame = pickle.load(open(self.test_x_path, 'rb'))
            else:
                # self._train_x, self._train_y, self._test_y, self._test_y are set here
                self._split_train_test_data()
            return self._test_x

    @property
    def test_y(self):
        try:
            return getattr(self, '_test_y')
        except AttributeError:
            if self.trust_files and os.path.exists(self.test_y_path):
                self._test_y: pd.DataFrame = pickle.load(open(self.test_y_path, 'rb'))
            else:
                # self._train_x, self._train_y, self._test_y, self._test_y are set here
                self._split_train_test_data()
            return self._test_y


if __name__ == '__main__':
    data = GenerateData()
    data._split_train_test_data()
