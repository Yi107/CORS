import os


class Config(object):
    def __init__(self, ddl):
        self.OIL_PRICE: float = 0.4 #The penalty setting of the order.
        self.ddl = ddl
        self.max_lng = 0
        self.min_lng = 0
        self.max_lat = 0
        self.min_lat = 0
        self.driver_speed = 60 / 3.6  # m/s.
        self.pr = "price"
        self.alpha = 1
        self.beta = 0.7
        self.gamma = 1.5
        self.R = 6371004  # earth radian, m.
        self.EPS = 1e-6
        self.simulate_day = '06' #The gps file of the simulated day.
        self.driver_num = 20
        self.driver_mix_ratio = 0.3

        #os.makedirs(self.data_folder, exist_ok=True)
        #The path of order file, gps file, and the generated data of the simulate day.
        self.order_file = 'order_201611{}'.format(self.simulate_day)
        self.gps_file = 'gps_201611{}'.format(self.simulate_day)
        self.order2len_file = 'order2len_201611{}'.format(self.simulate_day)
        self.order2vector_file = 'order2vec_201611{}'.format(self.simulate_day)
        self.processed_order_file = 'processed_order_201611{}'.format(self.simulate_day)
        #The path of the driver files of two platforms
        self.main_platform_driver_file = './synData/driver_201611061000p1.txt'
        self.friend_platform_driver_file = './synData/driver_201611061000p2.txt'
        #The path of the generated decision tree model.
        self.decision_tree_path = 'decision_tree.model'
        #The path of the figures of simulate results.
        self.simulate_fig_folder = './result/{}/{}_{}_{}_{}/fig/'. \
            format(self.simulate_day, self.driver_num, self.driver_mix_ratio,
                   self.beta, self.gamma)
        os.makedirs(self.simulate_fig_folder, exist_ok=True)
        #The path of the simulate results.
        self.simulate_log_folder = './result/{}/{}_{}_{}_{}/log/'. \
            format(self.simulate_day, self.driver_num, self.driver_mix_ratio,
                   self.beta, self.gamma)
        os.makedirs(self.simulate_log_folder, exist_ok=True)

#The setting of the deadline of every orders.
ddl_list = [x * 60 for x in [5, 10, 15, 20, 25]]
config = Config(ddl=ddl_list[3])
