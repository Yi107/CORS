import pickle

import numpy as np
import pandas as pd
from main import GenerateData
from Regresseors import DecisionTreeRegressor
from config import config
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    decision_tree = DecisionTreeRegressor()
    data = GenerateData()
    print('Training decision tree...', end='')
    decision_tree.fit(data.train_x, data.train_y)
    print('end. \nEvaluating...', end='')
    y_ = decision_tree.predict(data.test_x)
    mse = np.abs((y_ - data.test_y)) % 180
    avg_mse = np.mean(mse)
    print('loss is', avg_mse)
    print(mse.sort_values())
    mse.sort_values().to_csv('error')
    pickle.dump(decision_tree, open(config.decision_tree_path, 'wb'))

    sns.kdeplot(mse,clip=(0, 180.0),bw_adjust=0.1)

    matplotlib.pyplot.show()