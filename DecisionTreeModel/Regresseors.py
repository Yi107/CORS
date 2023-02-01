import pickle

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


def save(self, path):
    pickle.dump(self, open(path, 'wb'))


@staticmethod
def load(path):
    pickle.load(open(path, 'rb'))


regressors = [DecisionTreeRegressor, ExtraTreeRegressor]
names = [x.__name__ for x in regressors]
__all__ = names

for regressor in regressors:
    regressor.save = save
    regressor.load = load

# for name, regressor in zip(names, regressors):
#     globals()[name] = type(name, (regressor,), {'save': save, 'load':load})
