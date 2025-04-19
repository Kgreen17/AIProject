import numpy as np


def slice():
    random_scaler = np.random.rand(4,4)*10
    print(random_scaler)

    print(np.eye(4, k=2))
    print(np.identity(8))
    print(np.diag([1,2,3,4]))





if __name__ == '__main__':
    slice()