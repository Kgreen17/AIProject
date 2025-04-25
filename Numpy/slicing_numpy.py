import numpy as np


def slice():
    np1 = np.array([1, 2, 3, 4, 5])
    print(np1)

    # Slicing
    print(np1[1:3])  # [1 2]

    # Slicing with step
    print(np1[::2])  # [1 3 5]

    print(np1[-3:-1])

    np2 = np.array([[1,2,3,4,5],
                    [6,7,8,9,10]])
    print(np2[1,2])
    print(np2[0:1, 1:3]) # [[2 3]]
    print(np2[0:2, 1:3])





if __name__ == '__main__':
    slice()