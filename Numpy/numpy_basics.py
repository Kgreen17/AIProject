import numpy as np

def use_numpy():

    list1 = [1, 2, 3]
    print(list1)
    print(list[1])

    list2 = ["test", 1,2,3,True]
    print(list2)

    np1 = np.array([1, 2, 3])
    print(np1)
    print(type(np1))
    print(np1[0])
    print(np1.shape)

    np2 = np.arange(0,10,2)
    print(np2)

    np3 = np.zeros(10)
    print(np3)

    np4 = np.zeros((2,10))
    print(np4)

    np5 = np.full((10),6)
    print(np5)

    np6 = np.full((4,10),5)
    print(np6)

    list2 = [1, 2, 3]
    np7 = np.array(list2)
    print(np7)

    print(np7[1])


if __name__ == '__main__':
    use_numpy()