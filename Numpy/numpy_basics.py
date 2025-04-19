import numpy as np

def use_numpy():

    scale = np.array(5)
    print(scale)

    arr = np.array([1,2,3,4])
    print(arr)

    matrix = np.array([[1,2],[5,6]])
    print(matrix)

    tension = np.array([[[1,2],[3,4],[5,6]]])
    print(tension)
    print(tension.ndim)
    print(tension.shape)
    print(tension.dtype)

    print(np.__version__)


if __name__ == '__main__':
    use_numpy()