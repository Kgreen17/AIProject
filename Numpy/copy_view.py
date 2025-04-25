import numpy as np

def copy_view():

    np1 = np.array([[1, 2, 3], [4, 5, 6]])
    np2 = np1.view()
    np3 = np1.copy()



if __name__ =='__name__':
    copy_view()
