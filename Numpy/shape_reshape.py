import numpy as np

def shape_reshape():
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    list = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1)
            if i+j+k != n]
    print(list)

def arrays(arr):
    int_list = list(map(int, arr))
    rev_list = int_list[::-1]
    return np.array(rev_list)


if __name__ =='__main__':
    # Input: N rows, M columns
    n, m = map(int, input().split())

    # Read the matrix as a flat list of numbers
    data = []
    for _ in range(n):
        data.extend(map(int, input().split()))

    # Convert to NumPy array with shape (n, m)
    arr = np.array(data).reshape(n, m)

    # Transpose and flatten
    print(arr.T)
    print(arr.flatten())
