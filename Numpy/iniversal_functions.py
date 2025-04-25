import numpy as np
def universal_functions():

    # Create a 1D array
    arr = np.array([1, 2, 3, 4, 5])

    # Apply a universal function (ufunc)
    squared_arr = np.square(arr)

    # Print the original and squared arrays
    print("Original array:", arr)
    print("Squared array:", squared_arr)
    print(np.absolute(arr))

    print(np.exp(arr))

    print(np.max(arr))
    print(np.min(arr))
    print(np.sign(arr))

if __name__ == '__main__':
    universal_functions()
