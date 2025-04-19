import numpy as np

"""1. Introduction to NumPy - Basic Array Operations 
Import NumPy and create a 1D array of integers from 1 to 100. Calculate the sum, minimum,  maximum, and average of this array. """

def np_import():
    # Import NumPy
    import numpy as np

    # Create a 1D array of integers from 1 to 100
    arr = np.arange(1, 101)

    # Calculate the sum, minimum, maximum, and average of the array
    arr_sum = np.sum(arr)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_avg = np.mean(arr)

    # Print the results
    print(f"Sum: {arr_sum}, Min: {arr_min}, Max: {arr_max}, Average: {arr_avg}")

"""
2. What is NumPy & Installation - Verification and Setup 
Write the command to install NumPy via pip. Then, verify its installation by printing the version of the  NumPy library. """
def np_installation():
    # Command to install NumPy via pip
    # pip install numpy

    # Verify installation by printing the version of NumPy
    print(f"NumPy version: {np.__version__}")


"""
3. NumPy Scalars, Vectors, Matrices, and Tensors - Creating Dimensional Structures Create an example of each: a scalar, vector, matrix, and a 3D tensor, 
using NumPy. Print the shape of  each and confirm their dimensions. """
def np_dimensional_structures():
    # Scalar
    scalar = np.array(5)
    print(f"Scalar: {scalar}, Shape: {scalar.shape}, Dimensions: {scalar.ndim}")

    # Vector
    vector = np.array([1, 2, 3, 4, 5])
    print(f"Vector: {vector}, Shape: {vector.shape}, Dimensions: {vector.ndim}")

    # Matrix
    matrix = np.array([[1, 2], [3, 4]])
    print(f"Matrix: \n{matrix}, Shape: {matrix.shape}, Dimensions: {matrix.ndim}")

    # 3D Tensor
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"Tensor: \n{tensor}, Shape: {tensor.shape}, Dimensions: {tensor.ndim}")

"""
4. Why Use NumPy? - Performance Comparison with Python Lists 
Create two lists of one million random integers each and perform element-wise addition using  standard Python lists. Repeat the process with NumPy 
arrays and compare the time taken for each. """
def np_performance_comparison():
    import time

    # Create two lists of one million random integers
    list1 = list(np.random.randint(1, 100, size=1000000))
    list2 = list(np.random.randint(1, 100, size=1000000))

    # Element-wise addition using standard Python lists
    start_time = time.time()
    list_sum = [x + y for x, y in zip(list1, list2)]
    print(f"Time taken for Python lists: {time.time() - start_time} seconds")

    # Create two NumPy arrays of one million random integers
    arr1 = np.random.randint(1, 100, size=1000000)
    arr2 = np.random.randint(1, 100, size=1000000)

    # Element-wise addition using NumPy arrays
    start_time = time.time()
    arr_sum = arr1 + arr2
    print(f"Time taken for NumPy arrays: {time.time() - start_time} seconds")

"""
5. Indexing and Slicing in NumPy - Accessing Subarrays and Elements 
Generate a 6x6 matrix of random integers between 1 and 50. Extract and print the last two rows, the  first two columns, and a 3x3 sub-matrix from the center. """
def np_indexing_slicing():
    # Generate a 6x6 matrix of random integers between 1 and 50
    matrix = np.random.randint(1, 51, size=(6, 6))
    print(f"Original Matrix:\n{matrix}")

    # Extract and print the last two rows
    last_two_rows = matrix[-2:, :]
    print(f"Last Two Rows:\n{last_two_rows}")

    # Extract and print the first two columns
    first_two_columns = matrix[:, :2]
    print(f"First Two Columns:\n{first_two_columns}")

    # Extract and print a 3x3 sub-matrix from the center
    center_sub_matrix = matrix[1:4, 1:4]
    print(f"Center 3x3 Sub-Matrix:\n{center_sub_matrix}")

"""
6. NumPy Size, Shape, ndim, and dtype - Array Attributes Exploration 
Create a 3D array of shape (3, 4, 5) with random float values. Print its size, shape, ndim, and dtype.  Then, change its data type to integer and confirm the new dtype. """
def np_array_attributes():
    # Create a 3D array of shape (3, 4, 5) with random float values
    array_3d = np.random.rand(3, 4, 5)
    print(f"Original Array:\n{array_3d}")

    # Print its size, shape, ndim, and dtype
    print(f"Size: {array_3d.size}, Shape: {array_3d.shape}, Dimensions: {array_3d.ndim}, Data Type: {array_3d.dtype}")

    # Change its data type to integer
    array_3d_int = array_3d.astype(int)
    print(f"New Data Type: {array_3d_int.dtype}")

"""
7. Changing Data Types in NumPy - Type Conversion Using astype() and datetime64 Create an array of integers from 1 to 10 and convert it to float using astype(). 
Then, create an array  of date strings and convert them to datetime64 format. """
def np_data_type_conversion():
    # Create an array of integers from 1 to 10
    int_array = np.arange(1, 11)
    print(f"Integer Array: {int_array}")

    # Convert it to float using astype()
    float_array = int_array.astype(float)
    print(f"Float Array: {float_array}")

    # Create an array of date strings
    date_strings = np.array(['2023-01-01', '2023-02-01', '2023-03-01'])
    print(f"Date Strings: {date_strings}")

    # Convert them to datetime64 format
    date_array = date_strings.astype('datetime64[D]')
    print(f"Datetime64 Array: {date_array}")

"""
8. Slicing in NumPy Matrices - Extracting Matrix Sections 
Create a 5x5 matrix filled with integers from 10 to 34. Slice and display the following parts: • Top-left 3x3 submatrix 
• Last two rows 
• The diagonal elements """
def np_matrix_slicing():
    # Create a 5x5 matrix filled with integers from 10 to 34
    matrix = np.arange(10, 35).reshape(5, 5)
    print(f"Original Matrix:\n{matrix}")

    # Top-left 3x3 submatrix
    top_left_submatrix = matrix[:3, :3]
    print(f"Top-left 3x3 Submatrix:\n{top_left_submatrix}")

    # Last two rows
    last_two_rows = matrix[-2:, :]
    print(f"Last Two Rows:\n{last_two_rows}")

    # Diagonal elements
    diagonal_elements = np.diag(matrix)
    print(f"Diagonal Elements: {diagonal_elements}")

"""
9. NumPy Random - Generating Randomized Matrices 
Generate a 4x4 matrix of random integers between 5 and 15. Also, create a 2x5 matrix with a normal  distribution of random float numbers. Display both matrices. """
def np_random_matrices():
    # Generate a 4x4 matrix of random integers between 5 and 15
    int_matrix = np.random.randint(5, 16, size=(4, 4))
    print(f"Random Integer Matrix:\n{int_matrix}")

    # Create a 2x5 matrix with a normal distribution of random float numbers
    float_matrix = np.random.normal(loc=0.0, scale=1.0, size=(2, 5))
    print(f"Random Float Matrix:\n{float_matrix}")

"""
10. NumPy arange Method - Creating and Manipulating Ranges 
Use arange to create an array of values from 0 to 50 with a step of 5. Then, use this array to find the  square root of each element, rounding each result to 2 decimal places."""
def np_arange():
    # Use arange to create an array of values from 0 to 50 with a step of 5
    arr = np.arange(0, 51, 5)
    print(f"Array: {arr}")

    # Find the square root of each element, rounding each result to 2 decimal places
    sqrt_arr = np.round(np.sqrt(arr), 2)
    print(f"Square Root Array: {sqrt_arr}")

"""
11. NumPy linspace Method - Evenly Spaced Arrays 
Create an array with 15 evenly spaced values between 10 and 20 using linspace. Find the sum of all  elements and the cumulative sum across the array."""
def np_linspace():
    # Create an array with 15 evenly spaced values between 10 and 20 using linspace
    arr = np.linspace(10, 20, num=15)
    print(f"Array: {arr}")

    # Find the sum of all elements
    total_sum = np.sum(arr)
    print(f"Total Sum: {total_sum}")

    # Find the cumulative sum across the array
    cumulative_sum = np.cumsum(arr)
    print(f"Cumulative Sum: {cumulative_sum}")

"""
12. Using the NumPy Reshape Method - Reshaping Arrays 
Generate an array of 24 sequential numbers and reshape it into a 4x6 matrix. Then reshape it again  into a 3D array with dimensions (2, 3, 4) and print each new shape. """
def np_reshape():
    # Generate an array of 24 sequential numbers
    arr = np.arange(24)
    print(f"Original Array: {arr}")

    # Reshape it into a 4x6 matrix
    reshaped_matrix = arr.reshape(4, 6)
    print(f"Reshaped Matrix (4x6):\n{reshaped_matrix}")

    # Reshape it again into a 3D array with dimensions (2, 3, 4)
    reshaped_tensor = arr.reshape(2, 3, 4)
    print(f"Reshaped Tensor (2x3x4):\n{reshaped_tensor}")

"""
13. Using the NumPy View and Copy Methods - Understanding Memory Sharing Create a 1D array from 1 to 10. Use view to create a new array from it and modify 
the first element in  the view. Then use copy to create a separate array, modify its first element, and observe the  differences. """
def np_view_copy():
    # Create a 1D array from 1 to 10
    original_array = np.arange(1, 11)
    print(f"Original Array: {original_array}")

    # Use view to create a new array from it
    view_array = original_array.view()
    view_array[0] = 99
    print(f"Modified View Array: {view_array}")
    print(f"Original Array after modifying view: {original_array}")

    # Use copy to create a separate array
    copy_array = original_array.copy()
    copy_array[0] = 88
    print(f"Modified Copy Array: {copy_array}")
    print(f"Original Array after modifying copy: {original_array}")

"""
14. Using the NumPy Ones, Zeros, and Full Methods - Initializing Arrays 
Generate a 3x3 array filled with ones, a 4x4 array filled with zeros, and a 2x5 array filled with the  value 7. Display each array. """
def np_ones_zeros_full():
    # Generate a 3x3 array filled with ones
    ones_array = np.ones((3, 3))
    print(f"3x3 Array of Ones:\n{ones_array}")

    # Generate a 4x4 array filled with zeros
    zeros_array = np.zeros((4, 4))
    print(f"4x4 Array of Zeros:\n{zeros_array}")

    # Generate a 2x5 array filled with the value 7
    full_array = np.full((2, 5), 7)
    print(f"2x5 Array filled with 7:\n{full_array}")

"""
15. NumPy Diagonal Methods - Working with eye, diag, and identity 
Create a 5x5 identity matrix using eye. Then, create a 4x4 matrix with the diag method, setting the  diagonal elements to 3. Extract the diagonal 
elements from this matrix and print them. """
def np_diagonal_methods():
    # Create a 5x5 identity matrix using eye
    identity_matrix = np.eye(5)
    print(f"5x5 Identity Matrix:\n{identity_matrix}")

    # Create a 4x4 matrix with the diag method, setting the diagonal elements to 3
    diag_matrix = np.diag([3, 3, 3, 3])
    print(f"4x4 Diagonal Matrix:\n{diag_matrix}")

    # Extract the diagonal elements from this matrix
    diagonal_elements = np.diag(diag_matrix)
    print(f"Diagonal Elements: {diagonal_elements}")

"""
16. NumPy Aggregation and Statistics - Analyzing Data Arrays 
Create a 10x5 matrix of random integers from 1 to 100. Calculate the mean, median, standard  deviation, and sum for each column and row. """
def np_aggregation_statistics():
    # Create a 10x5 matrix of random integers from 1 to 100
    matrix = np.random.randint(1, 101, size=(10, 5))
    print(f"Original Matrix:\n{matrix}")

    # Calculate the mean, median, standard deviation, and sum for each column
    col_mean = np.mean(matrix, axis=0)
    col_median = np.median(matrix, axis=0)
    col_std = np.std(matrix, axis=0)
    col_sum = np.sum(matrix, axis=0)

    print(f"Column Mean: {col_mean}")
    print(f"Column Median: {col_median}")
    print(f"Column Standard Deviation: {col_std}")
    print(f"Column Sum: {col_sum}")

    # Calculate the mean, median, standard deviation, and sum for each row
    row_mean = np.mean(matrix, axis=1)
    row_median = np.median(matrix, axis=1)
    row_std = np.std(matrix, axis=1)
    row_sum = np.sum(matrix, axis=1)

    print(f"Row Mean: {row_mean}")
    print(f"Row Median: {row_median}")
    print(f"Row Standard Deviation: {row_std}")
    print(f"Row Sum: {row_sum}")

"""
17. NumPy Sorting and Searching - Organizing and Finding Data 
Generate a random array of 20 integers from 1 to 100. Sort the array in ascending and descending  order. Find and print the index of the number 50 in the sorted array 
(if present). """
def np_sorting_searching():
    # Generate a random array of 20 integers from 1 to 100
    arr = np.random.randint(1, 101, size=20)
    print(f"Original Array: {arr}")

    # Sort the array in ascending order
    sorted_asc = np.sort(arr)
    print(f"Sorted Array (Ascending): {sorted_asc}")

    # Sort the array in descending order
    sorted_desc = np.sort(arr)[::-1]
    print(f"Sorted Array (Descending): {sorted_desc}")

    # Find and print the index of the number 50 in the sorted array (if present)
    index_50 = np.where(sorted_asc == 50)[0]
    if index_50.size > 0:
        print(f"Index of 50 in Sorted Array: {index_50[0]}")
    else:
        print("50 is not present in the array.")

"""
18. NumPy Vectorized Operations - Arithmetic with Arrays 
Create two 3x3 matrices of random integers from 1 to 10. Perform and display element-wise  addition, subtraction, division, multiplication, and modulus between them. """
def np_vectorized_operations():
    # Create two 3x3 matrices of random integers from 1 to 10
    matrix1 = np.random.randint(1, 11, size=(3, 3))
    matrix2 = np.random.randint(1, 11, size=(3, 3))
    print(f"Matrix 1:\n{matrix1}")
    print(f"Matrix 2:\n{matrix2}")

    # Perform and display element-wise addition
    addition = matrix1 + matrix2
    print(f"Element-wise Addition:\n{addition}")

    # Perform and display element-wise subtraction
    subtraction = matrix1 - matrix2
    print(f"Element-wise Subtraction:\n{subtraction}")

    # Perform and display element-wise division
    division = matrix1 / matrix2
    print(f"Element-wise Division:\n{division}")

    # Perform and display element-wise multiplication
    multiplication = matrix1 * matrix2
    print(f"Element-wise Multiplication:\n{multiplication}")

    # Perform and display element-wise modulus
    modulus = matrix1 % matrix2
    print(f"Element-wise Modulus:\n{modulus}")

"""
19. Filtering and Logical Operators in NumPy Matrices - Applying Conditions Create a 5x5 matrix of random integers from 10 to 99. Use Boolean indexing 
to filter out and replace  all values greater than 50 with 0. Display the modified matrix.
"""
def np_filtering_logical_operators():
    # Create a 5x5 matrix of random integers from 10 to 99
    matrix = np.random.randint(10, 100, size=(5, 5))
    print(f"Original Matrix:\n{matrix}")

    # Use Boolean indexing to filter out and replace all values greater than 50 with 0
    matrix[matrix > 50] = 0
    print(f"Modified Matrix:\n{matrix}")

if __name__=='__main__':
    np_filtering_logical_operators()