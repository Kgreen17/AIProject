
def adding_numbers(a, b):
    a = a
    b = b
    a += b

def subtracting_numbers(a, b):
    a = a
    b = b
    a -= b

def divide_numbers(a, b):

    if b == 0:
        print("Error: Division by zero is not allowed.")

def modulus_numbers(a, b):
    a = a
    b = b
    a %= b

def comparison_operators(a, b):
    print( (5<10) or (10>9))
    print( a == b)
"""
1. Equal to operator
2. Not equal to operator
3. Greater than operator
4. Less than operator
5. Greater than or equal to operator
6. Less than or equal to operator
"""

def identity_operator(a, b):
    # is and is not operators --- more than value. is and is not operators checking for memory location.

    print(a is b)
    print(a is not b)

def membership_operator(a, b):
    # in and not in operators
    

    print(5 in a)
    print(5 not in a)

if __name__ == "__main__":
    identity_operator(5,5)