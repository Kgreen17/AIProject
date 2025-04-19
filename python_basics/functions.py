def one_ast(*args):
    print(sum(args))

def multiple_args(**kwargs):
    print(sum(kwargs.values()))

def lambda_function(x,y):
    adding = lambda x, y: x + y
    print(adding(x,y))

    lambda_test = lambda x: x*2
    print(lambda_test)
def high_order_function():
    # map()
    numbers = [1, 2, 3, 4, 5]
    squares = list(map(lambda x: x**2, numbers))
    print(squares)

    # filter()
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    print(even_numbers)

    # reduce()
    from functools import reduce
    numbers = [1, 2, 3, 4, 5]
    product = reduce(lambda x, y: x * y, numbers)
    print(product)

    # list comprehension
    numbers = [1, 2, 3, 4, 5]
    squares = [x**2 for x in numbers]
    print(squares)

    # generator expression
    numbers = (x**2 for x in numbers)

    #sorted
    sorted_numbers = sorted(numbers, reverse=True)
    print(sorted_numbers)

    # zip()
    numbers1 = [1, 2, 3]
    numbers2 = [4, 5, 6]
    zipped = list(zip(numbers1, numbers2))
    print(zipped)

    # enumerate()
    my_list = ['apple', 'banana', 'cherry']
    for i, fruit in enumerate(my_list):
        print(i, fruit)

    # set comprehension
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    squares = {x**2 for x in numbers}
    print(squares)

def decorators():
    def decorator_function(func):
        def wrapper_function(*args, **kwargs):
            print('Before function call')
            result = func(*args, **kwargs)
            print('After function call')
            return result
        return wrapper_function

    @decorator_function
    def add(x, y):
        return x + y

    print(add(5, 10))

def generators_yeild():
    def generator_function():
        yield 1
        yield 2
        yield 3

    generator = generator_function()
    print(next(generator))
    print(next(generator))
    print(next(generator))

def pass_statement():
    def pass_statement_function():
        pass

    print('This is a pass statement')

def attributes_function():
    class MyClass:
        def __init__(self):
            self.attribute = 'Hello'

    my_object = MyClass()
    print(my_object.attribute)
    my_object.attribute = 'World'
    print(my_object.attribute)
    del my_object.attribute
    print(hasattr(my_object, 'attribute'))


if __name__ == '__main__':
    generators_yeild()

    # multiple_args(one=1, two=2, three=3)