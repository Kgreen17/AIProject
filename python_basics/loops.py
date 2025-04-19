
def for_loop():
    height = 15
    for i in range(height):
        print(' '*(height-i-1)+"*" * (2*i+1))

    for i in range(4):
        print(i, end=' ')

def practice():
    str = input("Enter a string")
    for char in str:
        print(char.upper())

    strInt = int(input("Enter a number"))
    for i in range(strInt, -1, -1):
        print(i)

def while_loop():
    i = 0
    while i < 10:
        print(i)
        i += 1

    i = 0
    while True:
        print(i)
        i += 1
        if i == 10:
            break

    i = 0
    while i < 10:
        if i % 2 == 0:
            continue
        print(i)
        i += 1

    i = 0
    while i < 10:
        i += 1
        if i == 5:
            break
        print(i)
    # else with while
    else:
        print("Loop ended without breaking")

def nested_loops():
    for i in range(1,4):
        for j in range(1, i+1):
            print('*', end='')
        print()

def solution(str, b):
   if b>0:
       print(str[:3]*b)
   else:
       print('the integer is not positive')


def solution():
    sqr = [x*y for x in range(1,4) for y in range(1,4)]
    print(sqr)

def enumerate1():
    my_list = ['apple', 'banana', 'cherry']
    for i, fruit in enumerate(my_list):
        print(i+1, fruit)

def zipp():
    names = ['apple', 'banana', 'cherry']
    ages = [20, 15, 10, 11]

    for name, age in zip(names,ages):
        print(f'Name: {name}, Age: {age}')


if __name__ == '__main__':
    zipp()