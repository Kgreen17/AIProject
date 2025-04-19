
def if_conditions():
    a = 5
    b = 10
    if a < b:
        print("a is less than b")
    elif a == b:
        print("a is equal to b")
    else:
        print("a is greater than b")

def and_or_not_with_conditions():
            a = 5
            b = 10
            if a < b and a > 0:
                print("a is less than b and a is positive")
            elif a > b or a < 0:
                print("a is greater than b or a is not positive")
            else:
                print("a is equal to b and a is positive")

if __name__ == '__main__':
    if_conditions()