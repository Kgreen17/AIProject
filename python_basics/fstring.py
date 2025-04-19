
def fString():
    name = "John Doe"
    age = 30
    print(f"My name is {name}, and I am {age} years old.")

def fstring2(name, last_name):
    # name = "Kevin"
    # last_name = "Green"
    print(f"my name is {name}, and my last name is {last_name}.")

    first_name = "john"
    last_name = "no"
    greeting = "hello, {}, {} what the hell are on".format(name, last_name)
    print(greeting)

    age = 30

    greet2 = "hello you there %s and I thin you are %d years old" % (name, age)
    print(greet2)

    floatt = 3.1423456
    print("pi is approximately " + f"{floatt:.3f}")

    distance = 5.678
    print(f"you run {distance:.2f} miles today")

    apples = 10
    oranges = 15
    print(f"you have {apples} apples and {oranges} oranges so you got the total of {apples+oranges} fruits")

    namess = "jordan"
    print("welcome {}".format(namess))

    char = "A"
    charnum = 45
    print(chr(charnum))
    print(ord(char))

    print("hello "*10)
if __name__ == '__main__':
    fstring2("kevin", "Greenish")