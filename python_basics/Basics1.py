# single line comment

"""
this is multi-line comment
snake_case for variables, pascalCase for classes, snake_case for functions.
"""


def operators():
    str = "Hello, World! I know how to use Python operators."
    print(str.count("q"))  # True
    # replace()
    text = "Good morning, everyone!"
    text2 = text.replace("Goo", "Foo")
    print(text2)  # Foo morning, everyone!
    # join()
    words = ["hello", "you", "me"]
    print(" ".join(words))
    # split(), rsplit(), splitlines()
    # zfill()
    print("222".zfill(5))
    #center(), ljust(), rjust()
    print("python".center(8,"-"))
    print("tst".ljust(10,"-"))

    test = "123heelllo"
    print(test.isdigit())
    print(test.isalpha())
    print(test.isalnum())
    print(test.isspace())
    print(test.islower())
    print(test.isupper())

    print("I like to {} go to home and fix my {}.".format("mean","test"))




if __name__ == '__main__':
    operators()

