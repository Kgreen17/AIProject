
def set_list():
    list1 = [1, 2, 3, 4, 5,3]
    print(list1)
    list1.append(7)
    print(list1)
    print(len(list1))
    print(list1.count(3))
    print(list1[0:3])
    my_list = [1,2]
    my_list.insert(1,3)
    print(my_list)

def remove_items():
    my_list = [1, 2, 3, 4, 5, 3]
    my_list.remove(3)
    print(my_list)
    my_list.pop(1)
    print(my_list)
    del my_list[0:2]
    print(my_list)
    my_list.clear()
    print(my_list)

def sort_list():
    my_list = [1, 2, 4, 5, 3]
    my_list.sort()
    print(my_list)
    my_list.reverse()
    print(my_list)
    my_list.sort(reverse=True)
    print(my_list)
    my_list.sort(key=lambda x: x % 2)
    print(my_list)

def list_functions():
    my_list = [1, 2, 3, 4, 5]
    my_list.append(7)
    print(my_list)
    print(len(my_list))
    print(my_list.count(3))
    print(my_list[0:3])
    my_list.insert(1, 3)
    print(my_list)
    my_list.remove(3)
    print(my_list)
    my_list.pop(1)
    print(my_list)
    del my_list[0:2]
    print(my_list)
    my_list.clear()
    print(my_list)
    




if __name__ == "__main__":
    sort_list()