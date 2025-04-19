

def set_operation():
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}

    print(set1.union(set2))
    print(set1.intersection(set2))
    print(set1.difference(set2))
    print(set1.symmetric_difference(set2))

def set_iterator():
    set1 = {1, 2, 3, 4, 5}
    for i in set1:
        print(i)

def set_frozenset():
    frozenset1 = frozenset({1, 2, 3, 4, 5})
    print(frozenset1)
    for i in frozenset1:
        print(i)

def set_practice():
    set_name = set()
    print(type(set_name))

    number_set = {1,2,3,4,5,6,7,8,9}
    print(number_set)

    set_range = set(range(1, 11))
    print(set_range)

    set_mix = {1,2,3,4,5, "apple", "orange"}
    print("mix:", set_mix)
    print("length:", len(set_mix))
    print("apple", 'apple' in set_mix)
    for i in set_mix:
        print(i)

    set_mix.add("testFruit")
    print("mix after add:", set_mix)

    # set_mix.update("testFruit")
    # print("mix after update:", set_mix)

    set_mix.remove("apple")
    print("mix after remove:", set_mix)

    set_mix.discard("banana")
    print("mix after discard:", set_mix)

    set_mix.clear()
    print("mix after clear:", set_mix)

    animal_set = {"dog", "cat", "catfish"}
    print(sorted(animal_set))

if __name__ == '__main__':
    set_practice()