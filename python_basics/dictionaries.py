
def create_dictionary():
    dictionary = {}
    print("Creating dictionary", dictionary)
    dictionary['name'] = 'John'
    dictionary['age'] = 25
    print("Dictionary after adding elements", dictionary)

    car1 = dict(brand='tesla', model='modelS')
    print("Car 1 dictionary:", car1)

def access_dictionary_items():
    dictionary = {'name': 'John', 'age': 25}
    print("Name:", dictionary['name'])
    print("Age:", dictionary.get('age', 0))

    car1 = dict(brand='tesla', model='modelS')
    print("Car 1 brand:", car1['brand'])
    print("Car 1 model:", car1.get('model', 'Unknown'))

    print("Get all the keys",dictionary.keys())
    print("Get all the values", dictionary.values())
    print("Get all the key-value pairs", dictionary.items())

    print(dictionary.get('test'))

def changing_adding_item_dictionary():
    dictionary = {'name': 'John', 'age': 25}
    dictionary['name'] = 'Jane'
    print("Dictionary after changing name:", dictionary)

    dictionary['city'] = 'New York'
    print("Dictionary after adding city:", dictionary)

    dictionary.update({'address': '123 Main St', 'hobbies': ['reading', 'painting']})
    print("Dictionary after updating address and hobbies:", dictionary)

    del dictionary['age']
    print("Dictionary after deleting age:", dictionary)

def remove_dictionart_item():
    dictionary = {'name': 'John', 'age': 25, 'city': 'New York'}
    print("Dictionary before removing item:", dictionary)
    del dictionary['city']
    print("Dictionary after removing city:", dictionary)
    dictionary.pop('age')
    print("Dictionary after removing age:", dictionary)

    dictionary.popitem()
    print("Dictionary after removing last item:", dictionary)

    dictionary.clear()
    print("Dictionary after clearing all items:", dictionary)

if __name__ == '__main__':
    remove_dictionart_item()