import pandas as pd

def core_data_structure():
    pd1 = pd.Series([1, 2, 3, 4, 5])
    print(pd1)

def dataframe_2D_table_data():
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'Los Angeles', 'Chicago']
    }

    df = pd.DataFrame(data)
    print(df)

def reading_writing_files():
    df = pd.read_csv('')
    df1 = pd.read_excel('')

if __name__ == '__main__':
    dataframe_2D_table_data()