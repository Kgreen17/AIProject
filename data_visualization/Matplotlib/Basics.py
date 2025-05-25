from matplotlib import pyplot as plt
import numpy as np
import csv
from matplotlib.lines import lineStyles
from collections import Counter
import pandas as pd


def create_customized_plot(ages_x, py_dev_y, js_dev_y, dev_y):

  # plt.style.use("fivethirtyeight")
  plt.xkcd()
  plt.plot(ages_x, dev_y, color='#5a7d9a', label="all developer")

  plt.plot(ages_x, js_dev_y,  label="js developer", )

  plt.plot(ages_x, py_dev_y, color='#adad3b', linewidth=4, linestyle="--",label="python developer")


  plt.title("Python Developer Salaries")
  plt.xlabel("Ages")
  plt.ylabel("Salaries")

  plt.legend()

  # plt.grid(True)
  plt.tight_layout()

  # plt.savefig("plot.png", dpi=300)

  plt.show()

def bar_charts_analyzing_data(ages_x, py_dev_y, js_dev_y, dev_y):

  plt.style.use("fivethirtyeight")
  x_indexes = np.arange(len(ages_x))
  width = 0.25

  plt.bar(x_indexes-width, dev_y, width = width, color='#444444', label="all developer")

  plt.bar(x_indexes, py_dev_y, width = width, color='#008fd5', label="python developer")

  plt.bar(x_indexes+width, js_dev_y, width = width, color="#e5ae38",label="js developer", )

  plt.xticks(ticks=x_indexes, labels=ages_x)
  plt.title("Python Developer Salaries")
  plt.xlabel("Ages")
  plt.ylabel("Salaries")

  plt.legend()

  plt.tight_layout()

  plt.show()

def bar_charts_from_csv():
    plt.style.use("fivethirtyeight")

    # with open('data.csv') as csv_file:
    #   csv_reader = csv.DictReader(csv_file)
    data=pd.read_csv('data.csv')
    ids = data['Responder_id']
    lang_responses = data['LanguagesWorkedWith']

    language_counter = Counter()

    for response in lang_responses:
      language_counter.update(response.split(';'))

    languages = []
    popularity = []

    for item in language_counter.most_common(15):
      languages.append(item[0])
      popularity.append(item[1])

    languages.reverse()
    popularity.reverse()

    plt.barh(languages, popularity, color="#444444")
    plt.title("Most Popular Languages")
    # plt.ylabel("Languages")
    plt.xlabel("Popularity")

    plt.tight_layout

    plt.show()




def pie_charts():
  plt.style.use("fivethirtyeight")

  slices = [59219, 55466, 47544, 36443, 35917]
  labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
  explode = [0,0,0,0,0.1]


  plt.pie(slices, labels = labels,explode = explode, shadow=True, startangle=90, autopct= '%1.1f%%',wedgeprops={'edgecolor': 'black'})


  plt.title("Pie Chart")
  plt.tight_layout
  plt.show()



def stack_plots():
    plt.style.use("fivethirtyeight")

    minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    player1 = [1, 2, 3, 3, 4, 4, 4, 4, 5]
    player2 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
    player3 = [1, 1, 1, 2, 2, 2, 3, 3, 3]

    plt.stackplot(minutes, player1, player2, player3, labels=['Player 1', 'Player 2', 'Player 3'], colors=['#6d904f', '#fc4f30', '#008fd5'], alpha=0.8)

    plt.legend(loc=(0.07, 0.05), fontsize=14)

    plt.title("stack plots")
    plt.tight_layout()
    plt.show()


def filling_area():
    plt.style.use("fivethirtyeight")

    x = np.arange(0, 10, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.fill_between(x, y1, y2, where=(y1 > y2), color='green', alpha=0.5, label='y1 > y2')
    plt.fill_between(x, y1, y2, where=(y1 <= y2), color='red', alpha=0.5, label='y1 <= y2')

    plt.title("Filling Area Between Curves")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.tight_layout()
    plt.show()

def histogram():
    plt.style.use("fivethirtyeight")
    data = np.random.randn(1000)
    plt.hist(data, bins=30, color='#008fd5', edgecolor='black', alpha=0.7)
    plt.title("Histogram of Random Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def scatter_plot():
    plt.style.use("fivethirtyeight")
    x = np.random.rand(100)
    y = np.random.rand(100)
    sizes = np.random.randint(10, 100, size=100)
    colors = np.random.rand(100)

    plt.scatter(x, y, s=sizes, c=colors, alpha=0.5, cmap='viridis')
    plt.title("Scatter Plot with Varying Sizes and Colors")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar(label='Color Scale')
    plt.tight_layout()
    plt.show()

def ploting_time_series():
    plt.style.use("fivethirtyeight")
    dates = pd.date_range(start='2023-01-01', periods=100)
    values = np.random.randn(100).cumsum()

    plt.plot(dates, values, marker='o', linestyle='-', color='#008fd5')
    plt.title("Time Series Plot")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def data_use():
  ages_x = [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

  py_dev_y = [5372, 5425, 5720, 6320, 6420, 6430, 6440, 6450, 6460, 6470, 6480, 6490]

  js_dev_y = [4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300]

  dev_y = [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100]
  return ages_x, py_dev_y, js_dev_y, dev_y


if __name__ == "__main__":
  ages_x, py_dev_y, js_dev_y, dev_y = data_use()
  # create_customized_plot(ages_x, py_dev_y, js_dev_y, dev_y)
  # bar_charts_analyzing_data(ages_x, py_dev_y, js_dev_y, dev_y)
  # bar_charts_from_csv()
  # pie_charts()
  # stack_plots()
  # filling_area()
  # histogram()
  # scatter_plot()
  ploting_time_series()