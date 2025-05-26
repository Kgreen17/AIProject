import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load the example dataset
    tips = sns.load_dataset("tips")

    # Create a simple scatter plot
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")

    # Show the plot
    plt.show()