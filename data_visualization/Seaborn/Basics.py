import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # I will put a comment here as I am using my xr glass

    # Load the example dataset
    tips = sns.load_dataset("tips")

    # Create a simple scatter plot
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")

    plt.show()