import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

def Display_basic_information():
    print("First 5 rows of the dataset:")
    print(titanic.head())

    print("\nDataset Info:")
    print(titanic.info())

    print("\nStatistical Summary:")
    print(titanic.describe())

    print("\nMissing Values:")
    print(titanic.isnull().sum())

def Distribution_Plot_Age():
    plt.figure()
    sns.histplot(titanic['age'].dropna(), kde=False, bins=30, color='skyblue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def KDE_Plot_Age():
    plt.figure()
    sns.kdeplot(data=titanic, x='age', fill=True, color='purple')
    plt.title('Density Plot of Passenger Ages')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

def Bar_Plot_Average_Fare_by_Class():
    plt.figure()
    sns.barplot(x='class', y='fare', data=titanic, palette='pastel')
    plt.title('Average Fare by Passenger Class')
    plt.xlabel('Class')
    plt.ylabel('Average Fare')
    plt.tight_layout()
    plt.show()

def Count_Plot_Embark_Town():
    plt.figure()
    sns.countplot(x='embark_town', data=titanic, palette='Set2')
    plt.title('Number of Passengers by Embark Town')
    plt.xlabel('Embark Town')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def Box_Plot_Fare_by_Class():
    plt.figure()
    sns.boxplot(x='class', y='fare', data=titanic, palette='coolwarm')
    plt.title('Fare Distribution Across Classes')
    plt.xlabel('Class')
    plt.ylabel('Fare')
    plt.tight_layout()
    plt.show()

def Violin_Plot_Age_by_Class():
    plt.figure()
    sns.violinplot(x='class', y='age', data=titanic, palette='muted')
    plt.title('Age Distribution by Class')
    plt.xlabel('Class')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.show()

def Strip_Plot_Fare_vs_Class_by_Gender():
    plt.figure()
    sns.stripplot(x='class', y='fare', data=titanic, hue='sex', jitter=True, dodge=True, palette='husl')
    plt.title('Fare vs Class by Gender')
    plt.xlabel('Class')
    plt.ylabel('Fare')
    plt.legend(title='Sex')
    plt.tight_layout()
    plt.show()

def Swarm_Plot_Fare_vs_Class_by_Gender():
    plt.figure()
    sns.swarmplot(x='class', y='fare', data=titanic, hue='sex', palette='dark')
    plt.title('Swarm Plot of Fare vs Class by Gender')
    plt.xlabel('Class')
    plt.ylabel('Fare')
    plt.legend(title='Sex')
    plt.tight_layout()
    plt.show()

def Heatmap_Correlation_between_Numerical_Features():
    plt.figure(figsize=(10, 6))
    sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='YlGnBu', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def Cluster_Map_Numerical_Features():
    sns.clustermap(titanic.corr(numeric_only=True), cmap='coolwarm', annot=True)
    plt.suptitle('Cluster Map of Numerical Features', y=1.02)
    plt.show()

if __name__ == '__main__':
    Display_basic_information()
    Distribution_Plot_Age()
    KDE_Plot_Age()
    Bar_Plot_Average_Fare_by_Class()
    Count_Plot_Embark_Town()
    Box_Plot_Fare_by_Class()
    Violin_Plot_Age_by_Class()
    Strip_Plot_Fare_vs_Class_by_Gender()
    Swarm_Plot_Fare_vs_Class_by_Gender()
    Heatmap_Correlation_between_Numerical_Features()
    Cluster_Map_Numerical_Features()


