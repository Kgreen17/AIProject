"""
Objectives
1. Perform Exploratory Data Analysis (EDA)
o Load the dataset and check for missing values.
o Generate summary statistics (mean, median, standard deviation, etc.).
o Visualize the distribution of features using histograms and boxplots.
o Analyze correlations between features using a heatmap.
o Identify outliers and handle them appropriately.
o Check for multicollinearity among features.
2. Preprocess the Data
o Normalize or standardize numerical features if necessary.
o Encode categorical variables if present.
o Split the dataset into training and testing sets (80% train, 20% test).
3. Model Training & Evaluation
o Set BMI as the target variable.
o Train and evaluate the following regression models:
▪ Simple Linear Regression (using one independent variable)
▪ Multiple Linear Regression (using all independent variables)
▪ Ridge Regression (to handle multicollinearity)
▪ Lasso Regression (to perform feature selection)
▪ Elastic Net Regression (combination of Ridge and Lasso)
o Calculate and compare R² scores for each model.
o Analyze the effect of hyperparameter tuning for Ridge, Lasso, and Elastic Net.
4. Results & Interpretation
o Compare the performance of different models.
o Interpret the coefficients of the best-performing model.
o Discuss the impact of different features on BMI.
Deliverables
• Jupyter Notebook / Python Script containing:
o EDA and visualizations
o Regression model implementation and evaluation
• Presentation / Report summarizing findings, challenges, and conclusions
Bonus Challenges
• Try using Polynomial Regression for better accuracy.
• Use Feature Engineering to create new meaningful variables.
• Apply Cross-Validation to assess model generalizability.
Evaluation Criteria
• Completeness of EDA
• Proper implementation of regression models
• Interpretation of results
• Code clarity and documentation
• Creativity and additional insights

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ML
from sklearn.model_selection import train_test_split #split data into training and testing sets
from sklearn.linear_model import LinearRegression, Ridge, Lasso #linear regression models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #metrics for evaluating regression models #performance metrics
from sklearn.preprocessing import LabelEncoder
#setting default figure size for plots
plt.rcParams['figure.figsize'] = (10, 6) #ensures all plots have the same size

#suppressing unnecessary warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore') #ignores warning messages that may not impact functionality.
def  dataSet():
      # Load the dataset
    df = pd.read_csv('BMI_dataset.csv')
    # print(df.head())

      #Load the dataset and check for missing values.
    # print(df.isnull().sum())

      #Generate summary statistics (mean, median, standard deviation, etc.)
    # print(df.describe())

      #Visualize the distribution of features using histograms and boxplots.
    # df.hist(bins=30, figsize=(15, 10))
    # plt.tight_layout()
 jƒkdlkdfajsdlfedfkhfjf nvj;vjfl5j5rll  # sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
  uc.  v,b # plt.show()
 oufof¨√p
    #Identify outliers and handle them appropriately.

    # sns.boxplot(x=df['bmi_value'])
    # plt.title('Boxplot of BMI')
    # plt.show()

      #Check for multicollinearity among features.
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()



if __name__ == '__main__':
    dataSet()