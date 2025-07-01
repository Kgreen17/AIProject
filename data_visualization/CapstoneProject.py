# -------------------------
# 1. DATA PREPROCESSING & EXPLORATION
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
output_notebook()

# 1.1 Load Dataset
df = pd.read_csv('employee_data.csv')
print(df.head())
print(df.shape)
print(df.info())

# 1.2 Handle Missing Values
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# Fill missing numerical with median, categorical with mode
for col in df.select_dtypes(include='number'):
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object'):
    df[col] = df[col].fillna(df[col].mode()[0])

# 1.3 Clean Incorrect Entries
df['Gender'] = df['Gender'].replace({'Femelle': 'Female', 'Malle': 'Male'})
df['RemoteWork'] = df['RemoteWork'].str.capitalize()

# 1.4 Explore Categorical & Numerical Columns
print(df.select_dtypes(include='object').nunique())
print(df.describe())

# 1.5 Outlier Detection (IQR Method)
def detect_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

outliers_salary = detect_outliers('Salary')
print("Outliers in Salary:\n", outliers_salary)

# 1.6 Remove Duplicates
df = df.drop_duplicates()

# 1.7 Derived Columns
df['BonusToSalaryRatio'] = df['AnnualBonus'] / df['Salary']
df['ExperienceLevel'] = pd.cut(df['YearsAtCompany'], bins=[-1,5,15,100], labels=['Junior', 'Mid', 'Senior'])
df['AgeDecade'] = (df['Age'] // 10) * 10


# 2.1 GroupBy Analysis
print(df.groupby(['Department', 'Gender'])['Salary'].mean())
top_roles = df.groupby('JobRole')['PerformanceScore'].mean().sort_values(ascending=False).head(3)
print("Top 3 Roles by Performance:\n", top_roles)

# 2.2 Correlation
corr_matrix = df[['Salary', 'YearsAtCompany', 'PerformanceScore']].corr()
print("Correlations:\n", corr_matrix)

# 2.3 Crosstab: RemoteWork vs MaritalStatus
print(pd.crosstab(df['RemoteWork'], df['MaritalStatus'], normalize='index'))

# 2.4 Filtering & Ranking
top_bonus_ratio = df.sort_values(by='BonusToSalaryRatio', ascending=False).head(5)
print("Top 5 Bonus-to-Salary:\n", top_bonus_ratio[['Name', 'BonusToSalaryRatio']])

top_cities = df.groupby('City')[['Salary', 'PerformanceScore']].mean().sort_values(by='Salary', ascending=False).head(3)
print("Top 3 Cities:\n", top_cities)

# 2.5 Department Gender Balance
gender_counts = df.groupby('Department')['Gender'].value_counts().unstack().fillna(0)
gender_balance = abs(gender_counts['Male'] - gender_counts['Female'])
print("Most balanced department:\n", gender_balance.sort_values().head(1))

# 2.6 NumPy Calculations
median_salaries = df.groupby('JobRole')['Salary'].apply(np.median)
df['PerformanceScore_z'] = (df['PerformanceScore'] - df['PerformanceScore'].mean()) / df['PerformanceScore'].std()

# 2.7 Performance & Age
df['AgeGroup'] = pd.cut(df['Age'], bins=[20,30,40,50,60,70], labels=['20s','30s','40s','50s','60s'])
age_perf = df.groupby('AgeGroup')[['PerformanceScore', 'AnnualBonus']].mean()
print("Performance by AgeGroup:\n", age_perf)

exp_bonus = df.groupby('ExperienceLevel')['AnnualBonus'].mean()
print("Bonuses by Experience Level:\n", exp_bonus)


# Matplotlib / Seaborn
plt.figure(figsize=(8,5))
sns.barplot(x='EducationLevel', y='Salary', data=df)
plt.title("Average Salary by Education Level")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='Department', y='Salary', hue='Gender', data=df)
plt.xticks(rotation=45)
plt.title("Salary Distribution by Department & Gender")
plt.show()

plt.figure(figsize=(10,6))
avg_perf_trend = df.groupby('YearsAtCompany')['PerformanceScore'].mean()
plt.plot(avg_perf_trend)
plt.title("Performance vs Years at Company")
plt.xlabel("Years at Company")
plt.ylabel("Avg Performance Score")
plt.show()

# Plotly
fig = px.scatter(df, x='Salary', y='PerformanceScore', color='Department', hover_data=['Name'])
fig.update_layout(title="Salary vs Performance Score")
fig.show()

fig = px.choropleth(df, locations='Country', locationmode='country names', color='Salary',
                    title="Average Salary by Country")
fig.show()

fig = px.parallel_coordinates(df, dimensions=['Salary', 'AnnualBonus', 'PerformanceScore'],
                              color='PerformanceScore', color_continuous_scale='Viridis')
fig.show()

fig = px.histogram(df, x='YearsAtCompany', nbins=20, title="Distribution of Years at Company")
fig.show()

# Bokeh
source = ColumnDataSource(df)
p = figure(title="Bokeh: Salary vs Performance", x_axis_label='Salary', y_axis_label='PerformanceScore')
p.circle(x='Salary', y='PerformanceScore', size=10, source=source)
show(p)
