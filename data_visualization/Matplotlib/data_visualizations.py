import pandas as pd
import matplotlib.pyplot as plt

# Use a nice style
plt.style.use('ggplot')

# Load the dataset
df = pd.read_csv("country_data.csv")

# =======================
# Exploratory Data Analysis
# =======================
print("🔍 HEAD OF DATA:")
print(df.head())

print(" DATA INFO:")
print(df.info())

print("DESCRIPTIVE STATS:")
print(df.describe())

print(" MISSING VALUES:")
print(df.isnull().sum())

# Drop rows with missing data (optional)
df = df.dropna()


# 1. Histogram: Median Age
df["Median Age (Years)"].plot(kind='hist', bins=15, figsize=(8,5), title="Distribution of Median Age")
plt.xlabel("Median Age (Years)")
plt.show()

# 2. Bar chart: Average Population by Country
pop_avg = df.groupby("Country")["Population (Millions)"].mean().sort_values(ascending=False)
pop_avg.plot(kind='bar', figsize=(10, 5), title="Average Population by Country")
plt.ylabel("Population (Millions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Scatter Plot: GDP vs Life Expectancy
plt.figure(figsize=(8,6))
plt.scatter(df["GDP (Trillions)"], df["Life Expectancy (Years)"], alpha=0.7)
plt.xlabel("GDP (Trillions)")
plt.ylabel("Life Expectancy (Years)")
plt.title("GDP vs Life Expectancy")
plt.show()

# 4. Line Plot: Happiness Index by Country
happy_avg = df.groupby("Country")["Happiness Index"].mean().sort_values()
happy_avg.plot(kind='line', marker='o', figsize=(10, 5), title="Average Happiness Index by Country")
plt.ylabel("Happiness Index")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Pie Chart: Top 5 Countries by Renewable Energy %
renew_top5 = df.groupby("Country")["Renewable Energy (%)"].mean().nlargest(5)
renew_top5.plot.pie(autopct='%1.1f%%', figsize=(6,6), title="Top 5 Countries by Renewable Energy (%)")
plt.ylabel("")
plt.show()

# 6. Box Plot: Internet Users (%)
plt.figure(figsize=(8, 5))
plt.title("Internet Users (%) Distribution")
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(10, 8))
plt.title("Correlation Between Numeric Variables")
plt.show()

# 8. Stacked Bar: Population and CO2 Emissions
pop_co2 = df.groupby("Country")[["Population (Millions)", "CO2 Emissions (MT)"]].mean()
pop_co2.plot(kind='bar', stacked=True, figsize=(10,6), title="Population and CO2 Emissions by Country")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Area Plot: GDP by Country
gdp_series = df.groupby("Country")["GDP (Trillions)"].mean()
gdp_series.sort_values().plot(kind='area', figsize=(10, 5), title="Average GDP by Country", alpha=0.5)
plt.ylabel("GDP (Trillions)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10. Custom: Literacy Rate vs Happiness Index
plt.figure(figsize=(8,6))
plt.scatterplot(data=df, x="Literacy Rate (%)", y="Happiness Index", hue="Country", palette="tab10")
plt.title("Literacy Rate vs Happiness Index")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# =======================
# Insights Summary
# =======================
print("INSIGHTS:")
print("- USA, Brazil, Japan appear multiple times, indicating subregional entries or time-based data.")
print("- High GDP doesn’t always mean high Happiness Index — e.g., Germany, France.")
print("- Higher Renewable Energy % generally means lower CO2 emissions.")
print("- Literacy Rate and Life Expectancy correlate positively with Happiness Index.")
print("- Internet access is high in developed countries but uneven globally.")
