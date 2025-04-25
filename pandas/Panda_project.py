"""
Load your data with pandas, and give it a first look. Use info(), head(), and describe() to get an initial sense of what’s in front of you. Make sure columns
like Join_Date are in the correct formats so you're ready for analysis! Missing Data Analysis: Missing values? No problem. Check for any gaps, and decide if you
want to clean or fill them in. Data Selection and Filtering: Pull up information on employees over 30 with an "A" in Performance_Score, focusing on columns like Name,
Age, Department, and Performance_Score. Find who’s working in the "Engineering" department under "Manager_B". Working with Text Data: Let’s make sure names look neat!
Turn all names in the Name column to lowercase, and give each letter in Department a big first-letter boost. Date and Time Handling: Filter down to employees who joined
in 2021 or later. Display their Name, Join_Date, and Department. Data Grouping and Aggregation: Let’s crunch some numbers! Calculate the average salary per department and
count employees in each Performance_Score category. Sorting and Ordered Selection: Who’s making the big bucks? Sort employees by salary in descending order, and display
the top 10 with their Name and Salary. Data Merging and Joining: Create a department_leaders table listing each department and its leader. Merge it with the main table t
o see each employee’s department leader. Exporting Data: Wrap it up by saving employees who work over 40 hours a week to a new CSV file for further analysis.
"""


import pandas as pd

# Load the data from the CSV file
file_path = '158_775_attachment.csv'  # Ensure the file is in the correct directory or provide the full path
data = pd.read_csv(file_path)

# Give the data a first look
print("Data Info:")
print(data.info())  # Overview of the dataset

print("\nFirst 5 Rows:")
print(data.head())  # Display the first 5 rows

print("\nDescriptive Statistics:")
print(data.describe())  # Summary statistics for numerical columns

# Check for missing values in each column
missing_values = data.isnull().sum()

print("Missing Values in Each Column:")
print(missing_values)


# Check for missing values in each column
missing_values = data.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)

# Fill missing values with a default value (e.g., 0 for numerical columns, 'Unknown' for categorical columns)
data_filled = data.fillna({
    'Column1': 0,  # Replace 'Column1' with the actual column name
    'Column2': 'Unknown'  # Replace 'Column2' with the actual column name
})


# Verify that missing values are handled
print("\nMissing Values After Cleaning:")
print(data_filled.isnull().sum())


# Filter employees over 30 with an "A" in Performance_Score
filtered_data = data[(data['Age'] > 30) & (data['Performance_Score'] == 'A')]

# Select specific columns
filtered_data = filtered_data[['Name', 'Age', 'Department', 'Performance_Score']]

print("Employees over 30 with an 'A' in Performance_Score:")
print(filtered_data)

# Find employees in the "Engineering" department under "Manager_B"
engineering_under_manager_b = data[(data['Department'] == 'Engineering') & (data['Manager'] == 'Manager_B')]

print("\nEmployees in the 'Engineering' department under 'Manager_B':")
print(engineering_under_manager_b[['Name', 'Department', 'Manager']])


# Convert all names in the Name column to lowercase
data['Name'] = data['Name'].str.lower()

# Capitalize the first letter of each word in the Department column
data['Department'] = data['Department'].str.title()

# Display the updated data
print("Updated Data:")
print(data[['Name', 'Department']].head())

# Ensure Join_Date is in datetime format
data['Join_Date'] = pd.to_datetime(data['Join_Date'], errors='coerce')

# Filter employees who joined in 2021 or later
employees_2021_or_later = data[data['Join_Date'] >= '2021-01-01']
print("Employees who joined in 2021 or later:")
print(employees_2021_or_later[['Name', 'Join_Date', 'Department']])

# Grouping and aggregation: Average salary per department
average_salary_per_department = data.groupby('Department')['Salary'].mean()
print("\nAverage Salary Per Department:")
print(average_salary_per_department)

# Grouping and aggregation: Count employees in each Performance_Score category
performance_score_counts = data['Performance_Score'].value_counts()
print("\nEmployee Count Per Performance_Score Category:")
print(performance_score_counts)

# Grouping and aggregation: Average salary per department
average_salary_per_department = data.groupby('Department')['Salary'].mean()
print("Average Salary Per Department:")
print(average_salary_per_department)

# Grouping and aggregation: Count employees in each Performance_Score category
performance_score_counts = data['Performance_Score'].value_counts()
print("\nEmployee Count Per Performance_Score Category:")
print(performance_score_counts)

# Sort employees by salary in descending order
sorted_data = data.sort_values(by='Salary', ascending=False)

# Select the top 10 employees and display their Name and Salary
top_10_employees = sorted_data[['Name', 'Salary']].head(10)

print("Top 10 Employees by Salary:")
print(top_10_employees)

# Create the department_leaders table
department_leaders_data = {
    'Department': ['Engineering', 'HR', 'Sales', 'Marketing'],
    'Leader': ['Manager_B', 'Manager_HR', 'Manager_Sales', 'Manager_Marketing']
}
department_leaders = pd.DataFrame(department_leaders_data)

# Merge the main table with the department_leaders table
data_with_leaders = pd.merge(data, department_leaders, on='Department', how='left')

# Display the updated data with department leaders
print("Data with Department Leaders:")
print(data_with_leaders[['Name', 'Department', 'Leader']].head())

# Filter employees who work over 40 hours a week
employees_over_40_hours = data[data['Hours_Worked_Per_Week'] > 40]

# Save the filtered data to a new CSV file
output_file_path = 'employees_over_40_hours.csv'
employees_over_40_hours.to_csv(output_file_path, index=False)

print(f"Data saved to {output_file_path}")
