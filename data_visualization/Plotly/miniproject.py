import pandas as pd
import plotly.express as px

# Load dataset (replace with your path or download from Kaggle)
df = pd.read_csv("netflix_titles.csv")

# Data Preprocessing
df['date_added'] = pd.to_datetime(df['date_added'])
df['year_added'] = df['date_added'].dt.year

# Drop NaN years or countries
df = df.dropna(subset=['year_added', 'country'])

# Expand multiple countries per row
df['country'] = df['country'].str.split(', ')
df = df.explode('country')

# Group data
country_year = df.groupby(['year_added', 'country']).size().reset_index(name='count')

# Filter for top countries (optional: improve animation speed)
top_countries = country_year.groupby('country')['count'].sum().nlargest(10).index
country_year = country_year[country_year['country'].isin(top_countries)]

# Plot with Plotly Express
fig = px.bar(country_year,
             x='count',
             y='country',
             color='country',
             animation_frame='year_added',
             orientation='h',
             title='Netflix Content Growth by Country Over Time',
             labels={'count': 'Number of Titles', 'country': 'Country'},
             height=600)

fig.update_layout(showlegend=False)
fig.update_traces(marker_line_width=1.5, opacity=0.8)

# Save and show
fig.write_html("netflix_growth.html")
fig.show()