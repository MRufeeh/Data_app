import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

iris_df = pd.read_csv(url, header=None, names=columns)
st.write("## Raw Iris Dataset", iris_df)
st.write("---")

# To Show the Average Sepal Length for Each Species
avg_sepal_length = iris_df.groupby('species')['sepal_length'].mean()
st.write("## Average Sepal Length for Each Species")
st.table(avg_sepal_length)

# To scatter Plot Comparing Two Features with Plotly
st.write("## Scatter Plot Comparing Two Features")
x_axis= st.selectbox("Select the X-Axis feature",iris_df.columns[:-1])
y_axis= st.selectbox("Select the Y-Axis feature",iris_df.columns[:-1])

scatter_fig = px.scatter(iris_df, x=x_axis,y=y_axis, color="species",title="Scatter Plot of x_axis vs y_axis")
st.plotly_chart(scatter_fig)

#To Filter Data Based on Species
st.write("## Filter Data Based on Species")
species_selected = st.multiselect("Select Species", iris_df['species'].unique(),iris_df['species'].unique())

#To Display filtered data
filtered_data = iris_df[iris_df['species'].isin(species_selected)]
st.write(f"### Filtered Data (showing {len(filtered_data)} rows)", filtered_data)

# To plot for selected species using Seaborn
st.write("## Pairplot for Selected Species")
if len(species_selected) > 0:
    sns_fig = sns.pairplot(filtered_data, hue='species')
    st.pyplot(sns_fig)

# Distribution plot of a selected feature
st.write("## Distribution of Selected Feature")
selected_feature = st.selectbox("Select Feature for Distribution", iris_df.columns[:-1])

# Plot the distribution using Matplotlib and Seaborn
fig, ax = plt.subplots()
sns.histplot(filtered_data[selected_feature], kde=True, ax=ax)
ax.set_title(f'Distribution of {selected_feature}')
st.pyplot(fig)

