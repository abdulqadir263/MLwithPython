import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression

# App title and description
st.title("📈 Future Sales Prediction App")
st.write("This app predicts **sales** based on advertising budgets for TV, Radio, and Newspaper.")

# Loading dataset
data = pd.read_csv("advertising.csv")  

# Showing dataset
st.subheader("🗂️ Dataset Preview")
if st.checkbox("Show dataset"):
    st.write(data.head())

# Traditional plot using Seaborn
st.subheader("📉 Sales vs TV Spend (Seaborn Plot)")
fig, ax = plt.subplots()
sns.regplot(data=data, x='TV', y='Sales', ax=ax)
st.pyplot(fig)

# Plotly charts
st.subheader("📊 Interactive Visualizations (Plotly)")

# TV vs Sales
fig_tv = px.scatter(data_frame=data, x="Sales", y="TV", size="TV", trendline="ols", title="Sales vs TV")
st.plotly_chart(fig_tv)

# Newspaper vs Sales
fig_np = px.scatter(data_frame=data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols", title="Sales vs Newspaper")
st.plotly_chart(fig_np)

# Radio vs Sales
fig_radio = px.scatter(data_frame=data, x="Sales", y="Radio", size="Radio", trendline="ols", title="Sales vs Radio")
st.plotly_chart(fig_radio)

# Sidebar inputs
st.subheader("✍️ Enter Advertising Budgets")

tv = st.number_input("TV Budget", min_value=0.0, max_value=500.0, value=150.0)
radio = st.number_input("Radio Budget", min_value=0.0, max_value=50.0, value=25.0)
newspaper = st.number_input("Newspaper Budget", min_value=0.0, max_value=100.0, value=20.0)

# Training the model
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
model = LinearRegression()
model.fit(X, y)

# Prediction
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
predicted_sales = model.predict(input_data)[0]

# Showing prediction
st.subheader("🔮 Predicted Sales")
st.success(f"Estimated sales: **{predicted_sales:.2f} units**")
