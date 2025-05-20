import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load data (for visualizations) and the trained model along with its feature names
data = pd.read_csv("tips.csv") # Used for displaying the initial visualizations

# Load the dictionary containing both the model and the trained feature names
loaded_assets = joblib.load("tip_predictor.pkl")
model = loaded_assets["model"]
trained_feature_names = loaded_assets["feature_names"]

st.set_page_config(layout="wide") # Optional: Use wide layout for better visualization display
st.title("💸 Waiter Tips Predictor Dashboard")

# --- Display Visualizations ---
st.subheader("📊 Tip Trends Overview")

st.markdown("By Abdul Qadir")

col1, col2, col3 = st.columns(3) # Organize visualizations in columns

with col1:
    fig1 = px.scatter(data, x="total_bill", y="tip", size="size", color="day", trendline="ols", title="Tips by Day")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(data, x="total_bill", y="tip", size="size", color="sex", trendline="ols", title="Tips by Sex")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    fig3 = px.scatter(data, x="total_bill", y="tip", size="size", color="time", trendline="ols", title="Tips by Time")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---") # Separator

# --- User Input for Prediction (now below dashboards) ---
st.subheader("🔮 Predict Your Tip")
st.markdown("Please enter the details below to get a tip recommendation.")

# Use columns for better layout of input fields
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    total_bill = st.number_input("Total Bill ($)", min_value=0.01, value=20.00, step=0.50)
    sex = st.selectbox("Sex", ["Female", "Male"])

with input_col2:
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])

with input_col3:
    time = st.selectbox("Time", ["Lunch", "Dinner"])
    size = st.slider("Party Size", 1, 6, 2)

# --- Prediction Button and Result Display ---
# Create a single column for the prediction button and result
predict_col = st.columns(1)[0] # Just a single column

with predict_col:
    # Create a placeholder for the prediction result
    prediction_result_placeholder = st.empty()

    if st.button("Predict Tip", use_container_width=True):
        try:
            # Create a dictionary from the raw user inputs
            raw_input_data = {
                "total_bill": total_bill,
                "size": size,
                "sex": sex,
                "smoker": smoker,
                "day": day,
                "time": time
            }

            # Convert the raw input dictionary into a pandas DataFrame (single row)
            input_df = pd.DataFrame([raw_input_data])

            # Apply one-hot encoding to the user's input, dropping the first category
            input_encoded = pd.get_dummies(input_df, drop_first=True)

            # Reindex the encoded input DataFrame to match the exact columns and order
            final_input_data = input_encoded.reindex(columns=trained_feature_names, fill_value=0)

            # Make the prediction
            predicted_tip = model.predict(final_input_data)[0]

            # Display the result directly in the placeholder
            prediction_result_placeholder.success(f"**Recommended Tip: ${predicted_tip:.2f}**")

        except Exception as e:
            prediction_result_placeholder.error(f"An error occurred during prediction: {e}")
            prediction_result_placeholder.write("Please check the input values.")