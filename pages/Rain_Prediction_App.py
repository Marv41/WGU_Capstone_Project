import streamlit as st
import pandas as pd
import datetime
from Home import model

st.title("Rain Prediction")

st.write(
    "This is the Rain Prediction Tool. The machine learning model was trained to predict whether it will rain based on temperature, dewpoint, humidity, visibility, and the selected date.\n\n"
    "These values can either be obtained from a weather source or estimated by the user. The model was trained using real weather data from Austin, Texas, with some simulated data added to balance the number of rainy days.\n\n"
    "For more details, see the **Weather Dataset** section.\n\n"
    "**Note:** All input fields must be filled out before the **Predict Rain** button can be used."
)
# Input fields for prediction tool
temperature = st.number_input("Temperature (°F)", min_value=-30.0, max_value=130.0, value=None)
dewpoint = st.number_input("Dewpoint (°F)", min_value=-30.0, max_value=100.0, value=None)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=None)
visibility = st.number_input("Visibility (miles)", min_value=0.0, max_value=20.0, value=None)
selected_date = st.date_input("Select a Date", value=datetime.date.today())

# Create button for prediction tool
if st.button("Predict Rain"):
    if not temperature or not dewpoint or not visibility:
        st.warning("❗ Please fill in all fields before making a prediction.")
    else:
        try:
            temperature = float(temperature)
            dewpoint = float(dewpoint)
            visibility = float(visibility)
            day_of_year = selected_date.timetuple().tm_yday

            input_df = pd.DataFrame({
                "Temperature": [temperature],
                "Dewpoint": [dewpoint],
                "Humidity": [humidity],
                "day_of_year": [day_of_year],
                "Visibility": [visibility]
            })

            prob = model.predict_proba(input_df)[0][1]
            y_pred = model.predict(input_df) 
            result = "🌧️ It will Rain Today" if y_pred == 1 else "🌤️ No Rain Today"
            # Output Result
            st.success(f"Prediction: {result} (Probability: {prob:.2f})")

        except ValueError:
            st.error("Invalid input. Please enter numbers for temperature, dewpoint, and visibility.")
