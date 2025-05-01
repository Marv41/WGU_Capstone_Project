import streamlit as st
import pandas as pd


st.title("Weather Dataset")

st.write(
    "The weather dataset used in this project was sourced from Kaggle.com. It contains historical weather data from Austin, Texas, spanning from 2013 to 2017. For the purpose of this project, the dataset is being treated as if it represents weather data from Denver, Colorado.\n\n"
    "The data was cleaned using the **Pandas** library in Python. Unnecessary columns were removed, and all missing or invalid data was transformed. Additional features (columns) were created to enhance the model, including **Rained**, **Rained_Binary**, and **day_of_year**.\n\n"
    "To improve the training of the machine learning model, some data entries were modified to reflect rain on days that previously showed no rain. This adjustment was made to help balance the number of rainy and non-rainy days in the dataset."
    ""
)

st.markdown("Visit [Kaggle](https://www.kaggle.com) to explore the original weather dataset.")

df = pd.read_csv("weather_data.csv")

st.dataframe(df, use_container_width=True)