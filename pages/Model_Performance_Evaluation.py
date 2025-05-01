import streamlit as st
import pandas as pd
from main import accuracy, report, confusionMatrix


st.title("Rain Model Performance Evaluation")

st.write("Supervised learning was used to build this rain prediction model. The model was trained on labeled historical weather data, with the goal of learning the relationship between various weather conditions to make accurate predictions. The Random Forest Classifier algorithm, which was used, combines the results of many decision trees to make a final prediction. Each prediction is categorized as either 'Rained' or 'No Rain'. The model's performance was evaluated using accuracy, precision, recall, and F1-score metrics.")

st.subheader(f"Accuracy:  **{accuracy:.3f}**")
st.write("Accuracy was calculated by comparing the predictions made by the model (1 for rain, 0 for no rain) to the actual values in the dataset. This machine learning model predicted the correct outcome **86.7%** of the time.")
   
st.subheader("Classification Report")
st.write("The classification report shows several key metrics — **Precision**, **Recall**, **F1-Score** — presented in a table below.")
st.write("**Precision** – Of the times the model predicted rain, how many were actually correct?")
st.write("**Recall** – Out of all the actual rainy days in the dataset, how many did the model correctly predict?")
st.write("**F1 Score** – The harmonic mean of precision and recall. It balances both metrics into a single score.")
st.write("**Support** – This is not a metric. It represents the number of actual occurrences of each class (Rain / No Rain) in the dataset.")
st.dataframe(pd.DataFrame(report).transpose())
    
st.subheader("Confusion Matrix")
st.write("A confusion matrix shows how well the model's predictions match the actual outcomes. Starting from the top left, when the model predicated NO RAIN, it was correct 103 times, however when it predicted NO RAIN it was wrong 16 times. The model predicted RAIN correctly 126 times and falsly predicated rain 19 times.")
st.pyplot(confusionMatrix)