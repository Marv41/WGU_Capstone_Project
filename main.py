import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import streamlit as st
import plotly.express as px
import datetime

df = pd.read_csv("weather_data.csv")
df['Date'] = pd.to_datetime(df['Date']) 

df.info()
print(df.head(50))

# Correlation Heatmap
heatMap, ax1 = plt.subplots(figsize=(8, 6))# Subplots creates figure and axis
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="plasma", ax=ax1) # CMAP is the color gradient. Annot will display the actual numeric value on heatmap
ax1.set_title('Correlation Heatmap')

# Bar Chart Rained vs No Rain
rain_counts = df['Rained'].value_counts()
barChart, ax2 = plt.subplots()
rain_counts.plot(kind='bar', color=['skyblue', 'lightgray'], ax=ax2)
ax2.set_title('Rain vs. No Rain Balance')
ax2.set_xlabel('Rain Status')
ax2.set_ylabel('Number of Days')
ax2.set_xticks(range(len(rain_counts)))
ax2.set_xticklabels(rain_counts.index, rotation=0) # Adjust Ticks to be horizontal
barChart.tight_layout() #automatically adjusts the spacing of subplots and labels to prevent overlap

# Scatterplot Date vs Temperature
dateVtemp_scatterplot, ax3 = plt.subplots(figsize=(16,6))
ax3.scatter(df['Date'], df['Temperature'], alpha=0.6) # Alpha controls the transparency
ax3.set_title('Temperature Over Time', fontsize=32)
ax3.set_xlabel('Date', fontsize=24)
ax3.set_ylabel('Temperature', fontsize=24)
ax3.tick_params(axis='both', labelsize=18)
ax3.grid(True) # Displays grid
dateVtemp_scatterplot.tight_layout()

# Scatterplot Precipitation vs Humidity
pcpVhum_scatterplot, ax4 = plt.subplots(figsize=(16,6))
ax4.scatter(df['Humidity'], df['Precipitation'], alpha=0.6)
ax4.set_title('Humidity Impact on Rainfall', fontsize=32)
ax4.set_xlabel('Humidity', fontsize=24)
ax4.set_ylabel('Precipitation (in)', fontsize=24)
ax4.tick_params(axis='both', labelsize=18)
ax4.grid(True)
pcpVhum_scatterplot.tight_layout()

## Create 3D scatter plot
scatterplot3D = px.scatter_3d(
    df,
    x='Humidity',
    y='Dewpoint',
    z='Precipitation',
    color='Precipitation',  # color scale by value
    color_continuous_scale='BrBG',
    opacity=0.7,
    title='3D Scatter Plot: Humidity vs Dewpoint vs Precipitation'
)

# Optional layout tweaks
scatterplot3D.update_layout(
    width=1000,
    height=800,
    scene=dict(
        xaxis_title='Humidity',
        yaxis_title='Dewpoint',
        zaxis_title='Precipitation',
        xaxis=dict(autorange='reversed'),  # reverse X-axis if needed
        yaxis=dict(autorange='reversed')
    )
)

#Create Train/Testing data split
features = df[['Temperature', 'Dewpoint', 'Humidity', 'day_of_year','Visibility']]
target = df['Rained_Binary']               
F_train, F_test, pred_train, pred_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(F_train, pred_train)
#target_pred = model.predict(F_test) # not using target_pred. Prediction threshold was adjusted

# Adjust threshold so it predict rain more often
predictions_prob = model.predict_proba(F_test)[:, 1] # [:, 1] pull rain probablity in percent for ie .70
adjusted_predictions = (predictions_prob >= 0.4).astype(int) # If probablity is higher then .40 it will return 1

# Accuracy
accuracy = accuracy_score(pred_test, adjusted_predictions)

# Classification Report
report = classification_report(pred_test, adjusted_predictions, output_dict=True, target_names=["No Rain", "Rained"])

# Confusion Matrix
cm = confusion_matrix(pred_test, adjusted_predictions)
labels = ['No Rain', 'Rained']

confusionMatrix, ax5 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")
ax5.set_title("Confusion Matrix")

# Interface Starts Here
st.title("Rain Prediction Interface")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("👋 Hello and welcome to the Rain Prediction App!")
    st.write("Please enter the projected weather conditions for the day you'd like to predict. This app uses machine learning and historical weather data to estimate whether it will rain.")
    st.write("Below, you'll find a variety of charts used to explore and understand the data.")
    st.write("You’ll also see a performance evaluation of the machine learning algorithm used in this prediction.")

with col2:
    temperature = st.number_input("Temperature (°F)", min_value=-30.0, max_value=130.0, value=70.0)
    dewpoint = st.number_input("Dewpoint (°F)", min_value=-30.0, max_value=100.0, value=55.0)
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
    visibility = st.number_input("Visibility (miles)", min_value=0.0, max_value=20.0, value=10.0)
    selected_date = st.date_input("Select a Date", value=datetime.date.today())

    input_df = pd.DataFrame({
    "Temperature": [temperature],
    "Dewpoint": [dewpoint],
    "Humidity": [humidity],
    "day_of_year": [selected_date.timetuple().tm_yday],
    "Visibility": [visibility]
    })

with col3:
    if st.button("Predict Rain"):
        prob = model.predict_proba(input_df)[0][1]  # Get the probability for "Rained"
        prediction = int(prob >= 0.4) # Apply custom threshold (e.g., 0.4)
        result = "🌧️ It will Rain Today" if prediction == 1 else "🌤️ No Rain Today"
        st.subheader("Prediction Result")
        st.success(f"Prediction: {result}")


tab1, tab2, tab3, tab4, tab5 = st.tabs(["3D Scatterplot", "Heatmap & Bar Chart", "Temperature by Date", "Precipitation by Humidity", "Rain Model Preformance Evaluation"])

with tab1:
    st.write("This is a 3D scatterplot. This chart shows the correlation between humidity and dewpoint in relation to precipitation. It is interactive and allows you to zoom and select individual points.")
    st.plotly_chart(scatterplot3D)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(heatMap)
    with col2:
        st.pyplot(barChart)
    st.write("On the left is the **Correlation Heatmap**. This grid shows how strongly various features are correlated with one another. Values closer to **1** indicate a strong positive correlation, while values closer to **-1** indicate a strong negative correlation.")

    st.write("On the right is a simple **bar chart** showing the number of days with and without rain. A balanced dataset like this is better for training an effective machine learning model.")

with tab3:
    st.pyplot(dateVtemp_scatterplot)
    st.write("This scatterplot shows temperature over time. As expected, lower temperatures occur during the winter months, while higher temperatures are observed in the summer.")

with tab4:
    st.pyplot(pcpVhum_scatterplot)
    st.write("This scatterplot shows how humidity impacts total rainfall (precipitation). Higher levels of precipitation are typically observed when humidity exceeds 60%.")


with tab5:
    st.write("A Random Forest Classifier was used to train the rain prediction model. This algorithm combines the results of many decision trees to make a final prediction. The classifier categorizes each result as either 'Rained' or 'No Rain'. Accuracy, precision, recall, and F1-Score metrics were used to evaluate the performance of the machine learning model.")

    st.subheader(f"Accuracy:  **{accuracy:.3f}**")
    st.write("Accuracy was calculated by comparing the predictions made by the model (1 for rain, 0 for no rain) to the actual values in the dataset. This machine learning model predicted the correct outcome **86.7%** of the time.")
   
    st.subheader("Classification Report")
    st.write("The classification report shows several key metrics — **Precision**, **Recall**, **F1-Score** — presented in a table below.")
    st.write("**Precision** – Of the times the model predicted rain, how many were actually correct.")
    st.write("**Recall** – Out of all the actual rainy days in the dataset, how many did the model correctly predict.")
    st.write("**F1 Score** – The harmonic mean of precision and recall. It balances both metrics into a single score.")
    st.write("**Support** – This is not a metric. It represents the number of actual occurrences of each class (Rain / No Rain) in the dataset.")
    st.dataframe(pd.DataFrame(report).transpose())
    
    st.subheader("Confusion Matrix")
    st.write("A confusion matrix shows how well the model's predictions match the actual outcomes. Starting from the top left, when the model predicated NO RAIN, it was correct 103 times, however when it predicted NO RAIN it was wrong 16 times. The model predicted RAIN correctly 126 times and falsly predicated rain 19 times.")
    st.pyplot(confusionMatrix)
