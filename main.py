import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import plotly.express as px

df = pd.read_csv("weather_data.csv")
df['Date'] = pd.to_datetime(df['Date']) 

df.info()
print(df.head(50))

# Correlation Heatmap
heatMap, ax1 = plt.subplots(figsize=(8, 6))# Subplots creates figure and axis
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="plasma", ax=ax1) # CMAP is the color gradient. Annot will display the actual numeric value on heatmap
ax1.set_title('Correlation Heatmap')
st.session_state.heatMap = heatMap

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
st.session_state.barChart = barChart

# Scatterplot Date vs Temperature
dateVtemp_scatterplot, ax3 = plt.subplots(figsize=(16,6))
ax3.scatter(df['Date'], df['Temperature'], alpha=0.6) # Alpha controls the transparency
ax3.set_title('Temperature Over Time', fontsize=32)
ax3.set_xlabel('Date', fontsize=24)
ax3.set_ylabel('Temperature', fontsize=24)
ax3.tick_params(axis='both', labelsize=18)
ax3.grid(True) # Displays grid
dateVtemp_scatterplot.tight_layout()
st.session_state.dateVtemp_scatterplot = dateVtemp_scatterplot

# Scatterplot Precipitation vs Humidity
pcpVhum_scatterplot, ax4 = plt.subplots(figsize=(16,6))
ax4.scatter(df['Humidity'], df['Precipitation'], alpha=0.6)
ax4.set_title('Humidity Impact on Rainfall', fontsize=32)
ax4.set_xlabel('Humidity', fontsize=24)
ax4.set_ylabel('Precipitation (in)', fontsize=24)
ax4.tick_params(axis='both', labelsize=18)
ax4.grid(True)
pcpVhum_scatterplot.tight_layout()
st.session_state.pcpVhum_scatterplot = pcpVhum_scatterplot

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
st.session_state.scatterplot3D = scatterplot3D

#Create Train/Testing data split
features = df[['Temperature', 'Dewpoint', 'Humidity', 'day_of_year','Visibility']]
target = df['Rained_Binary']               
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.session_state.accuracy = accuracy

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True, target_names=["No Rain", "Rained"])
st.session_state.report = report

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['No Rain', 'Rain']

confusionMatrix, ax5 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")
ax5.set_title("Confusion Matrix")
st.session_state.confusionMatrix = confusionMatrix
