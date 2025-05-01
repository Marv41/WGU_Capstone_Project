import streamlit as st

st.title("Charts")

st.write(
    "Several charts were created to explore and better understand the dataset. The purpose of visualizing the data is to uncover patterns and correlations that help explain the relationships between different weather variables. Various Python libraries were used for data analysis and visualization, including **Pandas**, **Matplotlib**, **NumPy**, **Plotly**, and **Seaborn**. These libraries assist with computations, data manipulation, table creation, 2D charts, and 3D visualizations."
)

# Create Tab Section
tab1, tab2, tab3, tab4 = st.tabs(["3D Scatterplot", "Heatmap & Bar Chart", "Temperature by Date", "Precipitation by Humidity"])

with tab1:
    st.write("This interactive 3D scatterplot visualizes the relationship between humidity and dew point in the context of precipitation. You can zoom, rotate, and select individual data points to explore patterns more closely.")

    st.plotly_chart(st.session_state.scatterplot3D)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(st.session_state.heatMap)

    with col2:
        st.pyplot(st.session_state.barChart)

    st.write("On the left is the **Correlation Heatmap**. This grid shows how strongly various features are correlated with one another. Values closer to **1** indicate a strong positive correlation, while values closer to **-1** indicate a strong negative correlation.")

    st.write("On the right is a simple **bar chart** showing the number of days with and without rain. A balanced dataset like this is better for training an effective machine learning model.")

with tab3:
    st.pyplot(st.session_state.dateVtemp_scatterplot)

    st.write("This scatterplot shows temperature over time. As expected, lower temperatures occur during the winter months, while higher temperatures are observed in the summer.")

with tab4:
    st.pyplot(st.session_state.pcpVhum_scatterplot)

    st.write("This scatterplot shows how humidity impacts total rainfall (precipitation). Higher levels of precipitation are typically observed when humidity exceeds 60%.")

