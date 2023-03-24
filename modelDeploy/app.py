# Import Libraries and the Helper Functions
import pandas as pd
import numpy as np
import streamlit as st

import helperFunctions
import helperFunctions2022


# Read the data (df_comb)
@st.cache_data
def get_data():

    df_comb = pd.read_csv(
        "https://media.githubusercontent.com/media/sharmajee499/MinneMUDAC2023/main/modelDeploy/df_combUP.csv"
    )

    return df_comb


df = get_data()

st.title("MLB Game Attendance Prediction:baseball:")
# helperFunctions.add_bg_from_local("jose-francisco-morales-hKzmPs8Axh8-unsplash.jpg")

# Function for the different pages for the website
def bulkPredictPage():

    st.header("Bulk Prediction for Each Team")

    st.caption(
        "Bulk Prediction don't use the Weather Data. It uses the Game Statistics and Other factors.\
            The 'Model Perf' shows the chart of Predicted vs Actual Attendence for 2022 games"
    )
    # All the Teams Name
    teamName = df[df["Year"] == 2023]["HomeTeam"].unique()

    # Selected Name (Select Box)
    teamNameChosen = st.selectbox("Select the Team", options=np.sort(teamName))

    # -----------------------------
    # Prepare the Data------------
    # -----------------------------

    X_train, y_train, X_test, X_testDate = helperFunctions.prepareDataTeam(
        df, teamNameChosen, withWeather=False
    )

    # -----------------------------
    # Train the Model------------
    # -----------------------------
    modelPipe = helperFunctions.trainModel(X_train, y_train)

    # -----------------------------
    # Predict the Data------------
    # -----------------------------
    finalDf = helperFunctions.predict2023NoWeatherBulk(modelPipe, X_test, X_testDate)

    finalDf["Predicted_Attendance"] = finalDf["Predicted_Attendance"].astype("int")

    # -----------------------------
    # Radio Buttons to choose options------------
    # -----------------------------

    predOptions = st.radio("Prediction Option", options=["Bulk", "By Game Date"])

    if predOptions == "Bulk":
        btnOptions = st.button("Show Prediction")
        btn2022 = st.button("Model Perf")
        if btnOptions:
            st.table(finalDf)

        # if wanted to see model's perf graph
        if btn2022:

            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_testDate,
            ) = helperFunctions2022.prepareDataTeam2022(
                df, teamNameChosen, withWeather=False
            )

            modelPipe = helperFunctions2022.trainModel2022(X_train, y_train)

            finalDf = helperFunctions2022.predictNoWeatherBulk2022(
                modelPipe, X_test, X_testDate
            )

            finalDf["Predicted_Attendance"] = finalDf["Predicted_Attendance"].astype(
                "int"
            )

            fig, ax = helperFunctions2022.plot2022(finalDf)

            st.pyplot(fig)

    else:
        dateOptions = st.selectbox("Choose the Date", options=finalDf["Date"])
        df_byDate = finalDf[finalDf["Date"] == dateOptions]
        btnOptions = st.button("Show Prediction")
        if btnOptions:
            st.table(df_byDate)

    pass


def dynamicPredictPage():

    st.header("Dynamic Prediction")

    st.caption(
        "Dynamic Prediction uses the Game Stats as well as the Weather data too.\
             The major limitation is we are only able to get weather forecast for\
                  around 15 days. Any game after 15 days won't have weather forecast\
                    therfore not showing the predictions."
    )

    # All the Teams Name
    teamName = df[df["Year"] == 2023]["HomeTeam"].unique()

    # Selected Name (Select Box)
    teamNameChosen = st.selectbox("Select the Team", options=np.sort(teamName))

    # -----------------------------
    # Prepare the Data------------
    # -----------------------------
    X_train, y_train, X_test, X_testDate = helperFunctions.prepareDataTeam(
        df, teamNameChosen, withWeather=True
    )

    # Show the date in the select box
    dateOption = st.selectbox("Choose the Date", options=X_testDate["Date"])

    # -----------------------------
    # Train the Model--------------
    # -----------------------------
    modelPipe = helperFunctions.trainModel(X_train, y_train)

    # -----------------------------
    # Predict the Data------------
    # -----------------------------
    try:

        finalDf = helperFunctions.predictDynamic(modelPipe, X_testDate, dateOption)

        finalDf["Predicted_Attendance"] = finalDf["Predicted_Attendance"].astype("int")

        st.table(finalDf)

    except:
        st.write("Weather Forecast Not Available For that Future Date")

    pass


# For Making Different Pages

page_name_to_funcs = {
    "Bulk Prediction": bulkPredictPage,
    "Dynamic Prediction": dynamicPredictPage,
}

demo_name = st.sidebar.selectbox(
    "Choose the Prediction Page:", page_name_to_funcs.keys()
)
page_name_to_funcs[demo_name]()
