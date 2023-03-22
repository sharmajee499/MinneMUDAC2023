# Import Libraries and the Helper Functions
import pandas as pd
import numpy as np
import streamlit as st

import helperFunctions


# Read the data (df_comb)
@st.cache_data
def get_data():

    df_comb = pd.read_csv(
        r"C:\Users\Sande\OneDrive - MNSCU\4_Semester\MinneMUDAC2023\modelDeploy\df_colNeed.csv"
    )

    return df_comb


df = get_data()

# Function for the different pages for the website
def bulkPredictPage():

    # All the Teams Name
    teamName = df[df["Year_StdCap"] == 2023]["HomeTeam"].unique()

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

    finalDf["Attendance"] = finalDf["Attendance"].astype("int")

    # -----------------------------
    # Radio Buttons to choose options------------
    # -----------------------------

    predOptions = st.radio("Prediction Option", options=["Bulk", "By Game Date"])

    if predOptions == "Bulk":
        btnOptions = st.button("Show Prediction")
        if btnOptions:
            st.table(finalDf)

    else:
        dateOptions = st.selectbox("Choose the Date", options=finalDf["Date"])
        df_byDate = finalDf[finalDf["Date"] == dateOptions]
        btnOptions = st.button("Show Prediction")
        if btnOptions:
            st.table(df_byDate)

    pass


def dynamicPredictPage():

    # All the Teams Name
    teamName = df[df["Year_StdCap"] == 2023]["HomeTeam"].unique()

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

        finalDf = helperFunctions.predictDynamic(
            modelPipe, X_testDate, X_test, dateOption
        )

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
