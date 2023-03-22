# Libraries Import

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ML
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# weather
from geopy.geocoders import Nominatim
import requests

# ------------------------------------------------------------------------------------------------

# Function to subset the data by team (No Weather)


def prepareDataTeam(df_combined, teamName, withWeather: bool):

    """
    Input: Data-frame that is combined, team name and data with weather or not
    Output: X_train, y_train, X_test, X_testData
    """
    # Columns list for subsetting
    dropColX = [
        "Date",
        "Year_StdCap",
        "Attendance",
        "Capacity_StdCap",
        "CITY_StdCap",
        "Occ_Per",
    ]
    weatherCol = [
        "rain_sum",
        "snowfall_sum",
        "precipitation_sum",
        "temp_max",
        "temp_min",
    ]

    # When we don't need weather data
    if withWeather == False:

        X_train = (
            df_combined[
                (df_combined["Year_StdCap"] != 2023)
                & (df_combined["HomeTeam"] == teamName)
            ]
            .drop(dropColX + weatherCol, axis=1)
            .reset_index(drop=True)
        )
        X_train.fillna(value=0, inplace=True)

        y_train = df_combined[
            (df_combined["Year_StdCap"] != 2023) & (df_combined["HomeTeam"] == teamName)
        ]["Occ_Per"].reset_index(drop=True)

        X_test = (
            df_combined[
                (df_combined["Year_StdCap"] == 2023)
                & (df_combined["HomeTeam"] == teamName)
            ]
            .drop(dropColX + weatherCol, axis=1)
            .reset_index(drop=True)
        )
        X_test.fillna(value=0, inplace=True)

        # We will also output data for later purpose
        X_testDate = df_combined[
            (df_combined["Year_StdCap"] == 2023) & (df_combined["HomeTeam"] == teamName)
        ][["Date", "Capacity_StdCap", "CITY_StdCap"]].reset_index(drop=True)

        return X_train, y_train, X_test, X_testDate

    # When we need weather data
    else:

        X_train = (
            df_combined[
                (df_combined["Year_StdCap"] != 2023)
                & (df_combined["HomeTeam"] == teamName)
            ]
            .drop(dropColX, axis=1)
            .reset_index(drop=True)
        )

        # Fill the NAN to those columns that are not weather
        X_train[X_train.columns.difference(weatherCol)] = X_train[
            X_train.columns.difference(weatherCol)
        ].fillna(value=0)

        y_train = df_combined[
            (df_combined["Year_StdCap"] != 2023) & (df_combined["HomeTeam"] == teamName)
        ]["Occ_Per"].reset_index(drop=True)

        # Same with the y_train
        # y_train[y_train.columns.difference(weatherCol)] = y_train[y_train.columns.difference(weatherCol)].fillna(value=0)

        X_test = (
            df_combined[
                (df_combined["Year_StdCap"] == 2023)
                & (df_combined["HomeTeam"] == teamName)
            ]
            .drop(dropColX, axis=1)
            .reset_index(drop=True)
        )

        # Same with the X_test also
        X_test[X_test.columns.difference(weatherCol)] = X_test[
            X_test.columns.difference(weatherCol)
        ].fillna(value=0)

        # This stores the Date Column of the 2023
        # X_testDate = df_combined[(df_combined['Year_StdCap'] == 2023) & (df_combined['HomeTeam'] == teamName)][['Date', 'Capacity_StdCap', 'CITY_StdCap']].reset_index(drop=True)
        X_testDate = df_combined[
            (df_combined["Year_StdCap"] == 2023) & (df_combined["HomeTeam"] == teamName)
        ].reset_index(drop=True)

        # Return Values
        return X_train, y_train, X_test, X_testDate


# ------------------------------------------------------------------------------------------------
# Weather Fethching Function


def get_weather(cityName: str, datePlaying: str):

    temp_dict = {
        "rain_sum": [],
        "snowfall_sum": [],
        "precipitation_sum": [],
        "temp_max": [],
        "temp_min": [],
    }

    geolocator = Nominatim(user_agent="ballParkLocation")

    location = geolocator.geocode(cityName)

    lati = location.latitude
    longi = location.longitude

    temp_unit = "fahrenheit"
    start_date = datePlaying
    end_date = datePlaying

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lati}&longitude={longi}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum&temperature_unit=fahrenheit&start_date={start_date}&end_date={end_date}&timezone=America%2FChicago"

    try:

        response = requests.get(url)
        # Assign the data to the dictionary and it's respective keys
        temp_dict["rain_sum"].append(response.json()["daily"]["rain_sum"][0])
        temp_dict["snowfall_sum"].append(response.json()["daily"]["snowfall_sum"][0])
        temp_dict["precipitation_sum"].append(
            response.json()["daily"]["precipitation_sum"][0]
        )
        temp_dict["temp_max"].append(response.json()["daily"]["temperature_2m_max"][0])
        temp_dict["temp_min"].append(response.json()["daily"]["temperature_2m_min"][0])

    except:

        return "Some Error in Weather API"

    return temp_dict


# ------------------------------------------------------------------------------------------------
# Training Function


def trainModel(X_train, y_train):

    """
    Input: X_train and y_train
    Output: Model Object
    """

    # Make the pipeline

    # Columns with different types
    num_var = X_train.select_dtypes("number").columns.values
    cat_var = X_train.select_dtypes("object").columns.values

    # Columns Transformation
    colTrans = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_var),
            ("standardScale", StandardScaler(), num_var),
        ],
        remainder="passthrough",
        sparse_threshold=0,
    )

    # Pipeline with Model
    modelPipe = Pipeline(
        [
            ("coltr", colTrans),
            # ("pca", PCA(n_components=30, random_state=1122)),
            ("dt", DecisionTreeRegressor(random_state=1122)),
        ]
    )

    # Train the model
    modelPipe.fit(X_train, y_train)

    return modelPipe


# ------------------------------------------------------------------------------------------------
# Prediction function for No-Weather and Bulk


def predict2023NoWeatherBulk(modelPipe, X_test, X_testDate):

    """
    Input: modelPipe of the trained model, X_test: testing set (2023), date column
    Output:

    """
    # Predict the value
    y_pred = modelPipe.predict(X_test)

    # Predict the Attendance (We have predicted 'Occ_Per', to get atttendance we multiply with Capacity_StdCap)
    X_test["Attendance"] = np.round(y_pred * X_testDate["Capacity_StdCap"], 0)

    # Add the date column
    X_test["Date"] = X_testDate["Date"]

    # Add the Stadium Capacity
    X_test["Capacity_StdCap"] = X_testDate["Capacity_StdCap"]

    # Columns that we need
    colNeed = ["Date", "HomeTeam", "VisitingTeam", "Attendance"]

    finalDf = X_test[colNeed]

    return finalDf


# ------------------------------------------------------------------------------------------------
# Prediction function for Weather and Dynamics


def predictDynamic(modelPipe, X_testDate, datePlaying):

    # Get the weather for the specific
    cityName = X_testDate[X_testDate["Date"] == datePlaying]["CITY_StdCap"].values[0]

    # Get the weather
    weatherDict = get_weather(cityName, datePlaying)

    # Make the dataset for prediction
    predData = X_testDate[X_testDate["Date"] == datePlaying]

    # Replace the weather NaN with the fetched values
    predData["rain_sum"] = weatherDict["rain_sum"][0]
    predData["snowfall_sum"] = weatherDict["snowfall_sum"][0]
    predData["precipitation_sum"] = weatherDict["precipitation_sum"][0]
    predData["temp_max"] = weatherDict["temp_max"][0]
    predData["temp_min"] = weatherDict["temp_min"][0]

    # Drop the un-necessary columns
    dropColX = [
        "Date",
        "Year_StdCap",
        "Attendance",
        "Capacity_StdCap",
        "CITY_StdCap",
        "Occ_Per",
    ]
    finalPredData = predData.drop(dropColX, axis=1)

    # Fill the NAs with zero as decided
    finalPredData.fillna(value=0, inplace=True)

    # Make the prediction
    y_pred = modelPipe.predict(finalPredData)

    # Attach the prediction into the row that we are predicting
    predData["Predicted_Attendance"] = y_pred * predData["Capacity_StdCap"]

    # Col to output only
    colNeed = ["Date", "HomeTeam", "VisitingTeam", "Predicted_Attendance"]

    return predData[colNeed]
