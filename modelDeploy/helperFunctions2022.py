# Libraries Import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from lightgbm import LGBMRegressor

# weather
from geopy.geocoders import Nominatim
import requests


def prepareDataTeam2022(df_combined, teamName, withWeather: bool):

    """
    Input: Data-frame that is combined, team name and data with weather or not
    Output: X_train, y_train, X_test, X_testData
    """
    # Columns list for subsetting
    dropColX = [
        "Date",
        "Year",
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
                (df_combined["Year"] != 2023)
                & (df_combined["Year"] != 2022)
                & (df_combined["HomeTeam"] == teamName)
            ]
            .drop(dropColX + weatherCol, axis=1)
            .reset_index(drop=True)
        )
        X_train.fillna(value=0, inplace=True)

        y_train = df_combined[
            (df_combined["Year"] != 2023)
            & (df_combined["HomeTeam"] == teamName)
            & (df_combined["Year"] != 2022)
        ]["Occ_Per"].reset_index(drop=True)

        X_test = (
            df_combined[
                (df_combined["Year"] == 2022) & (df_combined["HomeTeam"] == teamName)
            ]
            .drop(dropColX + weatherCol, axis=1)
            .reset_index(drop=True)
        )
        X_test.fillna(value=0, inplace=True)

        y_test = df_combined[
            (df_combined["Year"] == 2022) & (df_combined["HomeTeam"] == teamName)
        ]["Occ_Per"].reset_index(drop=True)

        # We will also output data for later purpose
        X_testDate = df_combined[
            (df_combined["Year"] == 2022) & (df_combined["HomeTeam"] == teamName)
        ][["Date", "Capacity_StdCap", "CITY_StdCap", "Attendance"]].reset_index(
            drop=True
        )

        return X_train, y_train, X_test, y_test, X_testDate

    # When we need weather data
    else:

        X_train = (
            df_combined[
                (df_combined["Year"] != 2023)
                & (df_combined["HomeTeam"] == teamName)
                & (df_combined["Year"] != 2022)
            ]
            .drop(dropColX, axis=1)
            .reset_index(drop=True)
        )

        # Fill the NAN to those columns that are not weather
        X_train[X_train.columns.difference(weatherCol)] = X_train[
            X_train.columns.difference(weatherCol)
        ].fillna(value=0)

        y_train = df_combined[
            (df_combined["Year"] != 2023)
            & (df_combined["HomeTeam"] == teamName)
            & (df_combined["Year"] != 2022)
        ]["Occ_Per"].reset_index(drop=True)

        # Same with the y_train
        # y_train[y_train.columns.difference(weatherCol)] = y_train[y_train.columns.difference(weatherCol)].fillna(value=0)

        X_test = (
            df_combined[
                (df_combined["Year"] == 2022) & (df_combined["HomeTeam"] == teamName)
            ]
            .drop(dropColX, axis=1)
            .reset_index(drop=True)
        )

        # Same with the X_test also
        X_test[X_test.columns.difference(weatherCol)] = X_test[
            X_test.columns.difference(weatherCol)
        ].fillna(value=0)

        y_test = df_combined[
            (df_combined["Year"] == 2022) & (df_combined["HomeTeam"] == teamName)
        ]["Occ_Per"].reset_index(drop=True)

        # This stores the Date Column of the 2023
        # X_testDate = df_combined[(df_combined['Year_StdCap'] == 2023) & (df_combined['HomeTeam'] == teamName)][['Date', 'Capacity_StdCap', 'CITY_StdCap']].reset_index(drop=True)
        X_testDate = df_combined[
            (df_combined["Year"] == 2022) & (df_combined["HomeTeam"] == teamName)
        ].reset_index(drop=True)

        # Return Values
        return X_train, y_train, X_test, y_test, X_testDate


def trainModel2022(X_train, y_train):

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

    lgbmMdl = LGBMRegressor(
        boosting_type="gbdt",
        class_weight=None,
        colsample_bytree=1.0,
        importance_type="split",
        learning_rate=0.1,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=0.001,
        min_split_gain=0.0,
        n_estimators=100,
        n_jobs=-1,
        num_leaves=31,
        objective=None,
        random_state=746,
        reg_alpha=0.0,
        reg_lambda=0.0,
        silent="warn",
        subsample=1.0,
        subsample_for_bin=200000,
        subsample_freq=0,
    )

    # Pipeline with Model
    modelPipe = Pipeline(
        [
            ("coltr", colTrans),
            # ("pca", PCA(n_components=30, random_state=1122)),
            ("lgbm", lgbmMdl),
        ]
    )

    # Train the model
    modelPipe.fit(X_train, y_train)

    return modelPipe


def predictNoWeatherBulk2022(modelPipe, X_test, X_testDate):

    """
    Input: modelPipe of the trained model, X_test: testing set (2023), date column
    Output:

    """
    # Predict the value
    y_pred = modelPipe.predict(X_test)

    # Predict the Attendance (We have predicted 'Occ_Per', to get atttendance we multiply with Capacity_StdCap)
    X_test["Predicted_Attendance"] = np.round(y_pred * X_testDate["Capacity_StdCap"], 0)

    # Add the date column
    X_test["Date"] = X_testDate["Date"]

    # Add the Stadium Capacity
    X_test["Capacity_StdCap"] = X_testDate["Capacity_StdCap"]

    X_test["True_Attendance"] = X_testDate["Attendance"]

    # Columns that we need
    colNeed = [
        "Date",
        "HomeTeam",
        "VisitingTeam",
        "Predicted_Attendance",
        "True_Attendance",
    ]

    finalDf = X_test[colNeed]

    return finalDf


def plot2022(finalDf):

    # Change the finalDf
    finalDfMelt = finalDf.melt(
        id_vars=["Date", "HomeTeam", "VisitingTeam"],
        value_vars=["Predicted_Attendance", "True_Attendance"],
        var_name="Attendance Type",
        value_name="Attendance",
    )
    finalDfMelt["ShortDate"] = finalDfMelt["Date"].apply(lambda x: x[5:])

    # Team's name
    teamName = finalDf["HomeTeam"].unique()[0]
    pltTitle = f"Actual vs Predicted Attendance of {teamName} Team for 2022"

    # Create line plot using Seaborn
    fig, ax = plt.subplots()
    sns.lineplot(
        x="ShortDate", y="Attendance", hue="Attendance Type", data=finalDfMelt, ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize=4)
    ax.set_title(pltTitle)

    return fig, ax


