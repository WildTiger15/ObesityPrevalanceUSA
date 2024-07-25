# Import statements
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns
import numpy as np
import os
import tempfile
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Loading in csv datasets and assigning to variables
age_above_18_data = pd.read_csv("Obesity_by_age_18+.csv")
age_under_18_data = pd.read_csv("Obesity_by_age_under_18.csv")
income_data = pd.read_csv("Obesity_by_income.csv")
over_time_data = pd.read_csv("Obesity_over_time.csv")
# Filtering unnecessary columns
income_data = income_data[[
    "State", "Year", "Adult.Obesity*100", "Average.Age", "Average.Income",
    "Population", "Poverty.Rate*100", "Real.GDP", "Real.GDP.Growth*100",
    "Real.Personal.Income", "Region", "Real.GDP.Per.Capita"
]]


def check_state(state):
    """
    This method checks if the input is a state, or is Alaska/Hawaii
    We remove some states which ruin the geopstaial graphs
    Parameters: state (string name of the state in the series)
    Return: boolean
    """
    not_states = [
        "American Samoa", "District of Columbia", "Puerto Rico",
        "Virgin Islands", "Guam", "Palau", "Marshall Islands",
        "Northern Mariana Islands", "Fed States of Micronesia", "Hawaii",
        "Alaska"
    ]
    if state in not_states:
        return False
    return True


def obesity_v_income(df):
    """
    This method plots a scatterplot of Adult obesity prevalence (%)
    compared to the average income
    Parameters: df (DataFrame with income data)
    """
    # Creates a figure and axis to plot on
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Average.Income", y="Adult.Obesity*100")
    plt.title("Obesity Vs Income")
    plt.xlabel("Income")
    plt.ylabel("Obesity level (BMI)")
    # Saves plot to income.png
    fig.savefig("income.png", bbox_inches="tight")


def obesity_v_poverty(df):
    """
    This method plots a scactter plot comparing poverty rates and
    obesity Prevalance in percentage of population
    Parameters: df (DataFrame with income data)
    """
    # Creates a figure and axis to plot on
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Poverty.Rate*100", y="Adult.Obesity*100")
    plt.xlabel("Poverty rate in U.S. (%)")
    plt.ylabel("Average Obesity Prevalence (%)")
    plt.title("Poverty Rates vs Obesity Prevalence")
    # Saves plot to poverty.png
    fig.savefig("poverty.png", bbox_inches="tight")


def obesity_gdp_capita(df):
    """
    This method graphs scatter plot of GDP per capita and Obesity prevalance
    Parameters: df (DataFrame with income datat)
    """
    # Creates a figure and axis to plot on
    fig, ax = plt.subplots()
    # Applies a lambda function to get gdp per capita for each row
    df["Real.GDP.Per.Capita"] = df["Real.GDP.Per.Capita"].apply(
        lambda x: x * 70248.63)
    sns.scatterplot(data=df, x="Real.GDP.Per.Capita", y="Adult.Obesity*100")
    plt.xlabel("GDP per Capita ($)")
    plt.ylabel("Average BMI for U.S. Adults")
    plt.title("Obesity vs GDP per Capita")
    # Saves plot to gdp_per_capita.png
    fig.savefig("gdp_per_capita.png", bbox_inches="tight")


def obesity_over_18(df):
    """
    This method plots bar graph of year and prevalance of
    obesity over 18 population in the United States
    Parameters: df (over 18 dataset)
    """
    # Creates a figure and 2 axes to plot on
    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    sns.barplot(data=df,
                x="Year",
                y="United States, Male",
                # ChatGPT to understand how to use RGB values
                color=(.2, .4, .6),
                ax=ax[0])
    ax[0].set_xticks(np.arange(0, 41, 5))
    sns.barplot(data=df,
                x="Year",
                y="United States, Female",
                color=(.6, .4, .2),
                ax=ax[1])
    # Sets labels for plots and saves on over_18.png
    # We used ChatGPT to understand set_xticks with np.arange
    ax[1].set_xticks(np.arange(0, 41, 5))
    ax[0].set_ylabel("U.S. Obesity % (Male)")
    ax[1].set_ylabel("U.S. Obesity % (Female)")
    ax[0].set_title("U.S. Obesity in Adults 18+ Males")
    ax[1].set_title("U.S. Obesity in Adults 18+ Females")
    fig.savefig("over_18.png", bbox_inches="tight")


def obesity_under_18(df):
    """
    This method plots bar graph of year and prevalance of
    obesity under 18 population in the United States
    Parameters: df (of under_18 dataset)
    """
    # Creates a figure and axis to plot on
    fig, ax = plt.subplots()
    sns.barplot(x='YEAR', y='ESTIMATE', ax=ax, data=df, color=(.74, .71, .41))
    plt.xticks(rotation=50)
    plt.xlabel('Year')
    plt.ylabel('U.S. Obesity (%)')
    plt.title('Obesity among Children and Adolescents in U.S. (2-19 years)')
    fig.savefig("under_18.png", bbox_inches="tight")


def income_and_obesity_geospatial(income):
    """
    This method plots 2 geospatial 
    plots on same fig. First one showing representaiotn
    of average income in each state. Second plot showing prevalance of
    Obesity in state
    Parameters: income(dataframe of income.csv)
    """
    # Creates a figure and axes to plot on
    fig, [ax1, ax2] = plt.subplots(2, figsize=(70, 10))
    # loads in shape file for geospatial plotting
    shapefile = gpd.read_file("s_08mr23.shp")
    obesity = income.groupby("State")["Adult.Obesity*100"].mean()
    shapefile["is_state"] = shapefile["NAME"].apply(check_state)
    shapefile = shapefile[shapefile["is_state"]]
    obesity = obesity.drop("Alaska")
    obesity = obesity.drop("Hawaii")
    # merging
    merged = shapefile.merge(obesity, left_on="NAME", right_on="State")
    ax1.set_title("Average Obesity (BMI) By State")
    merged.plot(ax=ax1,
                edgecolor='black',
                column="Adult.Obesity*100",
                legend=True)
    # groupby
    income = income.groupby("State")["Average.Income"].mean()
    shapefile["is_state"] = shapefile["NAME"].apply(check_state)
    shapefile = shapefile[shapefile["is_state"]]
    income = income.drop("Alaska")
    income = income.drop("Hawaii")
    merged = shapefile.merge(income, left_on="NAME", right_on="State")
    ax2.set_title("Average Income By State")
    merged.plot(ax=ax2,
                edgecolor='black',
                column="Average.Income",
                legend=True)

    fig.savefig("geo_obesity_income.png", bbox_inches="tight")


def obesity_v_time(df):
    """
    This method plots a bar graph of all groups and obesity 
    prevalance over time
    Parameters: df(of over_time.csv)
    """
    # Creates a figure and axis to plot on
    fig, ax = plt.subplots()
    df = df.T
    df = df[[189, 190]]
    df.reset_index(inplace=True)
    df = df.iloc[1:, :2]
    df.columns = ["Year", "United States"]
    df["United States"] = df["United States"].apply(get_first)
    # Applies function to remove rows with years that have decimals
    df["is_year"] = df["Year"].apply(is_decimal)
    df = df[df["is_year"] == True]
    df = df.sort_values('Year', ascending=True)
    df = df.sort_index(ascending=True)
    df = df.reset_index(drop=True)
    df["Year"] = df["Year"].apply(convert_to_int)
    df["United States"] = df["United States"].apply(convert_to_float)
    # Plots barplot and sets color to rgb values
    sns.barplot(data=df, x="Year", y="United States", color=(.74, .71, .41))
    ax.set_xticks(np.arange(0, 41, 5))
    ax.set_yticks(np.arange(0, 40, 5))
    plt.ylabel("U.S. Obesity Percentage (%)")
    plt.title("Obesity over Time")
    fig.savefig("time.png")


def ml_model(df):
    """
    This method runs our machine learning model with features being
    all columns besides state, region, and population and labels being
    just the pravalance percentage
    """
    # Removes columns unnecessary for machine learning
    df = df.loc[:, df.columns != "State"]
    df = df.loc[:, df.columns != "Region"]
    df = df.loc[:, df.columns != "Population"]
    # Applies lambda fuction to round obesity percentages
    df["Adult.Obesity*100"] = df["Adult.Obesity*100"].apply(lambda x: round(x))
    model = DecisionTreeRegressor(max_depth=4)
    features = df.loc[:, df.columns != "Adult.Obesity*100"]
    labels = df["Adult.Obesity*100"]
    # Splits data and trains model
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=65)
    model.fit(features_train, labels_train)
    # Scores the training set and testing set then prints them
    training_score = model.score(features_train, labels_train)
    test_score = model.score(features_test, labels_test)
    print("Training accuracy: " + str(training_score))
    print("Test accuracy: " + str(test_score))


def convert_to_float(string_num):
    """
    This returns the inputted string to float
    """
    return float(string_num)


def convert_to_int(string_num):
    """
    This method returns string to num
    """
    return int(string_num)


def is_decimal(number):
    """
    This returns true if number is decimal otherwise no
    """
    number = float(number)
    if number % 1 == 0:
        return True
    else:
        return False


def get_first(input):
    """
    This method returns the first word of the array
    """
    return input.split()[0]


def main():
    """
    The main method calls all plot methods and machine learning method
    """
    obesity_v_income(income_data)
    obesity_v_poverty(income_data)
    obesity_gdp_capita(income_data)
    obesity_over_18(age_above_18_data)
    obesity_under_18(age_under_18_data)
    obesity_v_time(over_time_data)
    income_and_obesity_geospatial(income_data)
    ml_model(income_data)


if __name__ == '__main__':
    main()
