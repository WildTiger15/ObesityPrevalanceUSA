'''
Sneh Duggal and Muhammad Zaidi
Intermediate Data Programming
This file tests functions from main.py
'''
# Import statements
from cse163_utils import assert_equals
import pandas as pd
from main import get_first
import matplotlib.pyplot as plt
import seaborn as sns


def round_num(num):
    return round(num, 2)


def test_groupby_average(df):
    """
    We test groupby average to gain confidence for our own process
    """
    grouped_data = df.groupby("Age")["BMI"].mean().apply(round_num)
    index_group = []
    for index in range(0, 4):
        index_group.append(grouped_data.index[index])
    grouped_data = grouped_data.tolist()
    assert_equals([25, 35, 36, 49], index_group)
    assert_equals([27.17, 29.10, 20.50, 31.20], grouped_data)
    print("Passed test")


def test_merging(data, merge_data):
    """
    We test merging with 2 mock csv to gain our confidence for our datasets
    """
    merged_df = data.merge(merge_data, on="Name")
    expected_data = pd.DataFrame({
        "Name": ["Max", "Ryan", "Mike", "Joe", "Jack", "Kyle", "Russ", "Mack"],
        "Age": [25, 49, 36, 35, 25, 49, 25, 35],
        "Height": [68, 70, 74, 65, 64, 65, 69, 67],
        "Weight": [180, 215, 160, 200, 175, 197, 183, 223],
        "BMI": [27.4, 30.8, 20.5, 33.3, 23.5, 31.6, 30.6, 24.9],
        "Income": [50000, 75000, 40000, 60000, 55000, 90000, 65000, 80000],
        "Education": [
            "Bachelor's", "Master's", "High School", "Bachelor's", "Master's",
            "Ph.D.", "Bachelor's", "Master's"
        ]
    })
    assert_equals(expected_data, merged_df)
    print("Passed test")


def test_get_first():
    """
    We test our own method which should return the first word
    in sentence
    """
    sentence = "The cow jumped up"
    first_let = get_first(sentence)
    assert_equals("The", first_let)
    print("Passed test")


def test_filtering(df):
    """
    We test filtering to gain confidence for our own datasets as we filter them a lot
    """
    filtered_df = df[df["Age"] >= 35]
    expected_df = pd.DataFrame({
        'Name': ['Ryan', 'Mike', 'Joe', 'Kyle', 'Mack'],
        'Age': [49, 36, 35, 49, 35],
        "Height": [70, 74, 65, 65, 67],
        "Weight": [215, 160, 200, 197, 223],
        "BMI": [30.8, 20.5, 33.3, 31.6, 24.9]
    })
    assert_equals(expected_df, filtered_df.reset_index(drop=True))
    print("Passed test")


# TESTING BY VISUAL INSPECTION


def test_barplot():
    """
    Testing bar plot to gain confidence with our own bar plots 
    making sure we know how it works
    """
    # Towards the end it should spike in a bar graph and be highest point
    fig, ax = plt.subplots()
    df = pd.read_csv("mock_bar.csv")
    sns.barplot(x="Year", y="Pollution", data=df)
    fig.savefig('test_bar.png')


def test_scatterplot():
    """
    We test scatter plot to gain confidence with our own scatter plots 
    making sure we know how it works
    """
    # By visual inspection dots should be postive regression
    fig, ax = plt.subplots()
    df = pd.read_csv("mock_plot.csv")
    sns.scatterplot(x="A", y="B", data=df)
    fig.savefig('test_scatter.png')


def main():
    data = pd.read_csv("mock.csv")
    merge_data = pd.read_csv("merge.csv")
    test_groupby_average(data)
    test_get_first()
    test_filtering(data)
    test_merging(data, merge_data)
    test_barplot()
    test_scatterplot()


if __name__ == '__main__':
    main()
