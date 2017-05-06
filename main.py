# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def drop_unnecessary_features(df):
    drop_list = ['Ticket', 'Cabin', 'Name', 'PassengerId']
    print("Before", df.shape)
    df = df.drop(drop_list, axis=1)
    print("After", df.shape)

def enhance(combine):
    create_title(combine)

def create_title(combine):
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

def enhance_sex(combine):
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

def main():
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    combine = [train_df, test_df]

    enhance(combine)

    drop_unnecessary_features(train_df)
    drop_unnecessary_features(test_df)

    combine = [train_df, test_df]

main()