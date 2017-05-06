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
    drop_list = ['Ticket', 'Cabin', 'Name', 'PassengerId', 'Parch', 'SibSp', 'FamilySize']
    print("Before", df.shape)
    df = df.drop(drop_list, axis=1)
    print("After", df.shape)

def enhance(combine):
    create_title(combine)
    enhance_sex(combine)
    enhance_age(combine)
    enhance_family_size(combine)
    enhance_is_alone(combine)
    enhance_age_class(combine)
    enhance_embarked(combine)
    enhance_fare(combine)

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

def enhance_age(combine):
    guess_ages = np.zeros((2,3))

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                    (dataset['Pclass'] == j+1)]['Age'].dropna()

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
                
        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)
    
    combine[0]['AgeBand'] = pd.cut(combine[0]['Age'], 5)
    combine[0][['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']

    combine[0] = combine[0].drop(['AgeBand'], axis=1)

def enhance_family_size(combine):
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
def enhance_is_alone(combine):
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

def enhance_age_class(combine):
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

def enhance_embarked(combine):
    freq_port = combine[0].Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
def enhance_fare(combine):
    combine[1]['Fare'].fillna(combine[0]['Fare'].dropna().median(), inplace=True)
    
    combine[0]['FareBand'] = pd.qcut(combine[0]['Fare'], 4)

    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    combine[0] = combine[0].drop(['FareBand'], axis=1)

def solve(train_df, test_df):
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    acc_random_forest

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv('./data/submission.csv', index=False)

def main():
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    combine = [train_df, test_df]

    enhance(combine)

    train_df = combine[0]
    test_df = combine[1]

    train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin', 'Name', 'Parch', 'SibSp', 'FamilySize'], axis=1)

    solve(train_df, test_df)

main()