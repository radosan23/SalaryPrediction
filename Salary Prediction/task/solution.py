import os
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from itertools import combinations


def check_data():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')
    if 'data.csv' not in os.listdir('../Data'):
        url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/data.csv', 'wb').write(r.content)


def remove_correlated(X, y):
    correlated = X.corr()[(X.corr() > 0.2) & (X.corr() < 1)].dropna(how='all').index.tolist()
    correlated.extend(map(list, list(combinations(correlated, 2))))
    scores = {}
    for i in correlated:
        X_mod = X.drop(i, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X_mod, y, test_size=0.3, random_state=100)
        model = LinearRegression()
        model.fit(X_train, y_train)
        scores[i if type(i) == str else '_'.join(i)] = mape(y_test, model.predict(X_test))
    return X.drop(min(scores, key=lambda x: scores[x]).split('_'), axis=1)


def main():
    check_data()
    df = pd.read_csv('../Data/data.csv')
    X, y = df.drop('salary', axis=1), df['salary']
    X_mod = remove_correlated(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_mod, y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    scores = {}
    for replace in (0, y_train.median()):
        prediction_mod = prediction
        prediction_mod[prediction_mod < 0] = replace
        scores[replace] = mape(y_test, prediction_mod)
    print(f'{min(scores.values()):.5f}')


if __name__ == '__main__':
    main()
