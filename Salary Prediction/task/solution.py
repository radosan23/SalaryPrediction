import os
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


def check_data():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')
    if 'data.csv' not in os.listdir('../Data'):
        url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/data.csv', 'wb').write(r.content)


def main():
    check_data()
    df = pd.read_csv('../Data/data.csv')
    X, y = df[['rating']], df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = mape(y_test, prediction)
    print(f'{model.intercept_:.5f} {model.coef_[0]:.5f} {score:.5f}')


if __name__ == '__main__':
    main()
