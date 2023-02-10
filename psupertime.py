from sklearn import linear_model
import path
import pandas as pd


def lm():
    clf = linear_model.LinearRegression()



if __name__ == "__main__":
    data_path = path.Path(". / processed_data.csv")

    data = pd.read_csv(data_path, index_col=False, sep=",")
    print(data)
