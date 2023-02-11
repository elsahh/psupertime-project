from sklearn import linear_model
import path
import pandas as pd


def lm(X, y):
    clf = linear_model.LinearRegression()
    clf.fit(X, y)


def xy_fromdf(df):
    y = df["Age"].to_numpy()
    X = df.drop("Age", axis=1).to_numpy()
    # FIXME: convert both to numpy array
    return X, y


if __name__ == "__main__":
    # paths to data
    genes_path = path.Path("processed_data.csv")
    ages_path = path.Path("Ages.csv")

    # import data
    genes_df = pd.read_csv(genes_path, index_col=False, sep=",")
    ages_df = pd.read_csv(ages_path, index_col=False, sep=",")

    # set df-index to sample-labels
    ages_df = ages_df.set_index("Accession")

    # merge dfs, ensures correct ordering fo X and y
    joined_df = genes_df.join(ages_df)
    print(joined_df)

    # get X (features), and y (target labels) from df
    X, y = xy_fromdf(df=joined_df)

    # linear regression
    lm(X, y)
