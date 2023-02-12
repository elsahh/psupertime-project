import path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy.stats import kendalltau
from matplotlib import pyplot as plt


def lm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        linear_model.LinearRegression: {},
        linear_model.LassoCV: {},
        # linear_model.Lasso: {"alpha": 0.00005},
    }

    for model, args in models.items():

        clf = model(**args)
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_test)
        print(y_predicted)

        ktau = kendalltau(y_predicted, y_test)
        print(ktau)
        plt.plot(y_predicted, y_test, linestyle="", marker="s")
        plt.show()
        # print(clf.alphas_)


def xy_fromdf(df):
    y = df["Age"].to_numpy()
    X = df.drop("Age", axis=1).to_numpy()
    # print(X, y)
    return X, y


if __name__ == "__main__":
    # paths to data
    genes_path = path.Path("variable_genes.csv")
    ages_path = path.Path("Ages.csv")

    # import data
    genes_df = pd.read_csv(genes_path, index_col=False, sep=",")
    ages_df = pd.read_csv(ages_path, index_col=False, sep=",")

    # set df-index to sample-labels
    ages_df = ages_df.set_index("Accession")

    genes_df = genes_df.rename(columns={"Unnamed: 0": "Accession"})
    genes_df = genes_df.set_index("Accession")

    # merge dfs, ensures correct ordering of X and y
    joined_df = genes_df.join(ages_df)
    print("There are", joined_df.isna().sum().sum(), "NaNs in the data")

    # get X (features), and y (target labels) from df
    X, y = xy_fromdf(df=joined_df)

    # linear regression
    lm(X, y)
