import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from pathlib import Path

PSUPERTIME_PATH = Path(__file__).parent

FIGURE_PATH = PSUPERTIME_PATH / "figures"


def lm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    models = {
        linear_model.LinearRegression: {},
        # linear_model.LassoCV: {"tol": 0.01},
        # linear_model.Lasso: {"alpha": 0.0001, "tol": 0.01},
        # linear_model.RidgeCV: {},
    }

    for model, args in models.items():

        classifier = model(**args)
        classifier.fit(X_train, y_train)

        y_predicted = classifier.predict(X_test)
        # print(y_predicted)

        ktau = kendalltau(y_predicted, y_test)
        print(ktau)
        # plt.plot(y_predicted, y_test, linestyle="", marker="s")
        # plt.xlabel("Predicted Age [Yr]")
        # plt.ylabel("Age [Yr]")
        # plt.show()
        # print(classifier.alphas_)
    return ktau[0]


def cross_validation(X, y):
    results = []
    kf = KFold(n_splits=2, shuffle=True)
    alphas = np.logspace(-7, -3, 3)
    for alpha in alphas:
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            classifier = linear_model.Lasso(alpha=alpha)
            classifier.fit(X[train_index], y[train_index])
            y_predicted = classifier.predict(X[test_index])
            # f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            # f.suptitle(f"alpha: {alpha}")

            # ax.scatter(y_predicted, y[test_index])
            # ax.set_ylabel("Age", fontdict={"size": 18, "weight": "bold"})
            # ax.set_xlabel("Predicted Age", fontdict={"size": 18, "weight": "bold"})
            ktau = kendalltau(y_predicted, y[test_index])
            results.append({"Fold": i, "alpha": alpha, "KTau": ktau[0]})
            # plt.show()
    df = pd.DataFrame(results, columns=["Fold", "alpha", "KTau"])
    by_alpha = df[["alpha", "KTau"]].groupby("alpha")
    mean = by_alpha.mean().reset_index()
    sd = by_alpha.std().reset_index()

    return mean, sd


def xy_fromdf(df):
    y = df["Age"].to_numpy()
    X = df.drop("Age", axis=1).to_numpy()
    # print(X, y)
    return X, y


if __name__ == "__main__":
    # paths to data
    genes_path = PSUPERTIME_PATH / "variable_genes.csv"
    ages_path = PSUPERTIME_PATH / "Ages.csv"

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
    lin_ktau = lm(X, y)

    mean, sd = cross_validation(X, y)

    f_alpha, ax_alpha = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax_alpha.errorbar(mean["alpha"], mean["KTau"], yerr=sd["KTau"], label="Lasso Regression $\pm$ SD")
    ax_alpha.axhline(lin_ktau, linestyle="--", color="red", label="Linear Regression")

    ax_alpha.set_xscale("log")
    ax_alpha.set_ylim(0, 1)
    ax_alpha.set_ylabel("Kendall's Tau [-]")
    ax_alpha.set_xlabel("Alpha")
    ax_alpha.legend()
    plt.show()
