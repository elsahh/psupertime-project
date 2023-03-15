import path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


def lm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    models = {
        # linear_model.LinearRegression: {},
        # linear_model.LassoCV: {"tol": 0.01},
        # linear_model.Lasso: {"alpha": 0.0001, "tol": 0.01},
        linear_model.RidgeCV: {},
    }

    for model, args in models.items():

        classifier = model(**args)
        classifier.fit(X_train, y_train)

        y_predicted = classifier.predict(X_test)
        # print(y_predicted)

        ktau = kendalltau(y_predicted, y_test)
        print(ktau)
        plt.plot(y_predicted, y_test, linestyle="", marker="s")
        plt.xlabel("Predicted Age [Yr]")
        plt.ylabel("Age [Yr]")
        plt.show()
        # print(classifier.alphas_)


def cross_validation(X, y):
    results = []
    kf = KFold(n_splits=6, shuffle=True)
    alphas = np.logspace(-11, -4, 8)
    for alpha in alphas:
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            classifier = linear_model.Lasso(alpha=alpha)
            classifier.fit(X[train_index], y[train_index])
            y_predicted = classifier.predict(X[test_index])
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            f.suptitle(f"alpha: {alpha}")

            ax.scatter(y_predicted, y[test_index])
            ax.set_ylabel("Age", fontdict={"size": 18, "weight": "bold"})
            ax.set_xlabel("Predicted Age", fontdict={"size": 18, "weight": "bold"})
            ktau = kendalltau(y_predicted, y[test_index])
            results.append({"Fold": i, "alpha": alpha, "KTau": ktau})
            plt.show()
    df = pd.DataFrame(results, columns=["Fold", "alpha", "KTau"])
    print(df.sort_values(by="alpha"))


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
    # lm(X, y)

    cross_validation(X, y)

    # lasso = linear_model.Lasso(random_state=0, max_iter=10000)
    # alphas = np.logspace(-4, -0.5, 30)
    #
    # tuned_parameters = [{"alpha": alphas}]
    # n_folds = 5
    #
    # clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
    # clf.fit(X, y)
    # scores = clf.cv_results_["mean_test_score"]
    # scores_std = clf.cv_results_["std_test_score"]
    #
    # plt.figure().set_size_inches(8, 6)
    # plt.semilogx(alphas, scores)
    #
    # std_error = scores_std / np.sqrt(n_folds)
    #
    # plt.semilogx(alphas, scores + std_error, "b--")
    # plt.semilogx(alphas, scores - std_error, "b--")
    #
    # # alpha=0.2 controls the translucency of the fill color
    # plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
    #
    # plt.ylabel("CV score +/- std error")
    # plt.xlabel("alpha")
    # plt.axhline(np.max(scores), linestyle="--", color=".5")
    # plt.xlim([alphas[0], alphas[-1]])
    # plt.show()