import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from ordinal_classifier import OrdinalClassifier

PSUPERTIME_PATH = Path(__file__).parent

FIGURE_PATH = PSUPERTIME_PATH / 'figures'


def data():
    # paths to data
    genes_path = PSUPERTIME_PATH / 'variable_genes.csv'
    ages_path = PSUPERTIME_PATH / 'Ages.csv'

    # import data
    genes_df = pd.read_csv(genes_path, index_col=False, sep=',')
    ages_df = pd.read_csv(ages_path, index_col=False, sep=',')

    # set df-index to sample-labels
    ages_df = ages_df.set_index('Accession')
    genes_df = genes_df.rename(columns={'Unnamed: 0': 'Accession'})
    genes_df = genes_df.set_index('Accession')

    # merge dfs, ensures correct ordering of X and y
    joined_df = genes_df.join(ages_df)
    print('There are', joined_df.isna().sum().sum(), 'NaNs in the data')

    # get Features X and targets y
    y = joined_df['Age'].to_numpy()
    X = joined_df.drop('Age', axis=1).to_numpy()
    return X, y


def predictions(X, y):
    # models
    models = {
        "ordinall1": OrdinalClassifier(
            clf=LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=5000,
                C=10e4
            )
        ),

        "linearl1": Lasso(
            tol=0.01,
            alpha=np.power(10, -4.8)
        ),

        "ordinal": OrdinalClassifier(
            clf=LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=5000
            )
        ),

        "linear": LinearRegression()
    }

    ages = [1, 5, 6, 21, 22, 38, 44, 54]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = pd.DataFrame()
    for key, model in models.items():
        # fit model
        model.fit(X_train, y_train)
        # predicted test data
        y_predicted = model.predict(X_test)
        if key.startswith("ordinal"):
            ages_from_label = []
            for label in y_predicted:
                ages_from_label.append(ages[label])
            y_predicted = ages_from_label

        results[key] = y_predicted
    print(results)

    return X_test, results


if __name__ == '__main__':

    # get data
    features, labels = data()

    X_test, results = predictions(X=features, y=labels)

    titles = {
        "ordinall1": "OLR L1",
        "linearl1": "Linear L1",
        "ordinal": "OLR",
        "linear": "Linear"
    }

    pca = PCA(n_components=5)
    pca.fit(X_test)

    evr = pca.explained_variance_ratio_
    print(evr)

    components = pca.transform(X_test).T

    results["pca1"] = components[0]
    results["pca2"] = components[1]

    results = results[(results["linear"] >= -20) & (results["linear"] <= 65)]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 9), sharex='all', sharey='all')
    print(components[0].shape)
    for model, ax in zip(titles.keys(), [ax1, ax2, ax3, ax4]):
        ax.scatter(
            x=results["pca2"],
            y=results["pca1"],
            c=results[model],
            alpha=0.7,
            cmap="autumn",
            # edgecolors='black',
            linewidths=0.4,
            s=14,
        )
        ax.set_title(titles[model], weight='bold')

        ax.set_xlabel(f"PC 2 ({evr[1].round(2) * 100}%)")
        ax.set_ylabel(f"PC 1 ({evr[0].round(2) * 100}%)")

        ax.set_xlim(-0.005, 0.008)
        ax.set_ylim(-0.005, 0.013)
    plt.show()
