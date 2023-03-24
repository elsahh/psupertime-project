import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import kendalltau

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


def cross_validation_l1(X, y, alphas, n_folds):
    # Kfolds splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []
    for alpha in alphas:
        # models
        models = {
            "ordinall1": OrdinalClassifier(
                clf=LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    max_iter=5000,
                    C=1/alpha
                )
            ),

            "linearl1": Lasso(
                tol=0.01,
                alpha=alpha
            )
        }

        for key, model in models.items():
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                # fit model
                model.fit(X[train_index], y[train_index])
                # predicted test data
                y_predicted = model.predict(X[test_index])
                # Kendall's Tau between predicted age and measured age
                ktau = kendalltau(y_predicted, y[test_index])

                results.append({'Model': key, 'Fold': i+1, 'Alpha': alpha, 'KTau': ktau[0]})

    results_df = pd.DataFrame(results, columns=['Model', 'Fold', 'Alpha', 'KTau'])

    return results_df


def crossvalidation_noreg(X, y, n_folds):
    # Kfolds splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []
    # models
    models = {
        "ordinal": OrdinalClassifier(
            clf=LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=5000
            )
        ),

        "linear": LinearRegression()
    }

    for key, model in models.items():
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            # fit model
            model.fit(X[train_index], y[train_index])
            # predicted test data
            y_predicted = model.predict(X[test_index])
            # Kendall's Tau between predicted age and measured age
            ktau = kendalltau(y_predicted, y[test_index])

            results.append({'Model': key, 'Fold': i + 1, 'KTau': ktau[0]})

    results_df = pd.DataFrame(results, columns=['Model', 'Fold', 'KTau'])

    return results_df


def figure(results_noreg, results_l1):
    print(results_noreg, "\n", results_l1)

    # mean, sd of noreg results
    noreg_by_model = results_noreg.groupby("Model")
    noreg_means = noreg_by_model.mean()
    noreg_sds = noreg_by_model.std()

    linear_mean = noreg_means["KTau"]["linear"]
    linear_sd = noreg_sds["KTau"]["linear"]
    ordinal_mean = noreg_means["KTau"]["ordinal"]
    ordinal_sd = noreg_sds["KTau"]["ordinal"]

    f, (ax_linear, ax_ordlog) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # plot red reference values
    ax_linear.axhline(
        linear_mean,
        linestyle='--',
        color='red',
        label='No Regularization (Mean $\pm$ SD)'
    )
    ax_linear.fill_between(
        x=(1e-10, 1),
        y1=linear_mean + linear_sd,
        y2=linear_mean - linear_sd,
        color="red",
        alpha=0.3,
        edgecolor=None
    )

    ax_ordlog.axhline(
        ordinal_mean,
        linestyle='--',
        color='red',
        label='No Regularization (Mean $\pm$ SD)'
    )
    ax_ordlog.fill_between(
        x=(1e-10, 1),
        y1=ordinal_mean + ordinal_sd,
        y2=ordinal_mean - ordinal_sd,
        color="red",
        alpha=0.3,
        edgecolor=None
    )

    # plot alpha scan
    print(results_l1)
    for model, ax in zip(["ordinall1", "linearl1"], [ax_ordlog, ax_linear]):
        results_model = results_l1[results_l1.Model == model]
        by_alpha = results_model.groupby("Alpha")
        means = by_alpha.mean().reset_index()
        sds = by_alpha.std().reset_index()
        print(means, sds)
        ax.errorbar(
            x=means['Alpha'],
            y=means['KTau'],
            yerr=sds['KTau'],
            label='L1 regularized (Mean $\pm$ SD)'
        )

    # get x limits
    min_alpha = results_l1["Alpha"].min()
    max_alpha = results_l1["Alpha"].max()

    x_min = min_alpha - min_alpha / 2
    x_max = max_alpha + max_alpha / 2

    for ax in [ax_linear, ax_ordlog]:
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Kendall's Tau [-]")
        ax.set_xlabel('Alpha [-]')
        ax.legend()
    plt.show()


if __name__ == '__main__':

    # get data
    features, labels = data()

    # params for cv
    alphas = np.logspace(-7, -1, 10)
    n_folds = 5

    # cv of l1 regularized models
    results_l1 = cross_validation_l1(
        X=features,
        y=labels,
        alphas=alphas,
        n_folds=n_folds,
    )

    # cv of not-regularized models
    results_noreg = crossvalidation_noreg(
        X=features,
        y=labels,
        n_folds=n_folds,
    )

    figure(results_noreg=results_noreg, results_l1=results_l1)
