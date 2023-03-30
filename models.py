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


def data(genes_path):
    # path to ages
    ages_path = PSUPERTIME_PATH / 'datasets' / 'Ages.csv'

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


def cross_validation_l1(X, y, olr_alphas, linear_alphas, n_folds):
    # Kfolds splits, alphas
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []
    for olr_alpha, linear_alpha in zip(olr_alphas, linear_alphas):
        # models
        olrl1 = OrdinalClassifier(
            clf=LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=5000,
                C=1/olr_alpha
            )
        )

        linearl1 = Lasso(
            tol=0.01,
            alpha=linear_alpha
        )

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            # fit models
            olrl1.fit(X[train_index], y[train_index])
            linearl1.fit(X[train_index], y[train_index])

            # predicted test data
            y_predicted_olr = olrl1.predict(X[test_index])
            y_predicted_linear = linearl1.predict(X[test_index])

            # Kendall's Tau between predicted age and measured age
            ktau_olr = kendalltau(y_predicted_olr, y[test_index])
            ktau_linear = kendalltau(y_predicted_linear, y[test_index])

            results.append({'Model': "olrl1", 'Fold': i+1, 'Alpha': olr_alpha, 'KTau': ktau_olr[0]})
            results.append({'Model': "linearl1", 'Fold': i+1, 'Alpha': linear_alpha, 'KTau': ktau_linear[0]})

    results_df = pd.DataFrame(results, columns=['Model', 'Fold', 'Alpha', 'KTau'])

    return results_df


def crossvalidation_noreg(X, y, n_folds):
    # Kfolds splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []
    # models
    models = {
        "olr": OrdinalClassifier(
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
            # predict test data
            y_predicted = model.predict(X[test_index])
            # Kendall's Tau between predicted age and measured age
            ktau = kendalltau(y_predicted, y[test_index])

            results.append({'Model': key, 'Fold': i + 1, 'KTau': ktau[0]})

    results_df = pd.DataFrame(results, columns=['Model', 'Fold', 'KTau'])

    return results_df


def figure(filename, results_noreg, results_l1):
    # mean, sd of noreg results
    noreg_by_model = results_noreg.groupby("Model")
    noreg_means = noreg_by_model.mean()
    noreg_sds = noreg_by_model.std()

    linear_mean = noreg_means["KTau"]["linear"]
    linear_sd = noreg_sds["KTau"]["linear"]
    olr_mean = noreg_means["KTau"]["olr"]
    olr_sd = noreg_sds["KTau"]["olr"]

    f, (ax_linear, ax_olr) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

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

    ax_olr.axhline(
        olr_mean,
        linestyle='--',
        color='red',
        label='No Regularization (Mean $\pm$ SD)'
    )
    ax_olr.fill_between(
        x=(1e-10, 1),
        y1=olr_mean + olr_sd,
        y2=olr_mean - olr_sd,
        color="red",
        alpha=0.3,
        edgecolor=None
    )

    # plot alpha scan
    for model, ax in zip(["olrl1", "linearl1"], [ax_olr, ax_linear]):
        results_model = results_l1[results_l1.Model == model]
        by_alpha = results_model.groupby("Alpha")
        means = by_alpha.mean().reset_index()
        sds = by_alpha.std().reset_index()
        ax.errorbar(
            x=means['Alpha'],
            y=means['KTau'],
            yerr=sds['KTau'],
            label='L1 regularized (Mean $\pm$ SD)',
            mfc='tab:blue',
            mec='black',
            capsize=2,
            ecolor='black',
            marker='s',
            markersize=3,
            color='black'
        )

        # get x limits
        min_alpha = results_model["Alpha"].min()
        max_alpha = results_model["Alpha"].max()

        x_min = min_alpha - min_alpha / 2
        x_max = max_alpha + max_alpha / 2

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)

        ax.set_xscale('log')

    ax_linear.set_title("Linear Model")
    ax_olr.set_title("OLR Model")

    for ax in [ax_linear, ax_olr]:
        ax.set_ylabel("Kendall's Tau [-]")
        ax.set_xlabel('alpha [-]')
        ax.legend()
    plt.savefig(PSUPERTIME_PATH / 'figures' / f"{filename}.svg", dpi=300)
    plt.show()


if __name__ == '__main__':
    gene_paths = {
        "500genes": PSUPERTIME_PATH / 'datasets' / '500_variable_genes.csv',
        "50genes": PSUPERTIME_PATH / 'datasets' / '50_variable_genes.csv',
    }

    for dset_key, path in gene_paths.items():
        # get data
        features, labels = data(genes_path=path)

        # params for cv
        n_folds = 5
        n_alphas = 8
        alphass = {
            "low_resolution": {
                "olr": np.logspace(-7, -1, n_alphas),
                "linear": np.logspace(-7, -1, n_alphas),
            },
            "high_resolution": {
                "olr": np.logspace(-5, -2.5, n_alphas),
                "linear": np.logspace(-5.5, -4, n_alphas),
            },
        }

        for resolution, alphas in alphass.items():
            # cv of l1 regularized models
            results_l1 = cross_validation_l1(
                X=features,
                y=labels,
                olr_alphas=alphas["olr"],
                linear_alphas=alphas["linear"],
                n_folds=n_folds,
            )

            # cv of not-regularized models
            results_noreg = crossvalidation_noreg(
                X=features,
                y=labels,
                n_folds=n_folds,
            )

            figure(
                filename=f"Fig_{dset_key}_{resolution}",
                results_noreg=results_noreg,
                results_l1=results_l1
            )
