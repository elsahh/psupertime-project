import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso

PSUPERTIME_PATH = Path(__file__).parent

FIGURE_PATH = PSUPERTIME_PATH / 'figures'


def data(genes_path):
    """Helper Function for data import and formatting."""
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
    genes = joined_df.drop('Age', axis=1).columns
    return X, y, genes


def get_coeffs(X, y):
    """Get all beta coefficients of linear and linear l1 models."""
    linearl1 = Lasso(
        tol=0.01,
        alpha=np.power(10, -4.9)
    )
    linear = LinearRegression()

    # fit models
    linearl1.fit(X, y)
    linear.fit(X, y)

    # return betas
    return linear.coef_, linearl1.coef_


def get_max_betas(genes, betas, n):
    """Get the n maximal beta values and their corresponding genes"""
    abs_betas = np.abs(betas)
    max_beta_indices = np.argpartition(abs_betas, -n)[-n:]
    max_betas = betas[max_beta_indices]
    relevant_genes = genes[max_beta_indices]

    return relevant_genes, max_betas


if __name__ == '__main__':
    # data path
    genes_path = PSUPERTIME_PATH / 'datasets' / '500_variable_genes.csv'

    # get data
    features, labels, genes = data(genes_path=genes_path)

    # get betas
    linear_betas, l1_betas = get_coeffs(X=features, y=labels)

    # make histogram of all beta values for both models
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    bins = np.linspace(-25000, 20000, 25)
    ax.hist(linear_betas, bins=bins, label="Linear model", alpha=1, linestyle="-", edgecolor='black')
    ax.hist(l1_betas, bins=bins, label="Linear L1 model", alpha=0.8, linestyle="-", edgecolor='black')
    ax.set_xlabel("beta")
    ax.set_ylabel("Frequency")
    ax.legend()
    f.savefig(PSUPERTIME_PATH / 'figures' / f"histogram.svg")

    plt.show()

    betas = {
        "linear": linear_betas,
        "linearl1": l1_betas
    }

    # how many zeros are in l1 betas
    print(len(np.where(l1_betas == 0.0)[0]))

    # save results of most relevant genes
    for model, model_betas in betas.items():
        relevant_genes, max_betas = get_max_betas(genes=genes, betas=model_betas, n=50)
        df = pd.DataFrame({"Gene": relevant_genes, "Beta": max_betas}).sort_values(by="Beta")
        df.to_csv(
            PSUPERTIME_PATH / "results" / f"{model}_relevant_genes.tsv",
            sep='\t',
            index=False
        )
