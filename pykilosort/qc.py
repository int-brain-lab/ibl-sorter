from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_covariance_matrix(covariance_matrix, out_path):
    """
    Function to plot covariance matrix
    :param covariance_matrix:
    :param out_path:
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set(title="Data covariance matrix", ylabel='Channel', xlabel='Channel')
    plt.imshow(20 * np.log10(np.abs(covariance_matrix)), vmin=0, vmax=60), plt.colorbar()
    if out_path is not None:
        fig.savefig(Path(out_path).joinpath('qc_covariance_matrix.png'))
        plt.close(fig)
    else:
        return fig, ax


def plot_whitening_matrix(wrot, whitening_range=32, out_path=None):
    """
    Function to plot whitening matrix
    :param wrot:
    :param whitening_range:
    :return:
    """
    nc = wrot.shape[0]
    qc_diag = np.zeros((whitening_range * 2, nc))
    for i, d in enumerate(np.arange(-whitening_range + 1, whitening_range)):
        diag = np.diag(wrot, d)
        off = int(np.abs(d) / 2)
        qc_diag[i, off:(off + diag.size)] = diag
    fig, ax = plt.subplots()
    ax.set(title="Whitening matrix diagonals", ylabel='Channel', xlabel='Channel')
    plt.imshow(qc_diag, vmin=-.1, vmax=.1, cmap='PuOr'), plt.colorbar(location='bottom')
    if out_path is not None:
        fig.savefig(Path(out_path).joinpath('qc_whitening_matrix.png'))
        plt.close(fig)
    else:
        return fig, ax
