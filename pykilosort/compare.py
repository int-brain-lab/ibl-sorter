import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

from iblqm.spikeinterface_plugins.bincomparison import BinnedSymmetricSortingComparison
import spikeinterface.extractors as se


def kdes(spikes_a, spikes_b, clusters_a, clusters_b, title='', label_a='', label_b='', output_file=None):
    """
    Displays the KDEs of firing rates, unit amplitudes and spike amplitudes
    :param spikes_a:
    :param spikes_b:
    :param clusters_a:
    :param clusters_b:
    :param title:
    :param label_a:
    :param label_b:
    :param output_file:
    :return:
    """
    # the left figure is the distribution of the units amplitudes
    fig, axs = plt.subplots(1, 3, figsize=(16, 7))
    ax = axs[0]
    sns.kdeplot(x=clusters_a['amp_median'][clusters_a['label'] == 1] * 1e6, ax=ax, label=label_a + ' good',
                color=sns.color_palette()[0])
    sns.kdeplot(x=clusters_b['amp_median'][clusters_b['label'] == 1] * 1e6, ax=ax, label=label_b + ' good',
                color=sns.color_palette()[1])
    sns.kdeplot(x=clusters_a['amp_median'] * 1e6, ax=ax, label=label_a, color=sns.color_palette()[0], linestyle='--')
    sns.kdeplot(x=clusters_b['amp_median'] * 1e6, ax=ax, label=label_b, color=sns.color_palette()[1], linestyle='--')
    ax.set(title='Unit amplitude distribution', xlabel='Amplitude (uV)', ylabel='Density', xlim=[0, 500])

    # the middle figure is the distribution of units firing rates
    ax = axs[1]
    sns.kdeplot(x=np.log10(clusters_a['firing_rate'][clusters_a['label'] == 1]),
                ax=ax, label=label_a + ' good', color=sns.color_palette()[0])
    sns.kdeplot(x=np.log10(clusters_b['firing_rate'][clusters_b['label'] == 1]),
                ax=ax, label=label_b + ' good', color=sns.color_palette()[1])
    sns.kdeplot(x=np.log10(clusters_a['firing_rate']), ax=ax, label=label_a, color=sns.color_palette()[0],
                linestyle='--')
    sns.kdeplot(x=np.log10(clusters_b['firing_rate']), ax=ax, label=label_b, color=sns.color_palette()[1],
                linestyle='--')
    ax.set(title='Firing rates distribution', xlabel='Firing rates log10(Hz)', ylabel='Density', xlim=[-3, 3])

    # the middle figure is the distribution of units firing rates
    ax = axs[2]
    ns = np.maximum(spikes_a['amps'].size, spikes_b['amps'].size)
    count_a, edges = np.histogram(
        spikes_a['amps'][clusters_a['label'][spikes_a['clusters']] == 1] * 1e6, bins=500, range=[0, 500])
    count_b, edges = np.histogram(
        spikes_b['amps'][clusters_b['label'][spikes_b['clusters']] == 1] * 1e6, bins=500, range=[0, 500])
    sns.lineplot(x=edges[:-1], y=count_a / ns, ax=ax, label=label_a + ' good', color=sns.color_palette()[0])
    sns.lineplot(x=edges[:-1], y=count_b / ns, label=label_b + ' good', color=sns.color_palette()[1])
    count_a, edges = np.histogram(spikes_a['amps'] * 1e6, bins=500, range=[0, 500])
    count_b, edges = np.histogram(spikes_b['amps'] * 1e6, bins=500, range=[0, 500])
    sns.lineplot(x=edges[:-1], y=count_a / ns, ax=ax, label=label_a, color=sns.color_palette()[0], linestyle='--')
    sns.lineplot(x=edges[:-1], y=count_b / ns, ax=ax, label=label_b, color=sns.color_palette()[1], linestyle='--')
    ax.set(title='Spike amplitudes distribution', xlabel='Spike amplitude (uV)', ylabel='Density (%)', xlim=[0, 500])

    # add grids, legends, and tight layout
    for ax in axs:
        ax.grid(True)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file)
    return fig, ax


def fig_spike_holes(spikes_a, spikes_b, title='', label_a='', label_b='', output_file=None):
    """
    Returns the spike holes distribution figure
    :param spikes_a: dict
    :param spikes_b: dict
    :param title: str: title of the figure
    :param label_a: str: label of the first dataset, used in the legend
    :param label_b: str: label of the second dataset, used in the legend
    :param output_file: str | Pah | None: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    a = np.diff(spikes_a['times'][np.where(np.diff(spikes_a['times']) > (0.006))]) / 2.18689567
    count, batch = np.histogram(a, bins=500, range=[0, 10])
    ax.plot(batch[:-1], count, label=label_a, linewidth=2, alpha=1)

    a = np.diff(spikes_b['times'][np.where(np.diff(spikes_b['times']) > (0.006))]) / 2.18689567
    count, batch = np.histogram(a, bins=500, range=[0, 10])
    ax.plot(batch[:-1], count, label=label_b, linewidth=2, alpha=0.8)

    ax.set(xlabel='Spike gaps (kilosort batches)', ylabel='Count', title=f'Spike holes distribution\n{title}',
           xlim=[-0.1, 3.5])
    ax.legend()
    if output_file:
        fig.savefig(output_file)
    return fig, ax


def venn_units(spikes_a, spikes_b, clusters_a, clusters_b, fs=30_000, title='', label_a='', label_b='', output_file=None):
    """

    :param spikes_a:
    :param spikes_b:
    :param clusters_a:
    :param clusters_b:
    :param fs:
    :param title:
    :param label_a:
    :param label_b:
    :param output_file:
    :return:
    """

    sorting_a = se.NumpySorting.from_times_labels(spikes_a['samples'] / fs, spikes_a['clusters'], fs)
    sorting_b = se.NumpySorting.from_times_labels(spikes_b['samples'] / fs, spikes_b['clusters'], fs)
    bssc = BinnedSymmetricSortingComparison(sorting_a, sorting_b, sorting1_name=label_a, sorting2_name=label_b)

    iok_a = clusters_a['label'][spikes_a['clusters']] == 1
    iok_b = clusters_b['label'][spikes_b['clusters']] == 1
    sorting_a_good = se.NumpySorting.from_times_labels(
        spikes_a['samples'][iok_a] / fs, spikes_a['clusters'][iok_a], fs)
    sorting_b_good = se.NumpySorting.from_times_labels(
        spikes_b['samples'][iok_b] / fs, spikes_b['clusters'][iok_b], fs)
    bssc_good = BinnedSymmetricSortingComparison(sorting_a_good, sorting_b_good, sorting1_name=label_a,
                                                 sorting2_name=label_b)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": (1, 1)})
    num_match_best = np.sum(bssc.best_match_12 != -1.0)
    venn_dict = {
        "11": num_match_best,
        "10": sorting_a.get_num_units() - num_match_best,
        "01": sorting_b.get_num_units() - num_match_best,
    }
    venn2(subsets=venn_dict, set_labels=(label_a, label_b), ax=ax[0])
    ax[0].set(title="Cluster agreement - all units")

    num_match_best_good = np.sum(bssc_good.best_match_12 != -1.0)
    venn_dict = {
        "11": num_match_best_good,
        "10": sorting_a_good.get_num_units() - num_match_best_good,
        "01": sorting_b_good.get_num_units() - num_match_best_good,
    }
    venn2(subsets=venn_dict, set_labels=(label_a, label_b), ax=ax[1])
    ax[1].set(title="Cluster agreement - good units")

    if output_file:
        fig.savefig(output_file)
    return fig, ax