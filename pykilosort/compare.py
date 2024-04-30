import gc

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

from iblqm.spikeinterface_plugins.bincomparison import BinnedSymmetricSortingComparison
import spikeinterface.extractors as se

from brainbox.io.one import SpikeSortingLoader


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
    # first we compute a few indices and counts
    ns = np.maximum(spikes_a['amps'].size, spikes_b['amps'].size)
    nc = np.maximum(clusters_a['label'].size, clusters_b['label'].size)
    sok_a = clusters_a['label'][spikes_a['clusters']] == 1
    sok_b = clusters_b['label'][spikes_b['clusters']] == 1
    cok_a = clusters_a['label'] == 1
    cok_b = clusters_b['label'] == 1

    # create the figure
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    # the left figure is the distribution of spike amplitudes
    ax = axs[0]
    kwargs = dict(bins=500, range=[1, 3])
    count_a, edges = np.histogram(np.log10(spikes_a['amps'][sok_a] * 1e6), **kwargs)
    count_b, edges = np.histogram(np.log10(spikes_b['amps'][sok_b] * 1e6), **kwargs)
    sns.lineplot(x=edges[:-1], y=count_a / ns, ax=ax, label=label_a + ' good', color=sns.color_palette()[0])
    sns.lineplot(x=edges[:-1], y=count_b / ns, ax=ax, label=label_b + ' good', color=sns.color_palette()[1])
    count_a, edges = np.histogram(np.log10(spikes_a['amps'] * 1e6), **kwargs)
    count_b, edges = np.histogram(np.log10(spikes_b['amps'] * 1e6), **kwargs)
    # twin_axis = plt.twinx(ax)
    sns.lineplot(x=edges[:-1], y=count_a / ns, ax=ax, label=label_a, color=sns.color_palette()[0], linestyle='--')
    sns.lineplot(x=edges[:-1], y=count_b / ns, ax=ax, label=label_b, color=sns.color_palette()[1], linestyle='--')
    ax.set(title='Spike amplitudes distribution', xlabel='Spike amplitude [log10(uV)]', ylabel='Density (%)', xlim=kwargs['range'])

    # the middle figure is the distribution of the units amplitudes
    ax = axs[1]
    kwargs = dict(bins=25, range=[1, 3])
    count_a, edges = np.histogram(np.log10(clusters_a['amp_median'][cok_a] * 1e6), **kwargs)
    count_b, edges = np.histogram(np.log10(clusters_b['amp_median'][cok_b] * 1e6), **kwargs)
    sns.lineplot(x=edges[:-1], y=count_a / nc, ax=ax, label=label_a + ' good', color=sns.color_palette()[0])
    sns.lineplot(x=edges[:-1], y=count_b / nc, ax=ax, label=label_b + ' good', color=sns.color_palette()[1])
    count_a, edges = np.histogram(np.log10(clusters_a['amp_median'] * 1e6), **kwargs)
    count_b, edges = np.histogram(np.log10(clusters_b['amp_median'] * 1e6), **kwargs)
    sns.lineplot(x=edges[:-1], y=count_a / nc, ax=ax, label=label_a, color=sns.color_palette()[0], linestyle='--')
    sns.lineplot(x=edges[:-1], y=count_b / nc, ax=ax, label=label_b, color=sns.color_palette()[1], linestyle='--')
    ax.set(title='Unit amplitude distribution', xlabel='Amplitude [log10(uV)]', ylabel='Density', xlim=kwargs['range'])

    # the right figure is the distribution of units firing rates
    ax = axs[2]
    kwargs = dict(bins=25, range=[-3, 3])
    count_a, edges = np.histogram(np.log10(clusters_a['firing_rate'][cok_a]), **kwargs)
    count_b, edges = np.histogram(np.log10(clusters_b['firing_rate'][cok_b]), **kwargs)
    sns.lineplot(x=edges[:-1], y=count_a / nc, ax=ax, label=label_a + ' good', color=sns.color_palette()[0])
    sns.lineplot(x=edges[:-1], y=count_b / nc, ax=ax, label=label_b + ' good', color=sns.color_palette()[1])
    count_a, edges = np.histogram(np.log10(clusters_a['firing_rate']), **kwargs)
    count_b, edges = np.histogram(np.log10(clusters_b['firing_rate']), **kwargs)
    sns.lineplot(x=edges[:-1], y=count_a / nc, ax=ax, label=label_a, color=sns.color_palette()[0], linestyle='--')
    sns.lineplot(x=edges[:-1], y=count_b / nc, ax=ax, label=label_b, color=sns.color_palette()[1], linestyle='--')
    ax.set(title='Firing rates distribution', xlabel='Firing rates [log10(Hz)]', ylabel='Density', xlim=kwargs['range'])

    # %% add grids, legends, and tight layout
    for ax in axs:
        ax.grid(True)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file)
        plt.close(fig)
        gc.collect()
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
        plt.close(fig)
        gc.collect()
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
        plt.close(fig)
        gc.collect()
    return fig, ax


def reveal_pid(ssl: SpikeSortingLoader, spikes_a, spikes_b, clusters_a, clusters_b, channels, path_figures, drift_b):
    """
    :param ssl:
    :param spikes_a:
    :param spikes_b:
    :param clusters_a:
    :param clusters_b:
    :param channels:
    :param path_figures:
    :param drift_b:
    :return:
    """
    label_a = 'original'
    label_b = 'rerun'
    sr = ssl.raw_electrophysiology(band="ap", stream=False)
    ns_a = spikes_a['clusters'].size
    ns_b = spikes_b['clusters'].size
    nc_a = clusters_a['label'].size
    nc_b = clusters_b['label'].size
    nc_ok_a = np.sum(clusters_a['label'] == 1)
    nc_ok_b = np.sum(clusters_b['label'] == 1)
    spikes_a_good = {k: spikes_a[k][clusters_a['label'][spikes_a['clusters']] == 1] for k in spikes_a}
    spikes_b_good = {k: spikes_b[k][clusters_b['label'][spikes_b['clusters']] == 1] for k in spikes_b}
    # this title will be reused throughout the figures
    title = (f"{ssl.pid}"
             f"\n{nc_ok_a}/{nc_a} units {ns_a / 1e6:.2f} M spikes "
             f"\n{nc_ok_b}/{nc_b} units {ns_b / 1e6:.2f} M spikes")
    comp_kwargs = dict(title=title, label_a=label_a, label_b=label_b)

    # %% 4 driftmaps: A, B, all and good units
    # raster all spikes spike sorting A
    ssl.raster(spikes_a, channels, label=label_a, title=f"raster all {label_a} {title}",
               drift=None, weights=spikes_a['amps'], save_dir=path_figures.joinpath(f"00_raster_{label_a}.png"))
    # raster all spikes spike sorting B
    ssl.raster(spikes_b, channels, label=label_b, title=f"raster all {label_b} {title}",
               drift=drift_b, weights=spikes_b['amps'], save_dir=path_figures.joinpath(f"01_raster_{label_b}.png"))
    # raster only good units A
    ssl.raster(spikes_a_good, channels, label=label_a, title=f"raster good {label_a} {title}",
               drift=None, weights=spikes_a_good['amps'], save_dir=path_figures.joinpath(f"02_raster_good_{label_a}.png"))
    # raster only good units B
    ssl.raster(spikes_b_good, channels, label=label_b, title=f"raster good {label_b} {title}",
               drift=drift_b, weights=spikes_b_good['amps'], save_dir=path_figures.joinpath(f"03_raster_good_{label_b}.png"))

    # %% plot 3 raw data snippets
    t_snippets = np.linspace(100, sr.rl - 100, 7)[1::2].astype(int)  # we take 3 snippets evenly spaced
    c = 3
    for t in t_snippets:
        ssl.plot_rawdata_snippet(sr, spikes_a, clusters_a, t, channels=channels, label=label_a,
                                 title=f"{label_a} T{t:04d} {title}",
                                 save_dir=path_figures.joinpath(f"{(c := c + 1):02d}_voltage_T{t:04d}_{label_a}.png"))
        ssl.plot_rawdata_snippet(sr, spikes_b, clusters_b, t, channels=channels, label=label_b,
                                 title=f"{label_b} T{t:04d} {title}",
                                 save_dir=path_figures.joinpath(f"{(c := c + 1):02d}_voltage_T{t:04d}_{label_b}.png"))

    # %% KDEs of firing rates, unit amplitudes and spike amplitudes
    file_kde = path_figures.joinpath(f"{(c := c + 1):02d}_kdes.png")
    kdes(spikes_a, spikes_b, clusters_a, clusters_b, output_file=file_kde, **comp_kwargs)

    # Venn diagrams of matching units and matching spikes
    file_venn = path_figures.joinpath(f"{(c := c + 1):02d}_venns.png")
    venn_units(spikes_a, spikes_b, clusters_a, clusters_b, sr.fs, label_a, label_b, output_file=file_venn)

    # Spike holes: distribution of gaps as a function of the batch size
    file_holes = path_figures.joinpath(f"{(c := c + 1):02d}_spike_holes.png")
    fig_spike_holes(spikes_a, spikes_b, output_file=file_holes, **comp_kwargs)
