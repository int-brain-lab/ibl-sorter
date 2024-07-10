import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from ibldsp import voltage
from ibldsp.utils import rms
import spikeglx
import brainbox.plot as bbplot
import one.alf.io as alfio
from brainbox.metrics.single_units import spike_sorting_metrics
from viewephys.gui import viewephys

from iblutil.util import setup_logger
from iblutil.numerical import ismember
logger = setup_logger(__name__)


def make_raster_plot(spike_times, spike_depths, nclusters, label='raster', out_path=None, raster_start=0.0,
                     raster_len=1200.0, vmax=0.5, d_bin=10, t_bin=0.007):
    fig, ax = plt.subplots(figsize=(16, 9))
    bbplot.driftmap(spike_times, spike_depths, t_bin=t_bin, d_bin=d_bin, vmax=vmax, ax=ax)
    title_str = f"{spike_times.size:_} spikes, {nclusters:_} clusters"
    ax.title.set_text(title_str)
    ax.set_xlim(raster_start, raster_start + raster_len), ax.set_ylim(0, 3800)
    fig.savefig(out_path.joinpath(f"{label}.png"))
    plt.close(fig)


def qc_plots_metrics(bin_file=None, pykilosort_path=None, out_path=None,
                     raster_plot=True, raw_plots=True, summary_stats=True,
                     raster_start=0., raster_len=1200., raw_start=600., raw_len=0.04, vmax=0.5, d_bin=10, t_bin=0.007):
    """
    Function to create plots and statistics for basic qc of spike sorting results. Depending on inputs creates
    a raster plot, three snapshots of the "raw" data - after butterworth filtering, destriping and whitening -
    overlayed with the detected spikes (good units in green, others in red), and a json file with summary statistics
    (number of spikes, clusters and good units, mean rating, drift, matrix condition number)

    :param bin_file: str or pathlib.Path
        Location of binary ephys file, required for raw plots
    :param pykilosort_path: str or pathlib.Path
        Location of pykilosort outputs, required for raster plots, summary stats, plotting of whitened data and
        spike overlay on all raw plots.
    :param out_path: str or pathlib.Path
        Location of directory in which to store outputs. Note that outputs have simple labels so outputs for
        different probes should be stored in different directories. If None is given, try to use pykilosort_path
    :param raster_plot: boolean
        Whether to create raster plot, default is True
    :param raw_plots: boolean
        Whether to create raw data plots, default is True
    :param summary_stats: boolean
        Whether to compute and save summary statistics, default is True
    :param raster_start: float
        Time in seconds of session time (spikes.times) to use as start of raster plot, default is 0
    :param raster_len: float
        Time in seconds of session time (spikes.times) to use as length of raster plot, default is 1200
    :param raw_start: float
        Time in seconds of probe time (sr.ns / sr.fs) to use as start of raw plots, default is 600
    :param raw_len: float
        Time in seconds of probe time (sr.ns / sr.fs) to use as length of raw plots, default is 0.04
    """

    out_path = out_path or pykilosort_path
    if out_path is None:
        raise IOError('Either out_path or alf_path needs to be passed to know where to store outputs')
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    if pykilosort_path:
        pykilosort_path = Path(pykilosort_path)

    # Collect input data for summary stats and raster plot
    if raster_plot or summary_stats:
        if pykilosort_path is None:
            raise IOError('If raster_plot=True or summary_stats=True an alf_path has to be passed')
        spikes = alfio.load_object(pykilosort_path, 'spikes')
        clusters = alfio.load_object(pykilosort_path, 'clusters')
        wmat = np.load(pykilosort_path.joinpath('_kilosort_whitening.matrix.npy'))
        df_units, drift = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths)

    # Compute and save summary statistics
    if summary_stats:
        cond = np.linalg.cond(wmat)
        summary = {'mean_rating': np.mean(df_units.label),
                   'n_good_units': df_units[df_units['label'] == 1].shape[0],
                   'drift': np.median(np.abs(drift['drift_um'])),
                   'n_clusters': clusters.channels.size,
                   'n_spikes': spikes.times.size,
                   'matrix_cond': cond  # condition number of the whitening matrix
                   }
        with open(out_path.joinpath('summary_stats.json'), 'w') as outfile:
            json.dump(summary, outfile)

    if raster_plot:
        make_raster_plot(spikes.times, spikes.depths, nclusters=clusters.depths.size, label='raster',
                         out_path=out_path, raster_start=raster_start, raster_len=raster_len,
                         vmax=vmax, d_bin=d_bin, t_bin=t_bin)
        good_clusters = df_units.loc[df_units['label'] == 1, ['cluster_id']].to_numpy().flatten()
        good_spikes, _ = ismember(spikes.clusters, good_clusters)
        make_raster_plot(spikes.times[good_spikes], spikes.depths[good_spikes], nclusters=good_clusters.size,
                         label='raster_good_units', out_path=out_path, raster_start=raster_start, raster_len=raster_len,
                         vmax=vmax, d_bin=d_bin, t_bin=t_bin)

    if raw_plots:
        if bin_file is None:
            raise IOError('If raw_plots=True a bin_file must be passed')
        # Load a slice of raw data
        sr = spikeglx.Reader(bin_file)
        start = int(raw_start * sr.fs)
        end = int((raw_start + raw_len) * sr.fs)
        raw = sr[start:end, :-sr.nsync].T
        # Compute butterworth filtered and destriped data
        sos = scipy.signal.butter(N=3, Wn=300 / sr.fs * 2, btype='highpass', output='sos')
        butt = scipy.signal.sosfiltfilt(sos, raw)
        destripe = voltage.destripe(raw, fs=sr.fs)
        views = [butt, destripe]
        view_names = ['butterworth', 'destriped']
        # If whitening matrix is available, compute whitened data
        if pykilosort_path:
            whitened = np.matmul(wmat, destripe)
            whitened = whitened / np.median(rms(whitened)) * np.median(rms(destripe))
            views.append(whitened)
            view_names.append('whitened')
            slice_spikes = slice(np.searchsorted(spikes.samples, start), np.searchsorted(spikes.samples, end))
            slice_df = pd.DataFrame()
            slice_df['times'] = (spikes.samples[slice_spikes] - start) / sr.fs * 1e3
            slice_df['channels'] = clusters.channels[spikes.clusters[slice_spikes]]
            slice_df['clusters'] = spikes.clusters[slice_spikes]
            slice_df['labels'] = list(df_units.iloc[slice_df['clusters']]['label'])
        else:
            print('WARNING: No alf_path provided, cannot plot whitened data or spike overlay')

        for view, name in zip(views, view_names):
            eqc = viewephys(view, fs=sr.fs, title=f'{name}')
            if pykilosort_path:
                # Plot not good spikes in red
                eqc.ctrl.add_scatter(slice_df[slice_df['labels'] != 1]['times'],
                                     slice_df[slice_df['labels'] != 1]['channels'], (255, 0, 0, 100),
                                     label='spikes_bad')
                # Plot good spikes in green
                eqc.ctrl.add_scatter(slice_df[slice_df['labels'] == 1]['times'],
                                     slice_df[slice_df['labels'] == 1]['channels'], (50, 205, 50, 100),
                                     label='spikes_good')

            eqc.ctrl.set_gain(30)
            eqc.resize(1960, 1200)
            eqc.viewBox_seismic.setYRange(0, raw.shape[1])
            eqc.grab().save(str(out_path.joinpath(f"{name}.png")))
            eqc.close()
