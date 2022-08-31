import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal
from neurodsp import voltage
from neurodsp.utils import rms
import spikeglx
import brainbox.plot as bbplot
import one.alf.io as alfio
from brainbox.metrics.single_units import spike_sorting_metrics


def qc_plots_metrics(bin_file=None, alf_path=None, out_path=None,
                     raster_plot=True, raw_plots=True, summary_stats=True,
                     raster_start=0, raster_len=1200, raw_start=600, raw_len=0.04):

    out_path = out_path or alf_path
    if out_path is None:
        raise IOError('Either out_path or alf_path needs to be passed to know where to store outputs')
    out_path = Path(out_path)

    # Collect input data for summary stats and raster plot
    if raster_plot or summary_stats:
        if alf_path is None:
            raise IOError('If raster_plot=True or summary_stats=True an alf_path has to be passed')
        spikes = alfio.load_object(alf_path, 'spikes')
        clusters = alfio.load_object(alf_path, 'clusters')
        wmat = np.load(alf_path.joinpath('_kilosort_whitening.matrix.npy'))
        df_units, drift = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths)

    # Compute and save summary statistics
    if summary_stats:
        cond = np.linalg.cond(wmat)
        summary = {'mean_rating': np.mean(df_units.label),
                   'n_good_units': df_units[df_units['label'] == 1].shape[0],
                   'pid': pid,
                   'tag': tag,
                   'drift': np.median(np.abs(drift['drift_um'])),
                   'n_clusters': clusters.channels.size,
                   'n_spikes': spikes.times.size,
                   'matrix_cond': cond  # condition number of the whitening matrix
                   }
        with open(out_path.joinpath('summary_stats.json'), 'w') as outfile:
            json.dump(summary, outfile)

    if raster_plot:
        fig, ax = plt.subplots(figsize=(16, 9))
        bbplot.driftmap(spikes.times, spikes.depths, t_bin=0.007, d_bin=10, vmax=0.5, ax=ax)
        title_str = f"{spikes.clusters.size:_} spikes, {clusters.depths.size:_} clusters"
        ax.title.set_text(title_str)
        ax.set_xlim(raster_start, raster_start+raster_len), ax.set_ylim(0, 3800)
        fig.savefig(out_path.joinpath("raster.png"))
        plt.close(fig)

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
        if alf_path:
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
            if alf_path:
                # Plot good spikes in green
                eqc.ctrl.add_scatter(slice_df[slice_df['labels'] == 1]['times'],
                                     slice_df[slice_df['labels'] == 1]['channels'], (50, 205, 50, 200),
                                     label='spikes_good')
                # Plot not good spikes in red
                eqc.ctrl.add_scatter(slice_df[slice_df['labels'] != 1]['times'],
                                     slice_df[slice_df['labels'] != 1]['channels'], (255, 0, 0, 200),
                                     label='spikes_bad')
            eqc.ctrl.set_gain(25)
            eqc.resize(1960, 1200)
            eqc.viewBox_seismic.setYRange(0, nc)
            eqc.grab().save(str(out_path.joinpath(f"{name}.png")))
            eqc.close()
