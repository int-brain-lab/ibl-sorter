import matplotlib.pyplot as plt
import one.alf.io as alfio
from brainbox.plot import driftmap
from pathlib import Path
from ibllib.io import spikeglx
import numpy as np
from ibllib.plots.figures import ephys_bad_channels
from brainbox.metrics.single_units import quick_unit_metrics
from easyqc.gui import viewseis

import pandas as pd


INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
eqc_xrange = [880, 935]
eqc_gain = 1
runs = list(INTEGRATION_DATA_PATH.rglob('imec_385_100s'))


csv = []
eqcs = []
raw = True
for i, run in enumerate(runs):
    run_label = run.parts[-2]
    fig_file = INTEGRATION_DATA_PATH.joinpath('_'.join(run.parts[-2:]) + '.png')
    eqc_file = INTEGRATION_DATA_PATH.joinpath('eqc_' + '_'.join(run.parts[-2:]) + '.png')

    spikes = alfio.load_object(run.joinpath('alf'), 'spikes')
    clusters = alfio.load_object(run.joinpath('alf'), 'clusters')
    if run.joinpath('intermediate').exists() and not eqc_file.exists():
        bin_file = next(INTEGRATION_DATA_PATH.rglob("imec_385_100s.ap.cbin"))
        sr = spikeglx.Reader(bin_file)
        pre_proc_file = run.joinpath('intermediate', 'proc.dat')
        nc = 384
        ns = int(pre_proc_file.stat().st_size / 2 / nc)
        mmap = np.memmap(pre_proc_file, dtype=np.int16, mode='r', shape=(ns, nc))
        start = 0
        end = start + 80000
        print(run)
        eqc = viewseis(mmap[start:end, :] / 200, si= 1 / sr.fs * 1e3, taxis=0, title=run.parts[-2])
        eqc.ctrl.add_scatter(spikes['times']* 1e3, clusters['channels'][spikes['clusters']])
        eqcs.append(eqc)

        eqc.ctrl.set_gain(eqc_gain)
        eqc.resize(1960, 1200)
        eqc.viewBox_seismic.setXRange(*eqc_xrange)
        eqc.viewBox_seismic.setYRange(0, nc)
        eqc.grab().save(str(eqc_file))
        eqc.close()

    if fig_file.exists():
        continue
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    clusters.keys()
    driftmap(spikes['times'], spikes['depths'], plot_style='bincount', t_bin=0.1, d_bin=20,  vmax=5, ax=ax)
    ax.set(title=f"{run_label}",  ylim=[0, 3900], xlim=[0, 100])
    nspi = spikes.times.size
    nclu = clusters.channels.size
    qc = quick_unit_metrics(spikes['clusters'], spikes['times'], spikes['amps'], spikes['depths'])
    fig.savefig(fig_file)
    csv.append(dict(label=run_label, sorted=True, nspikes=nspi, nclusters=nclu, quality=np.mean(qc.label)))
pd.DataFrame(csv)
