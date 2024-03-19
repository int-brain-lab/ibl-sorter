import spikeglx
from pykilosort.preprocess import get_data_covariance_matrix
from pykilosort import ibl, run
from pykilosort import params
from ibldsp import voltage
params = ibl.ibl_pykilosort_params(1)
probe = params['probe']

from one.api import ONE

one = ONE(base_url="https://alyx.internationalbrainlab.org", cache_dir="/mnt/s1/pykilosort_reruns")
pid = 'f3988aec-b8c7-47d9-b1a0-5334eed7f92c'
pid = '3c283107-7012-48fc-a6c2-ed096b23974f'


eid, pname = one.pid2eid(pid)
session_path = one.eid2path(eid)
cbin_file = next(session_path.joinpath('raw_ephys_data', pname).glob('*.ap.cbin'))

sr = spikeglx.Reader(cbin_file)
butter_kwargs = {'N': 3, 'Wn': 300 / sr.fs * 2, 'btype': 'highpass'}


import scipy
from viewephys.gui import viewephys
t0 = 1254
first, last = (int(sr.fs * t0), int(sr.fs * (t0 + 1)))
butter_kwargs = {'N': 3, 'Wn': 300 / sr.fs * 2, 'btype': 'highpass'}
sos = scipy.signal.butter(**butter_kwargs, output='sos')

raw = sr[first:last, :-sr.nsync].T
butt = scipy.signal.sosfiltfilt(sos, raw)
destripe = voltage.destripe(raw, sr.fs, neuropixel_version=1, channel_labels=True)


h = sr.geometry
viewephys(butt, sr.fs, channels=sr.geometry, title='butter')
viewephys(destripe, sr.fs, channels=sr.geometry, title='destripe')
## %%

butter_kwargs = {'N': 3, 'Wn': 0.4, 'btype': 'highpass'}

kfilt = voltage.kfilt(destripe, collection=h['col'], ntr_pad=0, ntr_tap=None, lagc=None, butter_kwargs=butter_kwargs)
viewephys(kfilt, sr.fs, channels=h, title='kfilt')
viewephys(destripe - kfilt, sr.fs, channels=h, title='diff')

## %%


import numpy as np

def spiking_decon(x, nsop=36, white=.1):
    """
    Wiener deconvolution using predictive filters
    :param x:
    :param white:
    :return:
    """
    nc, ns = x.shape
    dec = x.copy()
    op = np.zeros((nc, nsop))
    for i in np.arange(nc):
        ac = np.fft.irfft(np.fft.rfft(x[i, :]) * np.conj(np.fft.rfft(x[i, :])))
        # ac = ac / ac[0]
        gtg = scipy.linalg.toeplitz(ac[:nsop])
        gtg = gtg + np.eye(nsop) * white * np.trace(gtg) / nsop
        gty = np.zeros(nsop)
        gty[0] = 1
        op = np.linalg.solve(gtg, gty)
        dec[i, :] = np.convolve(x[i, :], op, mode='full')[:ns]
    return dec

decon = spiking_decon(destripe.T, nsop=36, white=.1).T

viewephys(decon, sr.fs, channels=h, title='decon')

#
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(x)
# plt.plot(dec)
# plt.show()
# plt.figure()
#
# plt.psd(x)
# plt.psd(dec)
# plt.show()
