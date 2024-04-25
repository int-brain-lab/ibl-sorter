import numpy as np
import matplotlib.pyplot as plt


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
