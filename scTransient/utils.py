import numpy as np
from scTransient.metrics import transient_event_score
from scTransient.wavelets import WaveletTransform
from scTransient.windowing import ConfinedGaussianWindow, Window
import pandas as pd
from time import time



def score_signals(signals: np.array,
                  wavelet_transform: WaveletTransform) -> np.array:
    """
    Computes the scores for the input signals.
    Parameters
    ----------
    signals : np.array
        Array of shape (n_genes, n_timepoints).
    wavelet_transform : WaveletTransform
        Wavelet transform to apply to the signals.

    Returns
    -------
    np.array
        Transient event scores of length (n_genes,)
    """
    tes = np.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        coefs, _ = wavelet_transform.apply(signals[i,:])
        tes[i] = transient_event_score(coefs)
    return tes


def convert_to_signal(values: pd.DataFrame,
                      positions: np.array,
                      window: Window = None) -> np.array:
    """

    Parameters
    ----------
    values
    positions
    window

    Returns
    -------

    """
    if window is None:
        # TODO: make the parameters either tunable or learnable
        window = ConfinedGaussianWindow(n_windows=25,
                                        sigma=0.03,
                                        max_distance=0.11,
                                        signal_domain = (np.min(positions), np.max(positions)))
    gene_sigs = np.zeros((values.shape[1], window.n_windows))
    for idx_col, gene in enumerate(values.columns):
        gene_sigs[idx_col,:] = window.apply(positions=positions, values=values[gene].values)
    return gene_sigs


def permutation_dist(values: pd.DataFrame,
                     positions: np.array,
                     wavelet_transform: WaveletTransform,
                     n_permutations: int = 10) -> np.array:
    """
    Computes the TES permutation distribution for each gene.
    Parameters
    ----------
    values : pd.DataFrame
        Signal quantities. Expected to be indexed by sample, with columns as genes.
    positions : np.array
        The position of each sample along the x-axis.
    n_permutations : int, optional
        Number of permutations to perform, by default 1000.

    Returns
    -------
    np.array
        Transient Event Score of each gene.
    np.array
        p-values for each TES value.
    """
    # First compute the TES without permutation
    # Convert to time signal
    t_start = time()
    signals = convert_to_signal(values, positions)
    tes = score_signals(signals, wavelet_transform=wavelet_transform)
    t_end = time()

    gt = np.zeros(tes.shape)
    print(f"Starting permutations. Expected duration: {n_permutations*(t_end-t_start)/60:.2f} minutes.")
    t_start = time()
    tes_all = []
    try:
        for i in range(n_permutations):
            perm_positions = np.random.permutation(positions)
            signals = convert_to_signal(values, perm_positions)
            tes_perm = score_signals(signals, wavelet_transform=wavelet_transform)
            gt += tes < tes_perm
            tes_all.append(tes_perm)
            print(f"\r{(n_permutations - i) * ((time() - t_start) / (i + 1)) / 60:.2f} minutes remaining.", end='')
    except KeyboardInterrupt:  # in case process gets too long and user wants to stop early
        return tes, gt / (i+1), tes_all
    return tes, gt/(i+1), tes_all