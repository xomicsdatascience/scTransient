import numpy as np


def transient_event_score(wavelet_coefficients: np.array) -> float:
    """
    Computes the TES based on the wavelet transform.
    Parameters
    ----------
    wavelet_coefficients : np.array
        Coefficients of the wavelet transform.

    Returns
    -------
    float
        The transient event score.
    """
    zmod = modified_z_score(wavelet_coefficients)
    return np.max(np.abs(wavelet_coefficients * zmod))


def modified_z_score(wavelet_coefficients: np.array) -> float:
    coef_std = wavelet_coefficients.ravel()
    std = np.std(coef_std)
    if std == 0:
        return 0
    return (wavelet_coefficients - np.median(wavelet_coefficients)) / std