import numpy as np
from scTransient.windowing import GaussianWindow, RectWindow
from typing import List


def generate_synthetic_data(n_cells,
                            n_genes,
                            pseudotime_density: np.array = None,
                            n_signal_genes: int = None,
                            spike_times: List[float] = None,
                            spike_widths: List[float] = None,
                            spike_amplitudes: List[float] = None,
                            autocorr: float = 0.5,
                            noise_gene_fraction=0.1,
                            noise_gene_mean=0,
                            noise_gene_sd=1,
                            seed=0):
    data_array = np.zeros((n_cells, n_genes))
    if n_signal_genes is None:
        n_signal_genes = int(n_genes*0.01)
    # Assign pseudotimes to cells
    pseudotime_values = sample_from_pdf(pseudotime_density, n_cells)

    # For signal genes, generate pseudotimecourse
    # TODO: Apply to some subset of cells; not all cells will belong to the process
    for i in range(n_signal_genes):
        data_array[:,i] = generate_spike_signal(spike_times=spike_times,
                                                spike_amplitudes=spike_amplitudes,
                                                spike_widths=spike_widths,
                                                total_length=n_cells,
                                                x=pseudotime_values,
                                                seed=seed)
    spike_sigs = generate_spike_signal(spike_times=spike_times,
                                            spike_amplitudes=spike_amplitudes,
                                            spike_widths=spike_widths,
                                            total_length=n_cells,
                                            seed=seed)

    # Generate signal counts
    for i in range(n_genes):
        data_array[:,i] += np.random.normal(noise_gene_mean, noise_gene_sd, n_cells)

    # order samples by pseudotime
    if autocorr > 0:
        ordered_idx = np.argsort(pseudotime_values)
        for idx_cell in range(1, n_cells):
            data_array[ordered_idx[idx_cell], :] = data_array[ordered_idx[idx_cell-1], :] * autocorr + data_array[ordered_idx[idx_cell], :] * (1-autocorr)

    return data_array, pseudotime_values, spike_sigs


def generate_synthetic_data_nb_spikes(n_cells,
                                      n_genes,
                                      pseudotime_density: np.array,
                                      n_signal_genes: int,
                                      spike_times: List[float],
                                      spike_amplitudes: List[float],
                                      spike_widths: List[float],
                                      data_nb_r: int = 10,
                                      data_nb_p: float = 0.5,
                                      autocorr: float = 0.5,
                                      noise_gene_mean=0,
                                      noise_gene_sd=1,
                                      seed=None):
    """
    Generate synthetic data with negative binomial spike trains.

    Parameters
    ----------
    n_cells : int
        Number of cells to simulate; number of total samples.
    n_genes : int
        Number of genes to include.
    pseudotime_density : np.array
        Probability density function for pseudotime values.
    n_signal_genes : int
        Number of signal genes in n_genes.
    spike_r : List[float]
        Number of successes for the NB kernel; each value is for a different spike.
    spike_p : List[float]
        Probability of success for the NB kernel; each value is for a different spike.
    spike_amplitudes : List[float]
        Magnitude of the spikes; each value is for a different spike.
    autocorr : float
        Autocorrelation to between cells.
    noise_gene_mean : float
        Mean of white noise to add.
    noise_gene_sd : float
        Standard deviation of white noise to add.
    seed : int
        Random seed.

    Returns
    -------

    """
    data_array = np.zeros((n_cells, n_genes))
    if n_signal_genes is None:
        n_signal_genes = int(n_genes*0.01)
    # Assign pseudotimes to cells
    pseudotime_values = sample_from_pdf(pseudotime_density, n_cells)

    # For signal genes, generate pseudotimecourse
    for i in range(n_signal_genes):
        data_array[:,i] = generate_spike_signal(spike_times=spike_times,
                                                spike_amplitudes=spike_amplitudes,
                                                spike_widths=spike_widths,
                                                total_length=n_cells,
                                                x=pseudotime_values,
                                                seed=seed)
    spike_sigs = generate_spike_signal(spike_times=spike_times,
                                            spike_amplitudes=spike_amplitudes,
                                            spike_widths=spike_widths,
                                            total_length=n_cells,
                                            seed=seed)

    # Generate count data
    for i in range(n_genes):
        pdf_domain = np.linspace(0, 20, 21)
        nb_pdf = continuous_nb(pdf_domain, r=data_nb_r, p=data_nb_p)
        data_array[:, i] += np.random.choice(pdf_domain, p=nb_pdf / np.sum(nb_pdf), size=data_array.shape[0])
    # order samples by pseudotime; add correlation between nearby cells
    if autocorr > 0:
        ordered_idx = np.argsort(pseudotime_values)
        for idx_cell in range(1, n_cells):
            # Could add (pseudotime_values[idx_cell] - pseudotime_values[idx_cell-1]) as factor to weigh by pt distance
            data_array[ordered_idx[idx_cell], :] = data_array[ordered_idx[idx_cell-1], :] * autocorr + data_array[ordered_idx[idx_cell], :] * (1-autocorr)

    return data_array, pseudotime_values, spike_sigs


def continuous_nb(x: np.array,
                  r: float,
                  p: float) -> np.array:
    """Compute the continous approximation to the NB distribution at values `x`. Adapted from
    https://stats.stackexchange.com/questions/310676/continuous-generalization-of-the-negative-binomial-distribution
    by swapping p and (1-p) to match scipy notation."""
    c = np.zeros(len(x))
    for idx, xx in enumerate(x):
        c[idx] = gamma(x+r) / (gamma(x+1)*gamma(r)) * (1-p)**(x) * (p)**r
    return c


def gaussian_weight(delta, sigma):
    """
    Compute the Gaussian weight for the time difference.

    Parameters:
        delta (float): The time difference between current and previous datapoint.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        float: The computed Gaussian weight.
    """
    return np.exp(- (delta ** 2) / (2 * sigma ** 2))


def update_data(values, times, alpha, sigma):
    """
    Update each data point by adding a fraction of the previous value weighted by a Gaussian.

    Parameters:
        values (numpy.ndarray): 1D array of data values.
        times (numpy.ndarray): 1D array of time stamps corresponding to the values.
        alpha (float): The fraction factor determining how much of the previous value to add.
        sigma (float): Standard deviation for the Gaussian weight.

    Returns:
        numpy.ndarray: Updated values.
    """
    # Create a copy so the original data is preserved.
    updated = values.copy()

    # Process each data point starting from the second one.
    for i in range(1, len(values)):
        # Calculate the time difference between this and the previous datapoint.
        delta_t = times[i] - times[i - 1]
        # Compute the Gaussian weight for the time difference.
        weight = gaussian_weight(delta_t, sigma)
        # Add a fraction (alpha * weight) of the previous data value to the current value.
        updated[i] += alpha * weight * values[i - 1]
    return updated


def generate_synthetic_data_multiple_processes(n_cells,
                                                n_genes,
                                                pseudotime_density: np.array = None,
                                                n_signal_genes: int = None,
                                                spike_times: List[float] = None,
                                                spike_widths: List[float] = None,
                                                spike_amplitudes: List[float] = None,
                                                autocorr: float = 0.5,
                                                noise_gene_mean=0,
                                                noise_gene_sd=1,
                                                seed=0):
    data_array = np.zeros((n_cells, n_genes))
    if n_signal_genes is None:
        n_signal_genes = int(n_genes * 0.01)
    # Assign pseudotimes to cells
    # pdens = pseudotime_density("uniform")
    pseudotime_values = sample_from_pdf(pseudotime_density, n_cells)
    second_pseudotime_values = sample_from_pdf(pseudotime_density, n_cells)

    # For signal genes, generate pseudotimecourse
    # TODO: Apply to some subset of cells; not all cells will belong to the process
    for i in range(n_signal_genes):
        data_array[:, i] = generate_spike_signal(spike_times=spike_times,
                                                 spike_amplitudes=spike_amplitudes,
                                                 spike_widths=spike_widths,
                                                 total_length=n_cells,
                                                 x=pseudotime_values,
                                                 seed=seed)
    spike_sigs = generate_spike_signal(spike_times=spike_times,
                                       spike_amplitudes=spike_amplitudes,
                                       spike_widths=spike_widths,
                                       total_length=n_cells,
                                       seed=seed)
    # spike_sigs = data_array[:,:n_signal_genes].copy()
    # Assume all genes have a second, unrelated process that would be ordered different in PT.
    for i in range(n_genes):
        data_array[:, i] += generate_spike_signal(spike_times=spike_times,
                                                  spike_amplitudes=spike_amplitudes,
                                                  spike_widths=spike_widths,
                                                  total_length=n_cells,
                                                  x=second_pseudotime_values,
                                                  seed=seed)
    # Generate noise for the remaining ones
    for i in range(n_genes):
        data_array[:, i] += np.random.normal(noise_gene_mean, noise_gene_sd, n_cells)

    data_array = update_data(data_array, pseudotime_values, autocorr, 0.01)

    return data_array, (pseudotime_values, second_pseudotime_values), spike_sigs


def generate_spike_signal(spike_times: List[float],
                          spike_amplitudes: List[float],
                          spike_widths: List[float],
                          total_length: int,
                          x: np.array = None,
                          time_noise: dict = {"mean": 0, "std": 0},
                          amplitude_noise: dict = {"mean": 0, "std": 0},
                          width_noise: dict = {"mean": 0, "std": 0},
                          signal_noise: dict = {"mean": 0, "std": 0},
                          seed: int = None) -> np.ndarray:
    """
    Generate a signal represented as a numpy array with spikes defined by Gaussians centered at spike_times.

    Parameters
    ----------
    spike_times : List[float]
        List of times where spikes should appear (centered in the signal).
    spike_amplitudes : List[float]
        Amplitudes for each spike.
    spike_widths : List[float]
        Widths for each spike (standard deviation of the Gaussian).
    total_length : int
        Length of the output signal array.
    noise_level : float, optional
        Fraction of noise to add to the output signal (default is 0, no noise).
    noise_mean : float, optional
        Mean of the noise to be added (default is 0).
    noise_std : float, optional
        Standard deviation of the noise to be added (default is 1).
    seed : int, optional
        Seed for reproducibility of random noise (default is 0).

    Returns
    -------
    np.ndarray
        The generated signal array.
    """
    np.random.seed(seed)
    signal = np.zeros(total_length)
    if x is None:
        x = np.linspace(0, 1, total_length)

    # Set default noise dictionary values if None
    time_noise = time_noise or {"mean": 0, "std": 0}
    amplitude_noise = amplitude_noise or {"mean": 0, "std": 0}
    width_noise = width_noise or {"mean": 0, "std": 0}
    signal_noise = signal_noise or {"mean": 0, "std": 0}

    for clean_time, clean_amplitude, clean_width in zip(spike_times, spike_amplitudes, spike_widths):
        time = clean_time + (np.random.randn()+time_noise["mean"])*time_noise["std"]
        amplitude = clean_amplitude + (np.random.randn()+amplitude_noise["mean"])*amplitude_noise["std"]
        width = clean_width + (np.random.randn()+width_noise["mean"])*width_noise["std"]

        gaussian = amplitude * np.exp(-((x - time) ** 2) / (2 * width ** 2))
        signal += gaussian

    noise = (np.random.randn(total_length)+signal_noise["mean"])*signal_noise["std"]
    signal += noise

    return signal


from typing import List
def continuous_nb(x: np.array,
                  r: float,
                  p: float) -> np.array:
    """Compute the continous approximation to the NB distribution at values `x`. Adapted from
    https://stats.stackexchange.com/questions/310676/continuous-generalization-of-the-negative-binomial-distribution
    by swapping p and (1-p) to match scipy notation."""
    c = np.zeros(len(x))
    for idx, xx in enumerate(x):
        c[idx] = gamma(xx+r) / (gamma(xx+1)*gamma(r)) * (1-p)**(xx) * (p)**r
    return c


def generate_spike_signal_nb(spike_amplitudes: List[float],
                             spike_r: List[float],
                             spike_p: List[float],
                             total_length: int,
                             x: np.array = None,
                             x_remap_range: tuple = (0, 20),
                             seed: int = None) -> np.ndarray:
    """
    Generate a signal with spikes shaped by a Negative Binomial (NB) kernel. Each spike is centered at spike_times with
    scale given by spike_widths and overall amplitude spike_amplitudes.
    """
    # Set seed; set defaults
    np.random.seed(seed)
    if x is None:
        x = np.linspace(0, 1, total_length)

    # Validate parameters for NB kernel
    for r in spike_r:
        if r <= 0:
            raise ValueError("r must be > 0.")
    for p in spike_p:
        if not (0 < p < 1):
            raise ValueError("p must be in (0, 1).")

    signal = np.zeros(total_length)
    for amplitude, r, p in zip(spike_amplitudes, spike_r, spike_p):
        k_vals = (x + x_remap_range[0]) * (x_remap_range[1] - x_remap_range[0])
        nb_vals = continuous_nb(x=k_vals, r=r, p=p)

        # Set peak value to match the specified amplitude
        nb_vals = nb_vals * (amplitude / np.max(nb_vals))
        signal += nb_vals

    return signal

from scipy.special import gamma


def pseudotime_density(function_name: str,
                       function_params: dict = None,
                       num_window_samples: int = 500):
    if function_name == "gaussian":
        # more in the centre
        if function_params is None:
            function_params = {"n_windows": 25, "sigma": 0.1}
        return GaussianWindow(**function_params).get_center_window(num_window_samples)[1]
    elif function_name == "uniform":
        if function_params is None:
            function_params = {"n_windows": 1, "width": 2}
        return RectWindow(**function_params).get_center_window(num_window_samples)[1]
    elif function_name == "inverse_spike":
        if function_params is None:
            function_params = {"spike_times": [0.5], "spike_amplitudes": [1], "spike_widths": [0.05], "total_length": num_window_samples}
        min_fraction = 0
        # print("updated")
        if "min_fraction" in function_params:
            min_fraction = function_params["min_fraction"]
            del function_params["min_fraction"]
        spike = generate_spike_signal(**function_params)
        spike = (1-spike) / np.sum(1-spike)
        mmin = np.min(spike)
        mmax = np.max(spike)
        a = mmax * (1-min_fraction) / (mmax-mmin)
        b = mmax - a*mmax
        return a*spike + b


def sample_from_pdf(pdf: np.array,
                    num_samples: int = 100):
    """
    Get samples from a probability density function (PDF) using inverse transform sampling.
    Parameters
    ----------
    pdf : np.array
        Probability density function values. Assumes the underlying range is 0-1
    num_samples : int
        Number of samples to generate.

    Returns
    -------
    np.array
        Sampled from the pdf distribution.
    """

    # Define the x range corresponding to the pdf values:
    x_min = 0
    x_max = 1
    x = np.linspace(x_min, x_max, len(pdf))
    dx = (x_max - x_min) / (len(pdf) - 1)
    pdf_normalized = pdf / (np.sum(pdf) * dx)
    cdf = np.cumsum(pdf_normalized) * dx
    cdf = cdf / cdf[-1]

    # Inverse transform sampling:
    uniform_samples = np.random.rand(num_samples)  # Generate uniform random numbers in [0, 1]
    samples = np.interp(uniform_samples, cdf, x)  # Invert CDF to get samples
    return samples
