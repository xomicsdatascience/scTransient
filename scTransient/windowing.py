import numpy as np
from warnings import warn


class Window:
    def __init__(self,
                 n_windows: int,
                 signal_domain: tuple = (0,1)):
        self.n_windows = n_windows
        self.signal_domain = signal_domain
        self.window_centers = np.linspace(signal_domain[0], signal_domain[1], n_windows)
        self.window_function = None
        return

    def apply(self,
              positions: np.ndarray,
              values: np.ndarray,
              exclude_minimum: bool = False):
        """
        Applies the windowing function to the supplied samples.
        Parameters
        ----------
        positions : np.ndarray
            Position on x-axis (e.g. time) of the samples.
        values : np.ndarray
            Value of each sample, aligned with `positions`.
        exclude_minimum : bool, optional
            Whether to exclude the minimum value of each column before windowing, by default False. Useful for excluding
            zero-values that have been shifted by some preprocessing.
        Returns
        -------
        np.array
            Windowed signal.
        """

        weights = np.zeros((self.n_windows, len(positions)))
        if exclude_minimum:
            windowed_sums = np.zeros((self.n_windows, values.shape[1]))
            for idx_col in range(values.shape[1]):
                for idx_wc, wc in enumerate(self.window_centers):
                    weights[idx_wc, :] = self.window_function(positions, wc)
                    val_min = np.min(values[:, idx_col])
                    s_vec = weights[idx_wc, :] * (values[:, idx_col] > val_min)
                    s = (s_vec).sum()
                    if s != 0:
                        weights[idx_wc, :] = s_vec / s
                windowed_sums[:, idx_col] = weights @ values[:, idx_col]
            return windowed_sums
        else:
            for idx, wc in enumerate(self.window_centers):
                weights[idx, :] = self.window_function(positions, wc)
                weights[idx, np.isnan(weights[idx,:])] = 0
                s = np.sum(weights[idx,:])
                if s > 0:
                    weights[idx, :] /= s
            return weights @ values

    def get_center_window(self,
                          num_samples: int = 500):
        wc = self.window_centers[len(self.window_centers)//2]
        positions = np.linspace(self.signal_domain[0],self.signal_domain[1], num_samples)
        return positions, self.window_function(positions, wc)

    def get_summed_windows(self,
                           num_samples: int = 500):
        positions = np.linspace(self.signal_domain[0],self.signal_domain[1], num_samples)
        signal = np.zeros((num_samples,))
        for wc in self.window_centers:
            signal += self.window_function(positions, wc)
        return positions, signal

    def get_each_window(self,
                        num_samples: int = 500):
        positions = np.linspace(self.signal_domain[0],self.signal_domain[1], num_samples)
        signals = []
        for wc in self.window_centers:
            signals.append(self.window_function(positions, wc))
        return signals

    def is_coverage_complete(self):
        return np.min(self.get_summed_windows()) != 0

class GaussianWindow(Window):
    def __init__(self,
                 n_windows: int,
                 sigma: float = 1.0,
                 signal_domain: tuple = (0,1)):
        """
        Use Gaussian window to convert samples.
        Parameters
        ----------
        n_windows : int
            Number of windows to use.
        sigma : float
            Standard deviation of the Gaussian.
        """
        super().__init__(n_windows, signal_domain=signal_domain)
        self.sigma = sigma
        self.window_function = lambda position, center: np.exp(-((position-center) ** 2) / (2 * sigma ** 2))
        return

class RectWindow(Window):
    def __init__(self,
                 n_windows: int,
                 width: float = 0.01,
                 signal_domain: tuple = (0,1)):
        """
        Use a rectangular window to convert samples.
        Parameters
        ----------
        n_windows : int
            Number of windows to use.
        width : float
            Width of each window.
        """
        super().__init__(n_windows, signal_domain=signal_domain)
        self.width = width
        self.window_function = lambda position, center: np.heaviside(self.width/2 - np.abs(position-center), 1)

        # Check if all parts of 0-1 are covered by the windows
        if not self.is_coverage_complete():
            warn("The width and spacing of the window will exclude some samples.")
        return

class ConfinedGaussianWindow(Window):
    def __init__(self,
                 n_windows: int,
                 sigma: float = 1.0,
                 max_distance: float = 1.0,
                 signal_domain: tuple = (0,1)):
        """
        Use a confined Gaussian window to convert samples.
        Parameters
        ----------
        n_windows : int
            Number of windows to use.
        sigma : float
            Standard deviation of the Gaussian.
        max_distance : float
            Maximum distance from the center for the window to be non-zero.
        signal_domain : tuple
            Minimum and maximum values of the signal domain (i.e. min and max pseudotime values). Default: (0,1).
        """
        super().__init__(n_windows,
                         signal_domain=signal_domain)
        self.sigma = sigma
        def window_function(position, center):
            return np.exp(-((position-center) ** 2) / (2 * sigma ** 2)) * (np.abs(position-center) <= max_distance)
        self.window_function = window_function
        return
