# scTransient

`scTransient` is a Python package designed for detecting transient events in single-cell data along a pseudotime trajectory. It utilizes wavelet transforms and windowing techniques to identify and score gene expression patterns that are localized in pseudotime, which is particularly useful for identifying key transition events.

## Features

- **Transient Event Score (TES):** A metric to quantify the strength of transient signals.
- **Wavelet Transform:** Analyze signals at multiple scales to detect localized features.
- **Windowing:** Smooth single-cell samples into continuous signals.
- **Permutation Testing:** Statistical validation of transient events.

## Installation

You can install `scTransient` from source:

```bash
pip install git+https://github.com/xomicsdatascience/scTransient.git
```

## Quick Start

### 1. Preparing Signals via Windowing

Convert the sample-level expression data into continuous signals by applying a windowing function:

```python
import numpy as np
import pandas as pd
from scTransient.windowing import ConfinedGaussianWindow

# Example data: 1000 cells, expression of 5 genes
cells_pseudotime = np.linspace(0, 1, 1000)
expression_data = pd.DataFrame(
    np.random.poisson(1, (1000, 5)), 
    columns=[f"Gene_{i}" for i in range(5)]
)

# Initialize a Gaussian Window with 50 bins
window = ConfinedGaussianWindow(n_windows=50, sigma=0.05)

# Apply windowing to a gene's expression
gene_signal = window.apply(positions=cells_pseudotime, values=expression_data["Gene_0"].values)
```

### 2. Detecting Transient Events with Wavelet Transform

Once you have a smoothed signal, use the `WaveletTransform` to compute the Transient Event Score (TES).

```python
from scTransient.wavelets import WaveletTransform
from scTransient.metrics import transient_event_score

# Initialize Wavelet Transform
wt = WaveletTransform(scales=[1, 2, 4, 8])

# Apply transform to the smoothed signal
coefs, freqs = wt.apply(gene_signal)

# Calculate Transient Event Score
score = transient_event_score(coefs)
print(f"Transient Event Score: {score}")
```

### 3. Complete Workflow using Utilities

The `utils` module provides high-level functions to score multiple genes and perform permutation testing.

```python
from scTransient.utils import score_signals, convert_to_signal, permutation_dist
from scTransient.wavelets import WaveletTransform

# 1. Convert raw data to smoothed signals
signals = convert_to_signal(expression_data, cells_pseudotime)

# 2. Score all signals
wt = WaveletTransform()
scores = score_signals(signals, wt)

# 3. Perform permutation testing for statistical significance
tes, p_values = permutation_dist(
    expression_data, 
    cells_pseudotime, 
    wavelet_transform=wt, 
    n_permutations=100
)

results = pd.DataFrame({
    "Gene": expression_data.columns,
    "TES": tes,
    "p-value": p_values
})
print(results)
```

## License

This project is licensed under the GPLv3 License - see the `pyproject.toml` file for details.

## Authors

- **[Alexandre Hutton](https://github.com/AlexandreHutton)**
