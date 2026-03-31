# Plotting with sliding_plotter.py

## Usage

```bash
python sliding_plotter.py data/file1.jld2 [data/file2.jld2 ...] [--mode=auto]
```

The plotter auto-detects the simulation mode from the JLD2 file contents and dispatches to the appropriate plotting function. Multiple files can be overlaid as separate curves.

## Mode Detection

The mode is inferred from which top-level keys are present in the JLD2 file:

| Key present | Detected mode |
|-------------|---------------|
| `lc_values` | erosion_test |
| `mean_energies` | energy |
| `log_mixing_times` | ffs |
| `mean_mixing_times` | mixing |
| `magnetization_history` | history |

## Plot Types

### History
Spacetime heatmap of local magnetization $m_i(t) = \sigma_i^\text{top} + \sigma_i^\text{bottom}$. Time runs upward, site index on x-axis. Uses `RdBu` colormap with range $[-2, 2]$. Data is shifted into the co-moving frame at $v/2$.

### Mixing
Log-scale plot of mean mixing time $\tau$ vs $p = e^\beta$ (or vs $v$). Each file appears as a separate curve labeled by $(v, L)$.

### FFS
Same as mixing but with error bars ($\pm 1\sigma$ in $\log_{10}\tau$). Overlays an $e^{L\beta} = p^L$ reference curve (dotted red) when sweeping $p$, drawn only when the data exceeds it.

### Energy
Energy per spin vs $p$ or $v$. When plotting energy (not heat flow), values are normalized to the first data point.

### Erosion Test
Critical domain size $\ell_c$ vs $p$ or $v$. Includes a linear fit to the last 3/4 of data points with printed slope. Optionally plots raw shrinkage probability curves on a separate axis.

## Overriding Mode

```bash
python sliding_plotter.py data/file.jld2 --mode=mixing
```

Forces the plotter to use a specific mode regardless of auto-detection.
