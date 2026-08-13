## Commands for reproducing paper figures 

Velocities are stored in the data files as the *relative* velocity of the two chains, which is twice the paper's $v$ (the paper slides each chain at $\pm v$). The plotter divides by 2 on load, so every $v$ appearing on an axis, in a legend, or in the `--vmin/--vmax` arguments below is in the paper's convention, while the `v...` in a filename is twice that.

Each figure below names the PDF it produces. The plotter opens an interactive window; save from there.

### Main text

Memory time phase diagram (`phase_diagram.pdf`):
```
python3 sliding_plotter.py data/ising_sliding_phase_diagram_L1000_mixing.jld2 --pmin 1.5 --pmax 2.5 --vmin 0 --vmax 5
``` 

Example histories (`clean_history.pdf` / `annotated_clean_history.pdf`, and `noisy_history.pdf`): 
``` 
python3 sliding_plotter.py data/ising_sliding_history_L200_v2.00_beta200.000.jld2 (noiseless)
python3 sliding_plotter.py data/ising_sliding_history_L200_v2.00_beta2.000.jld2 (noisy)
```
The noiseless one auto-draws the $v\pm\delta$ arrow annotations (hard-coded for this L, v, beta).

Erosion length against $e^{\beta J}$ for different values of v (`xier_vs_p.pdf`): 
``` 
python3 sliding_plotter.py data/ising_sliding_erosion_v4.50.jld2 data/ising_sliding_erosion_v4.00.jld2 data/ising_sliding_erosion_v3.50.jld2 data/ising_sliding_erosion_v3.00.jld2 data/ising_sliding_erosion_v2.50.jld2 data/ising_sliding_erosion_v2.00.jld2 --cmap Blues --fit_inset
```

Memory time vs v for sliding (`tmem_vs_v.pdf`): 
```
python3 sliding_plotter.py data/ising_sliding_ffs_p0.18_v0.00to5.00_adaptiveLx16test.jld2 data/ising_sliding_ffs_p0.21_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.24_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.29_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.37_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.49_v0.00to5.00_adaptiveLx8test.jld2 --xr --fitrange .5
``` 
Uses `--n_configs_per_run=2500 --n_repeats=5 --M_threshold=.5 --adaptive_factor=8`

Memory time vs (beta J)^2 for sliding (`tmemtilde_vs_r.pdf`):
```
python3 sliding_plotter.py data/ising_sliding_ffs_v5.00_p0.06to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v4.00_p0.05to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v3.00_p0.03to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v2.00_p0.02to0.37_adaptiveLx8_seedsize2bigrange.jld2 --xr --fitrange 0
``` 
Uses `--n_configs_per_run=2500 --n_repeats=5 --M_threshold=.5 --adaptive_factor=8`

Average energy in steady state vs v for sliding (`E_vs_v.pdf`): 
``` 
python3 sliding_plotter.py data/ising_sliding_energy_L4000_p0.40.jld2 data/ising_sliding_energy_L4000_p0.45.jld2 data/ising_sliding_energy_L4000_p0.50.jld2 data/ising_sliding_energy_L4000_p0.55.jld2 --cmap Oranges --legloc right
```

### Supplement

Fit of ln t_mem to (beta J)^2, with residuals (`tmemtilde_vs_r_a2fit.pdf`) — same files as `tmemtilde_vs_r.pdf`:
```
python3 sliding_plotter.py data/ising_sliding_ffs_v5.00_p0.06to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v4.00_p0.05to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v3.00_p0.03to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v2.00_p0.02to0.37_adaptiveLx8_seedsize2bigrange.jld2 --xr --fitrange .1 --residuals
```

The same fit against (beta J)^1 (`tmemtilde_vs_r_a1fit.pdf`):
```
python3 sliding_plotter.py data/ising_sliding_ffs_v5.00_p0.06to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v4.00_p0.05to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v3.00_p0.03to0.37_adaptiveLx8_seedsize2bigrange.jld2 data/ising_sliding_ffs_v2.00_p0.02to0.37_adaptiveLx8_seedsize2bigrange.jld2 --xr --a 1 --fitrange .1 --residuals
```

Memory time vs v with residuals (`tmem_vs_v_with_residual.pdf`) — the first four files of the `tmem_vs_v.pdf` sweep:
```
python3 sliding_plotter.py data/ising_sliding_ffs_p0.18_v0.00to5.00_adaptiveLx16test.jld2 data/ising_sliding_ffs_p0.21_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.24_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.29_v0.00to5.00_adaptiveLx8test.jld2 --xr --fitrange .2 --residuals
```

Diffusion constant of GKL at zero bias (`gkl_diffusion_constant.pdf`): 
```
python3 sliding_plotter.py data/gkl_diffusion_L4000_eta0.000_p0.000to0.040_im+1_sync.jld2 --yscale linear
``` 

Erosion length of GKL (`gkl_ler.pdf`): 
``` 
python3 sliding_plotter.py data/gkl_ler_eta0.000_p0.000to0.040_sync.jld2 data/gkl_ler_eta0.500_p0.000to0.040_sync.jld2 data/gkl_ler_eta1.000_p0.000to0.040_sync.jld2
``` 

Memory time for GKL (`gkl_tmem_vsq.pdf` / `gkl_tmem_vslog.pdf`): 
``` 
python3 sliding_plotter.py data/gkl_ffs_L500_eta0.000_p0.003to0.107_adaptiveLx5_sync.jld2 data/gkl_ffs_L500_eta1.000_p0.003to0.107_adaptiveLx5_sync.jld2 --xq (or --xlogsqq)
``` 

Mean-field phase diagram in the v -> infinity limit (`sliding_mf_pd.pdf`):
```
python3 mean_field.py
```
