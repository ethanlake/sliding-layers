## Commands for reproducing paper figures 

Memory time phase diagram: 
```
python3 sliding_plotter.py data/ising_sliding_phase_diagram_L1000_mixing.jld2 --pmin 1.5 --vmin 0 --vmax 10 --pmax 2.5 
``` 

Example histories: 
``` 
python3 sliding_plotter.py data/ising_sliding_history_L200_v2.00_beta200.000.jld2 (noiseless)
python3 sliding_plotter.py data/ising_sliding_history_L200_v2.00_beta2.000.jld2 (noisy)
```

Erosion length against v for different values of T: 
``` 
python3 sliding_plotter.py data/ising_sliding_erosion_v4.50.jld2 data/ising_sliding_erosion_v4.00.jld2 data/ising_sliding_erosion_v3.50.jld2 data/ising_sliding_erosion_v3.00.jld2 data/ising_sliding_erosion_v2.50.jld2 data/ising_sliding_erosion_v2.00.jld2
```

Diffusion constant of GKL at zero bias: 
```
python3 sliding_plotter.py data/gkl_diffusion_L4000_eta0.000_p0.000to0.040_im+1_sync.jld2 --yscale linear
``` 

Erosion length of GKL: 
``` 
python3 sliding_plotter.py data/gkl_ler_eta0.000_p0.000to0.040_sync.jld2 data/gkl_ler_eta0.500_p0.000to0.040_sync.jld2 data/gkl_ler_eta1.000_p0.000to0.040_sync.jld2
``` 

Memory time for GKL: 
``` 
python3 sliding_plotter.py data/gkl_ffs_L500_eta0.000_p0.003to0.107_adaptiveLx5_sync.jld2 data/gkl_ffs_L500_eta1.000_p0.003to0.107_adaptiveLx5_sync.jld2 --xq (or --xlogsqq)
``` 

Memory time vs v for sliding: 
```
python3 sliding_plotter.py data/ising_sliding_ffs_p0.18_v0.00to5.00_adaptiveLx16test.jld2 data/ising_sliding_ffs_p0.21_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.24_v0.00to5.00_adaptiveLx8test.jld2 data/ising_sliding_ffs_p0.29_v0.00to5.00_adaptiveLx8test.jld2 --xr
``` 

Memory time vs (beta J)^2 for sliding: 