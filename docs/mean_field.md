# Mean-Field Theory

## Purpose

The script `mean_field.py` solves the self-consistent mean-field equation for the equilibrium magnetization of the two-chain Ising model and plots the result as a 2D phase diagram in the $(T, h)$ plane.

## Mean-Field Equation

The self-consistent equation for the average magnetization $\langle s \rangle$ is:

$$\langle s \rangle = \frac{\sinh(\beta(J_\perp \langle s \rangle + h))}{\sqrt{\cosh^2(\beta(J_\perp \langle s \rangle + h)) - 2 e^{-2\beta J_\parallel} \sinh(2\beta J_\parallel)}}$$

where $\beta = 1/T$, $J_\parallel = J_\perp = 1$ are the in-chain and inter-chain couplings, and $h$ is the external magnetic field.

This equation is derived by treating the inter-chain coupling at the mean-field level while exactly accounting for the 1D in-chain correlations via the transfer matrix. The denominator encodes the effect of in-chain correlations — in the limit $J_\parallel \to 0$, it reduces to $\cosh(\beta(J_\perp s + h))$ and the equation becomes the standard Curie-Weiss mean-field equation.

## Solution Method

The equation is solved by simple iteration: starting from an initial guess $s_0$, compute $s_{n+1} = f(s_n)$ until convergence ($|s_{n+1} - s_n| < 10^{-10}$). The initial guess is chosen with the **opposite sign** to $h$, which biases the iteration toward the metastable solution (the one that opposes the field). This reveals the first-order transition line where the metastable branch disappears — the spinodal.

## Output

A heatmap of $\langle s \rangle$ over a $\text{res} \times \text{res}$ grid in $(T, h)$ space, using the `coolwarm` colormap with $\langle s \rangle \in [-1, 1]$. The sharp color boundary traces the mean-field phase boundary.

## Usage

```bash
python mean_field.py --T_min=0.1 --T_max=3.0 --h_min=-1.0 --h_max=1.0 --res=200
```
