import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
try:
    import matplotlib.cbook
    if not hasattr(matplotlib.cbook, "_Stack"):
        class _Stack(list):
            def push(self, item):
                self.append(item)
                return item
            def pop(self):
                return super().pop() if self else None
            def current(self):
                return self[-1] if self else None
            def forward(self):
                pass
            def back(self):
                pass
        matplotlib.cbook._Stack = _Stack
except:
    pass

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13

J_par = 1.0
J_perp = 1.0


def mean_field_rhs(s, beta, h):
    """Right-hand side of the self-consistent equation for <s>."""
    arg = beta * (J_perp * s + h)
    numerator = np.sinh(arg)
    denominator = np.sqrt(np.cosh(arg)**2 - 2 * np.exp(-2 * beta * J_par) * np.sinh(2 * beta * J_par))
    return numerator / denominator


def solve_mean_field(beta, h, tol=1e-10, max_iter=10000):
    """Solve the mean-field equation by iteration from initial guess opposite to h."""
    epsilon = 1e-5
    if h >= 0: 
        s = -1+epsilon
    else:
        s = 1-epsilon

    for _ in range(max_iter):
        s_new = mean_field_rhs(s, beta, h)
        if abs(s_new - s) < tol:
            return s_new
        s = s_new

    return s


def main():
    parser = argparse.ArgumentParser(description='Mean-field phase diagram for sliding Ising ladder')
    parser.add_argument('--T_min', type=float, default=0.1)
    parser.add_argument('--T_max', type=float, default=3.0)
    parser.add_argument('--h_min', type=float, default=-1.0)
    parser.add_argument('--h_max', type=float, default=1.0)
    parser.add_argument('--res', type=int, default=200)
    args = parser.parse_args()

    T_vals = np.linspace(args.T_min, args.T_max, args.res)
    h_vals = np.linspace(args.h_min, args.h_max, args.res)

    magnetization = np.zeros((args.res, args.res))

    for i, T in enumerate(T_vals):
        beta = 1.0 / T
        for j, h in enumerate(h_vals):
            magnetization[i, j] = solve_mean_field(beta, h)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    im = ax.imshow(magnetization.T, origin='lower', aspect='auto',
                   extent=[args.T_min, args.T_max, args.h_min, args.h_max],
                   cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$h$')
    fig.colorbar(im, ax=ax, label=r'$\langle s \rangle$', shrink=0.9)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
