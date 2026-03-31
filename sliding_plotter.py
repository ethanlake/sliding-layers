import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import argparse
import os

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
# matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13
matplotlib.rcParams['legend.fontsize'] = 13 * 0.8
matplotlib.rcParams['figure.dpi'] = 75
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['lines.markeredgewidth'] = 0.5

linewidth = 2.75
marker_size = 6.5
cmap = plt.cm.coolwarm_r


def load_jld2_data(filename):
    """Load data from a sliding_ising_chain JLD2 file (HDF5 format).

    All params are saved as top-level scalars/arrays in the JLD2 file,
    so h5py can read them directly.
    """
    data = {}
    with h5py.File(filename, 'r') as f:
        top_keys = set(f.keys())

        def get_scalar(key):
            if key not in f:
                return None
            val = f[key]
            try:
                if len(val.shape) == 0:
                    return val[()]
                arr = val[:]
                return arr.item() if arr.size == 1 else arr
            except:
                return None

        def get_array(key):
            if key not in f:
                return None
            return np.array(f[key])

        # Read all scalar params
        for key in ['v', 'L', 'p', 'beta', 'num_trials', 'n_trials',
                     'T_equil', 'T_sample', 'T_steps', 'M_threshold', 'max_time']:
            val = get_scalar(key)
            if val is not None:
                data[key] = val

        # Detect mode from which top-level datasets exist
        if 'lc_values' in top_keys:
            data['mode'] = 'erosion_test'
            data['p_values'] = get_array('p_values')  # None if vary_v
            data['vs'] = get_array('vs')               # None if vary_p
            data['lc_values'] = get_array('lc_values')
            data['erode_l_values'] = get_array('erode_l_values')
            data['erode_probs'] = get_array('erode_probs')
            data['thresh_prob'] = get_scalar('thresh_prob')
        elif 'teff_values' in top_keys:
            data['mode'] = 'teff'
            data['T_values'] = get_array('T_values')  # None if vary_v
            data['vs'] = get_array('vs')               # None if vary_p
            data['teff_values'] = get_array('teff_values')
            # Demon data (may be None if FDR method)
            data['demon_bins'] = get_array('demon_bins')
            data['demon_histograms'] = get_array('demon_histograms')
            # FDR data (may be None if demon method)
            data['C_arrays'] = get_array('C_arrays')
            data['R_arrays'] = get_array('R_arrays')
            data['chi_arrays'] = get_array('chi_arrays')
            data['dC_arrays'] = get_array('dC_arrays')
            data['teff_pointwise'] = get_array('teff_pointwise')
        elif 'mean_energies' in top_keys:
            data['mode'] = 'energy'
            data['vs'] = get_array('vs')           # new format (v sweep)
            data['p_values'] = get_array('p_values')  # old format (p sweep)
            data['mean_energies'] = get_array('mean_energies')
            data['mean_heat_flows'] = get_array('mean_heat_flows')
        elif 'log_mixing_times' in top_keys:
            data['mode'] = 'ffs'
            data['p_values'] = get_array('p_values')  # None if vary_v
            data['vs'] = get_array('vs')               # None if vary_p
            data['mean_mixing_times'] = get_array('mean_mixing_times')
            data['log_mixing_times'] = get_array('log_mixing_times')
            data['log_mixing_times_std'] = get_array('log_mixing_times_std')
        elif 'mean_mixing_times' in top_keys:
            data['mode'] = 'mixing'
            data['p_values'] = get_array('p_values')  # None if vary_v
            data['vs'] = get_array('vs')               # None if vary_p
            data['mean_mixing_times'] = get_array('mean_mixing_times')
        elif 'magnetization_history' in top_keys:
            data['mode'] = 'history'
            data['magnetization_history'] = get_array('magnetization_history')
        else:
            data['mode'] = 'unknown'

    return data


def plot_erosion_mode(filenames, raw=False, small_stats=False):
    """Plot lc vs p or lc vs v, with each file as a separate curve.
    If erode_vs_l data is present, also plot survival probability vs l."""
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    colors = cmap(np.linspace(0, 1, max(len(filenames), 2)))

    xlabel = None
    has_erode_data = False

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'erosion_test':
            print(f"Warning: {filename} is not erosion_test mode, skipping...")
            continue

        lc_values = data['lc_values']

        if data.get('vs') is not None:
            # Varying v, fixed p
            x_values = data['vs']
            p = data.get('p', '?')
            p_label = f'{int(p)}' if isinstance(p, float) and p == int(p) else f'{p}'
            label = rf'$p={p_label}$'
            xlabel = r'$v$'
        else:
            # Varying p, fixed v
            x_values = data['p_values']
            v = data.get('v', '?')
            v_label = f'{int(v)}' if isinstance(v, float) and v == int(v) else f'{v}'
            label = rf'$v={v_label}$'
            xlabel = r'$e^{\beta J}$'

        ax.plot(x_values, lc_values, '-o', color=colors[idx],
                markerfacecolor=colors[idx], markeredgecolor='k',
                markersize=marker_size, linewidth=linewidth, alpha=0.7,
                label=label)

        # Linear fit to last 3/4 of data points
        n_fit = max(1, len(x_values) - len(x_values) // 4)
        x_fit_pts = x_values[-n_fit:]
        lc_fit_pts = lc_values[-n_fit:]
        coeffs = np.polyfit(x_fit_pts, lc_fit_pts, 1)
        slope = coeffs[0]
        print(f"{label.strip('$')}: slope = {slope:.4f}")
        x_fit = np.linspace(x_fit_pts.min(), x_fit_pts.max(), 100)
        lc_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, lc_fit, '--', color=colors[idx], linewidth=1.5, alpha=0.7)

        if data.get('erode_l_values') is not None:
            has_erode_data = True

    ax.set_xlabel(xlabel or r'$e^{\beta}$')
    ax.set_ylabel(r'$\ell_{\sf er}$')
    ax.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()

    # If any file has erode_vs_l data, plot P_shrink vs l/lc
    if has_erode_data:
        fig2, ax2 = plt.subplots(figsize=(4., 4.), dpi=100)
        ax2.minorticks_off()

        for idx, filename in enumerate(filenames):
            data = load_jld2_data(filename)
            if data['mode'] != 'erosion_test' or data.get('erode_l_values') is None:
                continue

            l_matrix = data['erode_l_values'].T    # Julia saves (max_len, n_curves), h5py reads transposed
            prob_matrix = data['erode_probs'].T
            lc_values = data['lc_values']
            thresh_prob = data.get('thresh_prob', 0.75)

            if data.get('vs') is not None:
                sweep_values = data['vs']
                sweep_key = 'v'
            else:
                sweep_values = data['p_values']
                sweep_key = 'p'

            n_curves = l_matrix.shape[1] if l_matrix.ndim == 2 else 1
            curve_colors = cmap(np.linspace(0, 1, max(n_curves, 2)))

            for i in range(n_curves):
                if l_matrix.ndim == 2:
                    ls = l_matrix[:, i]
                    ps = prob_matrix[:, i]
                else:
                    ls = l_matrix
                    ps = prob_matrix

                # Filter out zero-padded entries
                valid = ls > 0
                ls = ls[valid]
                ps = ps[valid]
                lc = lc_values[i] if i < len(lc_values) else lc_values[-1]

                if lc > 0:
                    sweep_val = sweep_values[i] if i < len(sweep_values) else sweep_values[-1]
                    if sweep_key == 'v':
                        curve_label = rf'$v={sweep_val:.1f}$'
                    else:
                        p_val = sweep_val
                        p_label = f'{int(p_val)}' if isinstance(p_val, float) and p_val == int(p_val) else f'{p_val:.2g}'
                        curve_label = rf'${p_label}$' if small_stats else rf'$p={p_label}$'

                    if small_stats:
                        mask = ls <= lc
                        x_plot = (1.0 - ls[mask] / lc)**2
                        y_plot = 1.0 - ps[mask]
                        if len(np.unique(ls[mask])) < 2:
                            continue
                    elif raw:
                        x_plot = ls
                        y_plot = 1.0 - ps
                    else:
                        x_plot = ls / lc
                        y_plot = 1.0 - ps

                    y_plot[y_plot <= 0] = np.nan  # avoid log(0)
                    ax2.plot(x_plot, y_plot, '-o', color=curve_colors[i],
                             markerfacecolor=curve_colors[i], markeredgecolor='k',
                             markersize=marker_size, linewidth=linewidth, alpha=0.7,
                             label=curve_label)

            if not small_stats:
                ax2.axhline(y=1.0 - thresh_prob, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        if small_stats:
            ax2.set_xlabel(r'$(1-\ell / \ell_{\sf er})^2$')
            ax2.set_yscale('log')
        elif raw:
            ax2.set_xlabel(r'$\ell$')
        else:
            ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
            ax2.set_xlabel(r'$\ell / \ell_c$')
        ax2.set_ylabel(r'$1-P_{\sf er}(\ell)$')
        legend_title = r'$e^{\beta J}$' if (small_stats and sweep_key == 'p') else None
        ax2.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9, title=legend_title)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_axisbelow(True)
        plt.tight_layout()
        plt.show()


def plot_energy_mode(filenames, heat=False):
    """Plot energy or heat flow data. New format: vs v (label by p). Old format: vs p (label by v)."""
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    colors = cmap(np.linspace(0, 1, max(len(filenames), 2)))

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'energy':
            print(f"Warning: {filename} is not energy mode, skipping...")
            continue

        if heat:
            y_values = data.get('mean_heat_flows')
            if y_values is None:
                print(f"Warning: {filename} has no heat flow data, skipping...")
                continue
        else:
            y_values = data['mean_energies']

        if data.get('vs') is not None:
            x = data['vs']
            p = data.get('p', '?')
            if isinstance(p, (int, float)):
                p_label = f'{int(p)}' if isinstance(p, float) and p == int(p) else f'{p:.3g}'
            else:
                p_label = str(p)
            label = rf'${p_label}$'
            xlabel = r'$v$'
        else:
            x = data['p_values']
            v = data.get('v', '?')
            if isinstance(v, float) and v == int(v):
                v_label = f'{int(v)}'
            else:
                v_label = f'{v}'
            label = rf'$v={v_label}$'
            xlabel = r'$e^{\beta J}$'

        if heat:
            ax.plot(x, y_values, '-o', color=colors[idx],
                    markerfacecolor=colors[idx], markeredgecolor='k',
                    markersize=marker_size, linewidth=linewidth, alpha=0.7,
                    label=label)
        else:
            ax.plot(x, y_values / y_values[0], '-o', color=colors[idx],
                    markerfacecolor=colors[idx], markeredgecolor='k',
                    markersize=marker_size, linewidth=linewidth, alpha=0.7,
                    label=label)

    ax.set_xlabel(xlabel)
    if heat:
        ax.set_ylabel(r'$\dot{Q}$')
    else:
        ax.set_ylabel(r'$\langle E \rangle / \langle E_{v=0}\rangle$')
    ax.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9, title=r'$e^{\beta J}$')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def plot_teff_mode(filenames):
    """Plot T_eff vs p or v, with each file as a separate curve."""
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    colors = cmap(np.linspace(0, 1, max(len(filenames), 2)))

    xlabel = None
    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'teff':
            print(f"Warning: {filename} is not teff mode, skipping...")
            continue

        teff_values = data['teff_values']

        if data.get('vs') is not None:
            x = data['vs']
            T_bath = data.get('T', '?')
            if isinstance(T_bath, (int, float)):
                T_label = f'{T_bath:.3g}'
            else:
                T_label = str(T_bath)
            label = rf'$T={T_label}$'
            xlabel = r'$v$'
            # Reference line: equilibrium temperature
            if isinstance(T_bath, (int, float)):
                ax.axhline(y=T_bath, color=colors[idx], linestyle=':', linewidth=1, alpha=0.5)
        else:
            x = data['T_values']
            v = data.get('v', '?')
            v_label = f'{int(v)}' if isinstance(v, (int, float)) and v == int(v) else f'{v}'
            label = rf'$v={v_label}$'
            xlabel = r'$T$'

        finite_mask = np.isfinite(teff_values)
        ax.plot(x[finite_mask], teff_values[finite_mask], '-o', color=colors[idx],
                markerfacecolor=colors[idx], markeredgecolor='k',
                markersize=marker_size, linewidth=linewidth, alpha=0.7,
                label=label)

    # Reference line: T_eff = T (diagonal) when sweeping T
    if xlabel == r'$T$':
        xlims = ax.get_xlim()
        ref = np.linspace(xlims[0], xlims[1], 100)
        ax.plot(ref, ref, ':', color='gray', linewidth=1, alpha=0.5)

    ax.set_xlabel(xlabel or r'$T$')
    ax.set_ylabel(r'$T_{\mathrm{eff}}$')
    ax.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()

    # FDR diagnostic plots (if data available)
    has_fdr = False
    for filename in filenames:
        data = load_jld2_data(filename)
        if data['mode'] == 'teff' and data.get('chi_arrays') is not None:
            has_fdr = True
            break

    if has_fdr:
        # Parametric plot: χ(τ) vs C(0) - C(τ), all sweep values overlaid
        fig2, ax2 = plt.subplots(figsize=(4., 4.), dpi=100)
        ax2.minorticks_off()

        for idx, filename in enumerate(filenames):
            data = load_jld2_data(filename)
            if data['mode'] != 'teff' or data.get('chi_arrays') is None:
                continue

            C_matrix = data['C_arrays'].T      # (T_response+1, n_sweep) -> transpose for h5py
            chi_matrix = data['chi_arrays'].T   # (T_response, n_sweep)
            teff_values = data['teff_values']

            if data.get('vs') is not None:
                sweep_values = data['vs']
                sweep_key = 'v'
            else:
                sweep_values = data['T_values']
                sweep_key = 'T'

            n_curves = chi_matrix.shape[1] if chi_matrix.ndim == 2 else 1
            curve_colors = cmap(np.linspace(0, 1, max(n_curves, 2)))

            for i in range(n_curves):
                if chi_matrix.ndim == 2:
                    C = C_matrix[:, i]
                    chi = chi_matrix[:, i]
                else:
                    C = C_matrix
                    chi = chi_matrix

                delta_C = C[0] - C[1:]  # C(0) - C(τ) for τ=1..T_response
                # Trim to same length as chi
                n = min(len(delta_C), len(chi))
                delta_C = delta_C[:n]
                chi_plot = chi[:n]

                sweep_val = sweep_values[i] if i < len(sweep_values) else sweep_values[-1]
                teff = teff_values[i] if i < len(teff_values) else np.nan
                if sweep_key == 'v':
                    curve_label = rf'$v={sweep_val:.1f}$'
                else:
                    curve_label = rf'$T={sweep_val:.3g}$'

                ax2.plot(delta_C, chi_plot, '-', color=curve_colors[i],
                         linewidth=linewidth, alpha=0.7, label=curve_label)

                # Overlay fit line
                if np.isfinite(teff) and teff > 0:
                    x_max = np.max(delta_C)
                    ax2.plot([0, x_max], [0, x_max / teff], '--', color=curve_colors[i],
                             linewidth=1, alpha=0.5)

        ax2.set_xlabel(r'$C(0) - C(\tau)$')
        ax2.set_ylabel(r'$\chi(\tau)$')
        ax2.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_axisbelow(True)
        plt.tight_layout()
        plt.show()

        # Pointwise T_eff(τ) vs τ
        fig3, ax3 = plt.subplots(figsize=(4., 4.), dpi=100)
        ax3.minorticks_off()

        for idx, filename in enumerate(filenames):
            data = load_jld2_data(filename)
            if data['mode'] != 'teff' or data.get('teff_pointwise') is None:
                continue

            teff_pw = data['teff_pointwise'].T  # transpose for h5py
            teff_values = data['teff_values']

            if data.get('vs') is not None:
                sweep_values = data['vs']
                sweep_key = 'v'
            else:
                sweep_values = data['T_values']
                sweep_key = 'T'

            n_curves = teff_pw.shape[1] if teff_pw.ndim == 2 else 1
            curve_colors = cmap(np.linspace(0, 1, max(n_curves, 2)))

            for i in range(n_curves):
                if teff_pw.ndim == 2:
                    pw = teff_pw[:, i]
                else:
                    pw = teff_pw

                taus = np.arange(1, len(pw) + 1)
                valid = np.isfinite(pw)
                sweep_val = sweep_values[i] if i < len(sweep_values) else sweep_values[-1]
                if sweep_key == 'v':
                    curve_label = rf'$v={sweep_val:.1f}$'
                else:
                    curve_label = rf'$T={sweep_val:.3g}$'

                ax3.plot(taus[valid], pw[valid], '-', color=curve_colors[i],
                         linewidth=linewidth, alpha=0.7, label=curve_label)

        ax3.set_xlabel(r'$\tau$')
        ax3.set_ylabel(r'$T_{\mathrm{eff}}(\tau) = \partial_\tau C / R$')
        ax3.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9)
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.set_axisbelow(True)
        plt.tight_layout()
        plt.show()


def _add_mixing_inset(ax, curve_data, colors):
    """Add inset plot showing exponential fit coefficients, and draw fit lines on main plot.

    curve_data: list of (x_values, log10_tau, param_value, is_vary_v, color_idx) tuples.
    If vary_v: fit log10(tau) = c_p * v * log10(e) + const for each curve, plot c_p vs p.
    If vary_p: fit log10(tau) = c_v * p * log10(e) + const for each curve, plot c_v vs v.
    """
    if len(curve_data) < 2:
        return

    is_vary_v = curve_data[0][3]
    params = []
    coeffs_list = []

    for x, log_tau, param_val, _, cidx in curve_data:
        finite = np.isfinite(log_tau)
        if np.sum(finite) < 2:
            continue
        x_fin = x[finite]
        y_fin = log_tau[finite]
        # Restrict fit range
        if is_vary_v:
            fit_mask = x_fin > 2
        else:
            n_fit = max(2, len(x_fin) - len(x_fin) // 4)
            fit_mask = np.zeros(len(x_fin), dtype=bool)
            fit_mask[-n_fit:] = True
        if np.sum(fit_mask) < 2:
            fit_mask = np.ones(len(x_fin), dtype=bool)  # fallback to all
        x_fit = x_fin[fit_mask]
        y_fit = y_fin[fit_mask]
        fit = np.polyfit(x_fit, y_fit, 1)
        slope, intercept = fit[0], fit[1]
        c = slope / np.log10(np.e)
        params.append(param_val)
        coeffs_list.append(c)

        # Draw fit line on main plot
        x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_line = 10.0 ** (slope * x_line + intercept)
        ax.plot(x_line, y_line, '--', color=colors[cidx], linewidth=1.5, alpha=0.75)

    if len(params) < 2:
        return

    params = np.array(params)
    coeffs = np.array(coeffs_list)

    inset = ax.inset_axes([0.15, 0.55, 0.35, 0.35])
    inset.plot(params, coeffs, 'o-', color='k', markersize=4, linewidth=1.5)
    if is_vary_v:
        inset.set_xlabel(r'$e^{\beta J}$', fontsize=10)
        inset.set_ylabel(r'$c_p$', fontsize=10)
    else:
        inset.set_xlabel(r'$v$', fontsize=10)
        inset.set_ylabel(r'$c_v$', fontsize=10)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


def plot_mixing_mode(filenames, inset=False):
    """Plot mixing time vs p, with each file (different v or L) as a separate curve."""
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    colors = cmap(np.linspace(0, 1, max(len(filenames), 2)))

    xlabel = None
    is_vary_v = None
    curve_data = []

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'mixing':
            print(f"Warning: {filename} is not mixing mode, skipping...")
            continue

        mean_mixing_times = data['mean_mixing_times']
        L = data.get('L', '?')

        if data.get('vs') is not None:
            x_values = data['vs']
            p = data.get('p', '?')
            is_vary_v = True
            if inset:
                label = rf'${p:.1f}$' if isinstance(p, (int, float)) else rf'${p}$'
            else:
                p_label = f'{int(p)}' if isinstance(p, float) and p == int(p) else f'{p}'
                label = rf'$p={p_label},\, L={L}$'
            xlabel = r'$v$'
            curve_data.append((x_values, np.log10(mean_mixing_times), p, True, idx))
        else:
            x_values = data['p_values']
            v = data.get('v', '?')
            is_vary_v = False
            if inset:
                label = rf'${v:.1f}$' if isinstance(v, (int, float)) else rf'${v}$'
            else:
                v_label = f'{int(v)}' if isinstance(v, float) and v == int(v) else f'{v}'
                label = rf'$v={v_label},\, L={L}$'
            xlabel = r'$e^{\beta J}$'
            curve_data.append((x_values, np.log10(mean_mixing_times), v, False, idx))

        ax.plot(x_values, mean_mixing_times, '-o', color=colors[idx],
                markerfacecolor=colors[idx], markeredgecolor='k',
                markersize=marker_size, linewidth=linewidth, alpha=0.7,
                label=label)

    if inset and len(curve_data) > 1:
        _add_mixing_inset(ax, curve_data, colors)

    ax.set_xlabel(xlabel or r'$e^{\beta J}$')
    ax.set_ylabel(r'$t_{\sf mem}$')
    ax.set_yscale('log')
    legend_title = None
    if inset and is_vary_v is not None:
        legend_title = r'$e^{\beta J}$' if is_vary_v else r'$v$'
    ax.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9,
              title=legend_title)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def plot_ffs_mode(filenames, inset=False):
    """Plot FFS mixing time vs p or v, with exp(L*beta) reference line."""
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    colors = cmap(np.linspace(0, 1, max(len(filenames), 2)))

    max_mixing_time = 0
    ref_lines = []
    xlabel = None
    is_vary_v = None
    curve_data = []

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'ffs':
            print(f"Warning: {filename} is not ffs mode, skipping...")
            continue

        mean_mixing_times = data['mean_mixing_times']
        L = data.get('L', '?')
        log_tau = data.get('log_mixing_times')

        if data.get('vs') is not None:
            x_values = data['vs']
            p = data.get('p', '?')
            is_vary_v = True
            if isinstance(p, (int, float)):
                label = rf'${p:.2f}$'
            else:
                label = rf'${p}$'
            xlabel = r'$v$'
            log_tau_arr = log_tau if log_tau is not None else np.log10(mean_mixing_times)
            curve_data.append((x_values, log_tau_arr, p, True, idx))
        else:
            x_values = data['p_values'] **.5 
            # ax.set_xscale('log')
            v = data.get('v', '?')
            is_vary_v = False
            if inset:
                label = rf'${v:.1f}$' if isinstance(v, (int, float)) else rf'${v}$'
            else:
                v_label = f'{int(v)}' if isinstance(v, float) and v == int(v) else f'{v}'
                label = rf'$v={v_label},\, L={L}$'
            xlabel = r'$e^{\beta J}$'
            if isinstance(L, (int, float)) and L > 0:
                ref_lines.append((L, x_values))
            log_tau_arr = log_tau if log_tau is not None else np.log10(mean_mixing_times)
            curve_data.append((x_values, log_tau_arr, v, False, idx))

        # Filter out Inf/NaN values for plotting
        log_tau = data.get('log_mixing_times')
        log_tau_std = data.get('log_mixing_times_std')
        finite_mask = np.isfinite(mean_mixing_times)

        if log_tau is not None and log_tau_std is not None and np.any(np.isfinite(log_tau_std)):
            tau_upper = 10.0 ** (log_tau + log_tau_std)
            tau_lower = 10.0 ** (log_tau - log_tau_std)
            mask = finite_mask & np.isfinite(log_tau_std)
            ax.errorbar(x_values[mask], mean_mixing_times[mask],
                        yerr=[mean_mixing_times[mask] - tau_lower[mask],
                              tau_upper[mask] - mean_mixing_times[mask]],
                        fmt='-o', color=colors[idx],
                        markerfacecolor=colors[idx], markeredgecolor='k',
                        markersize=marker_size, linewidth=linewidth, alpha=0.7,
                        capsize=3, label=label)
        else:
            ax.plot(x_values[finite_mask], mean_mixing_times[finite_mask], '-o', color=colors[idx],
                    markerfacecolor=colors[idx], markeredgecolor='k',
                    markersize=marker_size, linewidth=linewidth, alpha=0.7,
                    label=label)

        if np.any(finite_mask):
            max_mixing_time = max(max_mixing_time, np.max(mean_mixing_times[finite_mask]))

    # Draw exp(L * beta) = p^L reference curve (only when sweeping p)
    for i, (L, p_values) in enumerate(ref_lines):
        ref_values = p_values ** L
        if max_mixing_time > np.min(ref_values):
            ax.plot(p_values, ref_values, ':', color='red', linewidth=1.5,
                    label=rf'$e^{{L \beta}}$' if i == 0 else None)

    if inset and len(curve_data) > 1:
        _add_mixing_inset(ax, curve_data, colors)

    ax.set_xlabel(xlabel or r'$e^{\beta J}$')
    ax.set_ylabel(r'$t_{\sf mem}$')
    ax.set_yscale('log')
    legend_title = None
    if is_vary_v:
        legend_title = r'$e^{\beta J}$'
    elif inset and is_vary_v is not None:
        legend_title = r'$v$'
    leg = ax.legend(loc='best', frameon=not True, fancybox=False, edgecolor='black', framealpha=0.9,
                    title=legend_title)
    if leg.get_title():
        leg.get_title().set_fontsize(13 * 1.2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def plot_history_mode(filenames):
    """Plot spacetime magnetization heatmap from a single file."""
    for filename in filenames:
        data = load_jld2_data(filename)
        if data['mode'] != 'history':
            print(f"Warning: {filename} is not history mode, skipping...")
            continue

        mag = data['magnetization_history']
        v = data.get('v', 0)
        L = data.get('L', mag.shape[0])
        beta_val = data.get('beta', '?')

        # Shift into co-moving frame at velocity v/2
        T_steps = mag.shape[1]
        mag_shifted = np.empty_like(mag)
        for t in range(T_steps):
            shift = int(round(v * t / 2)) % L
            mag_shifted[:, t] = np.roll(mag[:, t], -shift)

        fig, ax = plt.subplots(figsize=(3., 6.), dpi=100)
        im = ax.imshow(mag_shifted.T, aspect='auto', origin='lower', cmap='RdBu',
                       vmin=-2, vmax=2)
        ax.set_xlabel(r'Site')
        ax.set_ylabel(r'$t$')
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot data from sliding_ising_chain.jl')
    parser.add_argument('files', nargs='+', help='JLD2 files to plot')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'erosion_test', 'energy', 'teff', 'mixing', 'ffs', 'history'],
                        help='Mode to plot (default: auto-detect from first file)')
    parser.add_argument('--raw', action='store_true',
                        help='Plot P_shrink vs l instead of l/lc (erosion_test mode)')
    parser.add_argument('--small_stats', action='store_true',
                        help='Plot 1-P_shrink vs (l-lc)^2 for l<=lc on log scale (erosion_test mode)')
    parser.add_argument('--heat', action='store_true',
                        help='Plot heat flow instead of energy (energy mode)')
    parser.add_argument('--inset', action='store_true',
                        help='Add inset showing exponential fit coefficients (mixing/ffs modes, requires multiple files)')

    args = parser.parse_args()

    # Detect mode from first file if auto
    if args.mode == 'auto':
        first_data = load_jld2_data(args.files[0])
        mode = first_data.get('mode', 'unknown')
    else:
        mode = args.mode

    if mode == 'erosion_test':
        plot_erosion_mode(args.files, raw=args.raw, small_stats=args.small_stats)
    elif mode == 'energy':
        plot_energy_mode(args.files, heat=args.heat)
    elif mode == 'teff':
        plot_teff_mode(args.files)
    elif mode == 'mixing':
        plot_mixing_mode(args.files, inset=args.inset)
    elif mode == 'ffs':
        plot_ffs_mode(args.files, inset=args.inset)
    elif mode == 'history':
        plot_history_mode(args.files)
    else:
        print(f"Unknown mode '{mode}'. Supported: erosion_test, energy, mixing, history")


if __name__ == "__main__":
    main()
