import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
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
# amsmath gives us \boldsymbol (needed for bolding math italic + Greek in
# the history-mode v±δ annotations). Harmless everywhere else.
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
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

# Set by main() when --cmap is provided. When non-None, it globally overrides
# every per-file color picker below (the default coolwarm_r, the GKL rainbow,
# the Oranges-for-v-sweep / Blues-for-p-sweep choices in plot_ffs_mode, etc.).
_USER_CMAP = None

# Set by main() when --legloc is provided. When non-None, every ax.legend()
# call below uses this loc string instead of its hard-coded default.
_USER_LEGLOC = None

# Set by main() when --legend_alpha is provided. When non-None, every legend
# call below uses this framealpha instead of its hard-coded default.
_USER_LEGEND_ALPHA = None


def _user_or(default_cmap_obj):
    """Return the --cmap override if set, otherwise `default_cmap_obj`."""
    return _USER_CMAP if _USER_CMAP is not None else default_cmap_obj


def _legloc(default='best'):
    """Return the --legloc override if set, otherwise `default`."""
    return _USER_LEGLOC if _USER_LEGLOC is not None else default


def _legalpha(default):
    """Return the --legend_alpha override if set, otherwise `default`."""
    return _USER_LEGEND_ALPHA if _USER_LEGEND_ALPHA is not None else default


# Set by main() when --add_lines is provided. When True, marker-only data
# series (ffs / mixing / erosion) draw a connecting line between points.
_USER_ADD_LINES = False


def _line_fmt(default='o'):
    """Return '-' + default-marker if --add_lines is set, else just the marker."""
    return ('-' + default) if _USER_ADD_LINES else default


def _line_lw(default=0):
    """Return the standard `linewidth` if --add_lines is set, else `default`."""
    return linewidth if _USER_ADD_LINES else default


def _legend_no_errorbars(ax, **kwargs):
    """Like ax.legend(**kwargs), but the legend marker for any ax.errorbar()
    curve drops the error-bar whiskers — only the bare marker is shown,
    matching how the same curve is rendered in the data area (now that
    connecting lines are off). Falls back to the default handle for anything
    that isn't an ErrorbarContainer."""
    handles, labels = ax.get_legend_handles_labels()
    clean = []
    for h in handles:
        if isinstance(h, ErrorbarContainer) and h.lines:
            ml = h.lines[0]
            clean.append(Line2D([], [],
                                marker=ml.get_marker(),
                                color=ml.get_color(),
                                markerfacecolor=ml.get_markerfacecolor(),
                                markeredgecolor=ml.get_markeredgecolor(),
                                markersize=ml.get_markersize(),
                                linestyle='none'))
        else:
            clean.append(h)
    return ax.legend(clean, labels, **kwargs)


def _pick_cmap(filenames):
    """Return plt.cm.rainbow when any file in `filenames` has dynamics='gkl',
    otherwise the default `cmap` (coolwarm_r). Used so GKL plots get rainbow
    colors across files while everything else keeps the default coolwarm.
    Overridden globally by --cmap."""
    if _USER_CMAP is not None:
        return _USER_CMAP
    try:
        for fn in filenames:
            if load_jld2_data(fn).get('dynamics') == 'gkl':
                return plt.cm.rainbow
    except Exception:
        pass
    return cmap


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
                     'T_equil', 'T_sample', 'T_steps', 'M_threshold', 'max_time',
                     'eta', 'p_noise']:
            val = get_scalar(key)
            if val is not None:
                data[key] = val

        # Dynamics tag (e.g. 'gkl' from gkl.jl); None for legacy / sliding-Ising files.
        dyn = get_scalar('dynamics')
        if isinstance(dyn, bytes):
            dyn = dyn.decode('utf-8')
        data['dynamics'] = dyn

        # Detect mode from which top-level datasets exist
        if 'v_onset_values' in top_keys:
            data['mode'] = 'phase_diagram'
            data['p_values'] = get_array('p_values')
            data['v_values'] = get_array('v_values')
            y_mat = get_array('y_matrix')
            data['y_matrix'] = y_mat.T if y_mat is not None else None  # restore (n_p, n_v) after h5py read
            data['v_onset_values'] = get_array('v_onset_values')
            obs = get_scalar('observable')
            if isinstance(obs, bytes):
                obs = obs.decode('utf-8')
            data['observable'] = obs
            data['onset_threshold'] = get_scalar('onset_threshold')
        elif 'log_mixing_times' in top_keys:
            # NOTE: this check must come before the lc_values one — FFS files
            # now save lc_values too (one per sweep point), so the erosion_test
            # branch would otherwise swallow them.
            data['mode'] = 'ffs'
            data['p_values'] = get_array('p_values')  # None if vary_v
            data['vs'] = get_array('vs')               # None if vary_p
            data['mean_mixing_times'] = get_array('mean_mixing_times')
            data['log_mixing_times'] = get_array('log_mixing_times')
            data['log_mixing_times_std'] = get_array('log_mixing_times_std')
            data['n_repeats'] = get_scalar('n_repeats')
            data['n_configs'] = get_scalar('n_configs')                 # old format
            data['n_configs_per_run'] = get_scalar('n_configs_per_run') # new format
            data['per_run_log_taus'] = get_array('per_run_log_taus')    # new format
            data['seed_droplet_size'] = get_scalar('seed_droplet_size') # may be None on pre-seeding files
        elif 'lc_values' in top_keys:
            data['mode'] = 'erosion_test'
            data['p_values'] = get_array('p_values')  # None if vary_v
            data['vs'] = get_array('vs')               # None if vary_p
            data['lc_values'] = get_array('lc_values')
            data['lc_stderrs'] = get_array('lc_stderrs')  # None on old files
            data['erode_l_values'] = get_array('erode_l_values')
            data['erode_probs'] = get_array('erode_probs')
            data['thresh_prob'] = get_scalar('thresh_prob')
            # first-passage escape mode (if present): mirrors erode_* layout
            data['escape_l_values'] = get_array('escape_l_values')
            data['escape_probs'] = get_array('escape_probs')
            data['min_doublons'] = get_scalar('min_doublons')
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
            data['mean_energies_std'] = get_array('mean_energies_std')      # may be None on old files
            data['mean_heat_flows_std'] = get_array('mean_heat_flows_std')  # may be None on old files
        elif 'mean_mixing_times' in top_keys:
            data['mode'] = 'mixing'
            data['p_values'] = get_array('p_values')  # None if vary_v
            data['vs'] = get_array('vs')               # None if vary_p
            data['mean_mixing_times'] = get_array('mean_mixing_times')
        elif 'magnetization_history' in top_keys:
            data['mode'] = 'history'
            data['magnetization_history'] = get_array('magnetization_history')
        elif 'D_values' in top_keys:
            data['mode'] = 'diffusion'
            data['p_values'] = get_array('p_values')  # None if vary_eta
            data['vs'] = get_array('vs')               # None if vary_p
            data['D_values'] = get_array('D_values')
            data['D_stderrs'] = get_array('D_stderrs')
            data['init_mag'] = get_scalar('init_mag')
            data['T_thermalize'] = get_scalar('T_thermalize')
            data['T_track'] = get_scalar('T_track')
            data['n_trials'] = get_scalar('n_trials')
            data['msd_curve_first'] = get_array('msd_curve_first')
            sm = get_scalar('sweep_mode')
            if isinstance(sm, bytes):
                sm = sm.decode('utf-8')
            data['sweep_mode'] = sm  # 'p', 'tau', 'eta' (None on older files)
        else:
            data['mode'] = 'unknown'

    return data


def plot_erosion_mode(filenames, raw=False, small_stats=False, fit_inset=False, logy=False):
    """Plot lc vs p or lc vs v, with each file as a separate curve.
    If erode_vs_l data is present, also plot survival probability vs l.
    If fit_inset is True, add an inset axes showing the per-file linear-fit
    slope vs the file's fixed parameter (v when sweeping p, p when sweeping v)."""
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    _local_cmap = _pick_cmap(filenames)
    colors = _local_cmap(np.linspace(0, 1, max(len(filenames), 2)))

    xlabel = None
    has_erode_data = False
    has_escape_data = False
    slope_records = []  # list of (fixed_param_value, slope, slope_stderr, color, sweep_kind)
    global_fit_records = []  # used when fit_inset=False: list of dicts per file
    sweep_kind_global = None

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'erosion_test':
            print(f"Warning: {filename} is not erosion_test mode, skipping...")
            continue

        lc_values = data['lc_values']
        lc_stderrs = data.get('lc_stderrs')
        has_errors = lc_stderrs is not None and np.any(np.isfinite(lc_stderrs))

        is_gkl = data.get('dynamics') == 'gkl'
        if data.get('vs') is not None:
            # Varying v (or eta for GKL), fixed p (or p_noise for GKL)
            x_values = data['vs']
            if is_gkl:
                p_noise = data.get('p_noise', '?')
                p_label = f'{p_noise:.2f}' if isinstance(p_noise, (int, float)) else f'{p_noise}'
                label = rf'$p={p_label}$'
                xlabel = r'$\eta$'
                fixed_param_val = p_noise if isinstance(p_noise, (int, float)) else None
            else:
                p = data.get('p', '?')
                p_label = f'{int(p)}' if isinstance(p, float) and p == int(p) else f'{p}'
                label = rf'$p={p_label}$'
                xlabel = r'$v$'
                fixed_param_val = p if isinstance(p, (int, float)) else None
            sweep_kind = 'vary_v'
        else:
            # Varying p (or p_noise for GKL), fixed v (or eta for GKL)
            x_values = data['p_values']
            if is_gkl:
                eta = data.get('eta', '?')
                eta_label = f'{eta:.2f}' if isinstance(eta, (int, float)) else f'{eta}'
                label = rf'$\eta={eta_label}$'
                xlabel = r'$\epsilon$'
                fixed_param_val = eta if isinstance(eta, (int, float)) else None
            else:
                v = data.get('v', '?')
                v_label = f'{int(v)}' if isinstance(v, float) and v == int(v) else f'{v}'
                label = rf'$v={v_label}$'
                xlabel = r'$e^{\beta J}$'
                fixed_param_val = v if isinstance(v, (int, float)) else None
            sweep_kind = 'vary_p'

        # GKL convention: plot against 1/sqrt(p_noise) when sweeping p_noise.
        # (Leading-order scaling argument predicts xi_er ~ 1/sqrt(ε).)
        if is_gkl and sweep_kind == 'vary_p':
            x_values = 1.0 / np.sqrt(np.asarray(x_values, dtype=float))
            xlabel = r'$1/\sqrt{\epsilon}$'

        # Filter: only plot/fit data points where the erosion length is >= 5.
        x_arr = np.asarray(x_values)
        lc_arr = np.asarray(lc_values)
        mask = lc_arr >= 5

        if has_errors:
            err = np.where(np.isfinite(lc_stderrs), lc_stderrs, 0.0)
            ax.errorbar(x_arr[mask], lc_arr[mask], yerr=err[mask], fmt=_line_fmt('o'),
                        color=colors[idx], markerfacecolor=colors[idx],
                        markeredgecolor='k',
                        markersize=0.75 * marker_size,
                        linewidth=_line_lw(0), alpha=1.0, capsize=3, label=label,
                        zorder=3)
        else:
            ax.plot(x_arr[mask], lc_arr[mask], _line_fmt('o'), color=colors[idx],
                    markerfacecolor=colors[idx], markeredgecolor='k',
                    markersize=0.75 * marker_size, linewidth=_line_lw(0),
                    alpha=1.0, label=label, zorder=3)

        sweep_kind_global = sweep_kind  # all files share the same sweep kind

        if fit_inset:
            # Per-file linear fit on points with lc >= 5 (weighted if we have stderrs)
            x_fit_pts = x_arr[mask]
            lc_fit_pts = lc_arr[mask]
            if len(x_fit_pts) < 2:
                continue
            slope_stderr = np.nan
            coeffs = None
            if has_errors:
                err_fit_pts = lc_stderrs[mask]
                valid = np.isfinite(err_fit_pts) & (err_fit_pts > 0)
                if np.sum(valid) >= 2:
                    try:
                        coeffs, cov = np.polyfit(x_fit_pts[valid], lc_fit_pts[valid],
                                                  1, w=1.0 / err_fit_pts[valid],
                                                  cov=True)
                        slope_stderr = float(np.sqrt(cov[0, 0]))
                    except (ValueError, np.linalg.LinAlgError):
                        coeffs = None
            if coeffs is None:
                coeffs = np.polyfit(x_fit_pts, lc_fit_pts, 1)
            slope = coeffs[0]
            print(f"{label.strip('$')}: slope = {slope:.4f}"
                  + (f" ± {slope_stderr:.4f}" if np.isfinite(slope_stderr) else ""))
            x_fit = np.linspace(x_fit_pts.min(), x_fit_pts.max(), 100)
            lc_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, lc_fit, '--', color='k', linewidth=1.2, alpha=0.85,
                    zorder=1)
            if fixed_param_val is not None:
                slope_records.append((fixed_param_val, slope, slope_stderr,
                                      colors[idx], sweep_kind))
        elif not is_gkl:
            # Collect data for the global two-parameter fit y = s * v * p + b.
            # Skipped for GKL files — that model doesn't apply.
            if sweep_kind == 'vary_v':
                v_arr = x_arr[mask].astype(float)
                p_arr = np.full_like(v_arr,
                                      float(fixed_param_val) if fixed_param_val is not None else np.nan)
            else:  # vary_p
                p_arr = x_arr[mask].astype(float)
                v_arr = np.full_like(p_arr,
                                      float(fixed_param_val) if fixed_param_val is not None else np.nan)
            err_arr = lc_stderrs[mask] if has_errors else None
            global_fit_records.append({
                'v': v_arr,
                'p': p_arr,
                'lc': lc_arr[mask].astype(float),
                'err': err_arr,
                'x_range': x_arr[mask].astype(float),
                'color': colors[idx],
            })

        if data.get('erode_l_values') is not None:
            has_erode_data = True
        if data.get('escape_l_values') is not None:
            has_escape_data = True

    ax.set_xlabel(xlabel or r'$p$')
    ax.set_ylabel(r'$\xi_{\sf er}$')
    if fit_inset:
        leg = _legend_no_errorbars(ax, loc=_legloc('lower right'), frameon=True,
                                   fancybox=False, edgecolor='none', framealpha=_legalpha(0.8))
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_linewidth(0)
    else:
        _legend_no_errorbars(ax, loc=_legloc(), frameon=not True, fancybox=False,
                             edgecolor='black', framealpha=_legalpha(0.9))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    if not fit_inset and global_fit_records:
        # Three-parameter weighted fit of y = s * (v - v0) * p + b across all files.
        # Linearize as y = s*(v*p) + c*p + b with c = -s*v0; recover v0 = -c/s.
        v_all = np.concatenate([r['v'] for r in global_fit_records])
        p_all = np.concatenate([r['p'] for r in global_fit_records])
        y_all = np.concatenate([r['lc'] for r in global_fit_records])
        all_have_err = all(r['err'] is not None for r in global_fit_records)
        X1 = v_all * p_all
        X2 = p_all
        X3 = np.ones_like(X1)
        finite = np.isfinite(X1) & np.isfinite(X2) & np.isfinite(y_all)
        if all_have_err:
            err_all = np.concatenate([r['err'] for r in global_fit_records])
            valid = finite & np.isfinite(err_all) & (err_all > 0)
            w = 1.0 / err_all[valid] ** 2
        else:
            valid = finite
            w = np.ones(int(np.sum(valid)))

        s_hat = v0_hat = b_hat = s_se = v0_se = b_se = np.nan
        if int(np.sum(valid)) >= 3:
            M = np.column_stack([X1[valid], X2[valid], X3[valid]])
            sw = np.sqrt(w)
            A = M * sw[:, None]
            rhs = y_all[valid] * sw
            try:
                beta, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                s_hat = float(beta[0]); c_hat = float(beta[1]); b_hat = float(beta[2])
                MtWM = M.T @ (w[:, None] * M)
                cov = np.linalg.inv(MtWM)
                if not all_have_err:
                    resid = y_all[valid] - M @ beta
                    dof = max(1, int(np.sum(valid)) - 3)
                    cov = cov * float(np.sum(resid ** 2) / dof)
                s_se = float(np.sqrt(max(cov[0, 0], 0)))
                b_se = float(np.sqrt(max(cov[2, 2], 0)))
                if s_hat != 0.0:
                    v0_hat = -c_hat / s_hat
                    # delta method: var(v0) with v0 = -c/s
                    dv_ds = c_hat / s_hat ** 2
                    dv_dc = -1.0 / s_hat
                    v0_var = (dv_ds ** 2 * cov[0, 0] + dv_dc ** 2 * cov[1, 1]
                              + 2 * dv_ds * dv_dc * cov[0, 1])
                    v0_se = float(np.sqrt(max(v0_var, 0)))
            except (ValueError, np.linalg.LinAlgError):
                pass

        print(f"global fit: s = {s_hat:.4f}"
              + (f" ± {s_se:.4f}" if np.isfinite(s_se) else "")
              + f", v0 = {v0_hat:.4f}"
              + (f" ± {v0_se:.4f}" if np.isfinite(v0_se) else "")
              + f", b = {b_hat:.4f}"
              + (f" ± {b_se:.4f}" if np.isfinite(b_se) else ""))

        for rec in global_fit_records:
            xs = rec['x_range']
            if xs.size == 0 or not (np.isfinite(s_hat) and np.isfinite(v0_hat)
                                    and np.isfinite(b_hat)):
                continue
            x_line = np.linspace(float(xs.min()), float(xs.max()), 100)
            if sweep_kind_global == 'vary_v':
                p_fixed = float(rec['p'][0]) if rec['p'].size else np.nan
                y_line = s_hat * (x_line - v0_hat) * p_fixed + b_hat
            else:  # vary_p
                v_fixed = float(rec['v'][0]) if rec['v'].size else np.nan
                y_line = s_hat * (v_fixed - v0_hat) * x_line + b_hat
            ax.plot(x_line, y_line, '--', color='k', linewidth=1.2,
                    alpha=0.85, zorder=1)

        if np.isfinite(s_hat):
            annot = rf'$s = {s_hat:.2f}$'
            if np.isfinite(v0_hat):
                annot += '\n' + rf'$v_0 = {v0_hat:.2f}$'
            if np.isfinite(b_hat):
                annot += '\n' + rf'$b = {b_hat:.2f}$'
            ax.text(0.05, 0.95, annot, transform=ax.transAxes,
                    ha='left', va='top', fontsize=13)

    if fit_inset and len(slope_records) >= 1:
        # Sort by the fixed parameter so the inset is a clean line
        slope_records.sort(key=lambda r: r[0])
        fxs = np.array([r[0] for r in slope_records])
        slopes = np.array([r[1] for r in slope_records])
        kind = slope_records[0][4]
        inset_xlabel = r'$v$' if kind == 'vary_p' else r'$p$'
        inset = ax.inset_axes([0.11, 0.66, 0.32, 0.32])
        inset.minorticks_off()
        inset.plot(fxs, slopes, '-', color='0.4', linewidth=1, zorder=1)
        for fx, sl, se, col, _ in slope_records:
            if np.isfinite(se):
                inset.errorbar(fx, sl, yerr=se, fmt='o', color=col,
                               markerfacecolor=col, markeredgecolor='k',
                               markersize=marker_size * 0.7, capsize=2,
                               zorder=2)
            else:
                inset.plot(fx, sl, 'o', color=col, markerfacecolor=col,
                           markeredgecolor='k', markersize=marker_size * 0.7,
                           zorder=2)
        inset.set_xlabel(inset_xlabel, fontsize=15, labelpad=-6)
        inset.set_ylabel(r'$s$', fontsize=15, labelpad=-6)
        inset.tick_params(labelsize=9)
        inset.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        inset.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    # Per-curve plot: either P_shrink-vs-l/lc (default) or P_escape-vs-l/lc
    # if any file contains first-passage escape data (auto-detect, replaces y-axis).
    if has_erode_data or has_escape_data:
        fig2, ax2 = plt.subplots(figsize=(4., 4.), dpi=100)
        ax2.minorticks_off()

        for idx, filename in enumerate(filenames):
            data = load_jld2_data(filename)
            if data['mode'] != 'erosion_test':
                continue
            # Prefer escape data when available (first-passage mode)
            use_escape = data.get('escape_l_values') is not None
            if use_escape:
                l_matrix = data['escape_l_values'].T
                prob_matrix = data['escape_probs'].T
            elif data.get('erode_l_values') is not None:
                l_matrix = data['erode_l_values'].T
                prob_matrix = data['erode_probs'].T
            else:
                continue
            lc_values = data['lc_values']
            thresh_prob = data.get('thresh_prob', 0.75)

            is_gkl = data.get('dynamics') == 'gkl'
            if data.get('vs') is not None:
                sweep_values = data['vs']
                sweep_key = 'v'
            else:
                sweep_values = data['p_values']
                sweep_key = 'p'

            n_curves = l_matrix.shape[1] if l_matrix.ndim == 2 else 1
            curve_colors = _pick_cmap(filenames)(np.linspace(0, 1, max(n_curves, 2)))

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
                        curve_label = (rf'$\eta={sweep_val:.2f}$' if is_gkl
                                       else rf'$v={sweep_val:.1f}$')
                    else:
                        p_val = sweep_val
                        p_label = f'{int(p_val)}' if isinstance(p_val, float) and p_val == int(p_val) else f'{p_val:.2g}'
                        curve_label = rf'${p_label}$' if small_stats else rf'$p={p_label}$'

                    if small_stats:
                        mask = ls <= lc
                        x_plot = (1.0 - ls[mask] / lc)**2
                        y_plot = (ps[mask] if use_escape else 1.0 - ps[mask])
                        if len(np.unique(ls[mask])) < 2:
                            continue
                    elif raw:
                        x_plot = ls
                        y_plot = ps if use_escape else 1.0 - ps
                    else:
                        x_plot = ls / lc
                        y_plot = ps if use_escape else 1.0 - ps

                    y_plot = np.array(y_plot, dtype=float)
                    y_plot[y_plot <= 0] = np.nan  # avoid log(0)
                    ax2.plot(x_plot, y_plot, '-o', color=curve_colors[i],
                             markerfacecolor=curve_colors[i], markeredgecolor='k',
                             markersize=marker_size, linewidth=linewidth, alpha=0.7,
                             label=curve_label)

            # Threshold line is only meaningful for the shrink view
            if not small_stats and not use_escape:
                ax2.axhline(y=1.0 - thresh_prob, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        if small_stats:
            ax2.set_xlabel(r'$(1-\ell / \ell_{\sf er})^2$')
            ax2.set_yscale('log')
        elif raw:
            ax2.set_xlabel(r'$\ell$')
        else:
            ax2.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
            ax2.set_xlabel(r'$\ell / \ell_c$')
        if logy and not small_stats:
            ax2.set_yscale('log')
        ax2.set_ylabel(r'$P_{\sf escape}(\ell)$' if has_escape_data else r'$1-P_{\sf er}(\ell)$')
        legend_title = r'$p$' if (small_stats and sweep_key == 'p') else None
        _legend_no_errorbars(ax2, loc=_legloc(), frameon=not True, fancybox=False, edgecolor='black', framealpha=_legalpha(0.9), title=legend_title)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.set_axisbelow(True)
        plt.tight_layout()
        plt.show()


def plot_energy_mode(filenames, heat=False, fitrange=None, residuals=False):
    """Plot energy or heat flow data. New format: vs v (label by p). Old format: vs p (label by v)."""
    _want_res = (fitrange is not None) and residuals
    fig, ax, residual_ax = _setup_residual_axes(_want_res)
    ax.minorticks_off()
    colors = _user_or(cmap)(np.linspace(0, 1, max(len(filenames), 2)))

    if fitrange is not None:
        print(f"Per-file linear fits ln(y) ~ x over upper {(1.0 - float(fitrange)) * 100:.0f}% of x:")

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'energy':
            print(f"Warning: {filename} is not energy mode, skipping...")
            continue

        if heat:
            y_values = data.get('mean_heat_flows')
            y_std = data.get('mean_heat_flows_std')
            if y_values is None:
                print(f"Warning: {filename} has no heat flow data, skipping...")
                continue
        else:
            y_values = data['mean_energies']
            y_std = data.get('mean_energies_std')

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
            xlabel = r'$p$'

        # Use errorbar when block-averaged stderrs are present (saved by the
        # post-2026 energy mode). Old files without these fields still plot
        # via a plain line.
        has_err = (y_std is not None and np.any(np.isfinite(y_std)))
        if heat:
            y_plot = y_values
            yerr = y_std if has_err else None
        else:
            # Energies are normalised to the v=0 (or p=p_min) reference value.
            # Stderr is first-order-propagated by the same factor; this ignores
            # the small uncertainty in y_values[0] itself, which is fine since
            # ⟨E⟩ at the reference is the best-determined point.
            denom = y_values[0]
            y_plot = y_values / denom
            yerr = (y_std / denom) if has_err else None

        if has_err:
            ax.errorbar(x, y_plot, yerr=yerr, fmt='-o', color=colors[idx],
                        markerfacecolor=colors[idx], markeredgecolor='k',
                        markersize=marker_size, linewidth=linewidth, alpha=1.0,
                        capsize=3, label=label)
        else:
            ax.plot(x, y_plot, '-o', color=colors[idx],
                    markerfacecolor=colors[idx], markeredgecolor='k',
                    markersize=marker_size, linewidth=linewidth, alpha=1.0,
                    label=label)

        if fitrange is not None:
            _do_fitrange_fit(ax, np.asarray(x), np.asarray(y_plot),
                             fitrange, label, filename,
                             residual_ax=residual_ax, color=colors[idx])

    ax.set_xlabel(xlabel)
    if heat:
        ax.set_ylabel(r'$\dot{Q}$')
    else:
        ax.set_ylabel(r'$E(v) / E(0)$')
    # Semitransparent white legend background (same style as the FFS / erosion
    # plots): visible frame off, no edge, α≈0.5 so curves behind remain
    # partially readable through it.
    leg = _legend_no_errorbars(ax, loc=_legloc(), frameon=True, fancybox=False,
                               edgecolor='none', framealpha=_legalpha(0.5),
                               title=r'$e^{-\beta J}$')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    # v-sweep: force a major tick at every integer v in the visible range
    # (matches the FFS-mode behaviour added earlier).
    if xlabel == r'$v$':
        try:
            xmin, xmax = ax.get_xlim()
            lo = int(np.ceil(xmin - 1e-9))
            hi = int(np.floor(xmax + 1e-9))
            if hi >= lo:
                ax.set_xticks(list(range(lo, hi + 1)))
        except Exception:
            pass
    if not _finalize_residual_panel(ax, residual_ax):
        plt.tight_layout()
    plt.show()


def plot_teff_mode(filenames):
    """Plot T_eff vs p or v, with each file as a separate curve."""
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    colors = _user_or(cmap)(np.linspace(0, 1, max(len(filenames), 2)))

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
    _legend_no_errorbars(ax, loc=_legloc(), frameon=not True, fancybox=False, edgecolor='black', framealpha=_legalpha(0.9))
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
            curve_colors = _user_or(cmap)(np.linspace(0, 1, max(n_curves, 2)))

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
        _legend_no_errorbars(ax2, loc=_legloc(), frameon=not True, fancybox=False, edgecolor='black', framealpha=_legalpha(0.9))
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
            curve_colors = _user_or(cmap)(np.linspace(0, 1, max(n_curves, 2)))

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
        _legend_no_errorbars(ax3, loc=_legloc(), frameon=not True, fancybox=False, edgecolor='black', framealpha=_legalpha(0.9))
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.set_axisbelow(True)
        plt.tight_layout()
        plt.show()


def _setup_residual_axes(want_residuals):
    """Build the figure used by ffs / mixing / energy plotters.

    Returns (fig, ax, residual_ax). When `want_residuals` is False this is a
    one-panel figure and residual_ax is None (preserves the original layout
    for the default code path). When True we make a 3:1 two-panel gridspec
    with sharex; the bottom panel is wired up with a y=0 guide, grid, and
    label so callers only need to scatter their per-file residuals onto it.
    """
    if want_residuals:
        # constrained_layout handles the two-panel sizing cleanly; matches
        # what the bare-ax path gets from plt.tight_layout() at the end.
        fig, axes = plt.subplots(2, 1, figsize=(4., 5.), dpi=100,
                                 gridspec_kw={'height_ratios': [3, 1],
                                              'hspace': 0.08},
                                 sharex=True, constrained_layout=True)
        ax, residual_ax = axes[0], axes[1]
        residual_ax.axhline(0, color='k', linewidth=0.8, alpha=0.6, zorder=0)
        residual_ax.set_ylabel(r'$\ln y - \mathrm{fit}$')
        residual_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        residual_ax.set_axisbelow(True)
        return fig, ax, residual_ax
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    return fig, ax, None


def _finalize_residual_panel(ax, residual_ax):
    """Move the x-label from the top panel to the residual panel and hide the
    top panel's x-tick labels. Called just before plt.show() in any plotter
    that opted into the two-panel layout via _setup_residual_axes.

    Returns True iff a residual panel is present, so callers can skip the
    plt.tight_layout() call (residual figures use constrained_layout and
    mixing the two is noisy)."""
    if residual_ax is None:
        return False
    xlabel = ax.get_xlabel()
    ax.set_xlabel('')
    residual_ax.set_xlabel(xlabel)
    plt.setp(ax.get_xticklabels(), visible=False)
    return True


def _do_fitrange_fit(ax, x_plot, y_vals, fitrange, label, filename,
                     residual_ax=None, color=None):
    """Linear fit of ln(y) vs x over the upper (1 − `fitrange`) of x.

    Used by --fitrange. `x_plot` should be the *plotted* x-coordinate so the
    fit-line lands on the same axis the curve was drawn on (so e.g. when the
    user passed --a, x_plot is already x**a). Draws a black dashed line over
    the fit range and prints slope ± se, intercept ± se, R², n to the
    terminal. No-ops with an explanatory print if there aren't ≥ 2 finite,
    positive-y points in the fit window.

    When `residual_ax` is given (from --residuals), also scatters
    r_i = ln(y_i) − (slope·x_i + intercept) for ALL finite, positive-y points
    (including those outside the fit window — that's the whole point of the
    diagnostic: see whether the model curves away on the un-fitted side).
    Per-file `color` is used so the residual mapping matches the top panel.
    """
    x = np.asarray(x_plot, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    label_str = label.strip('$') if label else filename
    if np.sum(mask) < 2:
        print(f"  --fitrange [{filename}]: <2 finite positive-y points, skipped.")
        return
    x_m, y_m = x[mask], y[mask]
    x_lo, x_hi = float(np.min(x_m)), float(np.max(x_m))
    x_cut = x_lo + float(fitrange) * (x_hi - x_lo)
    fmask = x_m >= x_cut
    if np.sum(fmask) < 2:
        print(f"  --fitrange [{filename}]: window x >= {x_cut:.4g} keeps "
              f"{int(np.sum(fmask))} points; need >=2, skipped.")
        return
    x_fit = x_m[fmask]
    ln_y_fit = np.log(y_m[fmask])
    slope, intercept = np.polyfit(x_fit, ln_y_fit, 1)
    y_pred = slope * x_fit + intercept
    ss_res = float(np.sum((ln_y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((ln_y_fit - np.mean(ln_y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = len(x_fit)
    slope_se = np.nan
    intercept_se = np.nan
    if n >= 3 and ss_res > 0:
        s_resid = np.sqrt(ss_res / (n - 2))
        x_var = float(np.sum((x_fit - x_fit.mean()) ** 2))
        if x_var > 0:
            slope_se = s_resid / np.sqrt(x_var)
            intercept_se = s_resid * np.sqrt(1.0 / n + x_fit.mean() ** 2 / x_var)
    se_slope = f" +/- {slope_se:.4g}" if np.isfinite(slope_se) else ""
    se_int = f" +/- {intercept_se:.4g}" if np.isfinite(intercept_se) else ""
    print(f"  --fitrange [{label_str}] ({filename}): "
          f"x in [{x_cut:.4g}, {x_hi:.4g}], n={n}, "
          f"slope = {slope:.4g}{se_slope}, "
          f"intercept = {intercept:.4g}{se_int}, "
          f"R^2 = {r2:.4g}")
    fit_color = color if color is not None else 'k'
    x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
    y_line = np.exp(slope * x_line + intercept)
    ax.plot(x_line, y_line, '--', color=fit_color, linewidth=1.5, alpha=0.85,
            zorder=1)

    if residual_ax is not None:
        # ALL points (in-window and outside) are scattered against the fit;
        # systematic drift outside the fit window is exactly what we want to
        # see for the "wrong exponent?" diagnostic.
        residuals = np.log(y_m) - (slope * x_m + intercept)
        col = color if color is not None else 'k'
        order = np.argsort(x_m)
        residual_ax.plot(x_m[order], residuals[order], '-o',
                         color=col, markerfacecolor=col, markeredgecolor='k',
                         markersize=marker_size * 0.75,
                         linewidth=linewidth * 0.6, alpha=1.0, zorder=2)


def _add_fit_inset(ax, curve_data, colors, x_transform=None, alpha=None,
                   xr=False, a_for_inset=None):
    """For each file, linearly fit log10(tau) vs the *plotted* x-coordinate on
    the last half of the data, draw the fit as a dashed black line over the
    fit range, and add an inset plotting the per-file slope s against the
    file-level fixed parameter (v when sweeping p; e^(beta J) or (beta J)^alpha
    when sweeping v).

    curve_data: list of (x_values, log10_tau, fixed_param, is_vary_v, idx).
    x_transform: callable applied to x_values to match the main-plot transform.
    alpha: if given and we are on a v-sweep, the inset x-axis is (beta J)^alpha
           rather than e^(beta J).
    xr: if True and we are on a v-sweep, interpret the fit as
            t_mem = exp((βJ)^a · v · s)
        with default exponent a = 2 unless `a_for_inset` overrides it. The
        inset then plots s = slope_log10 · ln 10 / (βJ)^a against (βJ)^a for
        the different files.
    a_for_inset: optional override for the exponent `a` in the τ = exp((βJ)^a·v·s)
        fit. Only used when xr=True and is_vary_v=True. Defaults to 2 if None.
    """
    if len(curve_data) < 1:
        return
    if x_transform is None:
        x_transform = lambda x: x  # noqa: E731

    is_vary_v = curve_data[0][3]
    params = []
    slopes = []

    for x_raw, log_tau, fixed_param, _, _ in curve_data:
        x_t = x_transform(np.asarray(x_raw, dtype=float))
        log_tau = np.asarray(log_tau, dtype=float)
        finite = np.isfinite(log_tau) & np.isfinite(x_t)
        if np.sum(finite) < 2:
            continue
        x_fin = x_t[finite]
        y_fin = log_tau[finite]
        # Sort by x so "last half" means largest x
        order = np.argsort(x_fin)
        x_fin, y_fin = x_fin[order], y_fin[order]
        n_half = max(2, len(x_fin) // 2)
        x_fit = x_fin[-n_half:]
        y_fit = y_fin[-n_half:]
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        params.append(fixed_param)
        slopes.append(slope)
        # Draw the fit on the main plot (over the fit range only)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_line = 10.0 ** (slope * x_line + intercept)
        ax.plot(x_line, y_line, '--', color='k', linewidth=1.2, alpha=0.8)

    if len(params) < 1:
        return

    p_arr = np.array(params, dtype=float)
    slopes_arr = np.array(slopes, dtype=float)  # d(log10 τ)/d(plotted x) per file
    if is_vary_v and xr:
        # v-sweep + xr: each file has fixed p. Fit τ = exp((βJ)^a · v · s),
        # with plotted x = v ⇒ d(log10 τ)/dv = ((βJ)^a · s) / ln 10 per file,
        # i.e. s = slope · ln 10 / (βJ)^a. Inset x-axis is (βJ)^a so a flat
        # line means the (βJ)^a · v scaling holds.
        a_fit = a_for_inset if a_for_inset is not None else 2.0
        bj_a = np.abs(np.log(p_arr)) ** a_fit   # |βJ|^a since βJ = −log p > 0
        inset_x = bj_a
        slopes_arr = slopes_arr * np.log(10.0) / bj_a
        a_str = f"{a_fit:g}"
        inset_xlabel = rf'$(\beta J)^{{{a_str}}}$'
    elif (not is_vary_v) and xr:
        # p-sweep + xr: each file has fixed v. Plotted x = (βJ)^a, so the fit
        # τ = exp(s · v · (βJ)^a) gives d(log10 τ)/d(βJ)^a = s·v/ln 10 per
        # file, i.e. s = slope · ln 10 / v. Inset x-axis is v, so different
        # files (different v) give one data point each; a flat curve means
        # the same fit constant s describes every v.
        v_arr = p_arr  # `params` here are the fixed v values per file
        inset_x = v_arr
        slopes_arr = slopes_arr * np.log(10.0) / v_arr
        inset_xlabel = r'$v$'
    elif is_vary_v:
        # Each file has a fixed p; inset x is e^(beta J) = p or (beta J)^alpha.
        if alpha is not None:
            inset_x = np.log(p_arr) ** alpha
            inset_xlabel = rf'$(\beta J)^{{{alpha:g}}}$'
        else:
            inset_x = p_arr
            inset_xlabel = r'$p$'
    else:
        inset_x = p_arr  # files distinguished by v
        inset_xlabel = r'$v$'

    order = np.argsort(inset_x)
    inset_x = inset_x[order]
    slopes_arr = slopes_arr[order]

    inset = ax.inset_axes([0.15, 0.55, 0.35, 0.35])
    inset.plot(inset_x, slopes_arr, 'o-', color='k', markersize=4, linewidth=1.5)
    inset.set_xlabel(inset_xlabel, fontsize=10)
    inset.set_ylabel(r'$s$', fontsize=10)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


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
        inset.set_xlabel(r'$p$', fontsize=10)
        inset.set_ylabel(r'$c_p$', fontsize=10)
    else:
        inset.set_xlabel(r'$v$', fontsize=10)
        inset.set_ylabel(r'$c_v$', fontsize=10)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


def plot_mixing_mode(filenames, inset=False, alpha=None, fit_inset=False,
                     fitrange=None, residuals=False):
    """Plot mixing time vs p, with each file (different v or L) as a separate curve.

    If alpha is given (p-sweeps only), transform x = p to
    (ln p)**alpha = (beta J)**alpha before plotting; ignored on v-sweeps.

    If fit_inset is True, fit log10(tau) = s * x_plotted + b on the last half
    of each curve, draw the fits as dashed black lines, and add an inset of
    s vs the file-level fixed parameter (v on p-sweeps; e^(beta J) or
    (beta J)^alpha on v-sweeps).
    """
    _want_res = (fitrange is not None) and residuals
    fig, ax, residual_ax = _setup_residual_axes(_want_res)
    ax.minorticks_off()
    colors = _user_or(cmap)(np.linspace(0, 1, max(len(filenames), 2)))

    xlabel = None
    is_vary_v = None
    curve_data = []

    def _xtransform(x):
        # Convention p = exp(-β J) ⇒ β J = -log(p). Raise (β J)^α.
        return (-np.log(x)) ** alpha if alpha is not None else x

    if fitrange is not None:
        print(f"Per-file linear fits ln(y) ~ x over upper {(1.0 - float(fitrange)) * 100:.0f}% of x:")

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'mixing':
            print(f"Warning: {filename} is not mixing mode, skipping...")
            continue

        mean_mixing_times = data['mean_mixing_times']
        L = data.get('L', '?')

        is_ising_glauber = data.get('dynamics') == 'ising_glauber'
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
            is_vary_v = False
            if is_ising_glauber:
                label = rf'$L={L}$ (direct)'
                xlabel = r'$p$'
                curve_data.append((x_values, np.log10(mean_mixing_times), L, False, idx))
            else:
                v = data.get('v', '?')
                if inset:
                    label = rf'${v:.1f}$' if isinstance(v, (int, float)) else rf'${v}$'
                else:
                    v_label = f'{int(v)}' if isinstance(v, float) and v == int(v) else f'{v}'
                    label = rf'$v={v_label},\, L={L}$'
                xlabel = r'$p$'
                curve_data.append((x_values, np.log10(mean_mixing_times), v, False, idx))

        x_plot = _xtransform(x_values) if (not is_vary_v) else x_values
        ax.plot(x_plot, mean_mixing_times, _line_fmt('o'), color=colors[idx],
                markerfacecolor=colors[idx], markeredgecolor='k',
                markersize=marker_size, linewidth=_line_lw(0), alpha=1.0,
                label=label)

        if fitrange is not None:
            _do_fitrange_fit(ax, np.asarray(x_plot),
                             np.asarray(mean_mixing_times),
                             fitrange, label, filename,
                             residual_ax=residual_ax, color=colors[idx])

    if inset and len(curve_data) > 1 and alpha is None and not fit_inset:
        _add_mixing_inset(ax, curve_data, colors)
    if fit_inset and len(curve_data) >= 1:
        _add_fit_inset(ax, curve_data, colors,
                       x_transform=(_xtransform if (is_vary_v is False) else (lambda x: x)),
                       alpha=alpha)

    if alpha is not None and is_vary_v is not None and not is_vary_v:
        ax.set_xlabel(rf'$(\beta J)^{{{alpha:g}}}$')
    else:
        ax.set_xlabel(xlabel or r'$p$')
    ax.set_ylabel(r'$t_{\sf mem}$')
    any_ig_mix = any(load_jld2_data(fn).get('dynamics') == 'ising_glauber' for fn in filenames)
    if not any_ig_mix:
        ax.set_yscale('log')
    legend_title = None
    if inset and is_vary_v is not None:
        legend_title = r'$p$' if is_vary_v else r'$v$'
    _legend_no_errorbars(ax, loc=_legloc(), frameon=not True, fancybox=False,
                         edgecolor='black', framealpha=_legalpha(0.9), title=legend_title)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    if not _finalize_residual_panel(ax, residual_ax):
        plt.tight_layout()
    plt.show()


def plot_ffs_mode(filenames, inset=False, ploglog=False, a_exp=None, alpha=None,
                  fit_inset=False, mixing_overlay=None, anchor_normalize=False,
                  xlogsqq=False, xq=False, xp=False, xr=False, fitrange=None,
                  residuals=False):
    """Plot FFS mixing time vs p or v, with exp(L*beta) reference line.

    If ploglog is True, plot log10(tau) (rather than tau) on a log y-axis
    against p on a log x-axis (only meaningful when sweeping p). A per-file
    linear fit in log-log space gives the exponent a in log(tau) ~ p^a.

    If a_exp is given, transform the x-axis from x to x**a_exp before
    plotting (a straight line on log-y / linear-x^a means log tau ~ x^a).
    Ignored when ploglog is True.

    If mixing_overlay is a list of mixing-mode JLD2 filenames, each is loaded
    and rendered as scatter points on top of the FFS curves (no fit lines,
    no errorbars). Used for the ising.jl FFS-vs-direct benchmark.

    If alpha is given (p-sweeps only), transform x = p to
    (ln p)**alpha = (beta J)**alpha. A straight line on log-y then means
    log tau ~ (beta J)^alpha. Takes precedence over --a; ignored when
    sweeping v or when ploglog is True.
    """
    if a_exp is not None and ploglog:
        print("Warning: --a and --ploglog are mutually exclusive; ignoring --a.")
        a_exp = None
    if alpha is not None and ploglog:
        print("Warning: --alpha and --ploglog are mutually exclusive; ignoring --alpha.")
        alpha = None
    if alpha is not None and a_exp is not None:
        print("Warning: --alpha takes precedence over --a; ignoring --a.")
        a_exp = None

    # --xlogsqq / --xq / --xp / --xr are mutually exclusive shorthand transforms.
    # Precedence: xlogsqq > xq > xp > xr; warn and drop the losers. (xlogsqq/xq/xp
    # carry GKL-flavored labels using ε; xr carries the sliding-Ising label
    # (β J)^2 for the same (−log p)^2 transform.)
    if sum(bool(x) for x in (xlogsqq, xq, xp, xr)) > 1:
        print("Warning: --xlogsqq, --xq, --xp, --xr are mutually exclusive; "
              "using the first set in (xlogsqq, xq, xp, xr).")
        if xlogsqq:
            xq = xp = xr = False
        elif xq:
            xp = xr = False
        elif xp:
            xr = False
    _new_xform = xlogsqq or xq or xp or xr
    if _new_xform and ploglog:
        print("Warning: --xlogsqq/--xq/--xp/--xr are ignored with --ploglog.")
        xlogsqq = xq = xp = xr = False
        _new_xform = False
    # Under --xr, --a controls the *exponent* of the (βJ)-axis transform:
    # main-plot x-axis becomes (βJ)^a instead of the default (βJ)^2, and the
    # fit-inset (when --fit_inset is set) uses the same exponent in
    # τ = exp((βJ)^a · v · s). For non-xr "shorthand" transforms (xlogsqq, xq,
    # xp), --a is still incompatible and gets dropped with a warning.
    if _new_xform and alpha is not None:
        print("Warning: --xlogsqq/--xq/--xp/--xr take precedence over --alpha.")
        alpha = None
    if _new_xform and a_exp is not None and not xr:
        print("Warning: --xlogsqq/--xq/--xp take precedence over --a "
              "(only --xr accepts --a as an exponent override).")
        a_exp = None
    elif xr and a_exp is not None:
        print(f"Note: --xr + --a={a_exp:g} → plotting against (βJ)^{a_exp:g} "
              "(both main axis and fit-inset).")

    def _xtransform(x):
        # Convention p = exp(-β J) ⇒ β J = -log(p). Raise (β J)^α.
        if xlogsqq:
            return (-np.log(x)) ** 2
        if xq:
            return 1.0 / np.asarray(x, dtype=float)
        if xp:
            return x
        if xr:
            # --a overrides the default exponent (2) when given.
            a_xr = 2.0 if a_exp is None else a_exp
            return (-np.log(x)) ** a_xr
        if alpha is not None:
            return (-np.log(x)) ** alpha
        if a_exp is not None:
            return x ** a_exp
        return x

    # Print a one-line summary of how many independent FFS runs went into each
    # file's mixing-time estimates.
    print("FFS run counts per file:")
    for fn in filenames:
        d = load_jld2_data(fn)
        if d['mode'] != 'ffs':
            continue
        nreq = d.get('n_repeats')
        ncpr = d.get('n_configs_per_run')
        ncfg = d.get('n_configs')
        prl = d.get('per_run_log_taus')
        budget = (f"n_configs_per_run={int(ncpr)}" if ncpr is not None
                  else f"n_configs={int(ncfg)} (legacy total)" if ncfg is not None
                  else "n_configs=?")
        nreq_str = f"n_repeats={int(nreq)}" if nreq is not None else "n_repeats=?"
        if prl is not None and prl.ndim >= 1:
            # per_run_log_taus is saved per sweep point; count finite entries
            arr = np.asarray(prl, dtype=float)
            # may be (n_repeats, n_sweep) after h5py read of julia (n_sweep, n_repeats);
            # axis with size n_repeats is the smaller of the two
            n_ok = np.isfinite(arr).sum(axis=int(np.argmin(arr.shape)))
            n_ok_str = f", n_ok={list(map(int, n_ok))}"
        else:
            n_ok_str = ""
        print(f"  {fn}: {nreq_str}, {budget}{n_ok_str}")

    _want_res = (fitrange is not None) and residuals and not ploglog
    fig, ax, residual_ax = _setup_residual_axes(_want_res)
    ax.minorticks_off()
    # Colormap by sweep type:
    #   v-sweep (x = v):                       Oranges
    #   p-sweep (x = temperature-like; p, ε, (βJ)^a, ...):  Blues
    # Both are sliced [0.35, 0.95] so the lightest curve is still legible on
    # a white background and the darkest doesn't crush to black.
    _any_vary_v = any(load_jld2_data(fn).get('vs') is not None for fn in filenames)
    if _USER_CMAP is not None:
        colors = _USER_CMAP(np.linspace(0.35, 0.95, max(len(filenames), 2)))
    elif _any_vary_v:
        colors = plt.cm.Oranges(np.linspace(0.35, 0.95, max(len(filenames), 2)))
    else:
        colors = plt.cm.Blues(np.linspace(0.35, 0.95, max(len(filenames), 2)))

    # Per-FFS-file multiplicative rescale so that the FFS τ at the smallest x
    # value matches the direct (mixing-overlay) τ at the same x. Only takes
    # effect when --anchor_normalize is set AND a mixing overlay is present.
    anchor_scale = {}
    if anchor_normalize and mixing_overlay:
        direct_xs, direct_ys = None, None
        for fn in mixing_overlay:
            d = load_jld2_data(fn)
            if d['mode'] != 'mixing':
                continue
            direct_xs = np.asarray(d.get('p_values') if d.get('vs') is None else d['vs'], dtype=float)
            direct_ys = np.asarray(d['mean_mixing_times'], dtype=float)
            break  # use first mixing file as the reference
        if direct_xs is None:
            print("Warning: --anchor_normalize requested but no mixing overlay file found; "
                  "FFS curves left unscaled.")
        else:
            for fn in filenames:
                d = load_jld2_data(fn)
                if d['mode'] != 'ffs':
                    continue
                xs = np.asarray(d.get('p_values') if d.get('vs') is None else d['vs'], dtype=float)
                ys = np.asarray(d['mean_mixing_times'], dtype=float)
                valid = np.isfinite(ys) & (ys > 0)
                if not np.any(valid):
                    continue
                i_anchor = np.where(valid)[0][np.argmin(xs[valid])]
                x_anchor = xs[i_anchor]
                ffs_at_anchor = ys[i_anchor]
                j_match = int(np.argmin(np.abs(direct_xs - x_anchor)))
                direct_at_anchor = direct_ys[j_match]
                if np.isfinite(direct_at_anchor) and direct_at_anchor > 0 and ffs_at_anchor > 0:
                    anchor_scale[fn] = direct_at_anchor / ffs_at_anchor
                    print(f"  anchor: {fn}: scale = {anchor_scale[fn]:.4g} "
                          f"(FFS={ffs_at_anchor:.3g}, direct={direct_at_anchor:.3g} at x={x_anchor:.3g})")

    max_mixing_time = 0
    ref_lines = []
    xlabel = None
    is_vary_v = None
    curve_data = []

    if fitrange is not None and ploglog:
        print("Warning: --fitrange is ignored with --ploglog (y-axis is already log10(tau)).")
    if fitrange is not None and not ploglog:
        print(f"Per-file linear fits ln(y) ~ x over upper {(1.0 - float(fitrange)) * 100:.0f}% of x:")

    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'ffs':
            print(f"Warning: {filename} is not ffs mode, skipping...")
            continue

        mean_mixing_times = data['mean_mixing_times']
        L = data.get('L', '?')
        log_tau = data.get('log_mixing_times')

        # Apply anchor normalization (multiplicative scale)
        _scale = anchor_scale.get(filename)
        if _scale is not None:
            mean_mixing_times = np.asarray(mean_mixing_times, dtype=float) * _scale
            if log_tau is not None:
                log_tau = np.asarray(log_tau, dtype=float) + np.log10(_scale)

        is_gkl = data.get('dynamics') == 'gkl'
        is_ising_glauber = data.get('dynamics') == 'ising_glauber'
        _ms = marker_size * (0.75 if is_gkl else 1.0)
        if data.get('vs') is not None:
            x_values = data['vs']
            is_vary_v = True
            if is_gkl:
                fixed = data.get('p_noise', '?')
                label = rf'${fixed:.3f}$' if isinstance(fixed, (int, float)) else rf'${fixed}$'
                xlabel = r'$\eta$'
            else:
                fixed = data.get('p', '?')
                # With --xr, label each curve by its value of r = (βJ)² instead
                # of by raw p (the sliding-Ising temperature variable the user
                # actually controlled the sweep with).
                if xr and isinstance(fixed, (int, float)) and fixed > 0:
                    r_val = (-np.log(fixed)) ** 2
                    label = rf'${r_val:.2f}$'
                else:
                    label = rf'${fixed:.2f}$' if isinstance(fixed, (int, float)) else rf'${fixed}$'
                xlabel = r'$v$'
            log_tau_arr = log_tau if log_tau is not None else np.log10(mean_mixing_times)
            curve_data.append((x_values, log_tau_arr, fixed, True, idx))
        else:
            x_values = data['p_values'] #**.5
            # ax.set_xscale('log')
            is_vary_v = False
            if is_ising_glauber:
                # No "fixed" param when sweeping p (h is the only other knob).
                fixed = data.get('L', '?')
                label = (rf'$L={fixed}$' if isinstance(fixed, (int, float))
                         else rf'${fixed}$')
                xlabel = r'$p$'
            elif is_gkl:
                fixed = data.get('eta', '?')
                label = (rf'${fixed:.1f}$' if isinstance(fixed, (int, float))
                         else rf'${fixed}$')
                xlabel = r'$\epsilon$'
            else:
                fixed = data.get('v', '?')
                label = (rf'${fixed:.1f}$' if isinstance(fixed, (int, float))
                         else rf'${fixed}$')
                xlabel = r'$p$'
                if isinstance(L, (int, float)) and L > 0:
                    ref_lines.append((L, x_values))
            log_tau_arr = log_tau if log_tau is not None else np.log10(mean_mixing_times)
            curve_data.append((x_values, log_tau_arr, fixed, False, idx))

        # Append anchor-scale annotation to the curve label, if any
        if _scale is not None:
            label = label + rf' ($\times{_scale:.2g}$)'

        # Filter out Inf/NaN values for plotting
        log_tau = data.get('log_mixing_times')
        if _scale is not None and log_tau is not None:
            log_tau = np.asarray(log_tau, dtype=float) + np.log10(_scale)
        log_tau_std = data.get('log_mixing_times_std')
        finite_mask = np.isfinite(mean_mixing_times)

        if ploglog:
            if is_vary_v:
                print(f"Warning: {filename} is a v-sweep; --ploglog only makes "
                      f"sense for p-sweeps, skipping fit.")
            y_vals = log_tau if log_tau is not None else np.log10(mean_mixing_times)
            has_std = (log_tau_std is not None
                       and np.any(np.isfinite(log_tau_std)))
            mask = finite_mask & (y_vals > 0)
            if has_std:
                mask = mask & np.isfinite(log_tau_std)
                yerr = log_tau_std[mask]
            else:
                yerr = None

            # Per-file power-law fit: log10(log_tau) = a * log10(p) + c.
            a_fit = np.nan
            if (not is_vary_v) and np.sum(mask) >= 2:
                lx = np.log10(x_values[mask])
                ly = np.log10(y_vals[mask])
                if has_std and yerr is not None:
                    # Propagate std on log_tau to std on log10(log_tau):
                    # d(log10(z))/dz = 1/(z ln 10), so sigma' = sigma/(z ln 10).
                    sig = yerr / (y_vals[mask] * np.log(10.0))
                    valid = np.isfinite(sig) & (sig > 0)
                    if np.sum(valid) >= 2:
                        coeffs = np.polyfit(lx[valid], ly[valid], 1,
                                             w=1.0 / sig[valid])
                    else:
                        coeffs = np.polyfit(lx, ly, 1)
                else:
                    coeffs = np.polyfit(lx, ly, 1)
                a_fit, c_fit = float(coeffs[0]), float(coeffs[1])
                print(f"{filename}: log10(log10 tau) ~ {a_fit:.3f} log10(p) "
                      f"+ {c_fit:.3f}  =>  log tau ~ p^{a_fit:.3f}")
                # Draw fit line on main plot
                x_line = np.linspace(x_values[mask].min(),
                                     x_values[mask].max(), 100)
                y_line = 10.0 ** (a_fit * np.log10(x_line) + c_fit)
                ax.plot(x_line, y_line, '--', color=colors[idx],
                        linewidth=1.2, alpha=0.85, zorder=1)
                # Append exponent to legend label
                label = rf'{label}, $a={a_fit:.2f}$'

            if has_std:
                ax.errorbar(x_values[mask], y_vals[mask], yerr=yerr,
                            fmt=_line_fmt('o'), color=colors[idx],
                            markerfacecolor=colors[idx], markeredgecolor='k',
                            markersize=_ms, linewidth=_line_lw(0),
                            alpha=1.0, capsize=3, label=label)
            else:
                ax.plot(x_values[mask], y_vals[mask], _line_fmt('o'), color=colors[idx],
                        markerfacecolor=colors[idx], markeredgecolor='k',
                        markersize=_ms, linewidth=_line_lw(0),
                        alpha=1.0, label=label)
            if np.any(mask):
                max_mixing_time = max(max_mixing_time, float(np.max(y_vals[mask])))
            continue

        x_plot = _xtransform(x_values) if (not is_vary_v) else x_values
        if log_tau is not None and log_tau_std is not None and np.any(np.isfinite(log_tau_std)):
            tau_upper = 10.0 ** (log_tau + log_tau_std)
            tau_lower = 10.0 ** (log_tau - log_tau_std)
            mask = finite_mask & np.isfinite(log_tau_std)
            ax.errorbar(x_plot[mask], mean_mixing_times[mask],
                        yerr=[mean_mixing_times[mask] - tau_lower[mask],
                              tau_upper[mask] - mean_mixing_times[mask]],
                        fmt=_line_fmt('o'), color=colors[idx],
                        markerfacecolor=colors[idx], markeredgecolor='k',
                        markersize=_ms, linewidth=_line_lw(0), alpha=1.0,
                        capsize=3, label=label)
        else:
            ax.plot(x_plot[finite_mask], mean_mixing_times[finite_mask], _line_fmt('o'), color=colors[idx],
                    markerfacecolor=colors[idx], markeredgecolor='k',
                    markersize=_ms, linewidth=_line_lw(0), alpha=1.0,
                    label=label)

        if fitrange is not None:
            _do_fitrange_fit(ax,
                             np.asarray(x_plot)[finite_mask],
                             np.asarray(mean_mixing_times)[finite_mask],
                             fitrange, label, filename,
                             residual_ax=residual_ax, color=colors[idx])

        if np.any(finite_mask):
            max_mixing_time = max(max_mixing_time, np.max(mean_mixing_times[finite_mask]))

    # Draw exp(L β) reference curve (sweeping p, not ploglog).
    # Convention p = exp(-β J) ⇒ exp(L β) = p^(-L).
    if not ploglog:
        for i, (L, p_values) in enumerate(ref_lines):
            ref_values = p_values ** (-L)
            if max_mixing_time > np.min(ref_values):
                x_ref = _xtransform(p_values)
                ax.plot(x_ref, ref_values, ':', color='red', linewidth=1.5,
                        label=rf'$e^{{L \beta}}$' if i == 0 else None)

    if inset and len(curve_data) > 1 and not ploglog and a_exp is None and alpha is None and not fit_inset:
        _add_mixing_inset(ax, curve_data, colors)
    if fit_inset and not ploglog and len(curve_data) >= 1:
        _add_fit_inset(ax, curve_data, colors,
                       x_transform=(_xtransform if (is_vary_v is False) else (lambda x: x)),
                       alpha=alpha, xr=xr, a_for_inset=a_exp)

    # Mixing-mode overlay (direct-trajectory benchmark): scatter, no fit.
    if mixing_overlay:
        overlay_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(mixing_overlay), 2)))
        for j, fn in enumerate(mixing_overlay):
            d = load_jld2_data(fn)
            if d['mode'] != 'mixing':
                continue
            mm = np.asarray(d['mean_mixing_times'], dtype=float)
            xs = np.asarray(d.get('p_values') if d.get('vs') is None else d['vs'], dtype=float)
            xs_plot = _xtransform(xs) if (a_exp is None or ploglog) else xs ** a_exp
            mm_plot = np.log10(mm) if ploglog else mm
            L_overlay = d.get('L', '?')
            mthr = d.get('M_threshold', '?')
            mthr_str = f'{mthr:.2f}' if isinstance(mthr, (int, float)) else str(mthr)
            lbl = rf'direct, $L={L_overlay}$, $M_{{\rm th}}={mthr_str}$'
            ax.scatter(xs_plot, mm_plot, marker='x', s=70, color=overlay_colors[j],
                       zorder=5, label=lbl, linewidth=2)
            # Errorbars from SoM if present
            sd = d.get('mixing_time_stds')
            if sd is not None:
                sd = np.asarray(sd, dtype=float)
                if ploglog:
                    yerr = sd / (mm * np.log(10.0))
                    ax.errorbar(xs_plot, mm_plot, yerr=yerr, fmt='none',
                                ecolor=overlay_colors[j], capsize=3, zorder=5)
                else:
                    ax.errorbar(xs_plot, mm_plot, yerr=sd, fmt='none',
                                ecolor=overlay_colors[j], capsize=3, zorder=5)

    base_xlabel = xlabel or r'$p$'
    _xsym = base_xlabel.strip('$')  # '\epsilon' or 'p' (whatever the base symbol is)
    if (xlogsqq or xq or xp or xr) and not is_vary_v:
        if xlogsqq:
            ax.set_xlabel(rf'$(\log(1/{_xsym}))^2$')
        elif xq:
            ax.set_xlabel(rf'$1/{_xsym}$')
        elif xr:
            a_xr = 2.0 if a_exp is None else a_exp
            ax.set_xlabel(rf'$(\beta J)^{{{a_xr:g}}}$')
        else:  # xp
            ax.set_xlabel(base_xlabel)
    elif alpha is not None and not is_vary_v:
        ax.set_xlabel(rf'$(\beta J)^{{{alpha:g}}}$')
    elif a_exp is not None:
        # Wrap the base xlabel in parentheses and raise to a_exp.
        ax.set_xlabel(rf'$({base_xlabel.strip("$")})^{{{a_exp:g}}}$')
    else:
        ax.set_xlabel(base_xlabel)
    dynamics_set = set(load_jld2_data(fn).get('dynamics') for fn in filenames)
    if mixing_overlay:
        dynamics_set |= set(load_jld2_data(fn).get('dynamics') for fn in mixing_overlay)
    any_gkl = 'gkl' in dynamics_set
    any_ig = 'ising_glauber' in dynamics_set
    # Seeded runs estimate a modified (clock-truncated) τ_mem; mark it with a
    # tilde. Trigger if ANY input file used non-zero seeding — mixed-seeding
    # plots are visually misleading but at least the label flags the issue.
    _any_seeded = any(
        (load_jld2_data(fn).get('seed_droplet_size') or 0) > 0
        for fn in filenames
    )
    _tmem_sym = r't_{\sf mem}'
    if _any_seeded:
        _tmem_sym = r'\widetilde ' + _tmem_sym
    if ploglog:
        ax.set_ylabel(rf'$\log_{{10}} {_tmem_sym}$')
        ax.set_xscale('log')
    else:
        ax.set_ylabel(rf'${_tmem_sym}$')
    if not any_ig:
        ax.set_yscale('log')
    legend_title = None
    if is_vary_v:
        # --xr reparametrises the fixed temperature variable for non-GKL curves
        # as r = (βJ)², matching the per-curve relabelling above.
        if xr and not any_gkl:
            legend_title = r'$(\beta J)^2$'
        else:
            legend_title = r'$\epsilon$' if any_gkl else r'$p$'
    elif is_vary_v is not None and not is_vary_v:
        if any_ig:
            legend_title = None  # L is in each label; no shared title
        elif any_gkl:
            legend_title = r'$\eta$'
        else:
            legend_title = r'$v$'
    # Translucent white legend background (matches the erosion-vs-v style):
    # white fill at α≈0.5 so legend text is readable when overlapping curves
    # without fully obscuring what's behind it.
    leg = _legend_no_errorbars(ax, loc=_legloc(), frameon=True, fancybox=False,
                               edgecolor='none', framealpha=_legalpha(0.5),
                               title=legend_title)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0)
    if leg.get_title():
        leg.get_title().set_fontsize(13 * 1.2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    # v-sweep: force a major tick at every integer v in the visible range so
    # half-integer sweeps don't display only at half-integer ticks (matplotlib
    # otherwise places them at multiples of 0.5 for evenly-spaced 0.5-step
    # sweeps). Don't touch other modes — base_xlabel != $v$ means we're against
    # p, ε, (βJ)², etc., where integer-only ticks would look wrong.
    if is_vary_v:
        try:
            xmin, xmax = ax.get_xlim()
            lo = int(np.ceil(xmin - 1e-9))
            hi = int(np.floor(xmax + 1e-9))
            if hi >= lo:
                ax.set_xticks(list(range(lo, hi + 1)))
        except Exception:
            pass
    if not _finalize_residual_panel(ax, residual_ax):
        plt.tight_layout()
    plt.show()


def plot_diffusion_mode(filenames):
    """Plot the GKL domain-wall diffusion constant D vs the swept parameter.

    Axis labels:
      - Sweep was in p_noise (sweep_mode='p' or unspecified, p-sweep file):
          x = p_noise (= ε),         labels y = D(ε),  x = ε
      - Sweep was in τ = 1/√p (sweep_mode='tau'):
          x = 1/√(p_noise) = 1/√ε,   labels y = D(ε),  x = 1/√ε
      - Sweep was in η (vary_eta=true):
          x = η,                     labels y = D(η),  x = η

    One curve per input file (typically different L or η).
    """
    fig, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()
    colors = _user_or(cmap)(np.linspace(0, 1, max(len(filenames), 2)))

    xlabel = None
    ylabel = r'$D(\epsilon)$'
    for idx, filename in enumerate(filenames):
        data = load_jld2_data(filename)
        if data['mode'] != 'diffusion':
            print(f"Warning: {filename} is not diffusion mode, skipping...")
            continue

        D = np.asarray(data['D_values'], dtype=float)
        D_err = np.asarray(data['D_stderrs'], dtype=float)
        L = data.get('L', '?')
        sweep_mode = data.get('sweep_mode')

        if data.get('vs') is not None:
            # η-sweep
            x = np.asarray(data['vs'], dtype=float)
            xlabel = r'$\eta$'
            ylabel = r'$D(\eta)$'
            p_fixed = data.get('p_noise', data.get('p', '?'))
            p_lab = f'{p_fixed:.3f}' if isinstance(p_fixed, (int, float)) else f'{p_fixed}'
            label = rf'$p={p_lab},\, L={L}$'
        else:
            p = np.asarray(data['p_values'], dtype=float)
            if sweep_mode == 'tau':
                # Plot on the 1/√ε axis (= τ).
                x = 1.0 / np.sqrt(p)
                xlabel = r'$1/\sqrt{\epsilon}$'
            else:
                # Default / sweep_mode='p': plot against ε directly.
                x = p
                xlabel = r'$\epsilon$'
            ylabel = r'$D(\epsilon)$'
            eta = data.get('eta', '?')
            eta_lab = f'{eta:.3f}' if isinstance(eta, (int, float)) else f'{eta}'
            label = rf'$\eta={eta_lab},\, L={L}$'

        finite = np.isfinite(D) & (D > 0)
        if not np.any(finite):
            print(f"Warning: {filename} has no finite positive D values.")
            continue
        ax.errorbar(x[finite], D[finite], yerr=D_err[finite], fmt='-o',
                    color=colors[idx], markerfacecolor=colors[idx],
                    markeredgecolor='k', markersize=marker_size,
                    linewidth=linewidth, alpha=0.7, capsize=3, label=label)

    ax.set_xlabel(xlabel or r'$\epsilon$')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    _legend_no_errorbars(ax, loc=_legloc(), frameon=not True, fancybox=False,
                         edgecolor='black', framealpha=_legalpha(0.9))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def plot_history_mode(filenames, t_max=None, hide_ticks=False):
    """Plot spacetime magnetization heatmap from a single file. t_max optionally
    truncates the time axis to [0, t_max]. hide_ticks removes axis ticks and
    tick labels."""
    for filename in filenames:
        data = load_jld2_data(filename)
        if data['mode'] != 'history':
            print(f"Warning: {filename} is not history mode, skipping...")
            continue

        mag = data['magnetization_history']
        if t_max is not None:
            t_cut = min(int(t_max) + 1, mag.shape[1])
            mag = mag[:, :t_cut]
        v = data.get('v', 0)
        L = data.get('L', mag.shape[0])
        beta_val = data.get('beta', '?')

        # Shift into co-moving frame at velocity v/2
        T_steps = mag.shape[1]
        mag_shifted = np.empty_like(mag)
        for t in range(T_steps):
            shift = int(round(v * t / 2)) % L
            mag_shifted[:, t] = np.roll(mag[:, t], -shift)

        # Discrete 3-color colormap (values are -2, 0, +2).
        rdbu = plt.cm.RdBu
        m_cmap = matplotlib.colors.ListedColormap(
            [rdbu(0.15), rdbu(0.5), rdbu(0.85)]
        )
        m_norm = matplotlib.colors.BoundaryNorm([-3, -1, 1, 3], m_cmap.N)

        fig, ax = plt.subplots(figsize=(3., 6.), dpi=100)
        im = ax.imshow(mag_shifted.T, aspect='auto', origin='lower',
                       cmap=m_cmap, norm=m_norm, alpha=0.6)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$')
        mathsf_fmt = matplotlib.ticker.FuncFormatter(
            lambda val, _pos: rf'$\mathsf{{{val:g}}}$'
        )
        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.minorticks_off()
            # Exactly 5 evenly-spaced x-ticks from 0 to the system size L.
            L_int = int(L) if isinstance(L, (int, float)) else mag.shape[0]
            ax.set_xticks(np.linspace(0, L_int, 5))
            ax.set_xlim(-0.5, L_int)
            # y-ticks every 25 time units (e.g. 0, 25, 50, ..., max).
            y_top = mag.shape[1] - 1
            ax.set_yticks(list(range(0, int(y_top) + 1, 25)))
            ax.xaxis.set_major_formatter(mathsf_fmt)
            ax.yaxis.set_major_formatter(mathsf_fmt)
        # Inset colorbar in the upper-right of the plot.
        cax = ax.inset_axes([0.76, 0.69, 0.06, 0.28])
        cbar = fig.colorbar(im, cax=cax, ticks=[-2, 0, 2])
        cbar.set_label(r'$m$', labelpad=-4)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.set_major_formatter(mathsf_fmt)
        cbar.ax.minorticks_off()

        # Custom v±δ arrow overlay for the specific (L=200, v=2, β=200) file
        # the user hand-annotated. The two "V" markers identify pairs of
        # domain walls travelling at slightly different velocities relative
        # to the co-moving frame; v-δ (slow) is the red one at the trajectory
        # base, v+δ (fast) is the purple one further up. Coordinates are in
        # data units (x: site index, t: time step).
        try:
            # int()/float() coerce both Python and numpy scalars; the bare
            # isinstance(L, (int, float)) check from earlier missed numpy
            # scalars (e.g. np.int64) returned by h5py.
            L_int = int(L)
            v_val = float(v)
            beta_val_f = float(beta_val)
            if (L_int == 200 and abs(v_val - 2.0) < 1e-6
                    and abs(beta_val_f - 200.0) < 1e-6):
                red_hex = '#FF0A29'
                purple_hex = '#DC35FF'
                arrow_kw = dict(arrowstyle='-|>', mutation_scale=15, lw=3)
                label_fs = 14 * 1.5
                # Bold math: \boldsymbol works via the rcParams text.usetex=True
                # path at the top of this script; matplotlib's mathtext doesn't
                # honour weight='bold' for math italic + Greek without it.
                # Falls back to plain italic if LaTeX isn't available (which
                # would just render as non-bold, not error out).
                # Lower V — red, v-δ pair. Vertex at (75, 0).
                ax.annotate('', xy=(56, 30), xytext=(75, 0),
                            arrowprops=dict(color=red_hex, **arrow_kw))
                ax.annotate('', xy=(91, 30), xytext=(75, 0),
                            arrowprops=dict(color=red_hex, **arrow_kw))
                # Label horizontally offset to x=65 (left of the vertex, above
                # the left arm's tip) and just above the arrow tips.
                ax.text(65, 33, r'$\boldsymbol{v - \delta}$', color=red_hex,
                        fontsize=label_fs, ha='center', va='bottom')
                # Upper V — purple, v+δ pair. Vertex at (102, 52).
                ax.annotate('', xy=(62, 79), xytext=(102, 52),
                            arrowprops=dict(color=purple_hex, **arrow_kw))
                ax.annotate('', xy=(140, 79), xytext=(102, 52),
                            arrowprops=dict(color=purple_hex, **arrow_kw))
                ax.text(102, 82, r'$\boldsymbol{v + \delta}$', color=purple_hex,
                        fontsize=label_fs, ha='center', va='bottom')
        except Exception:
            pass

        plt.tight_layout()
        plt.show()


def plot_phase_diagram_mode(filenames, raw=False, pmin=None, pmax=None,
                            vmin=None, vmax=None):
    """Plot v* vs p (one curve per file). With raw=True, plot the inner-sweep
    curves (observable vs v) for each p in the first file. pmin/pmax/vmin/vmax
    override the axis limits when provided."""
    if raw:
        data = load_jld2_data(filenames[0])
        if data['mode'] != 'phase_diagram':
            print(f"Warning: {filenames[0]} is not phase_diagram mode")
            return
        p_values = data['p_values']
        v_values = data['v_values']
        y_matrix = data['y_matrix']
        v_onset = data['v_onset_values']
        observable = data.get('observable', 'mixing')

        _, ax = plt.subplots(figsize=(5., 4.), dpi=100)
        colors = _user_or(cmap)(np.linspace(0, 1, max(len(p_values), 2)))
        ylab = r'$t_{\sf mem}$' if observable == 'mixing' else r'$\ell_c$'
        for i, p in enumerate(p_values):
            ax.plot(v_values, y_matrix[i, :], '-o', color=colors[i],
                    markerfacecolor=colors[i], markeredgecolor='k',
                    markersize=marker_size, linewidth=linewidth, alpha=0.7,
                    label=rf'${p:.2f}$')
            if np.isfinite(v_onset[i]):
                ax.axvline(v_onset[i], color=colors[i], linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(r'$v$')
        ax.set_ylabel(ylab)
        if observable == 'mixing':
            ax.set_yscale('log')
        leg = _legend_no_errorbars(ax, loc=_legloc(), frameon=False, title=r'$p$')
        if leg.get_title():
            leg.get_title().set_fontsize(13 * 1.2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.show()
        return

    _, ax = plt.subplots(figsize=(4., 4.), dpi=100)
    ax.minorticks_off()

    curves = []  # collected (p_filt, v_filt) for post-loop shading
    v_data_max = -np.inf
    p_data_min = np.inf
    p_data_max = -np.inf
    for filename in filenames:
        data = load_jld2_data(filename)
        if data['mode'] != 'phase_diagram':
            print(f"Warning: {filename} is not phase_diagram mode, skipping...")
            continue
        p_values = data['p_values']
        v_onset = data['v_onset_values']
        v_values = data.get('v_values')
        mask = np.isfinite(v_onset)
        p_filt = p_values[mask]
        v_filt = v_onset[mask]
        curves.append((p_filt, v_filt))
        if v_values is not None:
            v_data_max = max(v_data_max, float(np.max(v_values)))
        if len(p_values) > 0:
            p_data_min = min(p_data_min, float(np.min(p_values)))
            p_data_max = max(p_data_max, float(np.max(p_values)))

    # Shade the region above each curve (larger v than v*(p)), then draw
    # scatter points on top.
    if vmax is not None:
        y_top = vmax
    elif np.isfinite(v_data_max):
        y_top = v_data_max
    else:
        y_top = ax.get_ylim()[1]
    for p_filt, v_filt in curves:
        if len(p_filt) >= 2:
            ax.fill_between(p_filt, v_filt, y_top, color='lightblue',
                            alpha=0.55, linewidth=0)
    dot_color = '#ff8a8a'  # light red
    for p_filt, v_filt in curves:
        ax.plot(p_filt, v_filt, 'o', color=dot_color,
                markerfacecolor=dot_color, markeredgecolor='k',
                markersize=0.75 * marker_size, zorder=3)

    if pmin is not None or pmax is not None:
        ax.set_xlim(left=pmin, right=pmax)
    if vmin is not None or vmax is not None:
        ax.set_ylim(bottom=vmin, top=(vmax if vmax is not None else y_top))
    else:
        ax.set_ylim(top=y_top)

    # Annotations — positioned relative to the final axis bounds (after the
    # user's pmin/pmax/vmin/vmax have been applied), in axes-fraction coords.
    ax.text(0.66, 0.78, r'$t_{\sf mem} \sim e^{v (\beta J)^2 \ln v}$',
            transform=ax.transAxes, ha='center', va='center', fontsize=13 * 0.9)
    ax.text(0.07, 0.1, r'$t_{\sf mem} \sim e^{\beta J}$',
            transform=ax.transAxes, ha='left', va='center', fontsize=13)

    ax.set_xlabel(r'$e^{\beta J}$')
    ax.set_ylabel(r'$v$')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot data from sliding_ising_chain.jl')
    parser.add_argument('files', nargs='+', help='JLD2 files to plot')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'erosion_test', 'energy', 'teff', 'mixing', 'ffs', 'history', 'phase_diagram', 'diffusion'],
                        help='Mode to plot (default: auto-detect from first file)')
    parser.add_argument('--raw', action='store_true',
                        help='Plot P_shrink vs l instead of l/lc (erosion_test mode)')
    parser.add_argument('--small_stats', action='store_true',
                        help='Plot 1-P_shrink vs (l-lc)^2 for l<=lc on log scale (erosion_test mode)')
    parser.add_argument('--logy', action='store_true',
                        help='Use log y-scale on the 1-P_shrink vs l/lc plot (erosion_test mode)')
    parser.add_argument('--heat', action='store_true',
                        help='Plot heat flow instead of energy (energy mode)')
    parser.add_argument('--inset', action='store_true',
                        help='Add inset showing exponential fit coefficients (mixing/ffs modes, requires multiple files)')
    parser.add_argument('--fit_inset', action='store_true',
                        help='Add an inset showing the linear-fit slope vs the per-file fixed parameter. '
                             'Erosion mode: slope of lc vs e^(beta J). '
                             'ffs/mixing modes: fit log10(t_mem) = s*x + b on the last half of each curve '
                             '(in whatever x-axis is being plotted, including --alpha transforms), draw the fits '
                             'as dashed black lines, and the inset plots s vs v (for p-sweeps) or vs '
                             'e^(beta J) / (beta J)^alpha (for v-sweeps).')
    parser.add_argument('--pmin', type=float, default=None,
                        help='Lower x-limit on the phase_diagram plot')
    parser.add_argument('--pmax', type=float, default=None,
                        help='Upper x-limit on the phase_diagram plot')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Lower y-limit on the phase_diagram plot')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Upper y-limit on the phase_diagram plot')
    parser.add_argument('--t_max', type=int, default=None,
                        help='Truncate history-mode plot to t in [0, t_max]')
    parser.add_argument('--hide_ticks', action='store_true',
                        help='Hide axis ticks and tick labels (history mode)')
    parser.add_argument('--ploglog', action='store_true',
                        help='ffs mode: log-log plot of log10(tau) vs p, with '
                             'per-file power-law fit log tau ~ p^a')
    parser.add_argument('--anchor_normalize', action='store_true',
                        help='ffs overlay mode: rescale each FFS curve so '
                             'its value at the smallest x matches the direct '
                             '(mixing) value at the same x. Tests whether '
                             'FFS and direct differ by a constant factor.')
    parser.add_argument('--a', type=float, default=None,
                        help='ffs mode: plot tau against x**a instead of x '
                             '(useful for testing log tau ~ p^a)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='ffs/mixing mode: when sweeping p, plot tau '
                             'against (beta J)**alpha = (ln p)**alpha instead '
                             'of p. A straight line on log-y means '
                             'log tau ~ (beta J)^alpha.')
    parser.add_argument('--xlogsqq', action='store_true',
                        help='ffs mode (GKL, p-sweep): plot t_mem against '
                             '(log(1/p))^2.')
    parser.add_argument('--xq', action='store_true',
                        help='ffs mode (GKL, p-sweep): plot t_mem against 1/p.')
    parser.add_argument('--xp', action='store_true',
                        help='ffs mode (GKL, p-sweep): plot t_mem against p '
                             '(explicit default; useful to override an implicit '
                             'transform).')
    parser.add_argument('--xr', action='store_true',
                        help='ffs mode (sliding Ising, p-sweep): plot t_mem '
                             'against (beta J)^2 = (log(1/p))^2 (same transform '
                             'as --xlogsqq but labeled with the sliding-Ising '
                             'temperature variable).')
    parser.add_argument('--xscale', choices=['linear', 'log'], default=None,
                        help='Force the x-axis scale to linear or log on every '
                             'figure produced this run, overriding all other defaults.')
    parser.add_argument('--yscale', choices=['linear', 'log'], default=None,
                        help='Force the y-axis scale to linear or log on every '
                             'figure produced this run, overriding all other defaults.')
    parser.add_argument('--cmap', type=str, default=None,
                        help='Override the per-file colormap with the named '
                             'matplotlib colormap (e.g. "viridis", "magma", '
                             '"Greens"). Replaces the default coolwarm_r / '
                             'rainbow / Oranges / Blues choices for every '
                             'plot mode that colours multiple files.')
    parser.add_argument('--legloc', type=str, default=None,
                        help='Force the legend location on every figure '
                             '(e.g. "upper left", "lower center", '
                             '"center right"). Overrides the default '
                             '"best" / "lower right" choices in every '
                             'plot mode.')
    parser.add_argument('--legend_alpha', type=float, default=None,
                        help='Force the legend background opacity '
                             '(framealpha) on every figure. 0 = fully '
                             'transparent, 1 = fully opaque. Overrides '
                             "each plot mode's hard-coded default "
                             '(typically 0.5 or 0.9).')
    parser.add_argument('--add_lines', action='store_true',
                        help='Draw connecting lines between plotted data '
                             'points (ffs / mixing / erosion modes, which '
                             'are marker-only by default). Energy mode '
                             'already shows connecting lines.')
    parser.add_argument('--fitrange', type=float, default=None,
                        help='ffs / mixing / energy mode: per-file linear fit '
                             'of ln(y) to the plotted x-coordinate over the '
                             'upper (1 - fitrange) portion of x (e.g. '
                             '--fitrange=0.2 fits on x >= x_min + 0.2*(x_max '
                             '- x_min), i.e. the last 80 percent). Honours '
                             '--a / --alpha / --xr / --xq / --xp / --xlogsqq '
                             '(fitted x is whatever is on-screen). Prints '
                             'slope, intercept, stderrs, R^2, and n per file '
                             'and overlays each fit as a black dashed line. '
                             'Ignored with --ploglog.')
    parser.add_argument('--residuals', action='store_true',
                        help='Companion to --fitrange. Swap the figure for a '
                             'two-panel gridspec (3:1) with the data + fit '
                             'lines on top and per-file residuals ln(y) - '
                             '(slope*x + intercept) on the bottom panel. '
                             'Residuals are shown for ALL plotted points '
                             '(including those outside the fit window) so '
                             'systematic curvature outside the fit window is '
                             'visible. No-op without --fitrange.')

    args = parser.parse_args()

    if args.residuals and args.fitrange is None:
        print("Warning: --residuals is a no-op without --fitrange; ignoring.")
        args.residuals = False

    # Apply the --cmap override (if any) before any plot function runs. Sites
    # that pick per-file colors consult _USER_CMAP via _user_or().
    if args.cmap is not None:
        global _USER_CMAP
        try:
            _USER_CMAP = plt.get_cmap(args.cmap)
        except (ValueError, KeyError) as e:
            print(f"Warning: unknown colormap '{args.cmap}'; ignoring --cmap. ({e})")

    # Apply the --legloc override (if any). Every ax.legend() site below
    # consults _USER_LEGLOC via _legloc().
    if args.legloc is not None:
        global _USER_LEGLOC
        _USER_LEGLOC = args.legloc

    # Apply the --legend_alpha override (if any). Every legend site below
    # consults _USER_LEGEND_ALPHA via _legalpha().
    if args.legend_alpha is not None:
        global _USER_LEGEND_ALPHA
        _USER_LEGEND_ALPHA = args.legend_alpha

    # Apply the --add_lines toggle. Each marker-only series in ffs / mixing /
    # erosion consults _USER_ADD_LINES via _line_fmt / _line_lw.
    if args.add_lines:
        global _USER_ADD_LINES
        _USER_ADD_LINES = True

    # Global axis-scale override: monkey-patch plt.show so that every figure
    # produced by any plot function has its scales forced to the requested
    # value just before display. Applied to all axes in the figure (including
    # inset axes); failures on individual axes are silently ignored (e.g. a
    # data range that includes 0 or negative values when forcing log).
    if args.xscale is not None or args.yscale is not None:
        _orig_show = plt.show
        def _show(*a, **kw):
            for fignum in plt.get_fignums():
                for ax in plt.figure(fignum).get_axes():
                    if args.xscale is not None:
                        try:
                            ax.set_xscale(args.xscale)
                        except Exception:
                            pass
                    if args.yscale is not None:
                        try:
                            ax.set_yscale(args.yscale)
                        except Exception:
                            pass
            return _orig_show(*a, **kw)
        plt.show = _show

    # Detect mode from first file if auto
    if args.mode == 'auto':
        first_data = load_jld2_data(args.files[0])
        mode = first_data.get('mode', 'unknown')
    else:
        mode = args.mode

    # Auto-overlay: when the file list contains a mix of FFS and mixing files
    # (regardless of `mode`), route FFS files to plot_ffs_mode and pass mixing
    # files as a scatter-overlay set. Used for the ising.jl benchmark.
    per_file_modes = [load_jld2_data(fn).get('mode') for fn in args.files]
    if 'ffs' in per_file_modes and 'mixing' in per_file_modes:
        ffs_files = [fn for fn, m in zip(args.files, per_file_modes) if m == 'ffs']
        mixing_files = [fn for fn, m in zip(args.files, per_file_modes) if m == 'mixing']
        plot_ffs_mode(ffs_files, inset=args.inset, ploglog=args.ploglog,
                      a_exp=args.a, alpha=args.alpha, fit_inset=args.fit_inset,
                      mixing_overlay=mixing_files,
                      anchor_normalize=args.anchor_normalize,
                      xlogsqq=args.xlogsqq, xq=args.xq, xp=args.xp, xr=args.xr,
                      fitrange=args.fitrange, residuals=args.residuals)
        return

    if mode == 'erosion_test':
        plot_erosion_mode(args.files, raw=args.raw, small_stats=args.small_stats,
                          fit_inset=args.fit_inset, logy=args.logy)
    elif mode == 'energy':
        plot_energy_mode(args.files, heat=args.heat, fitrange=args.fitrange,
                         residuals=args.residuals)
    elif mode == 'teff':
        plot_teff_mode(args.files)
    elif mode == 'mixing':
        plot_mixing_mode(args.files, inset=args.inset, alpha=args.alpha,
                         fit_inset=args.fit_inset, fitrange=args.fitrange,
                         residuals=args.residuals)
    elif mode == 'ffs':
        plot_ffs_mode(args.files, inset=args.inset, ploglog=args.ploglog,
                      a_exp=args.a, alpha=args.alpha, fit_inset=args.fit_inset,
                      xlogsqq=args.xlogsqq, xq=args.xq, xp=args.xp, xr=args.xr,
                      fitrange=args.fitrange, residuals=args.residuals)
    elif mode == 'history':
        plot_history_mode(args.files, t_max=args.t_max, hide_ticks=args.hide_ticks)
    elif mode == 'phase_diagram':
        plot_phase_diagram_mode(args.files, raw=args.raw,
                                pmin=args.pmin, pmax=args.pmax,
                                vmin=args.vmin, vmax=args.vmax)
    elif mode == 'diffusion':
        plot_diffusion_mode(args.files)
    else:
        print(f"Unknown mode '{mode}'. Supported: erosion_test, energy, mixing, history, phase_diagram, diffusion")


if __name__ == "__main__":
    main()
