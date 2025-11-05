"""
Functions to forecast ice cap thickness in the future based on climate and thickness measurements.

Author: Erik Schytt Mannerfelt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

import itertools
import tqdm

def juvfonne_thickness_series(
    T: pd.Series,
    H_2011: float,
    H_2025: float,
    tau: float
) -> tuple[pd.Series, tuple[float, float]]:
    """
    Forward-only thickness model for Juvfonne based on a first-order lag:
        H_{y+1} = (1 - 1/tau) * H_y + (1/tau) * (a + b*T_y)

    Assumptions
    -----------
    - Near-equilibrium at 2011: H_2011 = a + b*T_2011.
    - H_y represents *end-of-year* thickness.
    - Uses T[y] to step H[y] -> H[y+1].
    - No backcasting; output starts at 2011.

    Parameters
    ----------
    T : pd.Series
        JJA temperatures indexed by *integer years* (e.g., 2001..2100).
        Must contain a contiguous run of years from 2011..Ymax *and*
        at least 2011..2024 for calibration (since we step 2011->2025).
    H_2011 : float
        Observed thickness (m) at end of 2011.
    H_2025 : float
        Observed thickness (m) at end of 2025.
    tau : float
        Response time (years). Must be > 0.

    Returns
    -------
    A series of the thickness timeseries, and a tuple of slope/intercept parameters for the eq. thickness function.

    Raises
    ------
    ValueError, TypeError
        If inputs are inconsistent (missing years, non-integer index, tau<=0, etc.).
    """
    # --- validation ---
    if not isinstance(T, pd.Series):
        raise TypeError("T must be a pandas Series.")
    if not np.issubdtype(T.index.dtype, np.integer):
        raise TypeError("T.index must be integer years (e.g., 2001, 2002, ...).")
    if tau <= 0.0:
        raise ValueError("tau must be > 0.")

    # Ensure forward range (2011..Ymax) is contiguous
    years = np.asarray(T.index, dtype=int)
    y_min, y_max = int(years.min()), int(years.max())
    if 2011 < y_min:
        raise ValueError(f"T must include 2011; found min year {y_min}.")
    if any(y not in T.index for y in range(2011, y_max + 1)):
        missing = [y for y in range(2011, y_max + 1) if y not in T.index]
        raise ValueError(f"T must be contiguous from 2011..{y_max}. Missing: {missing}")

    # Ensure calibration window exists (2011..2024 temperatures step us to 2025)
    calib_years = range(2011, 2025)
    missing_calib = [y for y in calib_years if y not in T.index]
    if missing_calib:
        raise ValueError(f"T is missing required calibration years: {missing_calib}")

    lam = 1.0 - 1.0 / tau
    T_2011 = float(T.loc[2011])

    # --- Build linear mapping H_2025 = A*a + B*b + C, assuming equilibrium at end-2011 ---
    # Start sensitivities at end of 2011: H_2011 = a + b*T_2011  =>  A0=1, B0=T_2011, C0=0
    A, B, C = 1.0, T_2011, 0.0
    for y in range(2011, 2025):  # steps 2011->2012, ..., 2024->2025
        T_k = float(T.loc[y])
        A = lam * A + (1.0 / tau) * 1.0
        B = lam * B + (1.0 / tau) * T_k
        C = lam * C  # stays 0 with equilibrium-at-start assumption

    # Solve the 2x2 system:
    # [1, T_2011] [a] = [H_2011]
    # [A,    B  ] [b]   [H_2025 - C]
    M = np.array([[1.0, T_2011],
                  [A,   B     ]], dtype=float)
    rhs = np.array([H_2011, H_2025 - C], dtype=float)
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    a, b = float(sol[0]), float(sol[1])

    # --- Forward integrate thickness from 2011 to Ymax ---
    H = pd.Series(index=pd.Index(range(2011, y_max + 1), dtype=int), dtype=float)
    H.loc[2011] = a + b * T_2011  # equilibrium at 2011 by construction

    for y in range(2011, y_max):
        H_eq = a + b * float(T.loc[y])
        H.loc[y + 1] = lam * H.loc[y] + (1.0 / tau) * H_eq

    H.name = "thickness_m"
    return H, (a, b)


def main(n_steps_per_member: int = 6, thickness_uncertainty: float = 1.1):
    import climate
    temp_data = climate.main()

    thickness_vals = pd.Series({2011: 5.6, 2025: 3.5})

    diff_per_year = (thickness_vals.diff() / thickness_vals.index.diff()).iloc[-1]

    min_tau = thickness_vals[2025] / np.abs(diff_per_year)
    max_tau = thickness_vals[2011] / np.abs(diff_per_year)

    print(f"Tau ranges between {min_tau:.1f} and {max_tau:.1f}")

    temp_data = temp_data.groupby(level=["model", "ssp"]).transform(lambda s: s.rolling(window=3, min_periods=1).mean())

    taus = np.linspace(min_tau, max_tau, n_steps_per_member)
    h2011s = thickness_vals[2011] + np.linspace(-thickness_uncertainty, thickness_uncertainty, n_steps_per_member)
    h2025s = thickness_vals[2025] + np.linspace(-thickness_uncertainty, thickness_uncertainty, n_steps_per_member)

    combos = list(itertools.product(taus, h2011s, h2025s))

    temp_ensemble = temp_data.groupby(level=["ssp", "year"]).quantile([0.05, 0.25, 0.75, 0.95])

    coords = {k: temp_data.index.get_level_values(k).unique() for k in temp_data.index.names} | {"i": np.arange(len(combos))}

    arrs = []
    for i, (tau, h2011, h2025) in tqdm.tqdm(enumerate(combos), total=len(combos)):
        for j, ssp_str in enumerate(temp_data.index.get_level_values("ssp").unique()):

            for model in temp_data.index.get_level_values("model").unique():
                h_series, (a, b) = juvfonne_thickness_series(temp_data.loc[(model, ssp_str, slice(None))], H_2011=h2011, H_2025=h2025, tau=tau)
                arr = xr.DataArray(h_series.clip(lower=0).values, name="h", coords=(("year", h_series.index),)).to_dataset()

                arr["a"] = a
                arr["b"] = b

                arrs.append(arr.expand_dims({"model": [model], "ssp": [ssp_str], "i": [i]}))

    out = xr.combine_by_coords(arrs)


    out["q"] = out["h"].quantile([0.05, 0.25, 0.5, 0.75, 0.95], dim=["model", "i"])

    quantiles = [0.25, 0.5, 0.75]
    q_a = out["a"].quantile(quantiles)
    q_b = out["b"].quantile(quantiles)

    avg_study_temp = temp_data.loc[("noresm2_mm", "ssp1_2_6", slice(None))].loc[2011:2025].mean()
    print(f"Avg temp 2011-2025: {avg_study_temp:.1f} deg C")
    for quantile in quantiles:
        
        a = q_a.sel(quantile=quantile).item()
        b = q_b.sel(quantile=quantile).item()

        t_zero = -a / b

        print(f"{quantile * 100}th percentile:")
        print(f"\tH_eq = {b:.1f} * T + {a:.1f}")
        print(f"\tH_eq = 0 if T ~ {t_zero:.1f}")
        print(f"\tH_eq for 2011-2025 temp: {b * avg_study_temp + a:.1f}")

   
    fig = plt.figure(figsize=(8.3, 4.5))
    axes = fig.subplots(2, 3, sharex=True, sharey="row")
    ssps = temp_data.index.get_level_values("ssp").unique()
    era5 = temp_ensemble.loc[(ssps[0], slice(None), 0.25)].loc[:2025]

    for i, ssp_str in enumerate(sorted(ssps)):
        ssp_fmt = "{} {}.{}".format(*str(ssp_str).upper().split("_"))

        ssp_temp = temp_ensemble.loc[(ssp_str, slice(None), slice(None))]

        axes[0, i].fill_between(ssp_temp.index.get_level_values("year").unique(), temp_ensemble.loc[(ssp_str, slice(None), 0.05)], temp_ensemble.loc[(ssp_str, slice(None), 0.95)], label="5-95th percentile", alpha=0.4, color="pink") 
        axes[0, i].fill_between(ssp_temp.index.get_level_values("year").unique(), temp_ensemble.loc[(ssp_str, slice(None), 0.25)], temp_ensemble.loc[(ssp_str, slice(None), 0.75)], label="25-75th percentile", alpha=0.5, color="red") 

        axes[0, i].plot(era5, label="Reanalysis", color="black")

        axes[0, i].set_title(ssp_fmt)

        axes[1, i].fill_between(out["year"].values, out["q"].sel(quantile=0.05, ssp=ssp_str), out["q"].sel(quantile=0.95, ssp=ssp_str), color="purple", alpha=0.2, label="5-95th percentile")
        axes[1, i].fill_between(out["year"].values, out["q"].sel(quantile=0.25, ssp=ssp_str), out["q"].sel(quantile=0.75, ssp=ssp_str), color="blue", alpha=0.5, label="25-75th percentile")
        axes[1, i].scatter(thickness_vals.index, thickness_vals, label="Measurements", color="black", marker="x")

    axes[1, 0].set_ylabel("Mean thickness (m)")
    axes[0, 0].set_ylabel("Air temperature (°C)")
    axes[1, 1].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    axes[1, 1].set_xlabel("Year")
    axes[0, 1].set_ylim(0, 8)
    plt.xlim(2010, 2070)
    plt.tight_layout()
    Path("figures/").mkdir(exist_ok=True)
    plt.savefig("figures/thickness_forecast.jpg", dpi=400)
    plt.show()

    
    

if __name__ == "__main__":
    main()
    
