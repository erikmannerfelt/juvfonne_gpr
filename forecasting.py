import numpy as np
import pandas as pd
from pathlib import Path

import itertools
import tqdm

def juvfonne_thickness_series(
    T: pd.Series,
    H_2011: float,
    H_2025: float,
    tau: float
) -> pd.Series:
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
        (If you want to use 10-yr means, compute them before calling.)
    H_2011 : float
        Observed thickness (m) at end of 2011.
    H_2025 : float
        Observed thickness (m) at end of 2025.
    tau : float
        Response time (years). Must be > 0.

    Returns
    -------
    pd.Series
        Modeled thickness (m) for years 2011..max(T.index), inclusive.

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
    return H


def main(min_tau=10, max_tau=20, n_steps_per_member: int = 5, thickness_uncertainty: float = 1.):
    import matplotlib.pyplot as plt

    import xarray as xr
    import climate
    # temp_data = climate.main(models=["noresm2_mm", "cesm2"])
    temp_data = climate.main()

    # temp_data.to_csv("temp_data.csv")
    # temp_data = pd.read_csv("temp_data.csv", index_col=0)

    # temp_data = temp_data[sorted(temp_data.columns)]

    thickness_vals = pd.Series({2011: 5.6, 2025: 3.4})

    # temp_smoothed = temp_data.rolling(window=5, min_periods=1, center=True).mean()
    # # temp_data.loc[2011] = temp_data.loc[2001:2011].mean(axis="rows")
    # # temp_data.loc[2024] = temp_data.loc[2014:2024].mean(axis="rows")
    # temp_data.loc[:2025] = temp_smoothed.loc[:2025]
    # temp_data = temp_smoothed
    temp_data = temp_data.groupby(level=["model", "ssp"]).transform(lambda s: s.rolling(window=5, min_periods=1).mean())

    taus = np.linspace(min_tau, max_tau, n_steps_per_member)
    h2011s = thickness_vals[2011] + np.linspace(-thickness_uncertainty, thickness_uncertainty, n_steps_per_member)
    h2025s = thickness_vals[2025] + np.linspace(-thickness_uncertainty, thickness_uncertainty, n_steps_per_member)

    combos = list(itertools.product(taus, h2011s, h2025s))

    temp_ensemble = temp_data.groupby(level=["ssp", "year"]).quantile([0.05, 0.25, 0.75, 0.95])

    # colors = dict(zip(sorted(temp_data.columns), ["blue", "green", "red"]))
    #
    coords = {k: temp_data.index.get_level_values(k).unique() for k in temp_data.index.names} | {"i": np.arange(len(combos))}
    # out = xr.Dataset(coords=)

    arrs = []
    for i, (tau, h2011, h2025) in tqdm.tqdm(enumerate(combos), total=len(combos)):
        for j, ssp_str in enumerate(temp_data.index.get_level_values("ssp").unique()):

            for model in temp_data.index.get_level_values("model").unique():
                h_series = juvfonne_thickness_series(temp_data.loc[(model, ssp_str, slice(None))], H_2011=h2011, H_2025=h2025, tau=tau)
                # h_series = juvfonne_thickness_series_constrained(temp_data[ssp_str], H_2011=thickness_vals[2011], H_2025=thickness_vals[2025], tau=5)


                arr = xr.DataArray(h_series.clip(lower=0).values, name="h", coords=(("year", h_series.index),)).expand_dims({"model": [model], "ssp": [ssp_str], "i": [i]})

                arrs.append(arr)

    out = xr.combine_by_coords(arrs)


    out["q"] = out["h"].quantile([0.05, 0.25, 0.5, 0.75, 0.95], dim=["model", "i"])
    # print(out)
    # return
    # out["median"] = out["h"].median(["model", "i"])


    
    fig = plt.figure(figsize=(8, 6))
    axes = fig.subplots(2, 3, sharex=True, sharey="row")
    ssps = temp_data.index.get_level_values("ssp").unique()
    era5 = temp_ensemble.loc[(ssps[0], slice(None), 0.25)].loc[:2025]

    for i, ssp_str in enumerate(sorted(ssps)):
        # ssp = temp_data.index.get_level_values("ssp").unique()[i]
        ssp_fmt = "{} {}.{}".format(*str(ssp_str).upper().split("_"))

        ssp_temp = temp_ensemble.loc[(ssp_str, slice(None), slice(None))]

        axes[0, i].fill_between(ssp_temp.index.get_level_values("year").unique(), temp_ensemble.loc[(ssp_str, slice(None), 0.05)], temp_ensemble.loc[(ssp_str, slice(None), 0.95)], label="5-95th percentile", alpha=0.4, color="pink") 
        axes[0, i].fill_between(ssp_temp.index.get_level_values("year").unique(), temp_ensemble.loc[(ssp_str, slice(None), 0.25)], temp_ensemble.loc[(ssp_str, slice(None), 0.75)], label="25-75th percentile", alpha=0.5, color="red") 

        axes[0, i].plot(era5, label="ERA5")

        axes[0, i].set_title(ssp_fmt)

        axes[1, i].fill_between(out["year"].values, out["q"].sel(quantile=0.05, ssp=ssp_str), out["q"].sel(quantile=0.95, ssp=ssp_str), color="purple", alpha=0.2, label="5-95th percentile")
        axes[1, i].fill_between(out["year"].values, out["q"].sel(quantile=0.25, ssp=ssp_str), out["q"].sel(quantile=0.75, ssp=ssp_str), color="blue", alpha=0.5, label="25-75th percentile")

        # out["median"].sel(ssp=ssp_str).plot(ax=axes[i])
    
        axes[1, i].scatter(thickness_vals.index, thickness_vals, label="Measurements", color="black", marker="x")

    axes[1, 0].set_ylabel("Mean thickness (m)")
    axes[0, 0].set_ylabel("1900-m air temperature (°C)")
    axes[1, 1].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    axes[1, 1].set_xlabel("Year")
    axes[0, 1].set_ylim(0, 8)
    plt.xlim(2010, 2070)
    plt.tight_layout()
    Path("figures/").mkdir(exist_ok=True)
    plt.savefig("figures/thickness_forecast.jpg", dpi=400)
    # plt.legend()
    plt.show()

    
    

if __name__ == "__main__":
    main()
    
    # # Example (replace with your preprocessed JJA series):
    # years = np.arange(2001, 2101)
    # T = pd.Series(index=years, data=np.linspace(5.0, 9.0, len(years)))  # placeholder temps

    # H_series = juvfonne_thickness_series(
    #     T=T,
    #     H_2011=16.0,
    #     H_2025=8.0,
    #     tau=20.0
    # )
    # print(H_series.loc[2011:2026])

