import zipfile
from pathlib import Path
from typing import Literal
import math
import shutil
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd


def get_area(product: Literal["cmip6", "era5-land", "era5"], lat: float = 61.6769475, lon: float = 8.3489006) -> list[float]:
    if product == "cmip6":
        res = 1.
    elif product == "era5-land":
        res = 0.25
    elif product == "era5":
        res = 0.25
    else:
        raise NotImplementedError()

    lat_min = math.floor(lat / res) * res
    lon_min = math.floor(lon / res) * res
    return [lat_min, lon_min, lat_min + res, lon_min + res]


    
        

def get_cds_data(out_path: Path, dataset: str, request: dict[str, object]) -> None:
    import cdsapi

    temp_path = Path("download.zip")
    client =cdsapi.Client()
    client.retrieve(
        dataset,
        request=request,
    ).download(temp_path)

    try:
        with zipfile.ZipFile(temp_path) as zip_file:
            for file_info in zip_file.filelist:
                if file_info.filename.endswith(".nc"):
                    with open(out_path, "wb") as outfile:
                        outfile.write(zip_file.read(file_info))
                    break
    except zipfile.BadZipFile:
        shutil.move(temp_path, out_path)
        

def get_cmip(
    experiment: Literal["ssp1_2_6", "ssp5_8_5", "ssp2_4_5"] | str,
    variable: str = "air_temperature",
    pressure_levels: list[str] = ["700", "850"],
    months: list[str] = ["06", "07", "08"],
    year_start: int = 2015,
    year_end: int = 2100,
    model: str = "noresm2_mm",
    area: list[float] | None = None, 
):
    data_path = Path(f"data/cmip6/{model}-{year_start}_{year_end}-{experiment}-{variable}.nc")

    if data_path.is_file():
        return data_path

    if area is None:
        area = get_area(product="cmip6")


    data_path.parent.mkdir(exist_ok=True, parents=True)

    request={
        "experiment": experiment,
        "temporal_resolution": "monthly",
        "variable": variable,
        "level": pressure_levels,
        "model": model,
        "month": months,
        "year": list(map(lambda y: str(y), range(year_start, year_end + 1))),
        "data_format": "netcdf",
        "area": area,
    }

    get_cds_data(data_path, "projections-cmip6", request)

    return data_path

def get_era5(
    variable: str = "temperature",
    pressure_levels: list[str] = ["775", "800"],
    months: list[str] = ["06", "07", "08"],
    year_start: int = 1995,
    year_end: int = 2025,
    area: list[float] | None = None, 
    ):

    data_path = Path(f"data/era5-{year_start}_{year_end}.nc")

    if data_path.is_file():
        return data_path

    data_path.parent.mkdir(exist_ok=True, parents=True)

    if area is None:
        area = get_area(product="era5")
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [variable],
        "pressure_level": pressure_levels,
        "month": months,
        "year": list(map(lambda y: str(y), range(year_start, year_end + 1))),
        "time": ["00:00"],
        "data_format": "netcdf",
        "area": area,
    }

    get_cds_data(data_path, "reanalysis-era5-pressure-levels-monthly-means", request=request)

    return data_path

def process_climate_data(data: xr.Dataset, variable: str, plev_col: str, pressure_level: float, lat_col: str, lon_col: str) -> pd.Series:

    data = data.isel({lat_col: 0, lon_col: 0})

    plev0 = data[plev_col].isel({plev_col: 0}).item()
    plev1 = data[plev_col].isel({plev_col: 1}).item()

    if data[plev_col].shape[0] != 2 or not ((plev0 < pressure_level < plev1) or (plev1 < pressure_level < plev0)):
        raise ValueError(f"Incompatible pressure levels (must be just 2) and contain {pressure_level}: {data[plev_col].values}")
    w0 = (plev0 - pressure_level) / (plev0 - plev1)

    data["weight"] = plev_col, [w0, 1 - w0]

    data = data[variable].weighted(data["weight"]).mean(plev_col).to_pandas() - 273.15
    data.index = data.index.year
    data = data.groupby(data.index).mean()

    return data


def main(altitude: float = 1900., models=["noresm2_mm", "cesm2", "ec-earth3", "mpi-esm1-2-lr", "ukesm1-0-ll"]):

    pressure_level = (1013 - (1900 * 0.12))

    filepath = get_era5()

    with xr.open_dataset(filepath) as data:
        era5 = process_climate_data(data, "t", plev_col = "pressure_level", pressure_level=pressure_level, lat_col="latitude", lon_col="longitude")

    ssps = ["ssp1_2_6", "ssp5_8_5", "ssp2_4_5"]

    ssp_temps = {}
    for model in models:
        # model_ssp = {}
        # ssp_temps[model] = model_ssp
        for ssp_str in ssps:
            filepath = get_cmip(ssp_str, model=model)

            with xr.open_dataset(filepath) as data:
                ssp = process_climate_data(data=data, variable="ta", plev_col="plev", pressure_level=pressure_level * 100, lat_col="lat", lon_col="lon")

            diff = (era5.loc[ssp.index.min():] - ssp).dropna().mean()
            ssp += diff
            ssp = pd.concat([era5, ssp])
            ssp = ssp[~ssp.index.duplicated()]

            ssp_temps[(model, ssp_str)] = ssp


    ssps = pd.concat(ssp_temps, names=["model", "ssp", "year"])

    return ssps


if __name__ == "__main__":
    main()
