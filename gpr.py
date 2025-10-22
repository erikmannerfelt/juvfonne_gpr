import segysak.segy
import segysak
import segyio
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import scipy.interpolate
import shutil

from pathlib import Path

def prepare_dataset(filepath: Path, crs_epsg: int) -> Path:

    out_path = Path(f"proc/{filepath.stem}.nc")

    if out_path.is_file():
        return out_path

    print(f"Processing {filepath}")
    coord_fp = filepath.with_name(filepath.stem.replace("-gpr", "-position.csv"))

    if "09-18-08" in filepath.stem:
        coord_fp = coord_fp.with_stem(coord_fp.stem.replace("09-18-08", "09-18-10"))
        print(f"Using {coord_fp.name} for coordinates")

    coords = pd.read_csv(coord_fp)
    coords = coords.dropna(subset=["zGPR:Trace", "Longitude"]).set_index("zGPR:Trace")
    coords.columns = [str(col).lower() for col in coords.columns]


    coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords["longitude rtk"], coords["latitude rtk"], crs=4326)).to_crs(crs_epsg)

    coords["easting"] = coords["geometry"].x
    coords["northing"] = coords["geometry"].y

    # print(coords.iloc[1000])

    # plt.plot(coords.index, coords["alt:altitude"])
    # plt.show()
    # return


    relevant_traces = {
        "2025-09-03-08-14-45": (7000, 76500),
        "2025-09-03-08-47-05": (5000, 110000),
        "2025-09-03-09-18-08": (6700, 127800),
    }
    # print(coords.iloc[0])
    # return

    # print(coord_fp)
    # return



    # print(coord_fp)
    # print(coords.shape)
    # print(coords.iloc[10])
    # return

    # filepath = Path("input/gpr/Juvfonna_GPR_20250903/2025-09-03-08-37-27-gpr.sgy")

    # with segysak.segy.segy_loader(filepath, endian="little") as data:
    #     print(data)
    # with xr.open_dataset(filepath, engine="sgy_engine", endian="little") as data:
        # print(data)
    with segyio.open(filepath, endian="little", strict=False) as f:
        sample_time_ps = f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        sample_time_t0 = f.header[0][segyio.TraceField.DelayRecordingTime]

        n_samples = f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT]

    twtt_ns = np.arange(n_samples) * sample_time_ps / 1000 -sample_time_t0 / 1000 

    with xr.open_dataset(filepath, engine="sgy_engine",segyio_kwargs={"endian": "little"}, dim_byte_fields={"trace_n": 21}) as data:

        # data["trace_n"] = data["trace_n"].astype("int64")
        for key in relevant_traces:
            if key in filepath.stem:
                data = data.sel(trace_n=slice(*relevant_traces[key]))
                print("Sliced with preset values")
                break


        for col in ["easting", "northing", "altitude", "alt:altitude"]:
            data[col] = "trace_n", scipy.interpolate.interp1d(coords.index, coords[col], fill_value="extrapolate")(data["trace_n"])

        if data.data.shape[1] >= n_samples:
            data = data.isel(samples=slice(n_samples))

        data.coords["twtt"] = "samples", twtt_ns
        data = data.swap_dims(samples="twtt").drop_vars(["samples"])

        # Make it (sample, trace_n) instead of (trace_n, sample)
        data["data"] = data["data"].T

        diffs = data[["easting", "northing"]].diff("trace_n").broadcast_like(data["easting"]).fillna(0)
        data["distance"] = ((diffs["easting"] ** 2 + diffs["northing"] ** 2) ** 0.5).cumsum("trace_n")

        # data["distance"] = data["easting"].diff(

        # window_length = 5000
        data.attrs = {}
        # win_start = 10000

        data["data"] -= data["data"].median("trace_n")
        data["data"] /= 1e8
        # data["data"] = data["data"]

        tmp_path = out_path.with_suffix(".nc.tmp")
        tmp_path.parent.mkdir(exist_ok=True, parents=True)

        print(data)
        print("Saving")
        data.to_netcdf(tmp_path, encoding={v: {"zlib": True, "complevel": 9} for v in data.data_vars}, engine="netcdf4")

        shutil.move(tmp_path, out_path)

    return out_path


def proj_digitized(data_filepath: Path, digitized_surface_filepath: Path,digitized_bed_filepath: Path, crs_epsg: int):

    digitized_bed = gpd.read_file(digitized_bed_filepath)

    digitized_surface = gpd.read_file(digitized_surface_filepath)


    proj_list = []
    with xr.open_dataset(data_filepath, chunks={"data": "auto"}) as data:
        data["distance"].load()

        interp_points = np.unique(scipy.interpolate.interp1d(data["distance"], data["trace_n"])(np.arange(0, data["distance"].max().item(), 1)).astype(int))

        surface_coords = np.array(digitized_surface["geometry"].iloc[0].xy)
        surface_model = scipy.interpolate.interp1d(data.data.shape[1] - surface_coords[0, :], data.data.shape[0] + surface_coords[1, :])

        for _, line in digitized_bed["geometry"].items():
            coords = np.array(line.xy)

            model = scipy.interpolate.interp1d(data.data.shape[1] - coords[0, :], data.data.shape[0] + coords[1, :], bounds_error=False)

            pred = model(interp_points)

            mask = np.isfinite(pred)

            vals = pd.DataFrame({"x": interp_points[mask], "y": pred[mask].astype(int)}) 

            if vals.shape[0] == 0:
                continue

            for col in data.data_vars:
                if str(col) == "data":
                    continue
                vals[str(col)] = data[col].isel(trace_n=vals["x"])

            vals["bed_twtt"] = data["twtt"].isel(twtt=vals["y"])

            vals["surface_twtt"] = data["twtt"].isel(twtt=surface_model(vals["x"]).astype(int))

            # vals["peak_twtt"] = data["twtt"].isel(twtt=data["data"].isel(trace_n=vals["x"]).argmax("twtt").values)
            
            proj_list.append(vals)

            
    proj = pd.concat(proj_list)

    proj = gpd.GeoDataFrame(proj, geometry=gpd.points_from_xy(proj["easting"], proj["northing"], crs=crs_epsg))


    proj["ice_twtt"] = proj["bed_twtt"] - proj["surface_twtt"]

    proj["depth"] = proj["ice_twtt"] * 0.168 / 2
    # plt.scatter(proj["peak_twtt"], proj["airwave_twtt"])
    # plt.show()
    # print(proj.iloc[0])
    # return

    out_file = digitized_bed_filepath.parent / f"proj/{digitized_bed_filepath.stem}_proj.geojson"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    proj.to_file(out_file)

    return out_file



def main(crs_epsg: int = 25832):

    filepaths = []

    for filepath in Path("input/gpr/Juvfonna_GPR_20250903/").glob("*.sgy"):
        if any(s in filepath.stem.replace("-gpr", "") for s in ["09-18-08", "08-14-45", "08-47-05"]):
            filepaths.append(filepath)

    proj_list = []
    for filepath in filepaths:

        data_filepath = prepare_dataset(filepath, crs_epsg=crs_epsg)
        for digitized_bed_filepath in Path("digitized").glob("*bed.geojson"):

            digitized_surface_filepath = digitized_bed_filepath.with_stem(digitized_bed_filepath.stem.replace("_bed", "_surface"))
            if data_filepath.stem in digitized_bed_filepath.stem:
                proj = proj_digitized(data_filepath=data_filepath, digitized_bed_filepath=digitized_bed_filepath, digitized_surface_filepath=digitized_surface_filepath, crs_epsg=crs_epsg)

                proj_list.append(proj)

            
    pd.concat([gpd.read_file(fp) for fp in proj_list]).to_file(proj_list[0].parent / "combined_proj.geojson")

    return
    # filepaths = list()
    #
    # print(filepaths)

    filepath = filepaths[1]

    # print(filepath)
    # return

    print(filepath)
    coord_fp = filepath.with_name(filepath.stem.replace("-gpr", "-position.csv"))

    if "09-18-08" in filepath.stem:
        coord_fp = coord_fp.with_stem(coord_fp.stem.replace("09-18-08", "09-18-10"))

    coords = pd.read_csv(coord_fp)
    coords = coords.dropna(subset=["zGPR:Trace", "Longitude"]).set_index("zGPR:Trace")
    coords.columns = [str(col).lower() for col in coords.columns]


    coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords["longitude rtk"], coords["latitude rtk"], crs=4326)).to_crs(crs_epsg)

    coords["easting"] = coords["geometry"].x
    coords["northing"] = coords["geometry"].y

    # print(coords.iloc[1000])

    # plt.plot(coords.index, coords["alt:altitude"])
    # plt.show()
    # return


    relevant_traces = {
        "2025-09-03-08-14-45": (7000, 76500),
        "2025-09-03-08-47-05": (5000, 110000),
        "2025-09-03-09-18-08": (6700, 127800),
    }
    # print(coords.iloc[0])
    # return

    # print(coord_fp)
    # return



    # print(coord_fp)
    # print(coords.shape)
    # print(coords.iloc[10])
    # return

    # filepath = Path("input/gpr/Juvfonna_GPR_20250903/2025-09-03-08-37-27-gpr.sgy")

    # with segysak.segy.segy_loader(filepath, endian="little") as data:
    #     print(data)
    # with xr.open_dataset(filepath, engine="sgy_engine", endian="little") as data:
        # print(data)
    with segyio.open(filepath, endian="little", strict=False) as f:
        sample_time_ps = f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        sample_time_t0 = f.header[0][segyio.TraceField.DelayRecordingTime]

        n_samples = f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT]

    twtt_ns = np.arange(n_samples) * sample_time_ps / 1000 -sample_time_t0 / 1000 

    with xr.open_dataset(filepath, engine="sgy_engine",segyio_kwargs={"endian": "little"}, dim_byte_fields={"cdp": 21}) as data:

        for key in relevant_traces:
            if key in filepath.stem:
                data = data.sel(cdp=slice(*relevant_traces[key]))
                print("Slicing")
                break


        for col in ["easting", "northing", "altitude", "alt:altitude"]:
            data[col] = "cdp", scipy.interpolate.interp1d(coords.index, coords[col], fill_value="extrapolate")(data["cdp"])

        if data.data.shape[1] >= n_samples:
            data = data.isel(samples=slice(n_samples))

        data.coords["twtt"] = "samples", twtt_ns
        data = data.swap_dims(samples="twtt")

        # Make it (sample, trace_n) instead of (trace_n, sample)
        data["data"] = data["data"].T

        window_length = 5000
        win_start = 10000

        data["data"] -= data["data"].median("cdp")
        data["data"] /= 1e8

        data = data.isel(cdp=slice(win_start, min(data.cdp.shape[0], win_start + window_length)))

        # with xr.open_dataset(

        # print(data)
        # return

        # data["data"].values -= np.median(data["data"].values, axis=1, keepdims=True)


        plt.imshow(data.data, vmin=-1, vmax=1, cmap="Greys_r", extent=(data.cdp.min().item(), data.cdp.max().item(), data.twtt.max().item(), data.twtt.min().item()), aspect="auto")
        plt.show()
        print(data)
    return

    with segyio.open(filepath, ignore_geometry=True, strict=False, endian="big") as f:
        print(f.tracecount)
        print(f.samples.size)

    return

    with xr.open_dataset(filepath, dim_byte_fields={"iline": 189, "xline": 193}, extra_byte_fields={"cdp_x": 73, "cdp_y": 77}) as dataset:
        print(dataset)

    
    ...

if __name__ == "__main__":
    main()
