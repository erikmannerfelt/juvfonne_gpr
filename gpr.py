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

    # This file strangely had a different file stem for its valid coordinates!
    if "09-18-08" in filepath.stem:
        coord_fp = coord_fp.with_stem(coord_fp.stem.replace("09-18-08", "09-18-10"))
        print(f"Using {coord_fp.name} for coordinates")

    # Read and format the coordinate data
    coords = pd.read_csv(coord_fp)
    coords = coords.dropna(subset=["zGPR:Trace", "Longitude"]).set_index("zGPR:Trace")
    coords.columns = [str(col).lower() for col in coords.columns]

    # For this file, RTK stopped working mid-flight so it cannot be used.
    if "2025-09-03-08-14-45" in filepath.stem:
        c_cols = ["longitude", "latitude"]
    else:
        c_cols = ["longitude rtk", "latitude rtk"]

    coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords[c_cols[0]], coords[c_cols[1]], crs=4326)).to_crs(crs_epsg)

    coords["easting"] = coords["geometry"].x
    coords["northing"] = coords["geometry"].y

    # The files are huge so this part subsets the data to roughly where the glacier part starts and ends.
    relevant_traces = {
        "2025-09-03-08-14-45": (7000, 76500),
        "2025-09-03-08-47-05": (5000, 110000),
        "2025-09-03-09-18-08": (6700, 127800),
    }

    with segyio.open(filepath, endian="little", strict=False) as f:
        sample_time_ps = f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        sample_time_t0 = f.header[0][segyio.TraceField.DelayRecordingTime]

        n_samples = f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT]

    twtt_ns = np.arange(n_samples) * sample_time_ps / 1000 -sample_time_t0 / 1000 

    with xr.open_dataset(filepath, engine="sgy_engine",segyio_kwargs={"endian": "little"}, dim_byte_fields={"trace_n": 21}) as data:

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

        data.attrs = {}

        data["data"] -= data["data"].median("trace_n")
        data["data"] /= 1e8

        tmp_path = out_path.with_suffix(".nc.tmp")
        tmp_path.parent.mkdir(exist_ok=True, parents=True)

        print(data)
        print("Saving")
        data.to_netcdf(tmp_path, encoding={v: {"zlib": True, "complevel": 9} for v in data.data_vars}, engine="netcdf4")

        shutil.move(tmp_path, out_path)

    return out_path


def process_picks(data_filepath: Path, digitized_surface_filepath: Path,digitized_bed_filepath: Path, crs_epsg: int, spacing_m: float = 1.):

    digitized_bed = gpd.read_file(digitized_bed_filepath)

    if digitized_bed.shape[0] == 0:
        raise ValueError(f"Empty file: {digitized_bed_filepath}")

    digitized_surface = gpd.read_file(digitized_surface_filepath)

    if digitized_surface.shape[0] != 1:
        raise ValueError("Functionality not implemented for a split surface reflection (multiple lines). It needs to be exactly one line")


    proj_list = []
    with xr.open_dataset(data_filepath, chunks={"data": "auto"}) as data:
        data["distance"].load() # Needed for the next line to not error out

        # Generate {spacing_m} m separated trace number points to sample at.
        interp_points = np.unique(scipy.interpolate.interp1d(data["distance"], np.arange(data["trace_n"].shape[0]))(np.arange(0, data["distance"].max().item(), spacing_m)).astype(int))

        # Generate a surface model that can be called. The value will be the y index of the equivalent twtt
        surface_coords = np.array(digitized_surface["geometry"].iloc[0].xy)
        surface_model = scipy.interpolate.interp1d(surface_coords[0, :], data.data.shape[0] + surface_coords[1, :])

        # Loop through all digitized bed lines
        for i, line in digitized_bed["geometry"].items():
            coords = np.array(line.xy)
            model = scipy.interpolate.interp1d(coords[0, :], data.data.shape[0] + coords[1, :], bounds_error=False)

            # Interpolate points along the interp_points variable. Most of these will be nan because
            # they'll be outside the interpolation range.
            pred = model(interp_points.astype(float))

            # Find where valid data exist (i.e. where they were inside the line)
            mask = np.isfinite(pred)

            # Populate a dataframe with the valid interpolated points
            vals = pd.DataFrame({"x": interp_points[mask], "y": pred[mask].astype(int)}) 

            if vals.shape[0] == 0:
                print(f"Line {i} invalid")
                continue

            # Add dataframe columns for all data variables in the netcdf
            # TODO: Change to filter by data variables that only have the dim "trace_n"
            for col in data.data_vars:
                if str(col) == "data":
                    continue
                vals[str(col)] = data[col].isel(trace_n=vals["x"])

            # Get the twtt of the bed and the surface
            vals["bed_twtt"] = data["twtt"].isel(twtt=vals["y"])
            vals["surface_twtt"] = data["twtt"].isel(twtt=surface_model(vals["x"]).astype(int))

            proj_list.append(vals)

            
    proj = pd.concat(proj_list)

    proj = gpd.GeoDataFrame(proj, geometry=gpd.points_from_xy(proj["easting"], proj["northing"], crs=crs_epsg))

    # The twtt-part that's relevant is the difference between the bed and the surface.
    # Essentially, the surface becomes the time-0 part, to relate it to surface-coupled GPR.
    proj["ice_twtt"] = proj["bed_twtt"] - proj["surface_twtt"]
    proj["depth"] = proj["ice_twtt"] * 0.168 / 2

    out_file = digitized_bed_filepath.parent / f"proj/{digitized_bed_filepath.stem}_proj.geojson"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    proj.to_file(out_file)

    return out_file


def generate_track(data_filepath: Path, crs_epsg: int):
    
    with xr.open_dataset(data_filepath) as data:
        track_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(data["easting"], data["northing"], crs=crs_epsg))
        track_pts["trace_n"] = data["trace_n"].values

        track_pts.to_file(f"temp/{data_filepath.stem}_track_pts.geojson")


def main(crs_epsg: int = 25832):

    # Accumulate the three surveys from Juvfonne
    filepaths = []
    for filepath in Path("input/gpr/Juvfonna_GPR_20250903/").glob("*.sgy"):
        if any(s in filepath.stem.replace("-gpr", "") for s in ["09-18-08", "08-14-45", "08-47-05"]):
            filepaths.append(filepath)

    pick_list = []
    for filepath in filepaths:
        # Process the GPR data and save to a more usable netcdf
        data_filepath = prepare_dataset(filepath, crs_epsg=crs_epsg)

        # Generate a track file for the data (for visualization)
        generate_track(data_filepath, crs_epsg=crs_epsg)

        # For each _digitized_bed.geojson file that can be found, process these picks
        for digitized_bed_filepath in Path("digitized").glob("*bed.geojson"):
            digitized_surface_filepath = digitized_bed_filepath.with_stem(digitized_bed_filepath.stem.replace("_bed", "_surface"))

            if not digitized_surface_filepath.is_file():
                raise ValueError(f"No _surface.geojson equivalent for {digitized_bed_filepath}")

            if data_filepath.stem in digitized_bed_filepath.stem:
                picks = process_picks(
                    data_filepath=data_filepath,
                    digitized_bed_filepath=digitized_bed_filepath,
                    digitized_surface_filepath=digitized_surface_filepath,
                    crs_epsg=crs_epsg
                )

                pick_list.append(picks)

            
    pd.concat([gpd.read_file(fp) for fp in pick_list]).to_file(pick_list[0].parent / "combined_proj.geojson")

if __name__ == "__main__":
    main()
