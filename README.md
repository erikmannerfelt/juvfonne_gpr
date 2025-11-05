# Code for GPR and ice patch thickness forecasting for the Juvfonne ice cap, Norway

## Installation

This has been run on a nix-enabled system, meaning that the environment can be recreated with `nix develop`.
The `requirements.txt` should be compatible with pip, but may require small tweaks to work.
A CDS API account has to be registered beforehand to download climate data.

## `gpr.py`

Code for processing UAV GPR data and manual interpretations.

Usage: run `python gpr.py`

## `climate.py`

Code for downloading CMIP6 and ERA5 data for Juvfonne.

Usage: run `python climate.py`

## `forecasting.py`

Code for ice patch thickness forecasting using mean thickness measurements and climate data.

Usage: run `python forecasting.py`
