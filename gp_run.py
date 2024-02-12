import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
from eofs.xarray import Eof
import tensorflow as tf
import gpflow
import seaborn as sns
import cartopy.crs as ccrs
from utils import *
from glob import glob
import warnings
from matplotlib import transforms
import dataframe_image as dfi

from xskillscore import rmse, pearson_r, spearman_r, r2, smape, mae, me, mse

data_path = os.getcwd() + "/"

min_co2 = 0.
max_co2 = 9500
def normalize_co2(data):
    return data / max_co2

min_ch4 = 0.
max_ch4 = 0.8
def normalize_ch4(data):
    return data / max_ch4

def create_predictor_data(data_sets, n_eofs=5):
    # Create training and testing arrays
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    X = xr.concat([xr.open_dataset(data_path + f"inputs_{file}.nc") for file in data_sets], dim='time')
    X = X.assign_coords(time=np.arange(len(X.time)))

    # Compute EOFs for BC
    bc_solver = Eof(X['BC'])
    bc_eofs = bc_solver.eofsAsCorrelation(neofs=n_eofs)
    bc_pcs = bc_solver.pcs(npcs=n_eofs, pcscaling=1)

    # Compute EOFs for SO2
    so2_solver = Eof(X['SO2'])
    so2_eofs = so2_solver.eofsAsCorrelation(neofs=n_eofs)
    so2_pcs = so2_solver.pcs(npcs=n_eofs, pcscaling=1)

    # Convert to pandas
    bc_df = bc_pcs.to_dataframe().unstack('mode')
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    so2_df = so2_pcs.to_dataframe().unstack('mode')
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame({
        "CO2": normalize_co2(X["CO2"].data),
        "CH4": normalize_ch4(X["CH4"].data)
    }, index=X["CO2"].coords['time'].data)

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs, (so2_solver, bc_solver)



def create_predictdand_data(data_sets):
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    Y = xr.concat([xr.open_dataset(data_path + f"outputs_{file}.nc") for file in data_sets], dim='time').mean("member")
    # Convert the precip values to mm/day
    Y["pr"] *= 86400
    Y["pr90"] *= 86400
    return Y

def get_test_data(file, eof_solvers, n_eofs=5):
    # Create training and testing arrays
    X = xr.open_dataset(data_path + f"inputs_{file}.nc")
        
    so2_pcs = eof_solvers[0].projectField(X["SO2"], neofs=5, eofscaling=1)
    so2_df = so2_pcs.to_dataframe().unstack('mode')
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    bc_pcs = eof_solvers[1].projectField(X["BC"], neofs=5, eofscaling=1)
    bc_df = bc_pcs.to_dataframe().unstack('mode')
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame({
        "CO2": normalize_co2(X["CO2"].data),
        "CH4": normalize_ch4(X["CH4"].data)
    }, index=X["CO2"].coords['time'].data)

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs


def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon'])).data

# List of dataset to use for training
train_files = ["ssp126", "ssp370", "ssp585", "historical", "hist-GHG", "hist-aer"]

def tasGP():
    # Create training and testing arrays
    X_train, eof_solvers = create_predictor_data(train_files)
    y_train_tas = create_predictdand_data(train_files)['tas'].values.reshape(-1, 96 * 144)
    
    X_test = get_test_data('ssp245', eof_solvers)
    Y_test = xr.open_dataset(data_path + 'outputs_ssp245.nc').compute()
    tas_truth = Y_test["tas"].mean('member')
    
    # Drop rows including nans
    nan_train_mask = X_train.isna().any(axis=1).values
    X_train = X_train.dropna(axis=0, how='any')
    y_train_tas = y_train_tas[~nan_train_mask]
    assert len(X_train) == len(y_train_tas)
    nan_test_mask = X_test.isna().any(axis=1).values
    X_test = X_test.dropna(axis=0, how='any')
    tas_truth = tas_truth[~nan_test_mask]

    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = X_train['CO2'].mean(), X_train['CO2'].std()
    train_CH4_mean, train_CH4_std = X_train['CH4'].mean(), X_train['CH4'].std()
    X_train.CO2 = (X_train.CO2 - train_CO2_mean) / train_CO2_std
    X_train.CH4 = (X_train.CH4 - train_CH4_mean) / train_CH4_std
    X_test.CO2 = (X_test.CO2 - train_CO2_mean) / train_CO2_std
    X_test.CH4 = (X_test.CH4 - train_CH4_mean) / train_CH4_std

    # Standardize predictand fields
    train_tas_mean, train_tas_std = y_train_tas.mean(), y_train_tas.std()
    y_train_tas = (y_train_tas - train_tas_mean) / train_tas_std

    # Make kernel
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0])
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2

    # Make model
    np.random.seed(5)
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.GPR(data=(X_train.astype(np.float64), 
                                    y_train_tas.astype(np.float64)),
                              kernel=kernel,
                              mean_function=mean)

    # Define optimizer
    opt = gpflow.optimizers.Scipy()
    # Train model
    opt.minimize(model.training_loss,
                 variables=model.trainable_variables,
                 options=dict(disp=True, maxiter=1000))

    # predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(X_test.values)
    posterior_mean = standard_posterior_mean * train_tas_std + train_tas_mean
    posterior_stddev = np.sqrt(standard_posterior_var) * train_tas_std

    # put output back into xarray format for calculating RMSE/plotting
    posterior_tas = np.reshape(posterior_mean, [86, 96, 144])
    posterior_tas_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    
    posterior_tas_data = xr.DataArray(posterior_tas, dims=tas_truth.dims, coords=tas_truth.coords)
    posterior_tas_std_data = xr.DataArray(posterior_tas_stddev, dims=tas_truth.dims, coords=tas_truth.coords)

    # Compute RMSEs
    print(f"RMSE at 2050: {get_rmse(tas_truth[35], posterior_tas_data[35])}")
    print(f"RMSE at 2100: {get_rmse(tas_truth[85], posterior_tas_data[85])}")
    print(f"RMSE 2045-2055: {get_rmse(tas_truth[30:41], posterior_tas_data[30:41]).mean()}")
    print(f"RMSE 2090-2100: {get_rmse(tas_truth[75:], posterior_tas_data[75:]).mean()}")
    print(f"RMSE 2050-2100: {get_rmse(tas_truth[35:], posterior_tas_data[35:]).mean()}")
    
    # RMSE for average field over last 20 years
    print(f"RMSE average last 20y: {get_rmse(tas_truth[-20:].mean(dim='time'), posterior_tas_data[-20:].mean(dim='time'))}")

    # plotting predictions
    divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)
    diffnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2)
    
    ## Temperature
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 3))
    fig.suptitle('Temperature')
    
    # Test
    plt.subplot(131, projection=proj)
    tas_truth.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                                  cbar_kwargs={"label":"Temperature change / K"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='True')
    
    # Emulator
    plt.subplot(132, projection=proj)
    posterior_tas_data.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                           cbar_kwargs={"label":"Temperature change / K"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='GP posterior mean')
    
    # Difference
    difference = tas_truth - posterior_tas_data
    plt.subplot(133, projection=proj)
    difference.sel(time=slice(2050,None)).mean('time').plot(
        cmap="bwr",norm=diffnorm,cbar_kwargs={"label":"Temperature change / K"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='Difference')
    return posterior_tas_data

tas_preds = tasGP()


def dtrGP():
    # Create training and testing arrays
    X_train, eof_solvers = create_predictor_data(train_files)
    y_train_dtr = create_predictdand_data(train_files)['diurnal_temperature_range'].values.reshape(-1, 96 * 144)
    X_test = get_test_data('ssp245', eof_solvers)
    Y_test = xr.open_dataset(data_path + 'outputs_ssp245.nc').compute()
    dtr_truth = Y_test["diurnal_temperature_range"].mean('member')

    # Drop rows including nans
    nan_train_mask = X_train.isna().any(axis=1).values
    X_train = X_train.dropna(axis=0, how='any')
    y_train_dtr = y_train_dtr[~nan_train_mask]
    assert len(X_train) == len(y_train_dtr)
    nan_test_mask = X_test.isna().any(axis=1).values
    X_test = X_test.dropna(axis=0, how='any')
    dtr_truth = dtr_truth[~nan_test_mask]

    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = X_train['CO2'].mean(), X_train['CO2'].std()
    train_CH4_mean, train_CH4_std = X_train['CH4'].mean(), X_train['CH4'].std()
    X_train.CO2 = (X_train.CO2 - train_CO2_mean) / train_CO2_std
    X_train.CH4 = (X_train.CH4 - train_CH4_mean) / train_CH4_std
    X_test.CO2 = (X_test.CO2 - train_CO2_mean) / train_CO2_std
    X_test.CH4 = (X_test.CH4 - train_CH4_mean) / train_CH4_std

    # Standardize predictand fields
    train_dtr_mean, train_dtr_std = y_train_dtr.mean(), y_train_dtr.std()
    y_train_dtr = (y_train_dtr - train_dtr_mean) / train_dtr_std

    # Make kernel
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0])
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2

    # Make model
    np.random.seed(5)
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.GPR(data=(X_train.astype(np.float64), 
                                    y_train_dtr.astype(np.float64)),
                              kernel=kernel,
                              mean_function=mean)

    # Define optimizer
    opt = gpflow.optimizers.Scipy()
    # Train model
    opt.minimize(model.training_loss,
                 variables=model.trainable_variables,
                 options=dict(disp=True, maxiter=1000))

    # predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(X_test.values)
    posterior_mean = standard_posterior_mean * train_dtr_std + train_dtr_mean
    posterior_stddev = np.sqrt(standard_posterior_var) * train_dtr_std

    # put output back into xarray format for calculating RMSE/plotting
    posterior_dtr = np.reshape(posterior_mean, [86, 96, 144])
    posterior_dtr_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    posterior_dtr_data = xr.DataArray(posterior_dtr, dims=dtr_truth.dims, coords=dtr_truth.coords)
    posterior_dtr_std_data = xr.DataArray(posterior_dtr_stddev, dims=dtr_truth.dims, coords=dtr_truth.coords)

    # Compute RMSEs
    print(f"RMSE at 2050: {get_rmse(dtr_truth[35], posterior_dtr_data[35])}")
    print(f"RMSE at 2100: {get_rmse(dtr_truth[85], posterior_dtr_data[85])}")
    print(f"RMSE 2045-2055: {get_rmse(dtr_truth[30:41], posterior_dtr_data[30:41]).mean()}")
    print(f"RMSE 2090-2100: {get_rmse(dtr_truth[75:], posterior_dtr_data[75:]).mean()}")
    print(f"RMSE 2050-2100: {get_rmse(dtr_truth[35:], posterior_dtr_data[35:]).mean()}")
    # RMSE for average field over last 20 years
    print(f"RMSE average last 20y: {get_rmse(dtr_truth[-20:].mean(dim='time'), posterior_dtr_data[-20:].mean(dim='time'))}")

    # plotting predictions
    divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)
    diffnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2)
    
    ## Diurnal Temperature Range
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 3))
    fig.suptitle('DTR')
    
    # Test
    plt.subplot(131, projection=proj)
    dtr_truth.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                                  cbar_kwargs={"label":"DTR change / K"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='True')
    
    # Emulator
    plt.subplot(132, projection=proj)
    posterior_dtr_data.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                           cbar_kwargs={"label":"DTR change / K"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='GP posterior mean')
    
    # Difference
    difference = dtr_truth - posterior_dtr_data
    plt.subplot(133, projection=proj)
    difference.sel(time=slice(2050,None)).mean('time').plot(cmap="bwr", norm=diffnorm,
                    cbar_kwargs={"label":"DTR change / K"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='Difference')
    return posterior_dtr_data

dtr_preds = dtrGP()


def prGP():
    # Create training and testing arrays
    X_train, eof_solvers = create_predictor_data(train_files)
    y_train_pr = create_predictdand_data(train_files)['pr'].values.reshape(-1, 96 * 144)
    X_test = get_test_data('ssp245', eof_solvers)
    Y_test = xr.open_dataset(data_path + 'outputs_ssp245.nc').compute()
    pr_truth = Y_test["pr"].mean('member') * 86400
    
    # Drop rows including nans
    nan_train_mask = X_train.isna().any(axis=1).values
    X_train = X_train.dropna(axis=0, how='any')
    y_train_pr = y_train_pr[~nan_train_mask]
    assert len(X_train) == len(y_train_pr)
    nan_test_mask = X_test.isna().any(axis=1).values
    X_test = X_test.dropna(axis=0, how='any')
    pr_truth = pr_truth[~nan_test_mask]

    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = X_train['CO2'].mean(), X_train['CO2'].std()
    train_CH4_mean, train_CH4_std = X_train['CH4'].mean(), X_train['CH4'].std()
    X_train.CO2 = (X_train.CO2 - train_CO2_mean) / train_CO2_std
    X_train.CH4 = (X_train.CH4 - train_CH4_mean) / train_CH4_std
    X_test.CO2 = (X_test.CO2 - train_CO2_mean) / train_CO2_std
    X_test.CH4 = (X_test.CH4 - train_CH4_mean) / train_CH4_std

    # Standardize predictand fields
    train_pr_mean, train_pr_std = y_train_pr.mean(), y_train_pr.std()
    y_train_pr = (y_train_pr - train_pr_mean) / train_pr_std

    # Make kernel
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0])
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2

    # Make model
    np.random.seed(5)
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.GPR(data=(X_train.astype(np.float64), 
                                    y_train_pr.astype(np.float64)),
                              kernel=kernel,
                              mean_function=mean)

    # Define optimizer
    opt = gpflow.optimizers.Scipy()
    # Train model
    opt.minimize(model.training_loss,
                 variables=model.trainable_variables,
                 options=dict(disp=True, maxiter=1000))

    # predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(X_test.values)
    posterior_mean = standard_posterior_mean * train_pr_std + train_pr_mean
    posterior_stddev = np.sqrt(standard_posterior_var) * train_pr_std

    # put output back into xarray format for calculating RMSE/plotting
    posterior_pr = np.reshape(posterior_mean, [86, 96, 144])
    posterior_pr_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    posterior_pr_data = xr.DataArray(posterior_pr, dims=pr_truth.dims, coords=pr_truth.coords)
    posterior_pr_std_data = xr.DataArray(posterior_pr_stddev, dims=pr_truth.dims, coords=pr_truth.coords)

    # Compute RMSEs
    print(f"RMSE at 2050: {get_rmse(pr_truth[35], posterior_pr_data[35])}")
    print(f"RMSE at 2100: {get_rmse(pr_truth[85], posterior_pr_data[85])}")
    print(f"RMSE 2045-2055: {get_rmse(pr_truth[30:41], posterior_pr_data[30:41]).mean()}")
    print(f"RMSE 2090-2100: {get_rmse(pr_truth[75:], posterior_pr_data[75:]).mean()}")
    print(f"RMSE 2050-2100: {get_rmse(pr_truth[35:], posterior_pr_data[35:]).mean()}")
    # RMSE for average field over last 20 years
    print(f"RMSE average last 20y: {get_rmse(pr_truth[-20:].mean(dim='time'), posterior_pr_data[-20:].mean(dim='time'))}")

    # plotting predictions
    divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)
    diffnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2)
    
    ## Temperature
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 3))
    fig.suptitle('Precipitation')
    
    # Test
    plt.subplot(131, projection=proj)
    pr_truth.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                                  cbar_kwargs={"label":"Precip change / mm day-1"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='True')
    
    # Emulator
    plt.subplot(132, projection=proj)
    posterior_pr_data.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                           cbar_kwargs={"label":"Precip change / mm day-1"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='GP posterior mean')
    
    # Difference
    difference = pr_truth - posterior_pr_data
    plt.subplot(133, projection=proj)
    difference.sel(time=slice(2050,None)).mean('time').plot(cmap="bwr", norm=diffnorm,
                    cbar_kwargs={"label":"Precip change / mm day-1"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='Difference')
    return posterior_pr_data

pr_preds = prGP()



def pr90GP():
    # Create training and testing arrays
    X_train, eof_solvers = create_predictor_data(train_files)
    y_train_pr90 = create_predictdand_data(train_files)['pr90'].values.reshape(-1, 96 * 144)
    X_test = get_test_data('ssp245', eof_solvers)
    Y_test = xr.open_dataset(data_path + 'outputs_ssp245.nc').compute()
    pr90_truth = Y_test["pr90"].mean('member') * 86400
    
    # Drop rows including nans
    nan_train_mask = X_train.isna().any(axis=1).values
    X_train = X_train.dropna(axis=0, how='any')
    y_train_pr90 = y_train_pr90[~nan_train_mask]
    assert len(X_train) == len(y_train_pr90)
    nan_test_mask = X_test.isna().any(axis=1).values
    X_test = X_test.dropna(axis=0, how='any')
    pr90_truth = pr90_truth[~nan_test_mask]

    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = X_train['CO2'].mean(), X_train['CO2'].std()
    train_CH4_mean, train_CH4_std = X_train['CH4'].mean(), X_train['CH4'].std()
    X_train.CO2 = (X_train.CO2 - train_CO2_mean) / train_CO2_std
    X_train.CH4 = (X_train.CH4 - train_CH4_mean) / train_CH4_std
    X_test.CO2 = (X_test.CO2 - train_CO2_mean) / train_CO2_std
    X_test.CH4 = (X_test.CH4 - train_CH4_mean) / train_CH4_std

    # Standardize predictand fields
    train_pr90_mean, train_pr90_std = y_train_pr90.mean(), y_train_pr90.std()
    y_train_pr90 = (y_train_pr90 - train_pr90_mean) / train_pr90_std

    # Make kernel
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0])
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2

    # Make model
    np.random.seed(5)
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.GPR(data=(X_train.astype(np.float64), 
                                    y_train_pr90.astype(np.float64)),
                              kernel=kernel,
                              mean_function=mean)

    # Define optimizer
    opt = gpflow.optimizers.Scipy()
    # Train model
    opt.minimize(model.training_loss,
                 variables=model.trainable_variables,
                 options=dict(disp=True, maxiter=1000))

    # predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(X_test.values)
    posterior_mean = standard_posterior_mean * train_pr90_std + train_pr90_mean
    posterior_stddev = np.sqrt(standard_posterior_var) * train_pr90_std

    # put output back into xarray format for calculating RMSE/plotting
    posterior_pr90 = np.reshape(posterior_mean, [86, 96, 144])
    posterior_pr90_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    posterior_pr90_data = xr.DataArray(posterior_pr90, dims=pr90_truth.dims, coords=pr90_truth.coords)
    posterior_pr90_std_data = xr.DataArray(posterior_pr90_stddev, dims=pr90_truth.dims, coords=pr90_truth.coords)

    # Compute RMSEs
    print(f"RMSE at 2050: {get_rmse(pr90_truth[35], posterior_pr90_data[35])}")
    print(f"RMSE at 2100: {get_rmse(pr90_truth[85], posterior_pr90_data[85])}")
    print(f"RMSE 2045-2055: {get_rmse(pr90_truth[30:41], posterior_pr90_data[30:41]).mean()}")
    print(f"RMSE 2090-2100: {get_rmse(pr90_truth[75:], posterior_pr90_data[75:]).mean()}")
    print(f"RMSE 2050-2100: {get_rmse(pr90_truth[35:], posterior_pr90_data[35:]).mean()}")
    # RMSE for average field over last 20 years
    print(f"RMSE average last 20y: {get_rmse(pr90_truth[-20:].mean(dim='time'), posterior_pr90_data[-20:].mean(dim='time'))}")

    # plotting predictions
    divnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=5)
    diffnorm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2)
    
    ## Temperature
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 3))
    fig.suptitle('Extreme Precipitation')
    
    # Test
    plt.subplot(131, projection=proj)
    pr90_truth.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                                  cbar_kwargs={"label":"P90 change / mm day-1"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='True')
    
    # Emulator
    plt.subplot(132, projection=proj)
    posterior_pr90_data.sel(time=slice(2050,None)).mean('time').plot(cmap="coolwarm", norm=divnorm,
                           cbar_kwargs={"label":"P90 change / mm day-1"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='GP posterior mean')
    
    # Difference
    difference = pr90_truth - posterior_pr90_data
    plt.subplot(133, projection=proj)
    difference.sel(time=slice(2050,None)).mean('time').plot(cmap="bwr", norm=diffnorm,
                    cbar_kwargs={"label":"P90 change / mm day-1"})
    plt.gca().coastlines()
    plt.setp(plt.gca(), title='Difference')
    return posterior_pr90_data

pr90_preds = pr90GP()


SECONDS_IN_YEAR = 60*60*24*365 #s
convert = lambda x: x * SECONDS_IN_YEAR * 1e-12 # kg -> Gt

AREA_of_EARTH = 510.1 * 1e6 #million km²

inputs = glob(data_path + "inputs_s*.nc")

def global_mean(ds):
    weights = np.cos(np.deg2rad(ds.latitude))
    return ds.weighted(weights).mean(['latitude', 'longitude'])

def global_sum(ds):
    weights = np.cos(np.deg2rad(ds.latitude))
    return ds.weighted(weights).sum(['latitude', 'longitude'])




all_inputs = glob(data_path + "inputs_*.nc")
global_means = {}

def global_total(da):
    if 'latitude' in da.coords:
        if da.name in ['CO2', 'CH4', 'tas']:
            return global_mean(da)
        else:
            return convert(global_sum(da*AREA_of_EARTH*1000*100))
    else:
        return da

for inp in glob(data_path + "inputs_*.nc"):
    label=inp.split('_')[1][:-3]
    X = xr.open_dataset(inp)
    Y = xr.open_dataset(data_path + f"outputs_{label}.nc")
#     print(X.coords)
    print(label)
#     if label == "hist-aer":
#         X = X.rename_vars({"CO4": "CO2"})
    if 'lat' in X.coords:
        X = X.rename({'lat': 'latitude', 'lon': 'longitude'})
    if 'lat' in Y.coords:
        Y = Y.rename({'lat': 'latitude', 'lon': 'longitude'})

    if label == "abrupt-4xCO2":
        X = X.sel(time=slice(None, None, 5))
    X['tas'] = Y['tas'].mean('member')
    
    global_means[label] = X.map(global_total).to_pandas()


tas_preds.name = 'tas'
dtr_preds.name = 'diurnal_temperature_range'
pr_preds.name = 'pr'
pr90_preds.name = 'pr90'


X = xr.open_mfdataset([data_path + 'inputs_historical.nc', data_path + 'inputs_ssp245.nc']).compute()
Y = xr.open_dataset(data_path + 'outputs_ssp245.nc')

# Convert the precip values to mm/day
Y["pr"] *= 86400
Y["pr90"] *= 86400

gp_predictions = xr.merge([tas_preds, dtr_preds, pr_preds, pr90_preds])


variables = ['tas', 'diurnal_temperature_range', 'pr', 'pr90']
models = [gp_predictions, Y.mean('member')]
model_labels = ['Gaussian Process', 'NorESM2']
labels = ["Temperature (K)", "Diurnal temperature range (K)", "Precipitation (mm/day)", "Extreme precipitation (mm/day)"]
kwargs = [dict(cmap="coolwarm", vmax=6), dict(cmap="coolwarm", vmin=-2, vmax=2), dict(cmap="BrBG", vmin=-4, vmax=4), dict(cmap="BrBG", vmin=-8, vmax=8)]

proj = ccrs.PlateCarree()




with sns.plotting_context("talk"):
    fig, axes = plt.subplots(4, 2, subplot_kw=dict(projection=proj), figsize=(24, 18), constrained_layout=True)
    print(axes)
    for model_axes, var, label, kws in zip(axes, variables, labels, kwargs):
        for ax, model, model_label in zip(model_axes, models, model_labels):
            ax.set_title(model_label)
            if model_label == 'NorESM2':
                model[var].sel(time=slice(2080, 2100)).mean(['time']).plot(ax=ax, add_labels=False, transform=ccrs.PlateCarree(), cbar_kwargs={"label":label, "orientation":'vertical'}, **kws)
            else: 
                model[var].sel(time=slice(2080, 2100)).mean('time').plot(ax=ax, add_labels=False, transform=ccrs.PlateCarree(), add_colorbar=False, **kws)
            ax.coastlines()

    plt.savefig("GP_Nor.png")



def ttest_rel_from_stats(diff_mean, diff_std, diff_num):
    """
    Calculates the T-test for the means of *two independent* samples of scores.

    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.

    It is deliberately similar in interface to the other scipy.stats.ttest_... routines

    See e.g. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html
    and pg. 140 in Statistical methods in Atmos Sciences
    
    :param diff: The mean difference, x_d (|x1 - x1| == |x1| - |x2|)
    :param diff_std: The standard deviation in the difference, s_d (sqrt(Var[x_d]))
    :param diff_num: The number of points, n (n == n1 == n2)
    :return float, float: t-statistic, p-value
    """
    from scipy.stats import distributions

    z = diff_mean / np.sqrt(diff_std ** 2 / diff_num)
    # use np.abs to get upper tail, then multiply by two as this is a two-tailed test
    p = distributions.t.sf(np.abs(z), diff_num - 1) * 2
    return z, p

p_level = 0.05





kwargs = [dict(vmin=-1, vmax=1), dict(vmin=-0.5, vmax=0.5), dict(vmin=-1, vmax=1), dict(vmin=-2, vmax=2)]
with sns.plotting_context("talk"):
    fig, ax = plt.subplots(4, 1, subplot_kw=dict(projection=proj), figsize=(24, 18), constrained_layout=True)
    print(ax)
    model = models[0]
    model_label = model_labels[0]
    count = 0
    for var, label, kws in zip(variables, labels, kwargs):
        print(var, label, model_label)
        ax[count].set_title(model_label)
        diff = (model[var]-models[1][var]).sel(time=slice(2080, 2100)) # /models[-1][var]
        mean_diff = diff.mean('time')
        _, p = ttest_rel_from_stats(mean_diff, diff.std('time'), diff.count('time'))
        if model_label == 'Gaussian Process':
            mean_diff.where(p < p_level).plot(cmap="coolwarm", ax=ax[count], add_labels=False, transform=ccrs.PlateCarree(), cbar_kwargs={"label":label, "orientation":'vertical'}, **kws)
        else:
            mean_diff.where(p < p_level).plot(cmap="coolwarm", ax=ax[count], add_labels=False, transform=ccrs.PlateCarree(), add_colorbar=False, **kws)
            
        ax[count].coastlines()
        count += 1
    
    plt.savefig("GP_Diff.png")


def global_mean(ds):
    if 'lat' not in ds.coords:
        ds_ = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    else:
        ds_ = ds
    weights = np.cos(np.deg2rad(ds_.lat))
    return ds_.weighted(weights).mean(['lat', 'lon'])

weights = np.cos(np.deg2rad(Y['tas'].lat)).expand_dims(lon=144).assign_coords(lon=Y.lon)


R2E = pd.DataFrame({
    label: {variable: rmse( global_mean(Y.mean('member')[variable].sel(time=slice(2080, None))), 
                                global_mean(model[variable].sel(time=slice(2080, None)))).data/ np.abs(global_mean(Y.mean('member')[variable].sel(time=slice(2080, None)).mean('time')).data) for variable in variables} 
#                                 global_mean(model[variable].sel(time=slice(2080, None)))).data for variable in variables} 
                           for label, model in zip(model_labels[0:2], models[0:1])
})
R2E.T.round(3).style.highlight_min(subset=slice("Random Forest", None), axis = 0, props='font-weight: bold').format("{:.4f}")


NRMSE = pd.DataFrame({
    label: {variable: rmse(Y.mean('member')[variable].sel(time=slice(2080, None)).mean('time'), 
                               model[variable].sel(time=slice(2080, None)).mean('time'), weights=weights).data/ np.abs(global_mean(Y.mean('member')[variable].sel(time=slice(2080, None)).mean('time')).data) for variable in variables} 
    for label, model in zip(model_labels[:-1], models[:-1])
})
NRMSE.T.round(3).style.highlight_min(subset=slice("Random Forest", None), axis = 0, props='font-weight: bold').format("{:.4f}")


(NRMSE+5*R2E).T.round(3).style.highlight_min(subset=slice("Random Forest", None), axis = 0, props='font-weight: bold').format("{:.4f}")


combined_df = pd.concat([NRMSE, R2E, NRMSE+5*R2E], keys=['Spatial', 'Global', 'Total']).T
combined_df

dfi.export(combined_df, 'GP_NRMSE.png',table_conversion='matplotlib')




