def nino34_index(sst, clim_start, clim_end):
    nino34_region = sst.sel(lat=slice(-5,5), lon=slice(190,240)) #may need to alter this depending on dataset, if condition?
    climatology = nino34_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).groupby('time.month').mean()
    monthly_anomalies = (nino34_region.groupby('time.month')-climatology).mean(dim=['lat','lon'])
    monthly_anomalies_rolling = monthly_anomalies.rolling(time=5).mean()
    sst_std = nino34_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).std()
    nino34_index = monthly_anomalies_rolling/sst_std
    return nino34_index

def nino3_index(sst, clim_start, clim_end):
    nino3_region = sst.sel(lat=slice(-5,5), lon=slice(210,270)) #may need to alter this depending on dataset, if condition?
    climatology = nino3_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).groupby('time.month').mean()
    monthly_anomalies = (nino3_region.groupby('time.month')-climatology).mean(dim=['lat','lon'])
    monthly_anomalies_rolling = monthly_anomalies.rolling(time=5).mean()
    sst_std = nino3_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).std()
    nino3_index = monthly_anomalies_rolling/sst_std
    return nino3_index


def nino4_index(sst, clim_start, clim_end):
    nino4_region = sst.sel(lat=slice(-5,5), lon=slice(160,210)) #may need to alter this depending on dataset, if condition?
    climatology = nino4_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).groupby('time.month').mean()
    monthly_anomalies = (nino4_region.groupby('time.month')-climatology).mean(dim=['lat','lon'])
    monthly_anomalies_rolling = monthly_anomalies.rolling(time=5).mean()
    sst_std = nino4_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).std()
    nino4_index = monthly_anomalies_rolling/sst_std
    return nino4_index


def seasonal_mean_nino_index(nino_index):
    seasonyear = (nino_index.time.dt.year + (nino_index.time.dt.month//12))
    nino_index.coords['seasonyear'] = seasonyear
    
    yearly_seasonal_nino_index = nino_index.groupby('seasonyear').apply(seasonal_mean)
    return yearly_seasonal_nino_index

def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
        carrs = np.concatenate(arrs, axis)
    
        offsets=[]
        start=0
        for i in range(len(arrs)-1):
            start += arrs[i].shape[axis]
            offsets.append(start)
    
        def helperfunc(a, *args, **kwargs):
            arrs = np.split(a, offsets)
            return func1d(*[*arrs, *args], **kwargs)
  
        return np.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)

def detrend_2step(data):
    import scipy
    from scipy import stats
    import numpy as np
    try:
        x = np.arange(data.size)
        R = scipy.stats.linregress(x[np.isfinite(data)], data[np.isfinite(data)])
        return data - x*R.slope - R.intercept
    except ValueError:
        return data

def change_dec_years(dec_data):
    dec_data_copy = dec_data.copy()
    years_plus1 = dec_data_copy.time.values +1
    del dec_data_copy['time']
    dec_data_copy.coords['time'] = years_plus1
    last_year = dec_data_copy.time[-1].values
    dec_data_new = dec_data_copy.drop_sel(time=last_year)
    return dec_data_new

def find_event_years(nino_index, threshold, duration):
    
    import climtas
    import numpy as np
    
    #find all times that the nino index exceeds the threshold for the specified duration (usually 5 or 6 months)
    el_nino_events = climtas.event.find_events(nino_index >= threshold, min_duration=duration)
    la_nina_events = climtas.event.find_events(nino_index <= -threshold, min_duration=duration)
    
    #select out the time index for el nino years 
    nevents_en = np.shape(el_nino_events.event_duration)[0]

    time_index_list_en = []

    event_duration_months_en = el_nino_events.event_duration.values
    event_duration_years_en = np.ceil((event_duration_months_en-(duration-1))/12)

    for i in np.arange(0,nevents_en):
        event_index_en = np.arange(el_nino_events.iloc[i].time, el_nino_events.iloc[i].time+(event_duration_years_en[i]*12), 12)
        time_index_list_en.append(event_index_en)
    
    time_index_en = np.concatenate(time_index_list_en)
    
    #select out the time index for la nina years 
    nevents_ln = np.shape(la_nina_events.event_duration)[0]

    time_index_list_ln = []

    event_duration_months_ln = la_nina_events.event_duration.values
    event_duration_years_ln = np.ceil((event_duration_months_ln-(duration-1))/12)

    for j in np.arange(0,nevents_ln):
        event_index_ln = np.arange(la_nina_events.iloc[j].time, la_nina_events.iloc[j].time+(event_duration_years_ln[j]*12), 12)
        time_index_list_ln.append(event_index_ln)
    
    time_index_ln = np.concatenate(time_index_list_ln)
    
    #convert time index arrays from float to int
    time_index_en_int = time_index_en.astype(int)
    time_index_ln_int = time_index_ln.astype(int)
    
    #use time index to find event dates in datetime 
    nino_index_el_nino_years = nino_index.isel(time=time_index_en_int)
    nino_index_la_nina_years = nino_index.isel(time=time_index_ln_int)
    
    #select out just the year 
    el_nino_years = nino_index_el_nino_years.time.dt.year
    la_nina_years = nino_index_la_nina_years.time.dt.year
    
    return el_nino_years, la_nina_years

def seasonyear(data):
    seasonyear = (data.time.dt.year + (data.time.dt.month//12))
    data.coords['seasonyear'] = seasonyear
    return data 

def seasonal_mean(data):
    return data.groupby('time.season').mean()

def seasonal_max(data):
    return data.groupby('time.season').max()

def seasonal_min(data):
    return data.groupby('time.season').min()

def sign_corr(a,b):
    """
    Assesses if two input arrays a and b have the same sign at each gridpoint
    Returns array of integers from 0 to 4
    
    *input arrays must be on the same grid 
    """
    
    import numpy as np
    indicator_array = np.empty(shape=np.shape(a))
    for i in np.arange(0,np.shape(a)[0]):
        for j in np.arange(0,np.shape(a)[1]):
            if a.isel(lat=i).isel(lon=j) > 0 and b.isel(lat=i).isel(lon=j) > 0:
                indicator_array[i][j]=0
            elif a.isel(lat=i).isel(lon=j) < 0 and b.isel(lat=i).isel(lon=j) < 0:
                indicator_array[i][j]=1
            elif a.isel(lat=i).isel(lon=j) > 0 and b.isel(lat=i).isel(lon=j) < 0:
                indicator_array[i][j]=2
            elif a.isel(lat=i).isel(lon=j) < 0 and b.isel(lat=i).isel(lon=j) > 0:
                indicator_array[i][j]=3
            else: indicator_array[i][j]=4
    return indicator_array

def sign_corr_with_threshold(a,b,c,d,th):
    """
    a and b are input arrays to compare
    c and d are data arrays to get standard deviation 
    th is to set the threshold value of the amount of standard deviations 
    
    *need to make sure input arrays are the same size
    """

    import numpy as np
    indicator_array = np.empty(shape=np.shape(a))
    for i in np.arange(0,np.shape(a)[0]):
        for j in np.arange(0,np.shape(a)[1]):
            if a.isel(lat=i).isel(lon=j) > th*c.isel(lat=i).isel(lon=j).std() and b.isel(lat=i).isel(lon=j) > th*d.isel(lat=i).isel(lon=j).std():
                indicator_array[i][j]=0
            elif a.isel(lat=i).isel(lon=j) < th*-c.isel(lat=i).isel(lon=j).std() and b.isel(lat=i).isel(lon=j) < th*-d.isel(lat=i).isel(lon=j).std():
                indicator_array[i][j]=1
            elif a.isel(lat=i).isel(lon=j) > th*c.isel(lat=i).isel(lon=j).std() and b.isel(lat=i).isel(lon=j) < th*-d.isel(lat=i).isel(lon=j).std():
                indicator_array[i][j]=2
            elif a.isel(lat=i).isel(lon=j) < th*-c.isel(lat=i).isel(lon=j).std() and b.isel(lat=i).isel(lon=j) > th*d.isel(lat=i).isel(lon=j).std():
                indicator_array[i][j]=3
            else: indicator_array[i][j]=4
    return indicator_array

def detrend_by_month(data):
    import numpy as np
    import xarray as xr
    detrended_data = np.apply_along_axis(detrend_2step, data.get_axis_num('time'), data)
    return xr.DataArray(detrended_data, data.coords)

def detrend_by_month_ufunc(data):
    import xarray as xr
    detrended_data = xr.apply_ufunc(detrend_2step, data, 
                             input_core_dims=[['time']],
                             output_core_dims=[['time']],
                             vectorize=True,
                             dask='parallelized')

    return detrended_data.transpose(*data.dims)

def regress_nino(data, nino):
    """
    Returns the linear regression at each gridpoint of 'data' against 'nino'
    """
    
    import numpy as np
    import scipy.stats
    import xarray as xr
    
    # Get the nino values at the dates matching 'data'
    # nino = nino.sel(time=data.time)
    
    # Function to apply on each gridpoint
    def regress_gridpoint(data):
        return scipy.stats.linregress(nino, data)[0]
    
    # Apply the function on each gridpoint
    regression = np.apply_along_axis(regress_gridpoint, data.get_axis_num('time'), data)
    
    # This is just to get the correct coordinates for the output
    sample = data.mean('time')
    
    # Convert the numpy array back into xarray
    return xr.DataArray(regression, sample.coords)

def regress_nino_by_month(data, nino):
    """
    Runs 'regress_nino' on each month separately.
    If quarterly or seasonal averages are inputted, will run on each quarter or season seperately. 
    """
    
    return data.groupby('time.month').map(regress_nino, nino=nino)

def correlate_nino(data, nino):
    """
    Returns the Pearson correlation coefficient at each gridpoint of 'data' against 'nino'
    """
    
    import numpy as np
    import scipy.stats
    import xarray as xr
    
    # Get the nino values at the dates matching 'data'
    # nino = nino.sel(time=data.time)
    
    # Function to apply on each gridpoint
    def correlate_gridpoint(data):
        return scipy.stats.pearsonr(nino, data)[0]
    
    # Apply the function on each gridpoint
    pearsonr = np.apply_along_axis(correlate_gridpoint, data.get_axis_num('time'), data)
    
    # This is just to get the correct coordinates for the output
    sample = data.mean('time')
    
    # Convert the numpy array back into xarray
    return xr.DataArray(pearsonr, sample.coords)

def correlate_nino_by_month(data, nino):
    """
    Runs 'correlate_nino' on each month separately.
    If quarterly or seasonal averages are inputted, will run on each quarter or season seperately. 
    """
    
    return data.groupby('time.month').map(correlate_nino, nino=nino)


def read_in_cmip_models(directory, ensemble, var, start_date, end_date):
    """
    Reads in all files from mandy's CMIP collection where file path is structured as var_model_ensemble_year.nc 
    Files randomly in directory, not organised by model name 
    """
    
    import os
    import xarray as xr
    import pandas as pd
    import re
    
    # Get all the files in the directory 
    all_files = os.listdir(f'{directory}')
    # Loop through all files and check to see if ensemble is in them 
    model_files = []
    for file in all_files:
        if ensemble in file:
            model_files.append(file)
    # Get the paths 
    paths = [f'/g/data/eg3/mf3225/CMIP_TS/CMIP6/historical/{f}' for f in model_files]
    # Add model names instead of number as model dimension 
    model_names = []
    for file in model_files:
        model_name = re.search(f'{var}_(.+?)_{ensemble}', file).group(1)
        model_names.append(model_name)
    # Make into a dictionary 
    models = {model_names[i]: paths[i] for i in range(len(model_names))}
    # Read in all the models into one data set 
    names = []
    ds = []

    for name, path in models.items():
        try:
            d = xr.open_mfdataset(path, combine='by_coords', chunks={'time':-1, 'lat':110, 'lon':110})
            if len(d['time'])==1980:
                del d['time']
                del d['time_bnds']
                time_month = pd.date_range(start=f'{start_date}',end = f'{end_date}', freq ='M')
                d.coords['time'] = time_month
                ds.append(d)
                names.append(name)
            else:
                print(f'Model {name} has weird time')
        except OSError:
            # No files read, move on to the next
            continue 

    multi_model = xr.concat(ds, dim='model', coords='minimal')
    multi_model.coords['model'] = names
    return multi_model


def zonal_std(data):
    '''
    Takes three dimensional surface temperature xarray (time, lat, lon) and returns one dimensional np array of
    zonal standrd deviation from 160E to 280E avereged over 5S to 5N (region is set and not passed in as an arg)
    '''
    import numpy as np
    region = data.sel(lat=slice(-5,5)).sel(lon=slice(160, 280))
    std = np.apply_along_axis(np.std, region.get_axis_num('time'), region)
    zonal_mean = np.mean(std, 0) # 0 should refer to lat axis (but good to check) 
    return zonal_mean


def sst_comp(nino34, sst, season):
    '''
    Takes nino34 index, sst data and returns El Nino and La Nina sst composites for specified season
    '''
    # Calculate ENSO years with nino34 index
    el_nino_years, la_nina_years = find_event_years(nino34, 0.4, 6)
    # Offset years for djf and mam
    el_nino_years_offset = el_nino_years +1
    la_nina_years_offset = la_nina_years +1
    # Add season year axis to sst data 
    seasonyear(sst)
    # Calculate seasonal means 
    seasonal_sst = sst.groupby('seasonyear').apply(seasonal_mean)
    # Select out each season
    sst_season = seasonal_sst.sel(season=f'{season}')
    # If statement to determine whether to use offset
    if season == 'JJA' or season == 'SON':
        # Select El Nino years 
        sst_el_nino = sst_season.sel(seasonyear=el_nino_years) - sst_season.mean(dim='seasonyear')
        # Select La Nina years 
        sst_la_nina = sst_season.sel(seasonyear=la_nina_years) - sst_season.mean(dim='seasonyear')
    else:
        # Use offset  
        sst_el_nino = sst_season.sel(seasonyear=el_nino_years_offset) - sst_season.mean(dim='seasonyear')
        sst_la_nina = sst_season.sel(seasonyear=la_nina_years_offset) - sst_season.mean(dim='seasonyear')
    # Return average pattern by taking mean over time 
    return sst_el_nino.mean(dim='time'), sst_la_nina.mean(dim='time')


# make a function to return eofs, pcs and variance fraction 
def eof_and_pcs(data, n):
    import numpy as np
    import xarray as xr
    from eofs.xarray import Eof as Eof_xr
    from eofs.standard import Eof as Eof_np
    # xarray solver for eofs
    solver = Eof_xr(data)
    eofs = solver.eofsAsCorrelation(neofs=n)
    # numpy solver for pcs (need to convert to np due to bug in xarray interface for pcs)
    data_np = data.to_numpy()
    times = data.time.values
    solver_np = Eof_np(data_np)
    pcs = solver_np.pcs(npcs=n)
    pcs_xr = xr.DataArray(pcs, dims=['time', 'mode'], coords=dict(time=times, mode=np.arange(1,n+1)))
    # varianve fractions 
    variance_fractions = solver.varianceFraction(neigs=n)
    return eofs, pcs_xr, variance_fractions


