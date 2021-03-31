def nino34_index(sst, clim_start, clim_end):
    nino34_region = sst.sel(lat=slice(5,-5), lon=slice(190,240)) #may need to alter this depending on dataset, if condition?
    climatology = nino34_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).groupby('time.month').mean()
    monthly_anomalies = (nino34_region.groupby('time.month')-climatology).mean(dim=['lat','lon'])
    monthly_anomalies_rolling = monthly_anomalies.rolling(time=5).mean()
    sst_std = nino34_region.sel(time=slice(f'{clim_start}',f'{clim_end}')).std()
    nino34_index = monthly_anomalies_rolling/sst_std
    return nino34_index

def seasonal_mean_nino_index(nino_index):
    seasonyear = (nino_index.time.dt.year + (nino_index.time.dt.month//12))
    nino_index.coords['seasonyear'] = seasonyear
    
    def seasonal_mean(data):
        return data.groupby('time.season').mean()
    
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

def detrend_2step(sst):
    import scipy
    from scipy import stats
    import numpy as np
    try:
        x = np.arange(sst.size)
        R = scipy.stats.linregress(x[np.isfinite(sst)], sst[np.isfinite(sst)])
        return sst - x*R.slope - R.intercept
    except ValueError:
        return sst
    
def change_dec_years(dec_data):
    dec_data_copy = dec_data.copy()
    years_plus1 = dec_data_copy.time.values +1
    del dec_data_copy['time']
    dec_data_copy.coords['time'] = years_plus1
    last_year = dec_data_copy.time[-1].values
    dec_data_new = dec_data_copy.drop_sel(time=last_year)
    return dec_data_new
