def read_hist_models(institution_dir, variable_dir):
    
    import xarray as xr  
    import os
    import pandas as pd
    
    institutions = os.listdir(institution_dir)
    
    models = {}

    for institution in institutions:
        model_list = os.listdir(f'{institution_dir}{institution}')
        for model in model_list:
            # check if the historical model with the right variable exists and if so save the version number for the file
            if os.path.exists(f'{institution_dir}{institution}/{model}{variable_dir}'):
                version = os.listdir(f'{institution_dir}{institution}/{model}{variable_dir}')
                # for each version, call model_path to make the path and then store with the model in a dictionary 'models'
                for v in version:
                    path = f'{institution_dir}{institution}/{model}{variable_dir}{v}/*.nc'
                    models[model] = path
    
    names = []
    ds = []

    for name, path in models.items():
        try:
            d = xr.open_mfdataset(path, combine='by_coords')
            if len(d['time'])==1980:
                del d['time']
                del d['time_bnds']
                time_month = pd.date_range(start='1850-01',end = '2015-01', freq ='M')
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