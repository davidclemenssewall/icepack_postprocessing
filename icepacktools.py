"""Tools for examining Icepack output in python

@author: David Clemens-Sewall, NSF NCAR, NOAA
"""

import xarray as xr
import os
import matplotlib.pyplot as plt
import pandas as pd

# Function for reading history
def load_icepack_hist(run_name, icepack_dirs_path, hist_filename=None,
                      sst_above_frz=True, volp=False, snhf=False,
                      pnd_budget=False, compute_albedo_ai=False,
                      trcr_dict=None, trcrn_dict=None):
    """
    Load Icepack history output
    
    Parameters
    ----------
    run_name : str
        Name of the icepack run (directory name in RUN_DIR)
    icepack_dirs_path : str
        Path to root of icepack directory.
    hist_filename : str or None, optional
        Name of specific history file to load. If None load the first file 
        in history directory. Default is None.
    sst_above_frz : bool, optional
        Whether or not to compute the difference between mixed layer freezing 
        point and temperature. Default is True.
    volp : bool, optional
        Whether or not to compute the pond volume per grid cell area (units m).
        requires alvl and alvln tracers. Default is False.
    snhf : bool, optional
        Whether or not to compute the net surface heat flux, which is defined
        as: snhf = flw + flwout + fsens + flat + fswabs. Negative values
        are net flux from the ice to the atmosphere. Default is False.
    pnd_budget : bool, optional
        Whether or not to compute the pond budget, requires all meltwater flux
        diagnostics to be present. The default is False.
    compute_albedo_ai : bool, optional
        Whether or not to compute the albedo over the ice-covered area. The
        default is False. 
    trcr_dict : dict, optional
        Dict for which tracers to convert to data variables. Keys are tracer
        indices and values are names. Default is None.
    trcrn_dict : dict, optional
        Dict for which category tracers to convert to data variables. Keys 
        are tracer indices and values are names. Default is None.

    Returns
    -------
    xarray dataset with Icepack history output

    """

    # Open netCDF
    hist_path = os.path.join(icepack_dirs_path, "runs", run_name, "history")
    if hist_filename is None:
        hist_filename = os.listdir(hist_path)[0]
    ds = xr.open_dataset(os.path.join(hist_path, hist_filename))

    # Create mixed layer freezing point difference
    if sst_above_frz:
        ds['sst_above_frz'] = ds['sst'] - ds['Tf']

    # Create surface net heat flus
    if snhf:
        ds['snhf'] = (ds['flw'] + ds['flwout'] 
                      + ds['fsens'] + ds['flat']
                      + ds['fswabs'])
        
    # Compute albedo
    if compute_albedo_ai:
        ds['albedo_ai'] = (ds['fsw'] - ds['fswabs'])/ds['fsw']
     
    # Copy trcr and trcrn data variables
    if trcr_dict is not None:
        for key, value in trcr_dict.items():
            da = ds['trcr'].sel(ntrcr=key)
            da.name = value
            ds[value] = da
    if trcrn_dict is not None:
        for key, value in trcrn_dict.items():
            da = ds['trcrn'].sel(ntrcr=key)
            da.name = value
            ds[value] = da

    # Add pond volume per unit area
    if volp:
        if ('alvl' in ds.data_vars) and ('alvln' in ds.data_vars) and (
            'apnd' in ds.data_vars) and ('apndn' in ds.data_vars) and (
            'hpnd' in ds.data_vars) and ('hpndn' in ds.data_vars):
            ds['volp'] = ds['aice']*ds['alvl']*ds['apnd']*ds['hpnd']
            ds['volpn'] = ds['aicen']*ds['alvln']*ds['apndn']*ds['hpndn']
        else:
            raise RuntimeError("missing data variables needed for volp(n)")

    # Compute pond budget
    if pnd_budget:
        rhos = 330
        rhoi = 917
        rhofresh = 1000
        dt = (ds.time[1] - ds.time[0]).values.astype('timedelta64[s]').item(
            ).total_seconds()
        
        ds['liq_in'] = (ds['meltt']*rhoi + ds['melts']*rhos + ds['frain']*dt
                        )/rhofresh - ds['ilpnd'].where(ds['ilpnd']<0, 0)
        ds['liq_out'] = (ds['flpnd'] + ds['expnd'] + ds['frpnd'] + ds['rfpnd'] +
                        ds['ilpnd'].where(ds['ilpnd']>0, 0) + ds['mipnd'] 
                        + ds['rdpnd'])

        ds['liq_diff'] = ds['liq_in'] - ds['liq_out']
        ds['frshwtr_residual'] = ds['liq_diff'].cumsum('time') - ds['volp']

    # Convert time axis to datetimeindex
    try:
        datetimeindex = ds.indexes['time'].to_datetimeindex()
        ds['time'] = datetimeindex
    except AttributeError:
        pass
    
    # Add the run name as an attribute
    ds.attrs.update({'run_name': run_name})

    return ds

# Function for plotting single Icepack output
def plot_hist_var(ds, var_name, ni, ax, resample=None, cumulative=False,
                  mult=1, linestyle='-', color=None):
    """
    Plot a single variable from history DS on the given axis

    Parameters
    ----------
    ds : xarray DataSet
    var_name : str
    ni : int
        Which cell of the Icepack output to plot
    ax : matplotlib.pyplot.axes
        Axis object to plot on
    resample : str, optional
        If provided, frequency string for DataFrame.resample(). If None do not
        resample. The default is None.
    cumulative : bool, optional
        Whether the variable should be cumulative, useful for fluxes. The
        default is False.
    mult : float, optional
        Multiplier for values. Useful with cumulative to get the flux into
        correct units. The default is 1.
    linestyle : str, optional
        string specifying linestyle, the default is '-'
    color : str, optional
        color for the line, the default is None

    Returns
    -------
    handle for matplotlib plot object

    """

    # Get variable as Pandas DataFrame with time as a column
    df = ds[var_name].sel(ni=ni).to_pandas()
    if resample:
        df = df.resample(resample).mean()
    if cumulative:
        df = df.cumsum()
    df *= mult
    df = df.reset_index()

    # Display
    if df.shape[1] == 2:
        df.rename(columns={0: var_name}, inplace=True)
        label = ds.run_name + ' (' + str(ni) + ')'
        # Plot
        h = ax.plot('time', var_name, data=df, label=label, ls=linestyle, 
                    c=color)
    else:
        for col_name in df.columns:
            if col_name == 'time':
                continue
            label = ds.run_name + ' (' + str(ni) + ', ' + str(col_name) + ')'
            # Plot
            h = ax.plot(df['time'], df[col_name], label=label, ls=linestyle,
                        c=color)

    return h

def plot_forc_var(ds_forc, var_name, ax):
    """
    Plot a single forcing variable

    Parameters
    ----------
    ds : xarray DataSet
        MDF formatted
    var_name : str
    ax : matplotlib.pyplot.axes
        Axis object to plot on
    
    Returns
    -------
    handle for matplotlib plot object

    """

    # Get variable as Pandas DataFrame with time as a column
    df = ds_forc[var_name].to_pandas().reset_index()
    time_name = df.columns[0]
    df.rename(columns={time_name: 'time', 0: var_name}, inplace=True)
    # Plot
    h = ax.plot('time', var_name, data=df, linestyle=':', alpha=0.5)

    return h

def plot_ice_var(df_ice, var_name, site, ax, mean_only=False, linestyle=':',
                 color=None, semx2=False):
    """
    Plots the requested ice variable from an individual site
    
    Parameters
    ----------
    df_ice : Pandas dataframe
        A dataframe where the row index is site and date and the
        column index is var_name, and then mean and standard error of mean
    var_name : str
    site : str
    ax :
    mean_only : bool, optional
        Whether to just display mean of ice variable and not st. err. The 
        defaults is False.
    semx2 : bool, optional
        Whether to plot errorbars from semx2 column
    linestyle : str, optional
        string specifying linestyle, the default is '-'
    color : str, optional
        color for the line, the default is None

    """

    # Get dataframe with columns, time, mean, sem
    try:
        df = df_ice.loc[site, var_name].reset_index()
    except KeyError:
        return
    label = site + " " + var_name
    # plot
    if mean_only or df['sem'].isna().all():
        h = ax.plot('time', 'mean', data=df, linestyle=linestyle, label=label,
                    marker='o', c=color)
    else:
        if semx2:
            h = ax.errorbar('time', 'mean', yerr='semx2', data=df, capsize=5, 
                            linestyle=linestyle, label=label, c=color)
        else:
            h = ax.errorbar('time', 'mean', yerr='sem', data=df, capsize=5, 
                            linestyle=linestyle, label=label, c=color)


    return h

def plot_ice_varn(df_icen, var_name, site, ax):
    """
    Plots the requested category level ice variable from a site

    Parameters
    ----------
    df_icen : Pandas dataframe
        A dataframe where the row index is (site, date, category) and the
        column index is var_name
    var_name : str
    site : str
    ax :
    """

    handles = []
    # Get dataframe for just this site
    try:
        df = df_icen.loc[site].reset_index(level='time')
    except KeyError:
        return
    # Plot each category
    for nc in df.index:
        label = site + " " + var_name + " " + str(nc)
        try:
            handles.append(ax.plot('time', var_name, data=df.loc[nc], 
                                linestyle=':', label=label, marker='o'))
        except KeyError:
            return
    return handles

def plot_handler(run_plot_dict, var_names, hist_dict, forc_var_map={},
                 ds_forc=None, ice_var_map={}, ice_sites=[], df_ice=None,
                 df_icen=None,
                 figsize=None, ax_font=14, lfont=10, xlim=None,
                 mean_only=False, resample=None, cumulative=False,
                 mult=1):
    """
    Handler function for plotting different runs and variables

    Parameters
    ----------
    run_plot_dict : dict
        Dictionary where the keys are the names of the runs to plot and value
        is a list of the cells (ni) to plot
    var_names : iterable
        Variable names to plot
    hist_dict : dict
        Dict containing the Icepack output, keyed on run_name
    forc_var_map : dict, optional
        Dict keyed on variable name and values are lists of forcing variable
        names to compare with. Default is {}.
    ds_forc : xarray Dataset, optional
        MDF formatted forcing dataset. Default is None.
    ice_var_map : dict, optional
        Dict keyed on variable name and values are lists of ice df variable
        names to compare with. Default is {}.
    df_ice : Pandas dataframe, optional
        A dataframe where the row index is site and date and the
        column index is var_name, and then mean and standard error of mean
    df_icen : Pandas dataframe
        A dataframe where the row index is (site, date, category) and the
        column index is var_name
    mean_only : bool, optional
        Whether to just display mean of ice variable and not st. dev. The 
        defaults is False.
    resample : str, optional
        If provided, frequency string for DataFrame.resample(). If None do not
        resample. The default is None.
    cumulative : bool, optional
        Whether the variable should be cumulative, useful for fluxes. The
        default is False.
    mult : float, optional
        Multiplier for values. Useful with cumulative to get the flux into
        correct units. The default is 1.
    
    Returns
    -------
    Matplotlib figure object

    """

    # Create figsize
    if figsize is None:
        figsize = (10, 3*len(var_names))
    # Create figure and axes objects
    f, axs = plt.subplots(len(var_names), 1, sharex=True, figsize=figsize)

    # Loop through each variable
    for var_name, ax in zip(var_names, axs):
        # And through each run
        for run_name, nis in run_plot_dict.items():
            # and the desired cell(s) in each run
            for ni in nis:
                _ = plot_hist_var(hist_dict[run_name], var_name, ni, ax,
                                  resample=resample, cumulative=cumulative,
                                  mult=mult)
        
        # Plot forcing
        if var_name in forc_var_map:
            for forc_var_name in forc_var_map[var_name]:
                _ = plot_forc_var(ds_forc, forc_var_name, ax)
        
        # Plot ice variables
        for site in ice_sites:
            if var_name in ice_var_map:
                for ice_var_name in ice_var_map[var_name]:
                    if df_ice is not None:
                        _ = plot_ice_var(df_ice, ice_var_name, site, ax, 
                                        mean_only=mean_only)
                    if df_icen is not None:
                        _ = plot_ice_varn(df_icen, ice_var_name, site, ax)
                    del _
            
        # Axis labels
        ax.set_ylabel(var_name, fontsize=ax_font)
        ax.yaxis.set_tick_params(labelsize=lfont)
        ax.xaxis.set_tick_params(labelsize=lfont)
        ax.grid()
        # Legend
        ax.legend(fontsize=lfont, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    
    
    # xlimits on last plot
    if xlim is not None:
        axs[-1].set_xlim(xlim)

    #plt.show()
    return f, axs

def plot_freshwater_budget(ds, ni, ax):
    """Plot the freshwater budget as a stacked bar chart"""
    
    rhos = 330
    rhoi = 917
    rhofresh = 1000
    dt = (ds.time[1] - ds.time[0]).values.astype('timedelta64[s]').item(
        ).total_seconds()

    # Meltwater into ponds
    df_in = ds.sel(ni=ni)[['meltt', 'melts', 'frain', 'ilpnd']].to_pandas()
    df_in['ilpnd'] *= -1
    df_in['ilpnd'][df_in.ilpnd < 0] = 0
    df_in.drop(columns=['ni'], inplace=True)
    df_in['melts'] = df_in['melts'] * rhos/rhofresh
    df_in['meltt'] = df_in['meltt'] * rhoi/rhofresh
    df_in['frain'] *= dt
    df_in = df_in.cumsum()

    # Meltwater out of ponds
    df_out = ds.sel(ni=ni)[['flpnd', 'expnd', 'frpnd', 'rfpnd', 'ilpnd', 
                            'mipnd', 'rdpnd']].to_pandas()
    df_out['ilpnd'][df_out.ilpnd < 0] = 0
    df_out.drop(columns=['ni'], inplace=True)
    df_out *= -1
    df_out = df_out.cumsum()

    # Pond volume
    df_liq_diff = ds.sel(ni=ni)['liq_diff'].to_pandas().cumsum()
    df_volp = ds.sel(ni=ni)['volp'].to_pandas()

    df_in.plot.area(ax=ax)
    df_out.plot.area(ax=ax)
    df_liq_diff.plot.line(c='k', ax=ax, label='liq_diff')
    df_volp.plot.line(c='k', ls='--', ax=ax, label='volp')
    ax.set_ylim([df_out.iloc[-1].sum(), df_in.iloc[-1].sum()])
