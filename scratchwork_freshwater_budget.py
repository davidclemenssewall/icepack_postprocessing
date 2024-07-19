"""
Validate Icepack freshwater budget output.

"""

import xarray as xr
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

# Function for reading history
def load_icepack_hist(run_name, icepack_dirs_path, hist_filename=None,
                      sst_above_frz=True, volp=False,
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

    # Per category ice thickness
    ds['hin'] = ds['vicen'] / (ds['aicen'] + np.finfo(np.float64).eps)

    # Create mixed layer freezing point difference
    if sst_above_frz:
        ds['sst_above_frz'] = ds['sst'] - ds['Tf']
    
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

    # Add the run name as an attribute
    ds.attrs.update({'run_name': run_name})

    # Add pond volume per unit area
    if volp:
        if ('alvl' in ds.data_vars) and ('alvln' in ds.data_vars) and (
            'apnd' in ds.data_vars) and ('apndn' in ds.data_vars) and (
            'hpnd' in ds.data_vars) and ('hpndn' in ds.data_vars):
            ds['volp'] = ds['aice']*ds['alvl']*ds['apnd']*ds['hpnd']
            ds['volpn'] = ds['aicen']*ds['alvln']*ds['apndn']*ds['hpndn']
        else:
            raise RuntimeError("missing data variables needed for volp(n)")


    # Convert time axis to datetimeindex
    try:
        datetimeindex = ds.indexes['time'].to_datetimeindex()
        ds['time'] = datetimeindex
    except AttributeError:
        pass

    return ds

# Function for plotting single Icepack output
def plot_hist_var(ds, var_name, ni, ax, resample=None, cumulative=False,
                  mult=1):
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

    if df.shape[1] == 2:
        df.rename(columns={0: var_name}, inplace=True)
        label = ds.run_name + ' (' + str(ni) + ')'
        # Plot
        h = ax.plot('time', var_name, data=df, label=label)
    else:
        for col_name in df.columns:
            if col_name == 'time':
                continue
            label = ds.run_name + ' (' + str(ni) + ', ' + str(col_name) + ')'
            # Plot
            h = ax.plot(df['time'], df[col_name], label=label)

    return h

def plot_handler(run_plot_dict, var_names, hist_dict,
                 figsize=None, ax_font=14, lfont=10, xlim=None, resample=None,
                 cumulative=False, mult=1):
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
    
        # Axis labels
        if cumulative:
            ax.set_ylabel("cum. " + var_name, fontsize=ax_font)
        else:
            ax.set_ylabel(var_name, fontsize=ax_font)
        ax.grid()
        # Legend
        ax.legend(fontsize=lfont, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    
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
    df_out = ds.sel(ni=ni)[['flpnd', 'expnd', 'frpnd', 'rfpnd', 'ilpnd']
                          ].to_pandas()
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

# Load history output
ip_dirs_path = "/home/dcsewall/code/docker_icepack_interactive/icepack-dirs"
trcr_dict = {17: 'alvl',
             18: 'vlvl',
             19: 'apnd_t',
             20: 'hpnd_t',
             21: 'ipnd_t'}
trcrn_dict = {17: 'alvln',
              18: 'vlvln',
              19: 'apndn_t',
              20: 'hpndn_t',
              21: 'ipndn_t'}
run_dict = {"pndhist": None,
            "pndhist_norfrac": None,
            "mos_raph_fyi_frshbud": None,
            "pnd_hyps": None,
            "mos_raph_fyi_pnd_allmods": None,
            }

hist_dict = {}
for key, value in run_dict.items():
    hist_dict[key] = load_icepack_hist(run_name=key, 
                                       icepack_dirs_path=ip_dirs_path, 
                                       hist_filename=value,
                                       volp=True, 
                                       trcr_dict=trcr_dict, 
                                       trcrn_dict=trcrn_dict)

# Check that the pond history variables match values in the tracer arrays
ds = hist_dict['pndhist']
print("apndn: " + str(ds['apndn'].equals(ds['apndn_t'])))
print("hpndn: " + str(ds['hpndn'].equals(ds['hpndn_t'])))
print("ipndn: " + str(ds['ipndn'].equals(ds['ipndn_t'])))
print("apnd: " + str(ds['apnd'].equals(ds['apnd_t'])))
print("hpnd: " + str(ds['hpnd'].equals(ds['hpnd_t'])))
print("ipnd: " + str(ds['ipnd'].equals(ds['ipnd_t'])))

def pnd_budget(ds):
    """creating total liquid water input, drainage, and discrepancy"""

    rhos = 330
    rhoi = 917
    rhofresh = 1000
    dt = (ds.time[1] - ds.time[0]).values.astype('timedelta64[s]').item(
        ).total_seconds()
    
    ds['liq_in'] = (ds['meltt']*rhoi + ds['melts']*rhos + ds['frain']*dt
                    )/rhofresh - ds['ilpnd'].where(ds['ilpnd']<0, 0)
    ds['liq_out'] = (ds['flpnd'] + ds['expnd'] + ds['frpnd'] + ds['rfpnd'] +
                     ds['ilpnd'].where(ds['ilpnd']>0, 0))

    ds['liq_diff'] = ds['liq_in'] - ds['liq_out']

    return ds

# compute pond budget
for key in list(hist_dict.keys()):
    hist_dict[key] = pnd_budget(hist_dict[key])
    hist_dict[key]['liq_strg_frac'] = hist_dict[key]['volp']/(
        hist_dict[key]['liq_in'].cumsum(dim='time') + np.finfo(np.float64).eps)

# Plot pond budget
f, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

for i in [0, 1, 2]:
    ni = i + 1
    plot_freshwater_budget(hist_dict['pndhist'], ni, axs[i])
    axs[i].legend(loc='lower left')
    axs[i].set_ylabel("meltwater (m), ni:" + str(ni))
xlim = [datetime.datetime.fromisoformat('2015-06-01'),
                    datetime.datetime.fromisoformat('2015-08-15')]
axs[-1].set_xlim(xlim)
f.suptitle("Icepack Default")
plt.show()

f, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

for i in [0, 1, 2]:
    ni = i + 1
    plot_freshwater_budget(hist_dict['pndhist_norfrac'], ni, axs[i])
    axs[i].legend(loc='lower left')
    axs[i].set_ylabel("meltwater (m), ni:" + str(ni))
xlim = [datetime.datetime.fromisoformat('2015-06-01'),
                    datetime.datetime.fromisoformat('2015-08-15')]
axs[-1].set_xlim(xlim)
f.suptitle("rfracmin=1")
plt.show()

f, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

for i in [0, 1, 2]:
    ni = i + 1
    plot_freshwater_budget(hist_dict['mos_raph_fyi_frshbud'], ni, axs[i])
    axs[i].legend(loc='lower left')
    axs[i].set_ylabel("meltwater (m), ni:" + str(ni))
xlim = [datetime.datetime.fromisoformat('2020-05-15'),
                    datetime.datetime.fromisoformat('2020-08-15')]
axs[-1].set_xlim(xlim)
f.suptitle("MOSAiC level fyi")
plt.show()

f, axs = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
ni = 2
plot_freshwater_budget(hist_dict['mos_raph_fyi_frshbud'], ni, axs)
axs.legend(loc='lower left')
axs.set_ylabel("meltwater (m), ni:" + str(ni))
xlim = [datetime.datetime.fromisoformat('2020-05-15'),
                    datetime.datetime.fromisoformat('2020-08-15')]
axs.set_xlim(xlim)
f.suptitle("MOSAiC level fyi")
plt.show()

f, axs = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
ni = 2
plot_freshwater_budget(hist_dict["mos_raph_fyi_pnd_allmods"], ni, axs)
axs.legend(loc='lower left')
axs.set_ylabel("meltwater (m), ni:" + str(ni))
xlim = [datetime.datetime.fromisoformat('2020-05-15'),
                    datetime.datetime.fromisoformat('2020-08-15')]
axs.set_xlim(xlim)
f.suptitle("MOSAiC level fyi (sea level ponds)")
plt.show()

f, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

for i in [0, 1, 2]:
    ni = i + 1
    plot_freshwater_budget(hist_dict['pnd_hyps'], ni, axs[i])
    axs[i].legend(loc='lower left')
    axs[i].set_ylabel("meltwater (m), ni:" + str(ni))
xlim = [datetime.datetime.fromisoformat('2015-06-01'),
                    datetime.datetime.fromisoformat('2015-08-15')]
axs[-1].set_xlim(xlim)
f.suptitle("Hypsometric ponds")
plt.show()

# Plot variables
run_plot_dict = {"pndhist": [3],
                 "pndhist_norfrac": [3],
                }
var_names = ['aice', 'alvl','apnd', 'hpnd', 'ipnd', 'volp', 'liq_strg_frac']
xlim = [datetime.datetime.fromisoformat('2015-06-01'),
                  datetime.datetime.fromisoformat('2015-08-15')]

f, axs = plot_handler(run_plot_dict, var_names, hist_dict, xlim=xlim)

# Cumulative meltwater fluxes
run_plot_dict = {"pndhist": [1,2,3,4],
                }
var_names = ['melts', 'meltt', 'frain', 'flpnd', 'expnd', 'frpnd', 'rfpnd',
             'ilpnd', 'liq_in', 'liq_out', 'liq_diff']
xlim = [datetime.datetime.fromisoformat('2015-06-01'),
                  datetime.datetime.fromisoformat('2015-08-15')]

f, axs = plot_handler(run_plot_dict, var_names, hist_dict, xlim=xlim,
                      cumulative=True)

# Plot variables
run_plot_dict = {"pndhist": [1],
                }
var_names = ['melts', 'meltt', 'meltb', 'aicen', 'hin', 'apndn', 'hpndn', 
             'ipndn', 'flpndn', 'expndn', 'frpndn']

f, axs = plot_handler(run_plot_dict, var_names, hist_dict, xlim=xlim)
plt.show()

# Plot variables
run_plot_dict = {"pndhist_norfrac": [1],
                }
var_names = ['aicen', 'hin', 'apndn', 'hpndn', 'ipndn', 'frpndn']

f, axs = plot_handler(run_plot_dict, var_names, hist_dict, xlim=xlim)
plt.show()

# Plot variables
run_plot_dict = {"mos_raph_fyi_frshbud": [1, 2],
                }
var_names = ['vice', 'vsno', 'apnd', 'hpnd', 'ipnd', 'flpnd', 'expnd', 'frpnd']
xlim = [datetime.datetime.fromisoformat('2020-05-15'),
                  datetime.datetime.fromisoformat('2020-07-28')]
f, axs = plot_handler(run_plot_dict, var_names, hist_dict, xlim=xlim)


