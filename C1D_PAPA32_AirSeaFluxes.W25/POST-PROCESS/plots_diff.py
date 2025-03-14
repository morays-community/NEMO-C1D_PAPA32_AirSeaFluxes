# modules
import numpy as np
import xarray as xr
import f90nml as nml
import cmocean

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
matplotlib.use('Agg')

def make_plot(data,time_counter,depth,infos,output):
    # unpack args
    title, cmap, norm, tfs = infos
    data = tfs(data)
    # format time - isolate the 15th of each month
    idx = [i for i, t in enumerate(time_counter) if t.day == 15]
    time = [t.strftime("%Y-%m-%d") for t in time_counter]
    dates_counter = [time_counter[i] for i in idx]
    dates = [t.strftime("%Y-%m") for t in dates_counter]
    # figure
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    # color map
    pcm = ax.pcolormesh(time, depth, data, cmap=cmap, norm=norm)
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05, shrink=0.5)
    plt.title(title, fontsize=16)
    ax.invert_yaxis()
    ax.set_ylabel("Depth (m)", fontsize=14)
    ax.set_xticks(idx)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=14)
    # write fig
    plt.savefig(output, bbox_inches='tight')
    plt.close()


def main(filepath_ref, ref_name, config, file, var_name, fig_name, infos, freq):

    print(f'   Plotting difference for {var_name}')

    # read files
    try:
        ds = xr.open_dataset(config+file)
        ds_ref = xr.open_dataset(filepath_ref)
    except:
        return

    # get time and depth
    time_counter = ds.time_counter.values
    try:
        dpt = ds.deptht.values
    except Exception as e0:
        try:
            dpt = ds.depthu.values
        except Exception as e1:
            dpt = ds.depthv.values

    # get fields values
    var_ref = getattr(ds_ref,ref_name).values
    var_ref = var_ref[:,:,0,0]
    var_ref = var_ref.transpose()
    var = getattr(ds,var_name).values
    var = var[:,:,0,0]
    var = var.transpose()
    diff_var = var[:,:var_ref.shape[1]] - var_ref[:,:]
    time_counter = time_counter[:var_ref.shape[1]]

    # plot
    plotpath = fig_name + '_error_' + config + '_' + freq + '.png'
    make_plot(diff_var,time_counter,dpt,infos,plotpath)



if __name__=="__main__":

    # Config name
    # -----------
    try:
        namelist = nml.read('namelist_cfg')
        config = namelist['namrun']['cn_exp']
    except:
        config = 'C1D_PAPA32.W25ANN'

    print(f'Figures for config {config}')

    # temperature profiles error
    infos = [ 'T - T_OBS (ºC)' , cmocean.cm.balance , colors.Normalize(vmin=-1.5, vmax=1.5), lambda x: x ]
    main( filepath_ref='OSP32_obs_T.nc' , ref_name='T_20', config=config, file='_1d_20100615_20110614_grid_T.nc' , var_name='votemper' , fig_name='T' , infos=infos , freq='1d' )

    # salinity profiles error
    infos = [ 'S - S_OBS (psu)' , cmocean.cm.balance , colors.Normalize(vmin=-0.2, vmax=0.2), lambda x: x ]
    main( filepath_ref='OSP32_obs_S.nc' , ref_name='S_41', config=config, file='_1d_20100615_20110614_grid_T.nc' , var_name='vosaline' , fig_name='S' , infos=infos , freq='1d' )
