# modules
import xarray as xr
import f90nml as nml
import cmocean
from cftime import num2date

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


def main(config, file, var_name, fig_name, infos, freq):

    # read file
    try:
        ds = xr.open_dataset(config+file)
    except:
        return

    print(f'   Plotting {var_name}')

    # get time and depth
    time_counter = ds.time_counter.values
    try:
        dpt = ds.deptht.values
    except Exception as e0:
        try:
            dpt = ds.depthu.values
        except Exception as e1:
            dpt = ds.depthv.values

    # get field values
    var_val = getattr(ds,var_name).values
    var_val = var_val[:,:,0,0]
    var_val = var_val.transpose()

    # plot
    plotpath = fig_name + '_' + config +'_' + freq + '.png'
    make_plot(var_val,time_counter,dpt,infos,plotpath)



if __name__=="__main__":

    # Config name
    # -----------
    try:
        namelist = nml.read('namelist_cfg')
        config = namelist['namrun']['cn_exp']
    except:
        config = 'C1D_PAPA32.W25ANN'

    print(f'Figures for config {config}')

    # Field profiles
    # --------------
    # temperature
    infos = [ 'T (ºC)' , cmocean.cm.thermal , colors.Normalize(vmin=3.0, vmax=8.0), lambda x: x ]
    main( config=config, file='_1d_20100615_20110614_grid_T.nc' , var_name='votemper' , fig_name='T' , infos=infos , freq='1d' )

    # salinity
    infos = [ 'S (psu)' , cmocean.cm.haline , colors.Normalize(vmin=32, vmax=34), lambda x: x ]
    main( config=config, file='_1d_20100615_20110614_grid_T.nc' , var_name='vosaline' , fig_name='S' , infos=infos , freq='1d' )

    # U
    infos = [ 'Velocity U (m/s)' , cmocean.cm.balance , colors.Normalize(vmin=-0.15, vmax=0.15), lambda x: x ]
    main( config=config, file='_1d_20100615_20110614_grid_U.nc' , var_name='uo' , fig_name='u' , infos=infos , freq='1d' )

    # V
    infos = [ 'Velocity V (m/s)' , cmocean.cm.balance , colors.Normalize(vmin=-0.15, vmax=0.15), lambda x: x ]
    main( config=config, file='_1d_20100615_20110614_grid_V.nc' , var_name='vo' , fig_name='v' , infos=infos , freq='1d' )
