import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def plot_map():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()

    ax.add_wms(wms='http://vmap0.tiles.osgeo.org/wms/vmap0',
               layers=['basic'])

    plt.show()
    return ax


if __name__ == '__main__':
    plot_map()