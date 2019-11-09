import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# from owslib.wms import WebMapService
# wms = WebMapService('http://labs.metacarta.com/wms/vmap0')


def plot_map():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()

    ax.add_wms(wms='http://labs.metacarta.com/wms/vmap0',
               layers=['priroad'])

    plt.show()
    return ax


if __name__ == '__main__':
    plot_map()