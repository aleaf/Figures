import sys
sys.path.append('D:/ATLData/Documents/GitHub/Figures/')
import rasterio
from Figures import basemap

def test_init():
    tick_interval = 0.6
    Map = basemap(extent=(312629.9447000431, 5040393.68127088, 436629.9447000431, 5145393.68127088),
                  projection_shapefile='../Notebooks/data/WIMI16.shp', #epsg=26716,
                  #parallels=parallels, meridians=meridians,
                  subplots=(1,1), figsize=(15, 10),
                  tick_interval=tick_interval)
    Map.add_shapefile('../Notebooks/data/WIMI16.shp')
if __name__ == '__main__':
    test_init()
