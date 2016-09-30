
def test_map_imports():
    from mpl_toolkits.basemap import Basemap, pyproj
    import fiona
    from fiona.crs import to_string, from_epsg
    from shapely.ops import transform
    from shapely.affinity import translate
    from descartes import PolygonPatch
    from GISio import get_proj4, shp2df
    from GISops import project, projectdf

def test_map_init():
    from Figures import basemap

    extent = [500000, 2290000, 680000, 2560000]
    projection_shapefile='../Notebooks/data/hrus.shp'
    parallels=[43,44,45, 46]
    meridians=[-89,-88,-87]
    tick_interval=1
    Map = basemap(extent=extent, projection_shapefile=projection_shapefile,
                      parallels=parallels, meridians=meridians,
                      subplots=(1,2),
                      figsize=(8.5, 11),
                      tick_interval=tick_interval)

if __name__ == '__main__':
    test_map_imports()
    test_map_init()