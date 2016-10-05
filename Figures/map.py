import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from .report_figs import ReportFigures, Normalized_cmap
from .reference import getproj4

def errmsg(package):
    msg = 'This module needs basemap. If you are using Anaconda, '.format(package)
    msg += 'try:\n>conda config --add channels conda-forge\nthen:\n'
    msg += '>conda install {}\n'.format(package)
    print(msg)

try:
    from mpl_toolkits.basemap import Basemap, pyproj
except:
    errmsg('basemap')
try:
    import fiona
    from fiona.crs import to_string, from_epsg, from_string
except:
    errmsg('fiona')
try:
    from shapely.ops import transform
    from shapely.affinity import translate
except:
    errmsg('shapely')
try:
    from descartes import PolygonPatch
except:
    errmsg('descartes')
try:
    from GISio import get_proj4, shp2df
    from GISops import project, projectdf
except:
    msg = 'this package requires GISutils. try:\n'
    msg += 'pip install https://github.com/aleaf/GIS_utils/archive/master.zip'

class basemap:

    figsize = ReportFigures.doublecolumn_sizeT

    proj42basemap = {'utm': 'tmerc'}

    radii = {'NAD83': (6378137.00, 6356752.314245179),
             'NAD27': (6378206.4, 6356583.8)}

    drawparallels = np.array([True, False, False, False])
    drawmeridians = np.array([False, False, False, True])
    superlabels = [1, 0, 0, 1] # where to label subplot axes (l,r,t,b)
    graticule_text_color = 'k'
    graticule_label_style = None # None or '+/-'

    parallels_kw = {'linewidth': 0,
                    'rotation': 90.,
                    'color': '0.85',
                    'size': 8}

    meridians_kw = {'linewidth': 0,
                    'color': '0.85',
                    'size': 8}

    ticks_kw = {'linewidths': 0.5,
                'c': 'k',
                'zorder': 100}
    #resolution of boundary database to use.
    # Can be c (crude), l (low), i (intermediate), h (high), f (full) or None
    Basemap_kwargs = {'resolution': 'l',
                      'lat_0': 0,
                      'k_0': 0.99960000}

    scalebar_params = {'fontsize': 7, 'lineweight': 0.5, 'nticks': 3}


    def __init__(self, extent, ax=None,
                 crs=None, proj4=None, epsg=None, datum='NAD27', projection_shapefile=None,
                 tick_interval=0.2,
                 subplots=(1, 1), figsize='default', wspace=0.02, hspace=0.1,
                 parallels='default', meridians='default', parallels_kw={}, meridians_kw={},
                 **kwargs):

        self.fig = None

        if figsize != 'default':
            self.figsize = figsize
        if ax is None:
            # sharex and sharey arguments do not work with basemap!
            self.fig, self.axes = plt.subplots(*subplots, figsize=self.figsize)
            if sum(subplots) == 2:
                self.axes = np.array([self.axes])
        elif isinstance(ax, np.ndarray):
            self.axes = ax
            if len(np.shape(ax)) == 1:
                subplots = (1, np.shape(ax)[0])
            else:
                subplots = np.shape(ax)
        else:
            self.axes = np.array([ax])

        self.subplots = subplots

        self.maps = []

        self.extent_proj = extent
        self.proj4 = proj4
        self.epsg = epsg
        self.crs = crs
        self.projection_shapefile = projection_shapefile
        self.datum = datum

        # update default basemap setting with supplied keyword args
        self.Basemap_kwargs.update(kwargs)

        self._set_proj4() # establish a proj4 string for basemap projection regardless of input
        self._set_crs() # setup dictionary mapping of proj 4 string
        self._set_Basemap_kwargs()


        # decide with subplot axes edges to label (sharex/sharey don't work with basemap!)
        self.subplot_ticklabels()

        # set graticule positions
        if parallels == 'default':
            self.parallels = np.arange(self.llcrnrlat +.15, self.urcrnrlat, tick_interval)
        else:
            self.parallels = parallels
        if meridians == 'default':
            self.meridians = np.arange(self.llcrnrlon, self.urcrnrlon, tick_interval)
        else:
            self.meridians = meridians

        # update default graticule/grid formatting with supplied keywords
        self.parallels_kw.update(parallels_kw)
        self.meridians_kw.update(meridians_kw)

        # attributes for storing Map layers
        self.layers = {} # layer dataframes
        self.patches = {} # layer patches
        self.patches_inds = {} # indicies relating patches to attributes in dataframes

        '''
        lon_0 is the same as the Central_Meridian in ArcMap (see Layer Properties dialogue)
        lat_0 is same as Latitude_Of_Origin
        llcrnrlon, llcrnrlat, etc. all need to be specified in lat/lon

        Note: the major and minor axis radii for the rsphere argument are 6378137.00,6356752.314245179 for NAD83.
        '''

        for i, ax in enumerate(self.axes.flat):
            m = Basemap(ax=ax, **self.Basemap_kwargs)

            # labels option controls drawing of labels for parallels and meridians (l,r,t,b)
            y = m.drawparallels(self.parallels, labels=self.sp_ticklabels[i][1], ax=ax, **self.parallels_kw)
            x = m.drawmeridians(self.meridians, labels=self.sp_ticklabels[i][0], ax=ax, **self.meridians_kw)

            # add tickmarks for the graticules
            self.add_graticule_ticks(m, x, y)

            # add custom scalebar to plot (for subplots, scalebar is added only to last subplot)
            #if not subplots:
            #    add_scalebar(m, ax, scalebar_params, loc=scalebar_loc, color=scalebar_color, scalebar_pad=scalebar_pad)
            self.maps.append(m)

        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)

    def _convert_map_extent(self):

        # convert map units to lat / lons using pyproj
        #if self.epsg is not None:
        #    p1 = pyproj.Proj("+init=EPSG:{}".format(self.epsg))
        #else:
        p1 = pyproj.Proj(self.proj4, errcheck=True, preserve_units=True)

        extent = self.extent_proj

        self.llcrnrlon, self.llcrnrlat = p1(extent[0], extent[1], inverse=True)
        self.urcrnrlon, self.urcrnrlat = p1(extent[2], extent[3], inverse=True)

        self.Basemap_kwargs.update({'llcrnrlon': self.llcrnrlon,
                                    'llcrnrlat': self.llcrnrlat,
                                    'urcrnrlon': self.urcrnrlon,
                                    'urcrnrlat': self.urcrnrlat})

    def _read_shapefile_projection(self, shapefile):
        """sets the proj4 string from a shapefile"""
        crs = fiona.open(shapefile).crs
        init = crs.get('init', '')
        if 'epsg' in init:
            self.epsg = int(init.strip('epsg:'))

            if self.epsg is not None:
                self.proj4 = getproj4(self.epsg)
        else:
            self.proj4 = to_string(crs)


    def _set_proj4(self):

        # establish a proj4 string for basemap projection regardless of input
        if self.epsg is not None:
            self.proj4 = get_proj4(self.epsg)

        elif self.projection_shapefile is not None:
            self._read_shapefile_projection(self.projection_shapefile)

    def _set_crs(self):
        if self.crs is None and self.proj4 is not None:
            self.crs = from_string(self.proj4)

    def _set_Basemap_kwargs(self, use_epsg=False):
        """Sets the keyword arguments to Basemap from a crs mapping.

        Parameters
        ----------
        use_epsg : bool
            Use EPSG code for setting up basemap in lieu of all other parameters.
            (not recommended; utm projections are not well supported)
        """
        if use_epsg and self.epsg is not None:
            self.Basemap_kwargs = {'epsg': self.epsg}
        elif self.crs is not None:
            self.datum = str(self.crs.get('datum', self.datum))
            self.Basemap_kwargs['projection'] = self.projection
            self.Basemap_kwargs['ellps'] = self.crs.get('ellps', None)
            for latlon in ['lat_0', 'lat_1', 'lat_2', 'lon_0']:
                self.Basemap_kwargs[latlon] = self.crs.get(latlon,
                                                           self.Basemap_kwargs.get(latlon, None))
            # this is kind of ugly and needs some work for other coordinate systems
            if self.Basemap_kwargs['lon_0'] is None:
                self.Basemap_kwargs['lon_0'] = self._get_utm_lon_0(self.crs.get('zone', 15))

        self.Basemap_kwargs['rsphere'] = self.radii[self.datum]
        self._convert_map_extent() # convert supplied map extent to lat/lon; adds corners to Basemap kwargs
        '''
        {u'datum': u'NAD83',
         u'lat_0': 23,
         u'lat_1': 29.5
         u'lat_2': 45.5,
         u'lon_0': -96,
         u'no_defs': True,
         u'proj': u'aea',
         u'units': u'm',
         u'x_0': 0,
         u'y_0': 0}
         '''
    @property
    def projection(self):
        projection = self.proj42basemap.get(self.crs.get('proj', None), None)
        if projection is None:
            return 'tmerc'
        return projection

    def _get_utm_lon_0(self, zone=15):
        """get the longitude of origin for a given utm zone"""
        return -87. - (zone - 15) * 6


    def add_graticule_ticks(self, m, x, y):

        # add ticks for the graticules (basemap only gives option of lat/lon lines)
        # x and y are instances of drawmeridians and drawparallels objects in basemap

        # get the vertices of the lines where they intersect the axes by plumbing the dictionaries y, x, obtained above
        ylpts = [y[lat][0][0].get_data()[1][np.argmin(abs(y[lat][0][0].get_data()[0] - 0))] for lat in list(y.keys())]
        yrpts = [y[lat][0][0].get_data()[1][np.argmin(abs(y[lat][0][0].get_data()[0] - m.urcrnrx))] for lat in list(y.keys())]
        xtpts = [x[lat][0][0].get_data()[0][np.argmin(abs(x[lat][0][0].get_data()[1] - 0))] for lat in list(x.keys())]
        xbpts = [x[lat][0][0].get_data()[0][np.argmin(abs(x[lat][0][0].get_data()[1] - m.urcrnry))] for lat in list(x.keys())]

        # now plot the ticks as a scatter
        m.scatter(len(ylpts)*[0], ylpts, 100, marker='_', **self.ticks_kw)
        m.scatter(len(yrpts)*[m.urcrnrx], yrpts, 100, marker='_', **self.ticks_kw)
        m.scatter(xtpts, len(xtpts)*[0], 100, marker='|', **self.ticks_kw)
        m.scatter(xbpts, len(xbpts)*[m.urcrnry], 100, marker='|', **self.ticks_kw)

    def _get_xy_lables(self, drawlabels):
        drawlabels = np.array(drawlabels)
        return self.drawmeridians & drawlabels, self.drawparallels & drawlabels

    def subplot_ticklabels(self):

        nplots = len(self.axes.flat)
        rows, cols = self.subplots

        # determine which subplots are on the outside of the grid
        outside_left = list(range(nplots))[::cols]
        outside_right = list(range(nplots))[cols-1::cols]
        outside_top = list(range(nplots))[:cols]
        outside_bottom = list(range(nplots))[-cols:]

        sp_ticklabels = []
        for i in range(rows * cols):

            # set the tick labels based on whether the subplot is on the outside (tedious!)
            # (l,r,t,b)
            if nplots == 1:
                xl, yl = self._get_xy_lables([True, True, True, True])
            elif i in outside_left and i in outside_top and i in outside_bottom:
                xl, yl = self._get_xy_lables([True, False, True, True]) # L,R,T,B
            elif i in outside_left and i in outside_top:
                xl, yl = self._get_xy_lables([True, False, True, False])
            elif i in outside_top and i in outside_right and i in outside_bottom:
                xl, yl = self._get_xy_lables([False, True, True, True])
            elif i in outside_top and i not in outside_right:
                xl, yl = self._get_xy_lables([False, False, True, False])
            elif i in outside_top:
                xl, yl = self._get_xy_lables([False, True, True, False])
            elif i in outside_right and i not in outside_bottom:
                xl, yl = self._get_xy_lables([False, True, False, False])
            elif i in outside_right:
                xl, yl = self._get_xy_lables([False, True, False, True])
            elif i in outside_left and i not in outside_bottom:
                xl, yl = self._get_xy_lables([True, False, False, False])
            elif i in outside_left:
                xl, yl = self._get_xy_lables([True, False, False, True])
            else:
                xl, yl = self._get_xy_lables([False, False, False, False])

            sp_ticklabels.append([xl, yl])

        self.sp_ticklabels = sp_ticklabels

    def add_scalebar(self, ax=None, loc='outside_lower_right', color='k', scalebar_pad=0.08,
                     length_mi=None,
                     from_miles=1609.344):

        if ax is None:
            ax = self.axes.flat[-1]
        m = self.maps[-1]

        scalebar_params = self.scalebar_params

        previousfontfamily = plt.rcParams['font.family']
        plt.rcParams['font.family'] = plt.rcParams['font.family'] = 'Univers 57 Condensed'

        # scalebar properties
        units = ['Miles', 'Kilometers']


        # set scalebar length
        if length_mi is None:
            length_mi = (self.extent_proj[2] - self.extent_proj[0]) / from_miles/3

        major_tick_interval = length_mi // scalebar_params['nticks']
        multiplier2 = 0.621371 # multiplier to go from top units to bottom units (e.g. mi per km)
        scalebar_length_mu = major_tick_interval * scalebar_params['nticks'] * from_miles

        # scalebar position
        if isinstance(loc, tuple):
            position = [loc[0] * m.urcrnrx, loc[1] * m.urcrnry]
        elif loc == 'inside_upper_right':
            axis_pad = scalebar_pad * m.urcrnrx
            position = [m.urcrnrx - axis_pad * np.sqrt(scalebar_params['fontsize']/6.0) - scalebar_length_mu, m.urcrnry - 0.8*axis_pad]
        elif loc == 'inside_lower_right':
            axis_pad = scalebar_pad * m.urcrnrx
            position = [m.urcrnrx - axis_pad * np.sqrt(scalebar_params['fontsize']/6.0) - scalebar_length_mu, 0 + axis_pad]
        elif loc == 'outside_lower_right':
            axis_pad = scalebar_pad * m.urcrnr
            #position = [m.urcrnrx - axis_pad * np.sqrt(scalebar_params['fontsize']/6.0) - scalebar_length_mu, 0 - axis_pad] # 0.1 * m.urcrnrx]
            position = [0.67 * m.urcrnrx, 0 - scalebar_pad * m.urcrnry] # 0.1 * m.urcrnrx]
        elif loc == 'grid_outside_lower_right':
            axis_pad = scalebar_pad * m.urcrnrx
            position = [m.urcrnrx - axis_pad * np.sqrt(scalebar_params['fontsize']/6.0), 0 - axis_pad]
        elif loc == 'centered':
            axis_pad = 0.6 * scalebar_pad * m.urcrnrx
            position = [0.55 * (m.urcrnrx - scalebar_length_mu), 0 - axis_pad]
        elif loc == 'offcenter':
            axis_pad = 0.6 * scalebar_pad * m.urcrnrx
            position = [0.65 * (m.urcrnrx - scalebar_length_mu), 0 - axis_pad]
        else:
            axis_pad = scalebar_pad * m.urcrnrx
            position = [m.urcrnrx + axis_pad, 0 + axis_pad]

        # add bar line
        x = [position[0], position[0] + scalebar_length_mu]
        y = 2*[position[1]]

        sb = plt.Line2D(x, y, color=color, zorder=99, lw=scalebar_params['lineweight'])
        sb.set_clip_on(False)
        ax.add_line(sb)

        # setup position and length of ticks
        ticks = np.arange(0, length_mi, major_tick_interval, dtype=int)
        if ticks[-1] > length_mi:
            ticks = np.append(ticks, length_mi)
        tick_positions = np.array([ticks * from_miles + position[0], ticks * from_miles * multiplier2 + position[0]])
        tick_length = 0.05 * (tick_positions[0][-1] - tick_positions[0][0])
        y = np.array([[y[0], y[0] + tick_length], [y[0], y[0] - tick_length]])

        # convert over to axes fraction, for some reason mpl won't do 'data' coordinates
        # for annotations outside plot area
        tick_positions_ax = tick_positions/(m.urcrnrx)
        tick_length_ax = tick_length/(m.urcrnry)
        y_ax = y/(m.urcrnry)
        valignments = ['bottom','top']
        textpad = [0.5 * tick_length_ax, -0.5 * tick_length_ax] # 0 means that text is right on tick line

        # for each unit, then for each tick
        for n in range(2):
            for i in range(len(ticks)):

                # set ticks
                l = plt.Line2D(2*[tick_positions[n][i]], y[n], color=color,
                zorder=99, lw=scalebar_params['lineweight'])
                l.set_clip_on(False)
                ax.add_line(l)

                # add tick labels
                # -1 to tick labels is ugly kludge in response to changed annotation positioning in Mpl 1.4.3
                ax.annotate('{}'.format(ticks[i]), xy=(tick_positions_ax[n][i], y_ax[n][1]+textpad[n]),
                xycoords='axes fraction', fontsize=scalebar_params['fontsize'], horizontalalignment='center',
                color=color, verticalalignment=valignments[n])

            # add end l(unit) abels
            ax.annotate('    {}'.format(units[n]), xy=(tick_positions_ax[n][-1], y_ax[n][1]+textpad[n]),
            xycoords='axes fraction', fontsize=scalebar_params['fontsize'], horizontalalignment='left',
            color=color, verticalalignment=valignments[n])

        plt.rcParams['font.family'] = previousfontfamily

    def make_collection(self, shp, index_field=None,
                        s=20, fc='0.8', ec='k', lw=0.5, alpha=0.5,
                        color_field=None,
                        cbar=False, clim=(), cmap='jet', cbar_label=None,
                        simplify_patches=100,
                        zorder=5,
                        convert_coordinates=1,
                        remove_offset=True,
                        collection_name=None,
                        **kwargs):

        if collection_name is None:
            collection_name = os.path.split(shp)[-1].split('.')[0]
        df = shp2df(shp)

        if index_field is not None:
            df.index = df[index_field]

        proj4 = get_proj4(shp)

        if proj4 != self.proj4:
            df['geometry'] = projectdf(df, proj4, self.proj4)

        # convert projected coordinate units and/or get rid z values if the shapefile has them
        if convert_coordinates != 1 or df.iloc[0]['geometry'].has_z:
            df['geometry'] = [transform(lambda x, y, z=None: (x * convert_coordinates,
                                                              y * convert_coordinates), g)
                              for g in df.geometry]

        # remove model offset from projected coordinates (llcorner = 0,0)
        if remove_offset:
            df['geometry'] = [translate(g,
                                        -1 * self.extent_proj[0],
                                        -1 * self.extent_proj[1]) for g in df.geometry]

        if simplify_patches > 0:
            df['geometry'] = [g.simplify(simplify_patches) for g in df.geometry]

        if 'Polygon' in df.iloc[0].geometry.type:
            print("building PatchCollection...")
            inds = []
            patches = []
            for i, g in df.geometry.iteritems():
                if g.type != 'MultiPolygon':
                    inds.append(i)
                    patches.append(PolygonPatch(g))
                else:
                    for part in g.geoms:
                        inds.append(i)
                        patches.append(PolygonPatch(part))

            collection = PatchCollection(patches, cmap=cmap,
                                         facecolor=fc, linewidth=lw, edgecolor=ec, alpha=alpha,
                                         )

        elif 'LineString' in df.geometry[0].type:
            print("building LineCollection...")
            inds = []
            lines = []
            for i, g in df.geometry.iteritems():
                if 'Multi' not in g.type:
                    x, y = g.xy
                    inds.append(i)
                    lines.append(list(zip(x, y)))
                # plot each line in a multilinestring
                else:
                    for l in g:
                        x, y = l.xy
                        inds.append(i)
                        lines.append(list(zip(x, y)))

            collection = LineCollection(lines, colors=ec, linewidths=lw, alpha=alpha, zorder=zorder, **kwargs)
            #lc.set_edgecolor(ec)
            #lc.set_alpha(alpha)
            #lc.set_lw(lw)

            # set the color scheme (could set line thickness by same proceedure)
            if fc in df.columns:
                colors = np.array([df[fc][ind] for ind in inds])
                collection.set_array(colors)

        else:
            print("plotting points...")
            x = np.array([g.x for g in df.geometry])
            y = np.array([g.y for g in df.geometry])

            collection = self.ax.scatter(x, y, s=s, c=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder, **kwargs)
            inds = list(range(len(x)))

        self.layers[collection_name] = df
        self.collections[collection_name] = collection
        self.collection_inds[collection_name] = inds

        return collection

    def _load_shapefile(self, shp, index_field, convert_coordinates, remove_offset, simplify):

        df = shp2df(shp)

        if index_field is not None:
            df.index = df[index_field]

        proj4 = get_proj4(shp)

        if proj4 != self.proj4:
            df['geometry'] = projectdf(df, proj4, self.proj4)

        # convert projected coordinate units and/or get rid z values if the shapefile has them
        if convert_coordinates != 1 or df.iloc[0]['geometry'].has_z:
            df['geometry'] = [transform(lambda x, y, z=None: (x * convert_coordinates,
                                                              y * convert_coordinates), g)
                              for g in df.geometry]

        # remove model offset from projected coordinates (llcorner = 0,0)
        if remove_offset:
            df['geometry'] = [translate(g,
                                        -1 * self.extent_proj[0],
                                        -1 * self.extent_proj[1]) for g in df.geometry]

        if simplify > 0:
            df['geometry'] = [g.simplify(simplify) for g in df.geometry]
        return df

    def make_patches(self, shp, index_field=None,
                        simplify_patches=100,
                        convert_coordinates=1,
                        remove_offset=True,
                        layername=None,
                        **kwargs):

        if layername is None:
            layername = os.path.split(shp)[-1].split('.')[0]

        df = self._load_shapefile(shp, index_field, convert_coordinates, remove_offset, simplify_patches)

        print("building PatchCollection...")
        inds = []
        patches = []
        for i, g in df.geometry.iteritems():
            if g.type != 'MultiPolygon':
                inds.append(i)
                patches.append(PolygonPatch(g))
            else:
                for part in g.geoms:
                    inds.append(i)
                    patches.append(PolygonPatch(part))

        self.layers[layername] = df
        self.patches[layername] = patches
        self.patches_inds[layername] = inds # index values (for patches created from multipolygons)
        return patches, inds

    def add_shapefile(self, shp, index_field=None, axes=None,
                      s=20, fc='0.8', ec='k', lw=0.5, alpha=0.5,
                      color_field=None,
                      cbar=False, clim=(), cmap='jet', cbar_label=None,
                      simplify=100,
                      zorder=5,
                      convert_coordinates=1,
                      remove_offset=True,
                      layername=None,
                      **kwargs):

        if layername is None:
            layername = os.path.split(shp)[-1].split('.')[0]

        if layername not in list(self.layers.keys()):
            shpobj = fiona.open(shp)
            if 'Polygon' in shpobj.schema['geometry']:
                self.make_patches(shp, index_field=index_field,
                                  simplify_patches=simplify,
                                  zorder=zorder,
                                  convert_coordinates=convert_coordinates,
                                  remove_offset=remove_offset,
                                  layername=layername,
                                  **kwargs)

                self.plot_patches(layername,
                                  fc=fc, ec=ec, lw=lw, alpha=alpha,
                                  color_fields=color_field,
                                  cbar=cbar, clim=(), cmap=cmap, cbar_label=cbar_label,
                                  zorder=zorder)
            else:
                print("geometry type not implemented yet")
                return
        '''
        if axes is None:
            axes = self.axes.flat

        for ax in axes:
            df = self.layers[layername]
            patches = self.patches[layername]
            inds = self.patches_inds[layername]

            # set the color scheme
            if color_field is not None:
                colors = np.array([df[color_field][ind] for ind in inds])
                if len(clim) > 0:
                    collection.set_clim(clim[0], clim[1])
                collection.set_array(colors)

            ax.add_collection(collection)

        if cbar:
            self.fig.colorbar(collection, ax=axes, pad=0.03, label=cbar_label)
        '''
    def plot_patches(self, layername=None, patches=None, patches_inds=None,
                     color_fields=None, df=None,
                     fc='0.5', ec='k', lw=0.5, alpha=0.5, zorder=1,
                     clim=(), cmap='jet', normalize_cmap=False,
                     cbar=False, cbar_label=False,
                     axes=None,
                     cbar_kw={},
                     **kwargs):

        patches_cbar_kw = {'ax': self.axes.ravel().tolist(),
                           'fraction': 0.046,
                           'pad': 0.03,
                           'label': cbar_label}

        patches_cbar_kw.update(cbar_kw)

        if axes is None:
            axes = self.axes.flat

        if not isinstance(color_fields, list):
            color_fields = [color_fields]

        # plot patches from the basemap instance
        if layername is not None:
            if df is None:
                df = self.layers[layername]

            patches = self.patches[layername]
            inds = self.patches_inds[layername]
        # plot supplied patches (quicker for many basemap plots)
        else:
            patches = patches
            inds = patches_inds

        if color_fields[0] is None:
            color_fields = [None] * np.size(self.axes)
        elif len(clim) == 0 or clim == ('min', 'max'):
            clim = (df.min().min(), df.max().max())
        elif clim[0] == 'min':
            clim = (df.min().min(), clim[1])
        elif clim[1] == 'max':
            clim = (clim[0], df.max().max())
        else:
            pass

        if normalize_cmap and color_fields[0] is not None:
            cmap = Normalized_cmap(cmap, df[color_fields].values.ravel(), vmin=clim[0], vmax=clim[1]).cm

        for i, cf in enumerate(color_fields):

            collection = PatchCollection(patches, cmap=cmap,
                                         facecolor=fc, linewidth=lw, edgecolor=ec, alpha=alpha,
                                         **kwargs)
            # color patches by values in the included dataframe
            if cf is not None:
                colors = np.array([df[cf][ind] for ind in inds])
                collection.set_array(colors)
                if len(clim) > 0:
                    collection.set_clim(clim[0], clim[1])

            axes[i].add_collection(collection)

        fig = self.fig
        if cbar:
            self.colorbar = fig.colorbar(collection, **patches_cbar_kw)

        return collection

    def _set_shapefile_colors(self, df, column, inds):

        colors = [df[column][ind] for ind in inds]

