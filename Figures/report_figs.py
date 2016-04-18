from __future__ import print_function

__author__ = 'aleaf'

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection

try:
    from mpl_toolkits.basemap import Basemap, pyproj
    import fiona
    from fiona.crs import to_string, from_epsg
    from shapely.ops import transform
    from shapely.affinity import translate
    from descartes import PolygonPatch
    from GISio import get_proj4, shp2df
    from GISops import project, projectdf
except:
    pass
import textwrap

'''
try:
    import seaborn as sb
    seaborn = True
except:
    seaborn = False
'''

class ReportFigures(object):

    # figure sizes (based on 6 picas per inch, see USGS Illustration Standards Guide, p 34
    default_aspect = 6 / 8.0 # h/w
    tall_aspect = 7 / 8.0
    singlecolumn_width = 21/6.0
    doublecolumn_width = 42/6.0

    singlecolumn_size = (singlecolumn_width, singlecolumn_width * default_aspect)
    singlecolumn_sizeT = (singlecolumn_width, singlecolumn_width * tall_aspect)
    doublecolumn_size = (doublecolumn_width, doublecolumn_width * default_aspect)
    doublecolumn_sizeT = (doublecolumn_width, doublecolumn_width * tall_aspect)

    # title wraps
    singlecolumn_title_wrap = 50
    doublecolumn_title_wrap = 120

    # month abbreviations
    month = {1: 'Jan.', 2: 'Feb.', 3: 'Mar.', 4: 'Apr.', 5: 'May', 6: 'June',
             7: 'July', 8: 'Aug.', 9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Dec.'}

    # fonts
    default_font = 'Univers 57 Condensed'
    title_font = 'Univers 67 Condensed'
    title_size = 9
    legend_font = 'Univers 67 Condensed'
    legend_titlesize = 9
    legend_headingsize = 8
    basemap_credits_font = 'Univers 47 Condensed Light'
    basemap_credits_fontsize = 7
    
    ymin0=False

    # rcParams
    plotstyle = {'font.family': default_font,
                 'font.size' : 8.0,
                 'axes.linewidth': 0.5,
                 'axes.labelsize': 9,
                 'axes.titlesize': 9,
                 "grid.linewidth": 0.5,
                 'xtick.major.width': 0.5,
                 'ytick.major.width': 0.5,
                 'xtick.minor.width': 0.5,
                 'ytick.minor.width': 0.5,
                 'xtick.labelsize': 8,
                 'ytick.labelsize': 8,
                 'xtick.direction': 'in',
                 'ytick.direction': 'in',
                 'xtick.major.pad': 3,
                 'ytick.major.pad': 3,
                 'axes.edgecolor' : 'k',
                 'figure.figsize': doublecolumn_size}

    ymin0 = False

    def __init__(self):

        pass

    def title(self, ax, title, zorder=200, wrap=50,
                     subplot_prefix='',
                     capitalize=True):

        # save the defaults before setting
        old_fontset = mpl.rcParams['mathtext.fontset']
        old_mathtextit = mpl.rcParams['mathtext.it']
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.it'] = 'Univers 67 Condensed:italic'

        wrap = wrap
        title = "\n".join(textwrap.wrap(title, wrap)) #wrap title
        if capitalize:
            title = title.capitalize()
        '''
        ax.text(.025, 1.025, title,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes, zorder=zorder)
        '''
        if len(subplot_prefix) > 0:
            title = '$\it{}$. '.format(subplot_prefix) + title
        # with Univers 47 Condensed as the font.family, changing the weight to 'bold' doesn't work
        # manually specify a different family for the title
        ax.set_title(title, family=self.title_font, zorder=zorder, loc='left')

        # reinstate existing rcParams
        mpl.rcParams['mathtext.fontset'] = old_fontset
        mpl.rcParams['mathtext.it'] = old_mathtextit

    def legend(self, ax, handles, labels, **kwargs):
        '''Make a legend, following guidelines in USGS Illustration Standards, p. 14

        ax : matplotlib.pyplot axis object
        handles : list
            matplotlib.pyplot handles for each plot
        labels : list
            labels for each plot
        kwargs : dict
            keyword arguments to matplotlib.pyplot.legend()
        '''

        lgkwargs = {'title': 'EXPLANATION',
                    'fontsize': self.legend_headingsize,
                    'frameon': False,
                    'loc': 8,
                    'bbox_to_anchor': (0.5, -0.25)}
        lgkwargs.update(kwargs)

        mpl.rcParams['font.family'] = self.title_font


        lg = ax.legend(handles, labels, **lgkwargs)

        plt.setp(lg.get_title(), fontsize=self.legend_titlesize)
        #reset rcParams back to default
        mpl.rcParams['font.family'] = self.default_font
        return lg

    def axes_numbering(self, ax, format_x=False, enforce_integers=False):
        '''
        Implement these requirements from USGS standards, p 16
        * Use commas in numbers greater than 999
        * Label zero without a decimal point
        * Numbers less than 1 should consist of a zero, a decimal point, and the number
        * Numbers greater than or equal to 1 need a decimal point and trailing zero only where significant figures dictate
        '''

        # enforce minimum value of zero on y axis if data is positive
        if self.ymin0 and ax.get_ylim()[0] < 0:
            ax.set_ylim(0, ax.get_ylim()[1])

        # so clunky, but this appears to be the only way to do it
        def number_formatting(axis_limits, enforce_integers):
            if -10 > axis_limits[0] or axis_limits[1] > 10 or enforce_integers:
                fmt = '{:,.0f}'
            elif -10 <= axis_limits[0] < -1 or 1 < axis_limits[1] <= 10:
                fmt = '{:,.1f}'
            elif -1 <= axis_limits[0] < -.1 or .1 < axis_limits[1] <= 1:
                fmt = '{:,.2f}'
            else:
                fmt = '{:,.2e}'
                
            def format_axis(y, pos):
                y = fmt.format(y)
                return y
            return format_axis

        format_axis = number_formatting(ax.get_ylim(), enforce_integers) 
        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(format_axis))

        if format_x:
            format_axis = number_formatting(ax.get_xlim(), enforce_integers) 
            ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(format_axis))
        
        if enforce_integers:
            ax.get_yaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            if format_x:
                ax.get_xaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            
        # correct for edge cases of upper ylim == 1, 10
        # and fix zero to have no decimal places
        def fix_decimal(ticks):
            # ticks are a list of strings
            if float(ticks[-1]) == 10:
                ticks[-1] = '10'
            if float(ticks[-1]) == 1:
                ticks[-1] = '1.0'
            for i, v in enumerate(ticks):
                if float(v) == 0:
                    ticks[i] = '0'
            return ticks

        try:
            [float(l._text.replace('\u2212', '-')) for l in ax.get_xticklabels()]
            newxlabels = fix_decimal([fmt.format(float(l._text.replace('\u2212', '-'))) for l in ax.get_xticklabels()])
            ax.set_xticklabels(newxlabels)
        except:
            pass

        try:
            [float(l._text.replace('\u2212', '-')) for l in ax.get_yticklabels()]
            newylabels = fix_decimal([fmt.format(float(l._text.replace('\u2212', '-'))) for l in ax.get_yticklabels()])
            ax.set_yticklabels(newylabels)
        except:
            pass

    def basemap_credits(self, ax, text, wrap=50, y_offset=-0.01):
        """Add basemap credits per the Illustration Standards Guide p27
        """
        if wrap is not None:
            text = "\n".join(textwrap.wrap(text, wrap)) #wrap title
        ax.text(0.0, y_offset, text, family=self.basemap_credits_font, fontsize=self.basemap_credits_fontsize,
                transform=ax.transAxes, ha='left', va='top')

    def set_style(self, mpl=mpl, style='default', width='double', height='default'):
        """
        Set dimmensions of figures to standard report sizes

        Not quite sure yet about the cleanest way to implement multiple customizations of the base settings (styles)
        Seaborn's methodology is somewhat complicated, but may be the way to go for multiple figure styles
        (and it also provides the option of viewing the current style settings with the axes_style() command)


        Parameters
        ----------
        width : string
            Sets the figure width to single or double column
            'double' (default) or 'single'
        height : string
            Sets the aspect of the plot to the Matplotlib default (6:8), or a tall aspect (7:8)
            'default' (default) or 'tall'
        """


        if style == 'timeseries':
            self.plotstyle['grid.linewidth'] = 0
            if width == 'single':
                self.plotstyle['xtick.minor.size'] = 0

        if width == 'single' and height != 'tall':
            self.plotstyle['figure.figsize'] = self.singlecolumn_size

        elif height == 'tall' and width == 'double':
            self.plotstyle['figure.figsize'] = self.doublecolumn_sizeT

        elif height == 'tall' and width == 'single':
            self.plotstyle['figure.figsize'] = self.singlecolumn_sizeT

        else:
            pass

        mpl.rcParams.update(self.plotstyle)

        """Set the aesthetic style of the plots.

        This affects things like the color of the axes, whether a grid is
        enabled by default, and other aesthetic elements.

        Parameters
        ----------
        style : dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
            A dictionary of parameters or the name of a preconfigured set.
        rc : dict, optional
            Parameter mappings to override the values in the preset seaborn
            style dictionaries. This only updates parameters that are
            considered part of the style definition.

        Examples
        --------
        >>> set_style("whitegrid")

        >>> set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

        See Also
        --------
        axes_style : return a dict of parameters or use in a ``with`` statement
                     to temporarily set the style.
        set_context : set parameters to scale plot elements
        set_palette : set the default color palette for figures

        """

class basemap:

    figsize = ReportFigures.doublecolumn_sizeT

    projection = 'tmerc'
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

    #resolution of boundary database to use.
    # Can be c (crude), l (low), i (intermediate), h (high), f (full) or None
    Basemap_kwargs = {'resolution': 'i'}

    scalebar_params = {'fontsize': 7, 'lineweight': 0.5, 'nticks': 3}


    def __init__(self, extent, ax=None, proj4=None, epsg=None, datum='NAD27', projection_shapefile=None,
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
        self.projection_shapefile = projection_shapefile
        self.datum = datum

        # update default basemap setting with supplied keyword args
        self.Basemap_kwargs['rsphere'] = self.radii[self.datum]
        self.Basemap_kwargs.update(kwargs)

        self._set_proj4() # establish a proj4 string for basemap projection regardless of input

        self._convert_map_extent() # convert supplied map extent to lat/lon

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

        crs = fiona.open(shapefile).crs

        self.datum = str(crs.get('datum', self.datum))
        self.Basemap_kwargs['projection'] = str(crs.get('proj', None))

        for latlon in ['lat_0', 'lat_1', 'lat_2', 'lon_0']:
            self.Basemap_kwargs[latlon] = crs.get(latlon, None)

        '''
        {u'datum': u'NAD83',
         u'lat_0': 23,
         u'lat_1': 29.5,
         u'lat_2': 45.5,
         u'lon_0': -96,
         u'no_defs': True,
         u'proj': u'aea',
         u'units': u'm',
         u'x_0': 0,
         u'y_0': 0}
         '''

    def _set_proj4(self):

        # establish a proj4 string for basemap projection regardless of input
        if self.epsg is not None:
            self.proj4 = to_string(from_epsg(self.epsg))

        if self.projection_shapefile is not None:
            self._read_shapefile_projection(self.projection_shapefile)
            self.proj4 = get_proj4(self.projection_shapefile)

    def add_graticule_ticks(self, m, x, y):

        # add ticks for the graticules (basemap only gives option of lat/lon lines)
        # x and y are instances of drawmeridians and drawparallels objects in basemap

        # get the vertices of the lines where they intersect the axes by plumbing the dictionaries y, x, obtained above
        ylpts = [y[lat][0][0].get_data()[1][np.argmin(abs(y[lat][0][0].get_data()[0] - 0))] for lat in list(y.keys())]
        yrpts = [y[lat][0][0].get_data()[1][np.argmin(abs(y[lat][0][0].get_data()[0] - m.urcrnrx))] for lat in list(y.keys())]
        xtpts = [x[lat][0][0].get_data()[0][np.argmin(abs(x[lat][0][0].get_data()[1] - 0))] for lat in list(x.keys())]
        xbpts = [x[lat][0][0].get_data()[0][np.argmin(abs(x[lat][0][0].get_data()[1] - m.urcrnry))] for lat in list(x.keys())]

        # now plot the ticks as a scatter
        m.scatter(len(ylpts)*[0], ylpts, 100, c='k', marker='_', linewidths=1, zorder=100)
        m.scatter(len(yrpts)*[m.urcrnrx], yrpts, 100, c='k', marker='_', linewidths=1, zorder=100)
        m.scatter(xtpts, len(xtpts)*[0], 100, c='k', marker='|', linewidths=1, zorder=100)
        m.scatter(xbpts, len(xbpts)*[m.urcrnry], 100, c='k', marker='|', linewidths=1, zorder=100)

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
                ax.annotate('{}'.format(ticks[i]), xy=(tick_positions_ax[n][i], y_ax[n][1]+textpad[n] -1),
                xycoords='axes fraction', fontsize=scalebar_params['fontsize'], horizontalalignment='center',
                color=color, verticalalignment=valignments[n])

            # add end l(unit) abels
            ax.annotate('    {}'.format(units[n]), xy=(tick_positions_ax[n][-1], y_ax[n][1]+textpad[n] -1),
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
            for i, g in df.geometry.items():
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
            for i, g in df.geometry.items():
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
        for i, g in df.geometry.items():
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


class Normalized_cmap:

    def __init__(self, cmap, values, vmin=None, vmax=None):

        self.values = values
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        self.cmap = cmap

        self.normalize_cmap()
        #self.start = 0
        #self.stop = 1
        self.cm = self.shiftedColorMap(self.cmap, self.start, self.midpoint, self.stop)
        print(self.start, self.midpoint, self.stop)

    def normalize_cmap(self):
        """Computes start, midpoint, and end values (between 0 and 1), to make
        a colormap using shiftedColorMap(), which will be
        * centered on zero
        """

        if self.vmin is None:
            self.vmin = np.min(self.values)
        if self.vmax is None:
            self.vmax = np.max(self.values)

        self._compute_midpoint()

    def _compute_midpoint(self):

        low, high = self.vmin, self.vmax
        start, midpoint, stop = 0.0, 0.5, 1.0
        if high > 0 and low >= 0:
            midpoint = 0.5 + low / (2 * high)
            start = midpoint - 0.0001
        elif high <= 0 and low < 0:
            midpoint = (low - high) / (2 * low)
            stop = midpoint + 0.0001
        elif abs(low) > high:
            stop = (high-low) / (2 * abs(low))
            midpoint = start + abs(low) / (2*abs(low))
        elif abs(low) < high:
            start = (high - abs(low)) / (2 * high)
            midpoint = start + abs(low)/(2* high)
        else:
            pass
        self.start, self.midpoint, self.stop = float(start), float(midpoint), float(stop)

    def shiftedColorMap(self, cmap, start=0.0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero. From:
        http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''

        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap