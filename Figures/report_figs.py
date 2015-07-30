__author__ = 'aleaf'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from mpl_toolkits.basemap import Basemap, pyproj
    import fiona
    from GISio import get_proj4
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
            title = '$\it{}$ '.format(subplot_prefix) + title
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
            [float(l._text.replace(u'\u2212', '-')) for l in ax.get_xticklabels()]
            newxlabels = fix_decimal([fmt.format(float(l._text.replace(u'\u2212', '-'))) for l in ax.get_xticklabels()])
            ax.set_xticklabels(newxlabels)
        except:
            pass

        try:
            [float(l._text.replace(u'\u2212', '-')) for l in ax.get_yticklabels()]
            newylabels = fix_decimal([fmt.format(float(l._text.replace(u'\u2212', '-'))) for l in ax.get_yticklabels()])
            ax.set_yticklabels(newylabels)
        except:
            pass

    def basemap_credits(self, ax, text, wrap=50, y_offset=-0.01):
        """Add basemap credits per the Illustration Standards Guide p27
        """
        text = "\n".join(textwrap.wrap(text, wrap)) #wrap title
        ax.text(0.0, y_offset, text, family=self.basemap_credits_font, fontsize=self.basemap_credits_fontsize,
                transform=ax.transAxes, ha='left', va='top')

    def set_style(self, style='default', width='double', height='default'):
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

    projection = 'tmerc'
    radii = {'NAD83': (6378137.00, 6356752.314245179),
             'NAD27': (6378206.4, 6356583.8)}

    graticule_text_color = 'k'
    graticule_label_style = '+/-'

    Basemap_kwargs = {'projection': 'tmerc',
                      'lon_0': -93,
                      'lat_0': 0,
                      'lat_1':None,
                      'lat_2': None,
                      'lon_1': None,
                      'lon_2': None,
                      'k_0': None,
                      'resolution': 'i'}

    def __init__(self, ax, extent, proj4=None, epsg=None, datum='NAD27', shapefile=None,
                 tick_interval=0.2,
                  yticklabels=[True, False, True, False], xticklabels=[True, False, True, False],
                  subplots=False,
                  scalebar_loc='centered', scalebar_color='k', scalebar_pad=0.08, **kwargs):

        self.ax = ax
        self.extent_proj = extent
        self.proj4 = proj4
        self.epsg = epsg
        self.datum = datum

        if shapefile is not None:
            self._read_shapefile_proction(shapefile)
            self.proj4 = get_proj4(shapefile)

        self._convert_map_extent()

        self.Basemap_kwargs['rsphere'] = self.radii[self.datum]
        self.Basemap_kwargs.update(kwargs)


        '''
        lon_0 is the same as the Central_Meridian in ArcMap (see Layer Properties dialogue)
        lat_0 is same as Latitude_Of_Origin
        llcrnrlon, llcrnrlat, etc. all need to be specified in lat/lon

        Note: the major and minor axis radii for the rsphere argument are 6378137.00,6356752.314245179 for NAD83.
        '''
        self.m = Basemap(ax=ax, **self.Basemap_kwargs)

        # labels option controls drawing of labels for parallels and meridians (l,r,t,b)
        self.yticks = self.m.drawparallels(np.arange(self.lat_0 +.15, self.lat_1, tick_interval), linewidth=0, labels=yticklabels,
                            labelstyle=self.graticule_label_style, rotation=90., color='0.85', size=8, ax=ax)
        self.xticks = self.m.drawmeridians(np.arange(self.lon_0, self.lon_1, tick_interval), linewidth=0, labels=xticklabels,
                            labelstyle=self.graticule_label_style, color='0.85', size=8, ax=ax)

        # add tickmarks for the graticules
        self.add_graticule_ticks()

        # add custom scalebar to plot (for subplots, scalebar is added only to last subplot)
        #if not subplots:
        #    add_scalebar(m, ax, scalebar_params, loc=scalebar_loc, color=scalebar_color, scalebar_pad=scalebar_pad)

    def _convert_map_extent(self):

        # convert map units to lat / lons using pyproj
        if self.epsg is not None:
            p1 = pyproj.Proj("+init=EPSG:{}".format(self.epsg))
        else:
            p1 = pyproj.Proj(self.proj4)

        extent = self.extent_proj
        self.lon_0, self.lat_0 = p1(extent[0], extent[2], inverse=True)
        self.lon_1, self.lat_1 = p1(extent[1], extent[3], inverse=True)

        self.Basemap_kwargs.update({'llcrnrlon': self.lon_0,
                                    'llcrnrlat': self.lat_0,
                                    'urcrnrlon': self.lon_1,
                                    'urcrnrlat': self.lat_1})

    def _read_shapefile_proction(self, shapefile):

        crs = fiona.open(shapefile).crs

        self.datum = crs.get('datum', self.datum)
        self.Basemap_kwargs['projection'] = crs.get('proj', self.Basemap_kwargs['projection'])

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

    def add_graticule_ticks(self):

        # add ticks for the graticules (basemap only gives option of lat/lon lines)
        # x and y are instances of drawmeridians and drawparallels objects in basemap
        x, y = self.xticks, self.yticks
        # get the vertices of the lines where they intersect the axes by plumbing the dictionaries y, x, obtained above
        ylpts = [y[lat][0][0].get_data()[1][np.argmin(abs(y[lat][0][0].get_data()[0] - 0))] for lat in y.keys()]
        yrpts = [y[lat][0][0].get_data()[1][np.argmin(abs(y[lat][0][0].get_data()[0] - self.m.urcrnrx))] for lat in y.keys()]
        xtpts = [x[lat][0][0].get_data()[0][np.argmin(abs(x[lat][0][0].get_data()[1] - 0))] for lat in x.keys()]
        xbpts = [x[lat][0][0].get_data()[0][np.argmin(abs(x[lat][0][0].get_data()[1] - self.m.urcrnry))] for lat in x.keys()]

        # now plot the ticks as a scatter
        self.m.scatter(len(ylpts)*[0], ylpts, 100, c='k', marker='_', linewidths=1, zorder=100)
        self.m.scatter(len(yrpts)*[self.m.urcrnrx], yrpts, 100, c='k', marker='_', linewidths=1, zorder=100)
        self.m.scatter(xtpts, len(xtpts)*[0], 100, c='k', marker='|', linewidths=1, zorder=100)
        self.m.scatter(xbpts, len(xbpts)*[self.m.urcrnry], 100, c='k', marker='|', linewidths=1, zorder=100)

    def add_scalebar(self, scalebar_params, loc=None, color='k', scalebar_pad=0.08):

        m, ax = self.m, self.ax
        previousfontfamily = plt.rcParams['font.family']
        plt.rcParams['font.family'] = plt.rcParams['font.family'] = 'Univers 57 Condensed'

        # scalebar properties
        units = ['Miles', 'Kilometers']
        scalebar_length = scalebar_params['length'] # miles
        major_tick_interval = scalebar_params['major_tick_interval']
        multiplier = 1 * 5280 * 0.3048 # to convert scalebar length to map coordinate units
        multiplier2 = 0.621371 # multiplier to go from top units to bottom units (e.g. mi per km)
        scalebar_length_mu = scalebar_length * multiplier

        # scalebar position
        if loc == 'inside_upper_right':
            axis_pad = scalebar_pad * m.urcrnrx
            position = [m.urcrnrx - axis_pad * np.sqrt(scalebar_params['fontsize']/6.0) - scalebar_length_mu, m.urcrnry - 0.8*axis_pad]
        elif loc == 'inside_lower_right':
            axis_pad = scalebar_pad * m.urcrnrx
            position = [m.urcrnrx - axis_pad * np.sqrt(scalebar_params['fontsize']/6.0) - scalebar_length_mu, 0 + axis_pad]
        elif loc == 'outside_lower_right':
            axis_pad = scalebar_pad * m.urcrnrx
            position = [m.urcrnrx - axis_pad * np.sqrt(scalebar_params['fontsize']/6.0) - scalebar_length_mu, 0 - 0.1 * m.urcrnrx]
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
        ticks = np.arange(0,scalebar_length+major_tick_interval,major_tick_interval)
        tick_positions = np.array([ticks * multiplier + position[0], ticks * multiplier * multiplier2 + position[0]])
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
                ax.annotate('{}'.format(ticks[i]), xy=(tick_positions_ax[n][i], y_ax[n][1]+textpad[n]),
                xycoords='axes fraction', fontsize=scalebar_params['fontsize'], horizontalalignment='center',
                color=color, verticalalignment=valignments[n])

            # add end l(unit) abels
            ax.annotate('    {}'.format(units[n]), xy=(tick_positions_ax[n][-1], y_ax[n][1]+textpad[n]),
            xycoords='axes fraction', fontsize=scalebar_params['fontsize'], horizontalalignment='left',
            color=color, verticalalignment=valignments[n])

        plt.rcParams['font.family'] = previousfontfamily