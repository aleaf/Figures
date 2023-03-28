from __future__ import print_function

__author__ = 'aleaf'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import textwrap

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
                 'pdf.fonttype': 42,
                 'ps.fonttype': 42,
                 'figure.figsize': doublecolumn_size}

    ymin0 = False

    def __init__(self):

        pass

    def title(self, ax, title, zorder=200, wrap=None,
                     subplot_prefix='',
                     capitalize=False, **kwargs):

        # save the defaults before setting
        old_fontset = mpl.rcParams['mathtext.fontset']
        old_mathtextit = mpl.rcParams['mathtext.it']
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.it'] = 'Univers 67 Condensed:italic'
        #mpl.rcParams['mathtext.bf'] = 'Univers 67 Condensed:italic'

        if wrap is not None:
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
            title = f'$\it{{{subplot_prefix}}}$. ' + title
        # with Univers 47 Condensed as the font.family, changing the weight to 'bold' doesn't work
        # manually specify a different family for the title
        ax.set_title(title, family=self.title_font, zorder=zorder, loc='left', math_fontfamily='custom', 
                     **kwargs)
        
        # reinstate existing rcParams
        #mpl.rcParams['mathtext.fontset'] = old_fontset
        #mpl.rcParams['mathtext.it'] = old_mathtextit

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

    def axes_numbering(self, ax, format_x=False, enforce_integers=False,
                       x_fmt=None, y_fmt=None):
        """
        Implement these requirements from
        Standards for U.S. Geological Survey Page-Size Illustrations, p 36
        * Use commas in numbers greater than 999
        * Label zero without a decimal point
        * Numbers less than 1 should consist of a zero, a decimal point, and the number
        * Numbers greater than or equal to 1 need a decimal point and trailing zero only where significant figures dictate

        Parameters
        ----------
        ax : matplotlib axes object
        format_x : bool
            Option to format the x-axis (for figures where x-axis is dates or categories, etc.)
        enforce_integers : bool
            Force tick intervals to be at even integers
        x_fmt : format string
            Optional argument to explicitly specify desired axis formatting.
            (e.g. {:,.2f} for two decimal places with a thousands sep.)
        y_fmt : format string
            Optional argument to explicitly specify desired axis formatting.
        """

        # enforce minimum value of zero on y axis if data is positive
        if self.ymin0 and ax.get_ylim()[0] < 0:
            ax.set_ylim(0, ax.get_ylim()[1])

        # default number formatting for various axis limits
        def get_format(axis_limits, enforce_integers):
            if -10 > axis_limits[0] or axis_limits[1] > 10 or enforce_integers:
                fmt = '{x:,.0f}'
            elif -10 <= axis_limits[0] < -1 or 1 < axis_limits[1] <= 10:
                fmt = '{x:,.1f}'
            elif -1 <= axis_limits[0] < -.1 or .1 < axis_limits[1] <= 1:
                fmt = '{x:,.2f}'
            else:
                fmt = '{x:,.2e}'
            return fmt

        # formatting function for the ticker
        def number_formatting(axis_limits, enforce_integers, fmt=None):
            if fmt is None:
                fmt = get_format(axis_limits, enforce_integers)
            def format_axis(y, pos):
                y = fmt.format(y)
                return y
            return format_axis

        # correct for edge cases of upper ylim == 1, 10
        # and fix zero to have no decimal places
        def fix_decimal(ticks):
            # ticks are a list of strings
            if makefloat(ticks[-1]) == 10:
                ticks[-1] = '10'
            if makefloat(ticks[-1]) == 1:
                ticks[-1] = '1.0'
            for i, v in enumerate(ticks):
                if makefloat(v) == 0:
                    ticks[i] = '0'
            return ticks

        # strip unicode for minus sign and thousands separator
        def makefloat(text):
            txt = text.replace('\u2212', '-')
            txt = txt.replace(',', '')
            return float(txt)

        # apply the number formats
        #format_axis = number_formatting(ax.get_ylim(), enforce_integers, fmt=y_fmt)
        if y_fmt is None:
            y_fmt = get_format(ax.get_ylim(), enforce_integers)
        else:
            # add the variable to the string formatter if it wasn't included
            y_fmt = y_fmt.replace('{:', '{x:')
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(y_fmt))
        #ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(format_axis))

        # option to format x (may not want to mess with x if it has dates or other categories)
        if format_x:
            #format_axis = number_formatting(ax.get_xlim(), enforce_integers, fmt=x_fmt)
            if x_fmt is None:
                x_fmt = get_format(ax.get_xlim(), enforce_integers)
            else:
                # add the variable to the string formatter if it wasn't included
                x_fmt = x_fmt.replace('{:', '{x:')
            ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(x_fmt))
            #ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(format_axis))

        # enforce integer increments
        if enforce_integers:
            ax.get_yaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            if format_x:
                ax.get_xaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            
        # fix the min/max values
        # (don't do this if a format is explicitly specified)
        if format_x:
            if x_fmt is None:
                x_fmt = get_format(ax.get_xlim(), enforce_integers)
                newxlabels = fix_decimal([x_fmt.format(l) for l in ax.get_xticks()])
                ax.set_xticklabels(newxlabels)

        if y_fmt is None:
            y_fmt = get_format(ax.get_ylim(), enforce_integers)
            newylabels = fix_decimal([y_fmt.format(l) for l in ax.get_yticks()])
            ax.set_yticklabels(newylabels)

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

