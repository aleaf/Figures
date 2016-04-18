import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
from random import shuffle
import sys

class MidpointNormalize(Normalize):
      
    # stole this code from Stack Overflow
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
        
        
def make_cmap(colors, position=None, bit=False, webcolors=False):
    '''
    from: http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
    (added support for web colors)
    
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    '''
    if webcolors:
        bit = False
        try:
            for i in range(len(colors)):
                colors[i] = colorConverter.to_rgb(colors[i])
        except ValueError:
            print "invalid html web color {}.".format(colors[i])
    '''
    if bit:
        try:
            for i in range(len(colors)):
                colors[i] = (bit_rgb[colors[i][0]],
                            bit_rgb[colors[i][1]],
                            bit_rgb[colors[i][2]])
        except:
            try:
                for i in range(len(colors)):
                    colors[i] = colorConverter.to_rgb(colors[i])
            except ValueError:
                print("invalid html web color {}.".format(colors[i]))
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def discrete_colors(array, labels=None, colormap='jet', spccolors=None):
    
    if not labels:
        labels = dict(list(zip(np.unique(array).astype(int), np.unique(array).astype(str))))

    # respecify values/colors to improve contrast
    array = array.astype('float')
    categories = sorted(labels.keys()) # set categories from labels instead of raster, in case there are categories that shouldn't be plotted individually
    
    newvalues = list(range(len(categories)))
    d = dict(list(zip(categories, newvalues)))
    array_t = np.zeros(np.shape(array))

    for k, v in d.items(): array_t[array==k] = v
    
    # update labels and categories list with new values
    for c in categories:
        labels[d[c]] = labels[c]
        labels.pop(c)
    categories = newvalues
    
    # set colormap
    if spccolors:
        colors_list=[spccolors[k] for k in sorted(spccolors.keys())]
        cmap = make_cmap(colors_list, bit=True)
    else:
        cmap = plt.get_cmap(colormap)
           
    # now get corresponding colormap value and label for each category and make legend
    norm = Normalize(np.min(categories), np.max(categories)) # normalize to range of 0-1
    handles = []
    for c in categories:
        fc = cmap(norm(c))    
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=fc))
    labels_list = [labels[c] for c in categories]
    
    return array_t, handles, labels_list, cmap, categories