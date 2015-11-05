from warnings import warn
try:
    import flopy
except ImportError:
    warn('Could not import flopy; xsection.py module will not work.')
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_utils as mplu
from mpl_toolkits.axes_grid1 import ImageGrid
import sys
sys.path.append('../../../GitHub/Figures')
from report_figs import ReportFigures

# set the style for USGS reports
figures = ReportFigures()
figures.set_style()


class ModelInfo(object):
    
    def __init__(self, modelname, run_folder, disfile=None):
        
        # input files
        self.modelname = modelname
        self.namfile = os.path.join(run_folder, modelname + '.nam')
        self.run_folder = run_folder
        if not disfile:
            self.disfile = os.path.join(run_folder, modelname + '.dis')
        else:
            self.disfile = disfile
        
        # read in model grid info using flopy
        m = flopy.modflow.Modflow(model_ws=run_folder)
        nf = flopy.utils.mfreadnam.parsenamefile(self.namfile, {})
        dis = flopy.modflow.ModflowDis.load(self.disfile, m, nf)
        self.dis = dis
        self.elevs = np.zeros((dis.nlay + 1, dis.nrow, dis.ncol))
        self.elevs[0, :, :] = dis.top.array
        self.elevs[1:, :, :] = dis.botm.array
        
        # get left hand vertex coordinates (only works for rectilinear grids!)
        #self.X = np.append(np.array([0]), np.cumsum(self.dis.delc))
        #self.Y = np.append(np.array([0]), np.cumsum(self.dis.delr))
        self.X = np.cumsum(self.dis.delc)
        self.Y = np.cumsum(self.dis.delr)


class Xsection(ModelInfo):


    def makePDF(self, row=None, col=None, values=None, VE=20, reverse_dir=False,
                             labels=None, colors=None, ylims=None, equal_aspect_panel=True,
                             xunits='feet', yunits='feet', dirlabels=[],
                             ypad=0.2, explanation=True, outpdf=None,
                             **fig_kwargs):

        # values for each model cell (optional)
        self.values = values
        self.VE = VE
        self.reverse_dir = reverse_dir
        self.labels = labels
        self.colors = colors
        self.ylims = ylims
        self.yunits = yunits
        self.xunits = xunits
        self.dirlabels = dirlabels
        self.ypad = ypad # amount of whitespace to leave in figure beyond min and max elevation
        self.explanation = explanation

        if not outpdf and row:
            self.outpdf = self.modelname + '_xs_r{}.pdf'.format(row)
        elif not outpdf and col:
            self.outpdf = self.modelname + '_xs_c{}.pdf'.format(col)
        else:
            self.outpdf = outpdf


        self.plot(row, col, **fig_kwargs)

        plt.savefig(self.outpdf, dpi=300)



    def make_arrays(self, row=None, col=None):
    
        # if no values were supplied, make an array of the layer numbers
        if not self.values:
            self.values = np.ones((self.dis.nlay, self.dis.nrow, self.dis.ncol))
            for i in range(self.dis.nlay):
                self.values[i, :, :] = self.values[i, :, :] * (i+1)

        # elevations and values for cross section slice
        if col:
            self.z = self.elevs[:, :, col]
            self.H = np.zeros(np.shape(self.z)) # make array of horizontal spacing same shape as elevations
            self.H[:, :] = self.X
            self.xs_values = self.values[:, :, col]
        else:
            self.z = self.elevs[:, row, :]
            self.H = np.zeros(np.shape(self.z))
            self.H[:, :] = self.Y
            self.xs_values = self.values[:, row, :]

        if self.reverse_dir:
            self.z = np.fliplr(self.z)
            self.xs_values = np.fliplr(self.xs_values)


            
    def subplot(self, ax, aspect=None, ylims=None):

        if not aspect:
            aspect = self.VE
        
        # make a subplot
        ax.pcolormesh(self.H, self.z, self.arr, cmap=self.cmap) 

        ax.set_aspect(aspect)

        if ylims:
            ax.set_ylim(ylims)
        else:
            ylims = ax.get_ylims()

        if aspect == 1:
            velabel = 'No vertical exaggeration'
            vepad = 2
            
            # only three vertical ticks for thin equal aspect section
            ax.yaxis.set_ticks(np.array([ylims[0], 0, ylims[1]]))
            
        else:
            velabel = 'Vertical exaggeration: {}x'.format(aspect)
            vepad = 1
            
        ax.text(0.98, 0.05 * vepad, velabel, transform=ax.transAxes, ha='right', va='bottom',
                        family='Univers 47 Condensed Light', fontsize=7)

        
    def plot(self, row=None, col=None, equal_aspect_panel=True, **fig_kwargs):
        
        # make arrays of cell centers, elevations, and values
        self.make_arrays(row, col)
        
        # make array of discrete colors
        self.arr, h, l, self.cmap, categories = mplu.discrete_colors(self.xs_values, labels=self.labels, spccolors=self.colors)
            
        # include a top panel with the same section at equal aspect
        if equal_aspect_panel:
            # setup a grid of cross section plots
            F = plt.figure(1, **fig_kwargs)
            grid = ImageGrid(F, 111, # similar to subplot(111)
                            nrows_ncols = (2, 1),
                            axes_pad = 0.3,
                            add_all=True,
                            label_mode = "L",
                            )
            
            ylen = (np.max(self.z) - np.min(self.z))
            xlen = np.max(np.abs(self.H))
            plotaspect = xlen / ylen
            aspectcorrection = plotaspect / 10
            
            if not self.ylims:
                self.ylims = [(-10000, 10000), (np.min(self.z) - self.ypad * ylen, np.max(self.z) + self.ypad * ylen)]
            
            for i, aspect in enumerate([1, self.VE]):

                ax = grid[i]

                self.subplot(ax, aspect=aspect, ylims=self.ylims[i])

        else:
            F = plt.figure(1, **fig_kwargs)
            ax = F.add_subplot(111)
            self.subplot(ax)

        if self.explanation:
            # make the legend
            plt.rcParams['font.family'] = 'Univers 67 Condensed'
            lg = ax.legend(h,l,title='Explanation', fontsize=8, bbox_to_anchor=(0., -0.3), loc=2, ncol=3, 
                      borderaxespad=0.1, borderpad=1.0)
            lg.draw_frame(False)

        if len(self.dirlabels) == 2:
            ax.text(0, 1.01, self.dirlabels[0].upper(), transform=ax.transAxes, ha='left', va='bottom',
                    family='Univers 47 Condensed Light', fontsize=8)
            ax.text(1, 1.01, self.dirlabels[1].upper(), transform=ax.transAxes, ha='right', va='bottom',
                    family='Univers 47 Condensed Light', fontsize=8)

        # add in lines for layers
        for i in range(np.shape(self.z)[0]):
            ax.plot(self.H[0], self.z[i,:], c='k', linewidth=0.25)

        ax.set_ylabel('Elevation, {}'.format(self.yunits))
        ax.set_xlabel('Distance, {}'.format(self.xunits))

        plt.tight_layout()