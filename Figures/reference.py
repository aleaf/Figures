import sys
import os
import numpy as np

class epsgRef:
    """Sets up a local database of projection file text referenced by epsg code.
    The database is located in the site packages folder in epsgref.py, which
    contains a dictionary, prj, of projection file text keyed by epsg value.
    """
    def __init__(self):
        sp = [f for f in sys.path if f.endswith('site-packages')][0]
        self.location = os.path.join(sp, 'epsgref.py')

    def _remove_pyc(self):
        try: # get rid of pyc file
            os.remove(self.location + 'c')
        except:
            pass
    def make(self):
        if not os.path.exists(self.location):
            newfile = open(self.location, 'w')
            newfile.write('prj = {}\n')
            newfile.close()
    def reset(self, verbose=True):
        os.remove(self.location)
        self._remove_pyc()
        self.make()
        if verbose:
            print('Resetting {}'.format(self.location))
    def add(self, epsg, prj):
        """add an epsg code to epsgref.py"""
        with open(self.location, 'a') as epsgfile:
            epsgfile.write("prj[{:d}] = '{}'\n".format(epsg, prj))
    def remove(self, epsg):
        """removes an epsg entry from epsgref.py"""
        from epsgref import prj
        self.reset(verbose=False)
        if epsg in prj.keys():
            del prj[epsg]
        for epsg, prj in prj.items():
            self.add(epsg, prj)
    @staticmethod
    def show():
        import importlib
        import epsgref
        importlib.reload(epsgref)
        from epsgref import prj
        for k, v in prj.items():
            print('{}:\n{}\n'.format(k, v))


def getprj(epsg, addlocalreference=True):
    """Gets projection file (.prj) text for given epsg code from spatialreference.org
    See: https://www.epsg-registry.org/

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system
    addlocalreference : boolean
        adds the projection file text associated with epsg to a local
        database, epsgref.py, located in site-packages.

    Returns
    -------
    prj : str
        text for a projection (*.prj) file.
    """
    epsgfile = epsgRef()
    prj = None
    try:
        from epsgref import prj
        prj = prj.get(epsg)
    except:
        epsgfile.make()

    if prj is None:
        prj = get_spatialreference(epsg, text='prettywkt')
    if addlocalreference:
        epsgfile.add(epsg, prj)
    return prj

def get_spatialreference(epsg, text='prettywkt'):
    """Gets text for given epsg code and text format from spatialreference.org
    Fetches the reference text using the url:
        http://spatialreference.org/ref/epsg/<epsg code>/<text>/

    See: https://www.epsg-registry.org/

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system
    text : str
        string added to url

    Returns
    -------
    url : str

    """
    url = "http://spatialreference.org/ref/epsg/{0}/{1}/".format(epsg, text)
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen
    try:
        urlobj = urlopen(url)
        text = urlobj.read().decode()
    except:
        e = sys.exc_info()
        print(e)
        print('Need an internet connection to look up epsg on spatialreference.org.')
        return
    text = text.replace("\n", "")
    return text

def getproj4(epsg):
    """Gets projection file (.prj) text for given epsg code from spatialreference.org
    See: https://www.epsg-registry.org/

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system

    Returns
    -------
    prj : str
        text for a projection (*.prj) file.
    """
    return get_spatialreference(epsg, text='proj4')