from distutils.core import setup

DESCRIPTION = """\
Classes for making report-quality figures using Matplotlib.
"""

def run():
    setup(name="Figures",
          version="0.1",
          description="Classes for making report-quality figures using Matplotlib",
          author="Andy Leaf",
          packages=["Figures"],
          long_descripton=DESCRIPTION,
          )
          
if __name__ == "__main__":
    run()