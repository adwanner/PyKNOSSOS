import numpy as np;
from PyQt4 import QtCore, QtGui;
import vtk;
import os;
import glob
from collections import OrderedDict
from spyderlib.widgets.internalshell import InternalShell as SpyderShell

initialized=0
ariadne=None
CubeLoader=None

def init(main=None,loader=None):
    global ariadne
    global CubeLoader
    global initialized
    if not ariadne:
        ariadne=main
    if not CubeLoader:
        CubeLoader=loader
    if not ((not ariadne) or (not CubeLoader)):
        print "Initialized plugin {0}".format( __name__)
        initialized=1;
    return Plugin()

class Plugin():
    def __init__(self):
        1
        
    def runPlugin(self):
        global ariadne
        global CubeLoader

        ns = {'ariadne': ariadne, 'CubeLoader': CubeLoader}
        msg = "Try for example: ariadne.Neurons or CubeLoader.LoadDataset()"
        # Note: by default, the internal shell is multithreaded which is safer
        # but not compatible with graphical user interface creation.
        # For example, if you need to plot data with Matplotlib, you will need
        # to pass the option: multithreaded=False
        cons = SpyderShell(ariadne,multithreaded=False, namespace=ns, message=msg)
        ariadne.shell = cons
        
         # Create the console widget
        font = QtGui.QFont("Courier new")
        font.setPointSize(10)
        # Setup the console widget
        cons.set_font(font)
        cons.set_codecompletion_auto(True)
        cons.set_calltips(True)
        cons.setup_calltips(size=600, font=font)
        cons.setup_completion(size=(300, 180), font=font)
        ariadne.Console.setWidget(cons)
        ariadne.Console.setFloating(0)
        ariadne.Console.setVisible(1)
