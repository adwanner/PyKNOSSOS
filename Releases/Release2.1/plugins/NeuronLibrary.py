from PyQt4 import QtCore, QtGui, uic;
import numpy as np;
import sys;
import os;
from collections import OrderedDict
from QTableViewWithComboFilter import *

import csv

initialized=0
ariadne=None
CubeLoader=None
NeuronLibrary=None

# determine if application is a script file or frozen exe
#if getattr(sys, 'frozen', False):
#    application_path = os.path.dirname(sys.executable)
#elif __file__:
application_path = os.path.dirname(__file__)

if sys.platform.startswith('win'):
    win=1
else:
    win=0

def init(main=None,loader=None):
    global ariadne
    global CubeLoader
    global initialized
    if not ariadne:
        ariadne=main
    if not CubeLoader:
        CubeLoader=loader
        
    global NeuronLibrary
    NeuronLibrary = NeuronLibraryGUI(os.path.join(application_path,\
    "NeuronLibrary.ui"),ariadne)
         
        
    if not ((not ariadne) or (not CubeLoader)):
        print "Initialized plugin {0}".format( __name__)
        initialized=1;
    return NeuronLibrary

class NeuronLibraryGUI(QtGui.QDockWidget):
    _LibraryFile=""
    _DataSourcePath=""
    ValidFileExt=[".nmx",".mat",".nml",".txt",".csv",".ddx"]

    def __init__(self,uifile,parent=None):
        QtGui.QDockWidget.__init__(self, parent)
        uic.loadUi(uifile, self)
        
        self.setWindowTitle("Neuron library")        
        QtCore.QObject.connect(self.doubleSpinBox_ThresInnervLength,\
            QtCore.SIGNAL("editingFinished()"),self.update_thres)
            
        QtCore.QObject.connect(self.btn_LoadLibrary,\
            QtCore.SIGNAL("clicked()"),self.load_libraryfile);
        QtCore.QObject.connect(self.btn_ShowNeurons,\
            QtCore.SIGNAL("clicked()"),self.show_neurons);
            
        QtCore.QObject.connect(self.btn_DataSource,\
            QtCore.SIGNAL("clicked()"),self.set_data_source);

        # initiate table
        self.NeuronTable.doubleClicked.connect(self.show_neurons)            

        self.NeuronTable.verticalHeader().setVisible(False)
         
        self.NeuronTable.sourceModel.filterMode=["uniqueValues","Thres"]      
        
        self.NeuronTable.UpdateHeaderMenu()
        
        def contextMenuEvent(pos):
            self.NeuronTable.menu = QtGui.QMenu()
            chcolorAction = QtGui.QAction('Change color', self)
            chcolorAction.triggered.connect(self.changeNeuronColor)
            self.NeuronTable.menu.addAction(chcolorAction)
            # add other required actions
            self.NeuronTable.menu.popup(QtGui.QCursor.pos())
        
        self.NeuronTable.contextMenuEvent=contextMenuEvent

        self.show()
        
    def runPlugin(self):   
        self.load_libraryfile()
        
    def update_thres(self):
        thres=self.doubleSpinBox_ThresInnervLength.value()
        if not (not self.NeuronTable.proxy):
            self.NeuronTable.proxy.thres=thres;    
        self.NeuronTable.ApplyFilter(0)
        
    def load_libraryfile(self,filename=None):
        fileext="*.csv"
        currentPath=[""]
        if filename==None:
            if not (not self._LibraryFile):
                currentPath= os.path.split(unicode(self._LibraryFile))
            if os.path.isdir(currentPath[0]):
                filename = QtGui.QFileDialog.getOpenFileName(self,"Open file...",currentPath[0],fileext);
            else:
                if os.path.isdir(application_path):
                    filename= QtGui.QFileDialog.getOpenFileName(self,"Open file...",application_path,fileext);
                else:
                    filename= QtGui.QFileDialog.getOpenFileName(self,"Open file...","",fileext);
            if filename.__len__()==0:
                return
            
        allRows=[]     
        if not os.path.isfile(filename):
            print "Could not find file: ", filename
            return
        with open(filename, 'rb') as fid:
            reader = csv.reader(fid)
            allRows = list(reader)

        if allRows.__len__()<2:
            print "Invalid neuron library file."
            return
        self.NeuronTable.sourceModel.clear()
        self.NeuronTable.sourceModel.setRowCount(0)
        self.NeuronTable.sourceModel.setColumnCount(0)

        Header="";
        FilterMode=[]
        NRows=allRows.__len__()
        NColumns=0
        for irow, row in enumerate(allRows):
            NColumns=max(NColumns,row.__len__())
            for icol,value in enumerate(row):
                if irow==0:
                    Header+=value+";"
                elif irow==1:
                    FilterMode.append(value)
                else:
                    item=QtGui.QStandardItem(value)
                    item.setEditable(0)
                    self.NeuronTable.sourceModel.setItem(\
                        irow-2,icol,item)

        self.NeuronTable.sourceModel.setRowCount(NRows)
        self.NeuronTable.sourceModel.setColumnCount(NColumns)

        Header= Header.split(";")
        Header=Header[:-1]
        self.NeuronTable.sourceModel.filterMode=[]
        for filtermode in FilterMode:
            if filtermode in self.NeuronTable.validFilters:
                self.NeuronTable.sourceModel.filterMode.append(filtermode)
            else:
                self.NeuronTable.sourceModel.filterMode.append(self.NeuronTable.defaultFilterMode);
                
        self.NeuronTable.sourceModel.setHorizontalHeaderLabels(Header)   
        self.NeuronTable.sourceModel.setRowCount(allRows.__len__()-2)
        self.NeuronTable.sourceModel.setColumnCount(Header.__len__())
#        self.NeuronTable.sourceModel.layoutChanged.emit()
        print "********** LIBRARY FILE LOADED **********"
        print filename
        print "*****************************************"
        self._LibraryFile=unicode(str(filename))

        self.NeuronTable.UpdateHeaderMenu()
    
        self.NeuronTable.setVisible(0); #otherwise columns would only be resized to visible content
        self.NeuronTable.resizeColumnsToContents();
        self.NeuronTable.setVisible(1);
        
        basepath, currFile = os.path.split(unicode(filename))        
        Title="Neuron library: {0}".format(currFile)
        self.setWindowTitle(Title)
        
    def set_data_source(self):
        if not self._DataSourcePath:
            self._DataSourcePath=application_path;
        datapath = QtGui.QFileDialog.getExistingDirectory(self,"Choose data source directory...",self._DataSourcePath);             
        if not (not datapath):
            datapath=unicode(datapath)  
            self._DataSourcePath=datapath
        else:
            datapath=None
        return datapath
        
    def generate_neuron_list(self,basepath,NeuronIDs):
        filelist=list()
        for ID in NeuronIDs:
            found=False;
            for fileext in self.ValidFileExt:
                neuronfile=os.path.join(basepath,"Neuron_id{0:{fill}4}{1}".format(\
                    int(ID),fileext,fill=0))
                if os.path.exists(neuronfile):
                    found=True;
                    break;
            if not found:
                print "File Neuron_id{0:{fill}4} not found in source path {1}".format(\
                int(ID),basepath,fill=0);
                return None;
            filelist.append(neuronfile)
        return filelist
        
    def get_selected_neuron_IDs(self):
        model=self.NeuronTable.model();
        whichcol=None
        for col in range(model.columnCount()):
            if model.headerData(col,QtCore.Qt.Horizontal).toString()=="ID":
                whichcol=col
                break
        if whichcol==None:
            print "ERROR: No column with header ID found."
            return None,None
        rows = sorted(set(index.row() for index in
              self.NeuronTable.selectedIndexes()))
        SelectedNeuronIDs=[float(self.NeuronTable.proxy.index(irow,whichcol).data().toString())
            for irow in rows]
#        print SelectedNeuronIDs      
        return SelectedNeuronIDs,rows
    
    def get_fitered_neuron_IDs(self):
        model=self.NeuronTable.model();
        whichcol=None
        for col in range(model.columnCount()):
            if model.headerData(col,QtCore.Qt.Horizontal).toString()=="ID":
                whichcol=col
                break
        if whichcol==None:
            print "ERROR: No column with header ID found."
            return
        FilteredNeuronIDs=[float(self.NeuronTable.proxy.index(irow,whichcol).data().toString())
            for irow in xrange(model.rowCount())]
        return FilteredNeuronIDs
          
    def show_neurons(self,whichIndex=None):
        if whichIndex==None:
            FilteredNeuronIDs=self.get_fitered_neuron_IDs()
        else:
            model=self.NeuronTable.model();
            whichcol=None
            for col in range(model.columnCount()):
                if model.headerData(col,QtCore.Qt.Horizontal).toString()=="ID":
                    whichcol=col
                    break
            if whichcol==None:
                print "ERROR: No column with header ID found."
                return
            if not (whichIndex.__class__.__name__=='QModelIndex'):
                print "ERROR: Unknown index type: " , whichIndex.__class__.__name__
                return
            irow=whichIndex.row()
            FilteredNeuronIDs=[float(model.index(irow,whichcol).data().toString())]       
        global ariadne
        if not os.path.isdir(self._DataSourcePath):
            basepath=self.set_data_source();
        else:
            basepath=self._DataSourcePath;
        if not basepath:
            print "Data source path not found."
            return
            
        if not os.path.isdir(basepath):
            print "Data source path not found."
            return
            
        if ariadne==None:
            return
               
        filelist = self.generate_neuron_list(basepath,FilteredNeuronIDs)
        if filelist==None:
            filelist=list() 
        glomerulifile=os.path.join(basepath,"Glomeruli.nmx");
        if self.ckbx_IncludeGlomeruli.isChecked() and os.path.exists(glomerulifile):
            filelist.append(glomerulifile)

        if hasattr(ariadne,'Open'):
            ariadne.Open(filelist,'Overwrite');
        else:
            print "Could not show files because the parent application has no Open method."

        model=self.NeuronTable.model();
        whichcol=None
        for col in range(model.columnCount()):
            if model.headerData(col,QtCore.Qt.Horizontal).toString()=="ID":
                whichcol=col
                break
        if whichcol==None:
            print "ERROR: No column with header ID found."
            return
        changed=False;
        for irow in xrange(model.rowCount()):
            index=model.index(irow,whichcol)
            color=QtGui.QColor(model.data(index,QtCore.Qt.BackgroundColorRole))
            color=np.array(color.getRgb())/255.00
            if not hasattr(ariadne,'Neurons'):
                return
            neuronID=float(model.data(index).toString())
            if not neuronID in ariadne.Neurons:
                continue
            obj=ariadne.Neurons[neuronID]
            obj.change_color([color[0],color[1],color[2],color[3]])
            changed=True;
        if changed:
            ariadne.QRWin.Render()
            
        
    def changeNeuronColor(self):
        selectedNeuronIDs,selectedRows=self.get_selected_neuron_IDs()
        if not selectedNeuronIDs:
            return
        if selectedNeuronIDs.__len__()==0:
            return
        color=QtGui.QColor()
        obj=None
        if hasattr(ariadne,'Neurons'):
            if selectedNeuronIDs[0] in ariadne.Neurons:
                obj=ariadne.Neurons[selectedNeuronIDs[0]]
                color=obj.LUT.GetTableValue(obj.colorIdx)
                color=QtGui.QColor().fromRgb(*color)
        color=QtGui.QColorDialog.getColor(color,self, "Color")
        if not color.isValid():
            return
        for irow in selectedRows:
            for icol in xrange(self.NeuronTable.proxy.columnCount()):
                self.NeuronTable.proxy.setData(
                            self.NeuronTable.proxy.index(irow,icol),
                                color,
                                QtCore.Qt.BackgroundColorRole
                        )
        color=np.array(color.getRgb())/255.00
        color[3]=1.0
        if not hasattr(ariadne,'Neurons'):
            return
        changed=False;
        for neuronID in selectedNeuronIDs:
            if not neuronID in ariadne.Neurons:
                continue
            obj=ariadne.Neurons[neuronID]
            obj.change_color([color[0],color[1],color[2],color[3]])
            changed=True
        if changed:
            ariadne.QRWin.Render()
            
    
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(os.path.join('imageformats','PyKNOSSOS.ico')))
    
    main=QtGui.QMainWindow()
    NeuronLibrary=init(main)
    main.show()
    NeuronLibrary.runPlugin()

    sys.exit(app.exec_())
