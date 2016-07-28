#!/usr/bin/env python
#-*- coding:utf-8 -*-

from PyQt4 import QtCore, QtGui

class MyHeaderView(QtGui.QHeaderView): 
    labels=None
    
    def __init__(self,parent,*args):
        QtGui.QHeaderView.__init__(self,QtCore.Qt.Horizontal,parent,*args)
        self.labels=list()
        
    def paintSection(self,painter, rect, columnIndex):  
        pen =	QtGui.QPen (QtCore.Qt.white)         
        color=QtCore.Qt.gray;
        if self.headerMenus.__len__()>columnIndex:
            ActionList=self.headerMenus[columnIndex].actions()
            if ActionList.__len__()>0:
                AllAction=ActionList[0]
                if AllAction.isChecked()==0:  
                    color=QtCore.Qt.darkGray;
        painter.fillRect(rect, color); 
         
        painter.setPen(pen)     
        if self.labels.__len__()>columnIndex:
            painter.drawText(QtCore.QRectF(rect),self.labels[columnIndex])     


class MultipleColumnSortFilterProxyModel(QtGui.QSortFilterProxyModel):
    filterFcn=[];
    count=0
    thres=0.0
    defaultFilterMode="";
    
    def data(self, index, role=QtCore.Qt.DisplayRole):
#        if role == QtCore.Qt.BackgroundColorRole:
##            return QtCore.QVariant(QtGui.QColor(QtCore.Qt.blue)) 
#            return QtGui.QBrush()

        return super(MultipleColumnSortFilterProxyModel, self).data(index, role)

    def filterAcceptsRow(self, row_num, parent):
#        print "filterAcceptsRow"
        model = self.sourceModel() 
        while model.filterMode.__len__() < model.columnCount():
            model.filterMode.append(self.defaultFilterMode)
        
        while self.filterFcn.__len__() < model.columnCount():
            col= self.filterFcn.__len__()
            if model.filterMode[col]=="Thres":
                self.filterFcn.append([])
            elif model.filterMode[col]=="uniqueValues":
                self.filterFcn.append(QtCore.QRegExp( "(" ")",
                                        QtCore.Qt.CaseSensitive))
            else:
                print "ERROR: unknown filter mode: ", model.filterMode[col]
                return
                
        row = [model.item(row_num,icol) for icol in xrange(model.columnCount())]

        tests=[]
        for col in xrange(model.columnCount()):
            if not row[col]:
                continue
            if model.filterMode[col]=="Thres":
                testvalue=False;
                for filter in self.filterFcn[col]:
                    testvalue=testvalue or filter(float(row[col].text()),self.thres);
                tests.append(testvalue)
            elif model.filterMode[col]=="uniqueValues":
#                tests.append(QtGui.QRegExpValidator(\
#                    self.filterFcn[col],None).validate(row[col].text(),0)[0]>0)
                tests.append(self.filterFcn[col].exactMatch(row[col].text()))

        return all(tests) 

class QTableViewWithComboFilter(QtGui.QTableView):
    headerMenus=[]
    defaultFilterMode="uniqueValues"
    validFilters=["uniqueValues","Thres"]
    
    def __init__(self, parent=None, *args):
#        super(QTableViewWithComboFilter, self).__init__(parent, *args)
        QtGui.QTableView.__init__(self,parent, *args)
        
        self.setup()
        
    def setup(self):
#        print "setup"
        self.headerMenus=[]
        
        self.setHorizontalHeader(MyHeaderView(self))
        self.horizontalHeader().headerMenus=self.headerMenus
        self.horizontalHeader().setClickable(True)
        self.horizontalHeader().sectionClicked.connect(\
            self.horizontalHeader_clicked)        
        self.horizontalHeader().sectionPressed.disconnect(\
            self.selectColumn)

        self.setModel(QtGui.QStandardItemModel(self))
        self.sourceModel=self.model()
        self.sourceModel.filterMode=[]
        self.sourceModel.setHorizontalHeaderLabels=self.setHorizontalHeaderLabels
                
        self.proxy = MultipleColumnSortFilterProxyModel(self)
        self.proxy.defaultFilterMode=self.defaultFilterMode;
        self.proxy.setSourceModel(self.sourceModel)
        self.proxy.filterFcn=[];        
        self.setModel(self.proxy)
        
        self.UpdateHeaderMenu()
        
    def setHorizontalHeaderLabels(self,labels):
        self.horizontalHeader().labels=labels
        for icol,label in enumerate(labels):
            self.model().setHeaderData(icol,QtCore.Qt.Horizontal,label);
        self.horizontalHeader().update()

    def populate_menu(self,columnIndex):
#        print "populate_menu"
        while self.headerMenus.__len__() < self.sourceModel.columnCount():
            self.headerMenus.append(None)

        while self.sourceModel.filterMode.__len__() < self.sourceModel.columnCount():
            self.sourceModel.filterMode.append(self.defaultFilterMode)
                
    
        if not self.headerMenus[columnIndex]:
            self.headerMenus[columnIndex]=QtGui.QMenu(self)
            self.headerMenus[columnIndex].setStyleSheet("QMenu { menu-scrollable: 1; }");
            
        if self.sourceModel.filterMode[columnIndex]=="Thres":
            valuesUnique=[">","<="]
            
        elif self.sourceModel.filterMode[columnIndex]=="uniqueValues":
            valuesUnique=[];        
            for row in xrange(self.sourceModel.rowCount()):      
                item=self.sourceModel.item(row, columnIndex)
                if not item:
                    continue
                valuesUnique.append(item.text())
    
            valuesUnique = sorted(list(set(valuesUnique)))

        ActionList= [action.text() for action in self.headerMenus[columnIndex].actions()]   

        #add any new values to menu
        valueActions=[];
        for actionName in valuesUnique:        
            if actionName in ActionList[2:]:
                action=self.headerMenus[columnIndex].actions()[ActionList.index(actionName)]
            else:
                action = QtGui.QAction(actionName, self)
                action.setCheckable(True)
            action.setChecked(1)
            valueActions.append(action)
        
        ActionList= self.headerMenus[columnIndex].actions()
        for iaction,action in enumerate(ActionList):
            if iaction>1: #don't delete 'All' action and separator
                self.headerMenus[columnIndex].removeAction(action)

        ActionList= self.headerMenus[columnIndex].actions()
        if ActionList.__len__()==0:
            actionAll = QtGui.QAction("All", self)
            actionAll.setCheckable(True)
            if not actionAll.isChecked():
                actionAll.setChecked(1)
            QtCore.QObject.connect(actionAll,\
                QtCore.SIGNAL("triggered()"),lambda icol=columnIndex: \
                self.menuAll_clicked(icol,1))
            self.headerMenus[columnIndex].addAction(actionAll)
            self.headerMenus[columnIndex].addSeparator()
            
        for action in valueActions:
            self.headerMenus[columnIndex].addAction(action)            
            QtCore.QObject.connect(action,\
                QtCore.SIGNAL("triggered()"),lambda icol=columnIndex, \
                actionstr=action.text(): \
                self.menuValue_clicked(icol,actionstr))  
            self.headerMenus[columnIndex].addAction(action)
        

    
    def horizontalHeader_clicked(self, columnIndex):
#        print "horizontalHeader_clicked"
        if self.headerMenus.__len__()<(columnIndex+1):
            self.populate_menu(columnIndex)
        
        if not self.headerMenus[columnIndex]:
            self.populate_menu(columnIndex)
                    

        headerPos = self.mapToParent(self.horizontalHeader().cursor().pos())       
        self.headerMenus[columnIndex].popup(headerPos)

    
    def menuAll_clicked(self,columnIndex,triggerfilterupdate=1):
#        print "menuAll_clicked", triggerfilterupdate
        if self.headerMenus.__len__()<columnIndex+1:
            return
        ActionList=self.headerMenus[columnIndex].actions()
        if ActionList.__len__()>0:
            AllAction=ActionList[0]
        for action in ActionList[2:]:
            if action.isSeparator():
                continue
#            print action.text()
            if AllAction.isChecked()==1:
                if action.isChecked()==0:
                    action.setChecked(1)
            else:
                if action.isChecked()==1:
                    action.setChecked(0)
        
        if triggerfilterupdate==1:
            self.menuValue_clicked(columnIndex,'All')
                
    def update_filter(self,columnIndex):
#        print "update_filter"
        if self.headerMenus.__len__()<columnIndex+1:
            return
        while self.sourceModel.filterMode.__len__() < self.sourceModel.columnCount():
            self.sourceModel.filterMode.append(self.defaultFilterMode)

        while self.proxy.filterFcn.__len__() < self.sourceModel.columnCount():
            col= self.proxy.filterFcn.__len__()
            if self.sourceModel.filterMode[col]=="Thres":
                self.proxy.filterFcn.append([])
            elif self.sourceModel.filterMode[col]=="uniqueValues":
                self.proxy.filterFcn.append(QtCore.QRegExp( "(" ")",
                                        QtCore.Qt.CaseSensitive))
            else:
                print "Unknown filter mode: ", self.sourceModel.filterMode[col]
                return
            
        if self.sourceModel.filterMode[columnIndex]=="Thres":       
            filterFcn=[]
            allChecked=1;
            for iaction,action in enumerate(self.headerMenus[columnIndex].actions()):
                if iaction==0:
                    AllAction=action;
                    continue
                if action.isSeparator():
                    continue
    #            print action.text(), action.isChecked()
                if action.isChecked():
                    if action.text()=="<=":
                        filterFcn.append(lambda value,thres: (value<=thres))
                    elif action.text()==">":
                        filterFcn.append(lambda value,thres: (value>thres))
                else:
                    allChecked=0;
            if AllAction.isChecked() != allChecked: 
                AllAction.setChecked(allChecked)
                
            self.proxy.filterFcn[columnIndex]=filterFcn
            
        elif self.sourceModel.filterMode[columnIndex]=="uniqueValues":                
            filterFcn="";
            allChecked=1;
            for iaction,action in enumerate(self.headerMenus[columnIndex].actions()):
                if iaction==0:
                    AllAction=action;
                    continue
                if action.isSeparator():
                    continue
    #            print action.text(), action.isChecked()
                if action.isChecked():
                    if not filterFcn:
                        filterFcn=QtCore.QRegExp.escape(action.text())
                    else:
                        filterFcn+="|" + QtCore.QRegExp.escape(action.text())
                else:
                    allChecked=0;
            if AllAction.isChecked() != allChecked: 
                AllAction.setChecked(allChecked)
    #        print columnIndex, AllAction.isChecked(),allChecked
                                            
            self.proxy.filterFcn[columnIndex] = QtCore.QRegExp( "("+ filterFcn+")",
                                            QtCore.Qt.CaseSensitive);
        
    def menuValue_clicked(self,columnIndex,stringAction):
#        print "menuValue_clicked"
        #print 'columnIndex: ', columnIndex, "stringAction: " , stringAction
        self.update_filter(columnIndex)
        self.horizontalHeader().updateSection(columnIndex)
        self.ApplyFilter(columnIndex)

    
    def ApplyFilter(self,columnIndex):
        if self.sourceModel.columnCount()<columnIndex+1:
            return
        selection=self.selectionModel().selection()
        sourceselection=self.proxy.mapSelectionToSource(selection)
        self.proxy.setFilterKeyColumn(columnIndex)
        self.proxy.mapSelectionFromSource(sourceselection)
        

    def UpdateHeaderMenu(self):        
#        print "UpdateHeaderMenu"
        for col in xrange(self.sourceModel.columnCount()):
            self.populate_menu(col)
            if self.headerMenus.__len__()>=col+1:
                ActionList=self.headerMenus[col].actions()
                if ActionList.__len__()>0:                        
                    actionAll=ActionList[0]
                    if not actionAll.isChecked():
                        actionAll.setChecked(1)
            self.menuAll_clicked(col,0)
            self.update_filter(col)
        self.menuAll_clicked(0,1)
