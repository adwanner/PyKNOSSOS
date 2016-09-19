#PyKnossos has successfully been tested with Python 2.7

#PyKnossos requires the following extension packages/modules:
#ast, collections, configobj, Crypto.Cipher, cStringIO, ctypes, fnmatch, glob, 
#images2gif, imp, inspect, itertools, lxml, multiprocessing, numpy, os, PIL, 
#PyQt4, QxtSpanSlider, random, re, scipy, shutil, sip, struct, threading, 
#time, uuid, vtk, zipfile

#Non-standard extension packages/modules of PyKNOSSOS:
#QxtSpanSlider.py https://github.com/mkilling/QxtSpanSlider.py
#images2gif https://bitbucket.org/bench/images2gif.py 

#For Windows, many Python extension packages can easily be installed using the
# binaries provided here: http://www.lfd.uci.edu/~gohlke/pythonlibs/

#This has been used successfully with the following PyQt4 versions 
#Qt version:   4.8.6  (windows and linux)
#SIP version:  4.16.2 (windows) and 4.15.5 (linux)
#PyQt version: 4.11.1 (windows) and 4.10.4 (linux)

#developer switches
usermode=0 #0: expert mode, 1: user mode

doload=1 #start cube loader
mprocess=1 #use seperate process for cube loader
experimental=1 #show experimental features

#your encryption key used for encrypting and decrypting annotation files
encryptionkey='EncryptPyKnossos';
#AES key must be either 16, 24, or 32 bytes long

PyKNOSSOS_VERSION='PyKNOSSOS2.120160918'

if usermode==1:
    experimental=0
    mprocess=1
    doload=1

import sip
import random, struct
import fnmatch
import glob
import cStringIO

from configobj import ConfigObj
from collections import OrderedDict
import ast

from lxml import etree as lxmlET

from PyQt4 import QtGui, QtCore, uic

if not sip.getapi('QString')==1:
    QtCore.QString=unicode
    QtCore.QStringList=list


import sys

from QxtSpanSlider import QxtSpanSlider
import vtk

import multiprocessing.forking
from multiprocessing.sharedctypes import Value, RawArray
from multiprocessing import Pool
import threading

from ctypes import *
import time

if usermode==0:
    import scipy.io   
    try:
        from scipy.sparse.csgraph import _validation
    except:
        pass

import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import os
import itertools
import re
import zipfile
from zipfile import ZipFile
import shutil
import uuid
from Crypto.Cipher import AES

import images2gif
from PIL import Image
import imp

#global variable, makes sure that lines and vertices lying on the reslice plane are rendered on top of the reslices
#vtk.vtkDataSetMapper.SetResolveCoincidentTopologyToShiftZBuffer()
#vtk.vtkDataSetMapper.SetResolveCoincidentTopologyZShift(0.001)

#vtk.vtkDataSetMapper.SetResolveCoincidentTopology(1)
#vtk.vtkDataSetMapper.SetResolveCoincidentTopologyZShift(0.1);
#vtk.vtkDataSetMapper.SetResolveCoincidentTopologyToPolygonOffset()
#vtk.vtkDataSetMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0,500.0)
# Avoid z-buffer fighting

vtk.vtkPolyDataMapper().SetResolveCoincidentTopologyToPolygonOffset()
vtk.vtkAlgorithm().GlobalWarningDisplayOn()

colormap=[[0,0.6,0],[0,0.62667,0],[0,0.65333,0],[0,0.68,0],[0,0.70667,0],[0,0.73333,0],[0,0.76,0],[0,0.78667,0],[0,0.81333,0],[0,0.84,0],[0,0.86667,0],[0,0.89333,0],[0,0.92,0],[0,0.94667,0],[0,0.97333,0],[0,1,0],[0.0625,1,0],[0.125,1,0],[0.1875,1,0],[0.25,1,0],[0.3125,1,0],[0.375,1,0],[0.4375,1,0],[0.5,1,0],[0.5625,1,0],[0.625,1,0],[0.6875,1,0],[0.75,1,0],[0.8125,1,0],[0.875,1,0],[0.9375,1,0],[1,1,0],[1,0.9375,0],[1,0.875,0],[1,0.8125,0],[1,0.75,0],[1,0.6875,0],[1,0.625,0],[1,0.5625,0],[1,0.5,0],[1,0.4375,0],[1,0.375,0],[1,0.3125,0],[1,0.25,0],[1,0.1875,0],[1,0.125,0],[1,0.0625,0],[1,0,0],[0.96875,0,0],[0.9375,0,0],[0.90625,0,0],[0.875,0,0],[0.84375,0,0],[0.8125,0,0],[0.78125,0,0],[0.75,0,0],[0.71875,0,0],[0.6875,0,0],[0.65625,0,0],[0.625,0,0],[0.59375,0,0],[0.5625,0,0],[0.53125,0,0],[0.5,0,0]];

def isnan(num):
    return num != num
    
def isnumeric(string):
    try:
        float(string)
        return True
    except ValueError:
        return False       

def array2str(value):
    value=unicode(value)
    value=value.replace(' ',',')
    value=value.replace('\n',',')
    newvalue=value
    value=''
    while not newvalue==value:
        value=newvalue
        newvalue=value.replace(',,',',')
    value=value.replace('[,','[')
    value=value.replace(',]',']')
    value=value.replace('(,','[')
    value=value.replace(',)',']')
    value=value.replace(')',']')
    value=value.replace('(','[')
    return value
    

def encrypt_file(key, in_filename, out_filename=None, chunksize=64*1024):
    """ Encrypts a file using AES (CBC mode) with the
        given key.

        key:
            The encryption key - a string that must be
            either 16, 24 or 32 bytes long. Longer keys
            are more secure.

        in_filename:
            Name of the input file

        out_filename:
            If None, '<in_filename>.enc' will be used.

        chunksize:
            Sets the size of the chunk which the function
            uses to read and encrypt the file. Larger chunk
            sizes can be faster for some files and machines.
            chunksize must be divisible by 16.
    """
    if not out_filename:
        out_filename = in_filename + '.enc'

    iv = ''.join(chr(random.randint(0, 0xFF)) for i in range(16))
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    filesize = os.path.getsize(in_filename)

    with open(in_filename, 'rb') as infile:
        with open(out_filename, 'wb') as outfile:
            outfile.write(struct.pack('<Q', filesize))
            outfile.write(iv)

            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                elif len(chunk) % 16 != 0:
                    chunk += ' ' * (16 - len(chunk) % 16)

                outfile.write(encryptor.encrypt(chunk))


def decrypt_file(key, in_filename, out_filename=None, chunksize=24*1024):
    """ Decrypts a file using AES (CBC mode) with the
        given key. Parameters are similar to encrypt_file,
        with one difference: out_filename, if not supplied
        will be in_filename without its last extension
        (i.e. if in_filename is 'aaa.zip.enc' then
        out_filename will be 'aaa.zip')
    """
    if not out_filename:
        out_filename = os.path.splitext(in_filename)[0]

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key, AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)

def decrypt_string(key, in_string, chunksize=24*1024):
    """ Decrypts a string using AES (CBC mode) with the
        given key. Parameters are similar to decrypt_file,
    """
    if not in_string:
        return None
    infile = cStringIO.StringIO(in_string)
        
    origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
    iv = infile.read(16)
    decryptor = AES.new(key, AES.MODE_CBC, iv)
    
    output_str=''
    while True:
        chunk = infile.read(chunksize)
        if len(chunk) == 0:
            break
        output_str+=decryptor.decrypt(chunk)

    return output_str[:origsize]

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

if sys.platform.startswith('win'):
    win=1
else:
    win=0
    
print "application path: " + application_path

sys.path.append(os.path.join(application_path,'plugin'));

maxNStreamingChannels=50;
if win:
    maxContRAMBlock=5*5*5*128*128*128+1; #On windows we are limited by 32bit
    #Note: The number of cubes in memory: NCubesInMemory= 3 * NCubesPerEdge^3
    #Memory usage: NCubesInMemory*CubeSize^3
    print application_path;
    temp_dir='';
    lib_path=os.path.join(temp_dir,r"loaderlib.dll")
    loaderlib = cdll.LoadLibrary(lib_path)
    lib_path=os.path.join(temp_dir,r"extractROIlib.dll")
    extractROIlib = cdll.LoadLibrary(lib_path)
else:
    maxContRAMBlock=9*9*9*128*128*128+1; #No real limit, but 10 is already a lot...
    #Note: The number of cubes in memory = 3 * NCubesPerEdge^3
    #Memory usage: NCubesInMemory*CubeSize^3
    loaderlib = cdll.LoadLibrary(os.path.join(application_path, r"clibraries/loaderlib.so"))
    extractROIlib = cdll.LoadLibrary(os.path.join(application_path ,r"clibraries/extractROIlib.so"))


if usermode>0:
    availableFileFormats=["PyKnossos (*.nmx)"] 
else:
    availableFileFormats= ["PyKnossos (*.nmx)","KNOSSOS (*.nml)","Ariadne (*.amx)"]

selectColor=1
deleteColor=0

import inspect
    
def lineno():
    """Returns the current line number in our program."""
    print inspect.currentframe().f_back.f_lineno
    return inspect.currentframe().f_back.f_lineno

def invalidvector(value):
    class_str=value.__class__.__name__
    if class_str=='ndarray':
        value=value.tolist()
        if not value:
            return True
        else:
            return (None in value)
    elif class_str=='tuple' or class_str=='list':
        return (None in value)
    else:
        return (not value)

def str2array(value):
    if not (value.__class__.__name__=='str' or value.__class__.__name__=='unicode'):
        return value
    toreplace=['[',']','(',')']
    for ichar in toreplace: value=value.replace(ichar,'')
    value=value.split(',')
    return np.array(map(float,value))
    
def str2num(format,value):
    if not (value.__class__.__name__=='str' or value.__class__.__name__=='unicode'):
        return value
    if format.startswith('uint'):
        return uint(value)
    if format.startswith('int'):
        return int(value)
    if format.startswith('float'):
        return float(value)


def ParseNML(filename,root,parseflags=True):
    Data=dict()

    #assuming filename like dataset_parent_id***_objtype_flags_date.nml
    temppath, tempname = os.path.split(unicode(filename))
    tempname, ext = os.path.splitext(tempname)
    parts=tempname.split('_')
    NParts=parts.__len__()
#            if NParts>0:
#                dataset=parts[0]
#            else:
#                dataset=''
    if NParts>1:
        parent=parts[1]
    else:
        parent='neuron'
#            if NParts>2:
#                id=parts[2]
#            else:
#                id=None
    if NParts>3:
        objtype=parts[3]
    else:
        objtype='none'
    if NParts>4 and parseflags==1:
        flags=parts[4]
    else:
        flags=''

#            neuronId=float(id.replace('id',''))
#            if not parent in globals():
#                continue
    if not parent=='neuron' or parent=='area':
        parent='neuron'
    if not objtype in ['skeleton','synapse','soma','tag','region']:
        for iobjtype in ['skeleton','synapse','soma','tag','region']:
            if iobjtype in parts:
                objtype=iobjtype
                break
#                for ipart in parts:
#                    if 'id' in ipart:
#                        id=ipart
#                        break
        
    if not objtype in ['skeleton','synapse','soma','tag','region']:
        print "Unknown object type. Use default object type: skeleton"
        objtype='skeleton' #default objtype

    Data["flags"]=flags

    parameters=root.find('parameters')
    Data['parameters']=dict()
    try: 
        scale=parameters.find('scale')
        scale=(float(scale.get('x')),\
            float(scale.get('y')),\
            float(scale.get('z')))
    except:
        scale=(1.0,1.0,1.0)
    try: 
        editPosition=parameters.find('editPosition')
        editPosition=(float(editPosition.get('x')),\
            float(editPosition.get('y')),\
            float(editPosition.get('z')))
        Data['parameters']['editPosition']=editPosition
    except:
        1
        
    try: #dirty hacks for compability with previous datasets
        experiment=parameters.find('experiment')
        dataset=experiment.get('name')
        if (dataset=="E085L01_mag1" or dataset=="E085L01") and  (scale==(0.37008,0.37008,1.0) or scale==(0.37,0.37,1.0)):
            scale=(9.252,9.252,25.0)
        elif (dataset=="E046L01_mag1" or dataset=="E046L01") and  scale==(0.40,0.40,1.0):
            scale=(11.29,11.29,30)
        elif (dataset=="cube") and scale==(0.36,0.36,1.0):
            scale=(9.0,9.0,25.0)
        Data['parameters']['dataset']=dataset
    except:
        1
    Data['parameters']['scale']=scale
                    
    try: 
        activeNode=parameters.find('activeNode')
        activeNode=int(activeNode.get('id'))
    except:
        activeNode=None
    Data['parameters']['activeNode']=activeNode

    parentthing=dict()
    Data[parent]=parentthing
    for tree in root.iter('thing'):
        neuronId=None
        try:
            neuronId=float(tree.get('id'))
        except:
            try:
                neuronId=float(id.replace('id',''))
            except:
                continue

        if neuronId==None:
            continue
        parentthing[neuronId]=dict()
        thing=dict()
        parentthing[neuronId][objtype]=thing
        try: 
            thing['obj_color']=(float(tree.get('color.r')),\
                float(tree.get('color.g')),\
                float(tree.get('color.b')),\
                float(tree.get('color.a')))                  
        except:
            thing['obj_color']=None
        thing['Points']=[]
        thing['NodeID']=[]
        thing['attributes']=dict()
        thing['idxEdges']=list()
        thing['edges']=set()
        thing['comments']=dict()
        try:
            nodes=tree.find('nodes')
            for node in nodes.iter('node'):
                nodeId=int(node.attrib['id'])
                thing['NodeID'].append(nodeId)
                thing['Points'].extend([float(node.attrib['x'])*scale[0],\
                    float(node.attrib['y'])*scale[1],\
                    float(node.attrib['z'])*scale[2]])
                if node.attrib.__len__()>4:
                    attributes=dict(node.attrib)
                    del attributes['id']
                    del attributes['x']
                    del attributes['y']
                    del attributes['z']
                    if attributes.has_key('time'):
                        if not attributes['time']:
                            del attributes['time']
                        else:
                            nodetime=float(attributes['time'])/1000.0 #convert from ms to sec
                            if nodetime>0.0:
                                attributes['time']=nodetime
                            else:                           
                                del attributes['time']
                    if attributes.has_key('comment'):
                        value=unicode(attributes['comment'])     
                        if not value:
                            del attributes['comment']
                    
                    if attributes.__len__()>0:
                        thing['attributes'][nodeId]=attributes
        except:
            1
#                print "No nodes found."
            continue
        
        try:
            Edges=tree.find('edges')
            for edge in Edges.iter('edge'): 
                sourceID=int(edge.get('source'))
                targetID=int(edge.get('target'))
                thing['edges'].add((sourceID,targetID))
        except:
            1
#                print "No node id based edges found."
        try:
            Edges=tree.find('idxedges')
            for edge in Edges.iter('edge'): 
                sourceIdx=int(edge.get('source')) #0 offset indexing in vtk
#                    if sourceIdx<0:
#                        continue
                targetIdx=int(edge.get('target')) #0 offset indexing in vtk
#                    if targetIdx<0:
#                        continue
#                    if sourceIdx==targetIdx:
#                        continue
                thing['idxEdges'].extend([sourceIdx,targetIdx])
        except:
            1
#                print "No indexed edges found."
    try:
        comments=root.find('comments')
        for comment in comments.iter('comment'):
            if not (comment.attrib.has_key('node') and comment.attrib.has_key('content')):
                continue
            nodeId=int(comment.attrib['node'])
            value=unicode(comment.attrib['content'])     
            if not value=="":
                thing['comments'][nodeId]=value
    except:
        1
    return Data

#def parParseNML(input): 
#    Data={}
#    Data[input[0]]=ParseNML(input[0],input[1])
#    return Data

def extractObjInfo(data,iobj):
    #extracts information from vtk celldata or pointdata
    info_str=""
    NArrays=data.GetNumberOfArrays()
    NDataObjs=data.GetNumberOfTuples()
    for iarray in range(NArrays):
        array=data.GetArray(iarray)
        if not array:
            array=data.GetAbstractArray(iarray)
        if not array:
            continue
        if hasattr(array,'GetNumberOfTuples'):
            NValues=array.GetNumberOfTuples()
        elif hasattr(array,'GetNumberOfValues'):
            NValues=array.GetNumberOfValues()
        if NValues==NDataObjs:
            ituple=iobj
        else:
            ituple=0         
        if ituple>=NValues:
            continue
        NComponents=array.GetNumberOfComponents()
        info_str+="{0}: ".format(array.GetName())
        for icomp in range(NComponents):
            info_str+="{0},".format(array.GetComponent(ituple,icomp))
        info_str+="\n"
    return info_str

class IntersectionLine(vtk.vtkLineSource):
    def __init__(self,color,viewport):
        self.viewport=viewport
        self.SetResolution(2)

        self.mapper = vtk.vtkDataSetMapper()
#        self.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.mapper.SetInput(self.GetOutput())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.PickableOff()
        self.actor.GetProperty().LightingOff()
        self.actor.GetProperty().ShadingOff()
        self.actor.GetProperty().SetLineWidth(1)
        self.actor.GetProperty().SetDiffuseColor(np.array(color,dtype=np.float))

        self.SetVisibility(1)
        
    def SetVisibility(self,visible):
        self.Visible=visible
        if visible:
            self.actor.VisibilityOn()
        else:
            self.actor.VisibilityOff()

class PlaneIntersection(vtk.vtkPlane):

    def __init__(self,Plane1,Plane2):
        self.Plane1=Plane1
        self.Plane2=Plane2
        self.Intersections=list()
     
    def AddIntersection(self,viewport,color):
        Intersection=IntersectionLine(color,viewport)
        self.Intersections.append(Intersection)
        viewport.AddActor(Intersection.actor)
        viewport.Intersections.append(Intersection)
             
    def Intersect(self):
        t=vtk.mutable(0);
        x_1=np.array([[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]],dtype=np.float)
        
        PlaneNormal_1=np.array(self.Plane1.PlaneSource.GetNormal(),dtype=np.float)
        PlaneCenter_1=np.array(self.Plane1.PlaneSource.GetCenter(),dtype=np.float)
        pt0_1=np.array(self.Plane1.PlaneSource.GetOrigin(),dtype=np.float)
        pt1_1=np.array(self.Plane1.PlaneSource.GetPoint1(),dtype=np.float)
        pt2_1=np.array(self.Plane1.PlaneSource.GetPoint2(),dtype=np.float)        
        pt1_1-=pt0_1
        pt2_1-=pt0_1
        vtk.vtkMath.Normalize(pt1_1)
        vtk.vtkMath.Normalize(pt2_1)

        PlaneNormal_2=np.array(self.Plane2.PlaneSource.GetNormal(),dtype=np.float)
        PlaneCenter_2=np.array(self.Plane2.PlaneSource.GetCenter(),dtype=np.float)
        pt0_2=np.array(self.Plane2.PlaneSource.GetOrigin(),dtype=np.float)
        pt1_2=np.array(self.Plane2.PlaneSource.GetPoint1(),dtype=np.float)
        pt2_2=np.array(self.Plane2.PlaneSource.GetPoint2(),dtype=np.float)
        
        intersect=0;
        if self.IntersectWithLine(pt0_2,pt1_2,PlaneNormal_1,PlaneCenter_1,t,x_1[intersect]):
            proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt1_1)
            proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt2_1)
            if proj_pt1>=0.0 and proj_pt1<=(self.Plane1.ROISize[0]*self.Plane1.ROIRes[0]*self.Plane1._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane1.ROISize[1]*self.Plane1.ROIRes[1]*self.Plane1._ROIScale[1]+10e-5):
                intersect+=1
        if self.IntersectWithLine(pt2_2,pt1_2+pt2_2-pt0_2,PlaneNormal_1,PlaneCenter_1,t,x_1[intersect]):
            proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt1_1)
            proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt2_1)
            if proj_pt1>=0.0 and proj_pt1<=(self.Plane1.ROISize[0]*self.Plane1.ROIRes[0]*self.Plane1._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane1.ROISize[1]*self.Plane1.ROIRes[1]*self.Plane1._ROIScale[1]+10e-5):
                intersect+=1
        if intersect<2:
            if self.IntersectWithLine(pt0_2,pt2_2,PlaneNormal_1,PlaneCenter_1,t,x_1[intersect]):
                proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt1_1)
                proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt2_1)
                if proj_pt1>=0.0 and proj_pt1<=(self.Plane1.ROISize[0]*self.Plane1.ROIRes[0]*self.Plane1._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane1.ROISize[1]*self.Plane1.ROIRes[1]*self.Plane1._ROIScale[1]+10e-5):
                    intersect+=1
            if intersect<2:
                if self.IntersectWithLine(pt1_2,pt1_2+pt2_2-pt0_2,PlaneNormal_1,PlaneCenter_1,t,x_1[intersect]):
                    proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt1_1)
                    proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_1,pt2_1)
                    if proj_pt1>=0.0 and proj_pt1<=(self.Plane1.ROISize[0]*self.Plane1.ROIRes[0]*self.Plane1._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane1.ROISize[1]*self.Plane1.ROIRes[1]*self.Plane1._ROIScale[1]+10e-5):
                        intersect+=1

        if intersect<2:
            pt0_1=np.array(self.Plane1.PlaneSource.GetOrigin(),dtype=np.float)
            pt1_1=np.array(self.Plane1.PlaneSource.GetPoint1(),dtype=np.float)
            pt2_1=np.array(self.Plane1.PlaneSource.GetPoint2(),dtype=np.float)        
    
            pt0_2=np.array(self.Plane2.PlaneSource.GetOrigin(),dtype=np.float)
            pt1_2=np.array(self.Plane2.PlaneSource.GetPoint1(),dtype=np.float)
            pt2_2=np.array(self.Plane2.PlaneSource.GetPoint2(),dtype=np.float)

            pt1_2-=pt0_2
            pt2_2-=pt0_2
            
            vtk.vtkMath.Normalize(pt1_2)
            vtk.vtkMath.Normalize(pt2_2)
    
            if self.IntersectWithLine(pt0_1,pt1_1,PlaneNormal_2,PlaneCenter_2,t,x_1[intersect]):
                proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt1_2)
                proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt2_2)
                if proj_pt1>=0.0 and proj_pt1<=(self.Plane2.ROISize[0]*self.Plane2.ROIRes[0]*self.Plane2._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane2.ROISize[1]*self.Plane2.ROIRes[1]*self.Plane2._ROIScale[1]+10e-5):
                    intersect+=1
            if intersect<2:
                if self.IntersectWithLine(pt2_1,pt1_1+pt2_1-pt0_1,PlaneNormal_2,PlaneCenter_2,t,x_1[intersect]):
                    proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt1_2)
                    proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt2_2)
                    if proj_pt1>=0.0 and proj_pt1<=(self.Plane2.ROISize[0]*self.Plane2.ROIRes[0]*self.Plane2._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane2.ROISize[1]*self.Plane2.ROIRes[1]*self.Plane2._ROIScale[1]+10e-5):
                        intersect+=1
                if intersect<2:
                    if self.IntersectWithLine(pt0_1,pt2_1,PlaneNormal_2,PlaneCenter_2,t,x_1[intersect]):
                        proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt1_2)
                        proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt2_2)
                        if proj_pt1>=0.0 and proj_pt1<=(self.Plane2.ROISize[0]*self.Plane2.ROIRes[0]*self.Plane2._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane2.ROISize[1]*self.Plane2.ROIRes[1]*self.Plane2._ROIScale[1]+10e-5):
                            intersect+=1
                    if intersect<2:
                        if self.IntersectWithLine(pt1_1,pt1_1+pt2_1-pt0_1,PlaneNormal_2,PlaneCenter_2,t,x_1[intersect]):
                            proj_pt1=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt1_2)
                            proj_pt2=vtk.vtkMath.Dot(x_1[intersect]-pt0_2,pt2_2)
                            if proj_pt1>=0.0 and proj_pt1<=(self.Plane2.ROISize[0]*self.Plane2.ROIRes[0]*self.Plane2._ROIScale[0]+10e-5) and proj_pt2>=0.0 and proj_pt2<=(self.Plane2.ROISize[1]*self.Plane2.ROIRes[1]*self.Plane2._ROIScale[1]+10e-5):
                                intersect+=1   
        if intersect>1:
            for intersection in self.Intersections:
                if not intersection.Visible:
                    continue
                nDir=np.array(intersection.viewport.Camera.GetViewPlaneNormal(),dtype=np.float);
                vtk.vtkMath.Norm(nDir)
#                intersection.SetPoint1(x_1[0]+0.06*nDir)
#                intersection.SetPoint2(x_1[1]+0.06*nDir)
                intersection.SetPoint1(x_1[0]+0.06*nDir)
                intersection.SetPoint2(x_1[1]+0.06*nDir)
                intersection.Modified()
                intersection.actor.VisibilityOn()    
#            print "intersect on: {0},{1}".format(self.Plane1._Orientation,self.Plane2._Orientation)
        else:
            for intersection in self.Intersections:
                intersection.actor.VisibilityOff()
#            print "intersect off: {0},{1}".format(self.Plane1._Orientation,self.Plane2._Orientation)

class Dataset:
    _VisualizationType='Volumemetric'
    _DataScale=(9.252,9.252,25.0)
    _DataOrigin=(0.0,0.0,0.0)
    _FilePattern="%s%d.tif"
    _BaseName="ConsPlane"
    _DataExtent=(1000,1000,1000)
    _Color=(255,0,0,255)
    _Threshold=(60,80)
    _TargetResolution=200.0;
    _Smoothing=(2,3);

    def __init__(self,ariadne,filename):
        self.ariadne=ariadne
        self.viewports=list()
        self.SetupVisualization()
        self.filename=filename
        self.LoadDatasetInformation(filename)

    def SaveDataSetInformation(self,filename=None):
        if not filename:
            filename=self.filename
        DataSetConf=config(filename)        
        DataSetConf.SaveConfig(self,"Dataset")
        DataSetConf.write()
        
    def LoadDatasetInformation(self,filename):
        if not os.path.isfile(filename):
            print "Error: dataset file {0} not found.".format(filename)
            return None
        DataSetConf=config(filename)        
        DataSetConf.LoadConfig(self,"Dataset")
        self.BasePath=os.path.dirname(filename)
        QtCore.QObject.connect(self.ariadne.btn_DatasetColor,QtCore.SIGNAL("clicked()"),self.SetColor)

        QtCore.QObject.connect(self.ariadne.SpinBox_RangeLow,QtCore.SIGNAL("editingFinished()"),self.SetRange)        
        QtCore.QObject.connect(self.ariadne.SpinBox_RangeHigh,QtCore.SIGNAL("editingFinished()"),self.SetRange)        

        QtCore.QObject.connect(self.ariadne.SpinBox_SmoothingStd,QtCore.SIGNAL("editingFinished()"),self.SetSmoothing)        
        QtCore.QObject.connect(self.ariadne.SpinBox_SmoothingSize,QtCore.SIGNAL("editingFinished()"),self.SetSmoothing)        
        
        QtCore.QObject.connect(self.ariadne.SpinBox_TargetResolution,QtCore.SIGNAL("editingFinished()"),self.SetResolution)        
        
        self.SetColor(list(self._Color),1)
        self.SetRange(list(self._Threshold),1)
        self.SetSmoothing(list(self._Smoothing),1)
        self.SetResolution(self._TargetResolution,0) #render here
    
    def LoadData(self):
        if self._VisualizationType=='Volumemetric':
            return self.LoadVolumemetricData()
        else:
            return 0
    
    def LoadVolumemetricData(self):
#        self.TIFReader.DebugOn()
        self.TIFReader.SetDataScalarTypeToUnsignedChar();
        self.TIFReader.SetDataOrigin(self._DataOrigin);
        self.TIFReader.SetFilePrefix(os.path.join(self.BasePath,self._BaseName))
        self.TIFReader.SetFileNameSliceOffset(1);
        self.TIFReader.SetFileNameSliceSpacing(1);
        self.TIFReader.SetDataSpacing(np.float(self._DataScale[0]),np.float(self._DataScale[1]),np.float(self._DataScale[2]));
        self.TIFReader.SetFilePattern(r"{0}".format(self._FilePattern))
        self.TIFReader.SetDataExtent(0,self._DataExtent[0]-1,0,self._DataExtent[1]-1,0,self._DataExtent[2]-1);
        self.TIFReader.SpacingSpecifiedFlagOn()
#        self.volumeMapper.SetSampleDistance(np.min(self.ariadne.DataScale))
#        self.volumeMapper.Modified()
        self.TIFReader.Modified()
        
    def SetResolution(self,resolution=None,silent=0):        
        if resolution==None:
            resolution=self.ariadne.SpinBox_TargetResolution.value()
        if not (self.ariadne.SpinBox_TargetResolution.value()==resolution):
            self.ariadne.SpinBox_TargetResolution.setValue(resolution)
        resolution=np.float(resolution)
        self.VolSubSampling.SetShrinkFactors(resolution/self._DataScale[0],resolution/self._DataScale[1],resolution/self._DataScale[2])
        self._TargetResolution=resolution
#        self.VolSubSampling.Modified()
#        self.VolSubSampling.SetShift(resolution/2.0,resolution/2.0,resolution/2.0)
        if not silent:
            self.ariadne.QRWin.Render()
        
    def SetSmoothing(self,smoothing=[None,None],silent=0):
        smoothing=list(smoothing)
        print "pre", smoothing, self.ariadne.SpinBox_SmoothingStd.value(),self.ariadne.SpinBox_SmoothingSize.value()
        if smoothing[0]==None:
            smoothing[0]=self.ariadne.SpinBox_SmoothingStd.value()
        if smoothing[1]==None:
            smoothing[1]=self.ariadne.SpinBox_SmoothingSize.value()

        if smoothing[0]<0:
            smoothing[0]=0
        print "post", smoothing, self.ariadne.SpinBox_SmoothingStd.value(),self.ariadne.SpinBox_SmoothingSize.value()
            
#        if smoothing[1]>255:
#            smoothing[1]=255
            
        self.VolSmoothing.SetStandardDeviation(smoothing[0])
        self.VolSmoothing.SetRadiusFactor(smoothing[1])
        self.VolSmoothing.Modified()
        
        if not (smoothing[0]==self.ariadne.SpinBox_SmoothingStd.value()):
            print "std", smoothing[0], self.ariadne.SpinBox_SmoothingStd.value()
            self.ariadne.SpinBox_SmoothingStd.setValue(smoothing[0])
        if not (smoothing[1]==self.ariadne.SpinBox_SmoothingSize.value()):
            print "size",smoothing[1],self.ariadne.SpinBox_SmoothingSize.value()
            self.ariadne.SpinBox_SmoothingSize.setValue(smoothing[1])
        
        if not silent:
            self.ariadne.QRWin.Render()
        self._Smoothing=smoothing
        
    def SetRange(self,threshold=[None,None],silent=0):
        threshold=list(threshold)
        if threshold[0]==None:
            threshold[0]=self.ariadne.SpinBox_RangeLow.value()
        if threshold[1]==None:
            threshold[1]=self.ariadne.SpinBox_RangeHigh.value()

        if threshold[0]<0:
            threshold[0]=0
            
        if threshold[1]>255:
            threshold[1]=255
            
        self.contour.GenerateValues(2,threshold[0],threshold[1])
        
        if not (threshold[0]==self.ariadne.SpinBox_RangeLow.value()):
            self.ariadne.SpinBox_RangeLow.setValue(threshold[0])
        if not (threshold[1]==self.ariadne.SpinBox_RangeHigh.value()):
            self.ariadne.SpinBox_RangeHigh.setValue(threshold[1])

        print "low", threshold[0],self.ariadne.SpinBox_RangeLow.value()
        print "high", threshold[1],self.ariadne.SpinBox_RangeHigh.value()
        
        self.contour.Modified()
        
        if not silent:
            self.ariadne.QRWin.Render()
        self._Threshold=threshold

    def SetColor(self,color=None,silent=0):
        if color==None:           
            color=self._Color
            color=QtGui.QColor().fromRgb(*color)
            color=QtGui.QColorDialog.getColor(color,self.ariadne, "Color", QtGui.QColorDialog.ShowAlphaChannel)

            if not color.isValid():
                return
            color=color.getRgb()        
        self.ariadne.btn_DatasetColor.setStyleSheet("background-color: rgb({0}, {1}, {2})".format(color[0],color[1],color[2]))

        self.actor.GetProperty().SetColor(color[0]/255.0,color[1]/255.0,color[2]/255.0)
        self.actor.GetProperty().SetOpacity(color[3]/255.0)
        self.actor.Modified()
        
        self._Color=color
        if not silent:
            self.ariadne.QRWin.Render()
        
    def SetupVisualization(self):
        self.TIFReader=vtk.vtkTIFFReader()      

        self.VolSubSampling = vtk.vtkImageShrink3D()
#        self.VolSubSampling.SetInputConnection(self.TIFReader.GetOutputPort());
#        self.VolSubSampling.AveragingOn()
        
        self.VolSmoothing=vtk.vtkImageGaussianSmooth()
        self.VolSmoothing.SetDimensionality(3)
        self.VolSmoothing.SetInputConnection(self.TIFReader.GetOutputPort() )
        
        self.contour = vtk.vtkImageMarchingCubes()
        self.contour.SetInputConnection(self.VolSmoothing.GetOutputPort() )
        self.contour.ComputeNormalsOn()
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection( self.contour.GetOutputPort() )
        self.mapper.ScalarVisibilityOff()
        self.actor = vtk.vtkActor()
#        self.actor = vtk.vtkLODActor()
#        self.actor.SetNumberOfCloudPoints( 1000000 )
        self.actor.SetMapper( self.mapper )
        self.actor.GetProperty().SetSpecular(.3)
        self.actor.GetProperty().SetSpecularPower(30)

    def add_toviewport(self,viewport):
        if (not self.actor):
            return
        if viewport in self.viewports:
            return
        viewport.AddViewProp(self.actor)
        self.viewports.append(viewport)

        
#    def add_toviewport(self,viewport):
#        if (not self.volume):
#            return
#        if viewport in self.viewports:
#            return
#        viewport.AddVolume(self.volume)
#        self.viewports.append(viewport)

    def remove_fromviewport(self,viewport):
        if not hasattr(self,"renderer"):
            return
        if not viewport in self.viewports:
            return
        viewport.RemoveViewProp(self.actor)
        viewport.Modified()
        self.viewports.remove(viewport)

class Loader:
    filename=None
    _NCubesPerEdge=[5,5,5]
    _NStreamingChannels=5;
    _CubeSize=(128,128,128)
    _NumberofCubes=(120,120,120)
    _BaseName=""
    _BaseExt=".raw"
    _BasePath=""
    _BaseURL=""
    _UserName=""
    _Password=""
    _ServerFormat=0;
    _DataScale=(1.0,1.0,1.0)
    _FileType=0
    _Extent=[120*128,120*128,120*128]
    _Origin=[0.0,0.0,0.0]
    _Magnification=1;
    doload=1;
    InterPolFactor=1.0; #optimally 2.0

    def __init__(self,doload,multiprocessing):
        self.doload=doload
        self.multiprocessing=multiprocessing
        self.LoaderProcess=None
        self.LoaderState=RawArray(c_int,[0])
        self.WorkingOffline=RawArray(c_int,[0])
        self.NCubes2Load=RawArray(c_int,[0])
        self.ROIState=Value('i',0)
        self.Magnification=RawArray(c_int,[self._Magnification])
        self.Position=RawArray(c_float,(1000.0,1000.0,1000.0))
        self.ShortestEdge=[]

    def LoadDatasetInformation(self,filename):
        NPixels=self._CubeSize[0]*self._CubeSize[1]*self._CubeSize[2];
        maxNCubesPerEdge=np.floor((float(maxContRAMBlock)/float(NPixels))**(1.0/3.0))
        window1.SpinBox_CubesPerEdge.setMaximum(maxNCubesPerEdge)
        NCubesPerEdge=int(window1.SpinBox_CubesPerEdge.value())
        if NCubesPerEdge>maxNCubesPerEdge:
            NCubesPerEdge=int(maxNCubesPerEdge);
            window1.SpinBox_CubesPerEdge.setValue(NCubesPerEdge)
        if NCubesPerEdge<1:
            NCubesPerEdge=1;
            window1.SpinBox_CubesPerEdge.setValue(NCubesPerEdge)
        self._NCubesPerEdge=[NCubesPerEdge,NCubesPerEdge,NCubesPerEdge]
        
        NStreamingChannels=int(window1.SpinBox_StreamingSlots.value())
        if NStreamingChannels>maxNStreamingChannels:
            NStreamingChannels=int(maxNStreamingChannels);
            window1.SpinBox_StreamingSlots.setMaximum(maxNStreamingChannels)
            window1.SpinBox_StreamingSlots.setValue(NStreamingChannels)
        if NStreamingChannels<1:
            NStreamingChannels=1;
            window1.SpinBox_StreamingSlots.setValue(NStreamingChannels)
        self._NStreamingChannels=NStreamingChannels
        
        if not filename:
            print "No dataset specified."
            return None
        if not os.path.isfile(filename):
            print "Error: dataset file {0} not found.".format(filename)
            return None
        self.filename=filename;
            
        self._BaseName=""
        self._BaseExt=".raw"
        self._BasePath=""
        self._BaseURL=""
        self._UserName=""
        self._Password=""
        self._ServerFormat=0;
        self._Extent=[]
        self._Origin=[0.0,0.0,0.0]
        self._NumberofCubes=(120,120,120)
        self._DataScale=(1.0,1.0,1.0)
        self._CubeSize=(128,128,128)
        DataSetConf=config(filename)        
        DataSetConf.LoadConfig(self,"Dataset")
        self._BasePath=os.path.dirname(filename)
        if not self._Extent:     
            self._Extent=[\
            float(self._CubeSize[0])*self._NumberofCubes[0],\
            float(self._CubeSize[1])*self._NumberofCubes[1],\
            float(self._CubeSize[2])*self._NumberofCubes[2]]

        NPixels=self._CubeSize[0]*self._CubeSize[1]*self._CubeSize[2];
        maxNCubesPerEdge=np.floor((float(maxContRAMBlock)/float(NPixels))**(1.0/3.0))
        window1.SpinBox_CubesPerEdge.setMaximum(maxNCubesPerEdge)
        NCubesPerEdge=int(window1.SpinBox_CubesPerEdge.value())
        if NCubesPerEdge>maxNCubesPerEdge:
            NCubesPerEdge=int(maxNCubesPerEdge);
            window1.SpinBox_CubesPerEdge.setMaximum(maxNCubesPerEdge)
            window1.SpinBox_CubesPerEdge.setValue(NCubesPerEdge)
        if NCubesPerEdge<1:
            NCubesPerEdge=1;
            window1.SpinBox_CubesPerEdge.setValue(NCubesPerEdge)
        self._NCubesPerEdge=[NCubesPerEdge,NCubesPerEdge,NCubesPerEdge]

        return self._BaseName
   
    def LoadDataset(self):     
        self.StopLoader()
        self.ResetLoader()
        if self._ServerFormat==0:
            window1.ActionWorkingOffline.setEnabled(0)
        else:
            window1.ActionWorkingOffline.setEnabled(1)
        self.StartLoader()
        
    def CheckMagnification(self,maxROIEdge):   
#        #check which data set version is closest to this scale.
#        for idx in range(self.ShortestEdge.__len__()):
##            if maxROIEdge*1.2<self.ShortestEdge[idx]:
#            #sqrt(2)*arbitrary_orientation_diagonal=shortest_hypercube_edge 
#            if maxROIEdge*np.sqrt(2.0)<self.ShortestEdge[idx]:
#                break;
        #check which data set version is closest to this scale.
        for idx in range(self.DataScale.__len__()/3):
#            if maxROIEdge*1.2<self.ShortestEdge[idx]:
            #sqrt(2)*arbitrary_orientation_diagonal=shortest_hypercube_edge 
            CubeRes=min([self.DataScale[idx*3],self.DataScale[idx*3+1],self.DataScale[idx*3+2]])
            if np.round(maxROIEdge,3)<=np.round(self.InterPolFactor*CubeRes,3):
                break;

        whichMag=1+idx
#        print [(0.5-roiScaleNorm/x) for x in self.ShortestEdge]
        if not (whichMag==self.Magnification[0]):
            self.Magnification[0]=whichMag
            if self.LoaderState[0]>0:
                self.LoaderState[0]=2;            
#        print "Target resolution: ", maxROIEdge ,"; Magnification: ", whichMag, "; Cube resolution: " ,CubeRes, "; Interpol factor: ", self.InterPolFactor#, [roiScaleNorm/x for x in self.ShortestEdge]
        return whichMag
            
    def StopLoader(self):
        window1.QRWin.Timer.stop()
        
        if self.ROIState.value>0:
            self.ROIState.value=0
            extractROIlib.release_ROI(None)   
        #stop previous cubeloader instance
        if self.multiprocessing:
            if self.LoaderState[0]>0:        
                self.LoaderState[0]=0
                step=0
                if not self.LoaderProcess==None:
                    if CubeLoader.LoaderProcess.is_alive():
                        while (self.LoaderState[0]>-1 and step<100):
                            step+=1
                            print "wait, loader state: {0}".format(self.LoaderState[0])
                            time.sleep(0.1)
                        print "loader state: {0}".format(self.LoaderState[0])
            if not (not self.LoaderProcess):
                if CubeLoader.LoaderProcess.is_alive():
                    print "joining LoaderProcess"
                    self.LoaderProcess.join(5)
                    
                if CubeLoader.LoaderProcess.is_alive():
                    print "joining LoaderProcess timeout"
                self.LoaderProcess.terminate()     
                print "terminated LoaderProcess"
                del self.LoaderProcess
                self.LoaderProcess=None
        else:    
            print "Loader loop quited..."        
            loaderlib.release_loader(None)
            self.LoaderState[0]=-1
            
#        if hasattr(self,'AllCubes'):
#            del self.AllCubes
#            self.AllCubes=None
            
        print "Loader stopped."
   
    def StartLoader(self):       
        if self.multiprocessing:
            #Somehow starting/connecting the loader in a separate process is kind of non deterministic on windows. 
            #Probably it takes very long on windows to start a seprate process and therefore we have to wait
            #up to several seconds...
            self.LoaderProcess = multiprocessing.Process(target=self.LoaderLoop, \
                args=(self.Position,self.LoaderState,self.HyperCube0,self.HyperCube1,self.HyperCube2,self.AllCubes,\
                self.DataScale,self.CubeSize,self.NMags,self.Magnification,self.NumberofCubes,\
                self.BaseName,self.BaseExt,self.BasePath,self.NCubesPerEdge,self.FileType,\
                self.BaseURL,self.UserName,self.Password,self.ServerFormat,\
                self.NStreamingChannels,self.WorkingOffline,self.NCubes2Load))
            self.LoaderProcess.daemon=True 
            self.LoaderProcess.start()    
            print "Initializing loader process..."
            step=0
            while (self.LoaderProcess.is_alive() and self.LoaderState[0]<1 and step<100):
                step+=1
                print "...loader state: ", self.LoaderState[0]
#                time.sleep(0.05)
                time.sleep(0.1)
            if self.LoaderState[0]==5:
                if self.LoaderProcess.is_alive():
                    self.LoaderState[0]=1
                    print "...loader alive..."
                    self.LoaderProcess.join(0.25)
                    print "...joined..."
                else:
                    self.LoaderState[0]=-1
                    print "...loader is dead."
            if self.LoaderState[0]>0 and self.LoaderProcess.is_alive():
                self.LoaderProcess.join(0.1)
                print "Loader process successfully initialized."
            else:
                print "Error: Could not joint the loader process."
        else:
            if self.LoaderState[0]==0:
                print "Start loader..."            
                loaderlib.init_loader(self.Position,self.HyperCube0,self.HyperCube1,self.HyperCube2,self.AllCubes,\
                    self.NCubesPerEdge,self.BasePath,self.BaseName,self.BaseExt,self.NMags,self.DataScale,self.CubeSize,self.NumberofCubes,\
                    self.Magnification,self.FileType,self.BaseURL,self.UserName,self.Password,self.ServerFormat,\
                    self.NStreamingChannels,self.WorkingOffline,self.NCubes2Load,self.LoaderState)
            self.LoaderState[0]=1
            
        if self.LoaderState[0]>0:
            self.StartROI_Extraction()               
            if window1.QRWin.Timer==None:
                window1.QRWin.InitTimer()
                window1.QRWin.InitCubeQueueTimer()
        else:
            print "Error: Could not start loader."
         
    def ResetLoader(self):
        if self.ROIState.value>0:
            extractROIlib.release_ROI(None);
        loaderlib.release_loader(None);
        
#        if not hasattr(self,'LoaderState'):
        self.LoaderState=RawArray(c_int,[0])
#        else:
#            self.LoaderState[0]=int(0)
        self.ROIState=Value('i',0)

        self.NStreamingChannels=RawArray(c_int,[self._NStreamingChannels])

        if win:
            self.BaseName=RawArray(c_char,str(self._BaseName)+'\x00')
            self.BaseExt=RawArray(c_char,str(self._BaseExt)+'\x00')
            self.BasePath=RawArray(c_char,str(self._BasePath[:])+'\x00')
            self.BaseURL=RawArray(c_char,str(self._BaseURL[:])+'\x00')
            self.UserName=RawArray(c_char,str(self._UserName[:])+'\x00')
            self.Password=RawArray(c_char,str(self._Password[:])+'\x00')
        else:
            self.BaseName=RawArray(c_char,str(self._BaseName)+'\0')
            self.BaseExt=RawArray(c_char,str(self._BaseExt)+'\0')
            self.BasePath=RawArray(c_char,str(self._BasePath)+'\0')
            self.BaseURL= RawArray(c_char,str(self._BaseURL)+'\0')
            self.UserName=RawArray(c_char,str(self._UserName)+'\0')
            self.Password=RawArray(c_char,str(self._Password)+'\0')
        
        self.Magnification=RawArray(c_int,[self._Magnification])
        self.DataScale=RawArray(c_float,self._DataScale[:])
        self.CubeSize=RawArray(c_int,self._CubeSize[:])
        self.NumberofCubes=RawArray(c_int,self._NumberofCubes[:])
        self.FileType=RawArray(c_int,[self._FileType])
        self.ServerFormat=RawArray(c_int,[self._ServerFormat])

        self.ShortestEdge=[]
        NMags=np.int(self._DataScale.__len__()/3.0)
        for imag in range(NMags):
            self.ShortestEdge.append(\
                np.min(np.multiply(self._NCubesPerEdge,\
                        np.multiply([float(self._CubeSize[0]),float(self._CubeSize[1]),float(self._CubeSize[2])],\
                            self._DataScale[imag*3:(imag+1)*3]))))
        self.NMags=RawArray(c_int,[NMags])
        
        if not hasattr(self,'Position'):
            self.Position=RawArray(c_float,(1000.0,1000.0,1000.0))
        
        NPixels=(self._CubeSize[0]*self._CubeSize[1]*self._CubeSize[2])
        allocated=0
        for NCubesPerEdge in range(self._NCubesPerEdge[0],0,-1):        
            try:
                if not hasattr(self,'HyperCube0'):
                    self.HyperCube0=RawArray(c_ubyte,(NCubesPerEdge**3)*NPixels)
                else:
                    if sizeof(c_ubyte)*(NCubesPerEdge**3)*NPixels>sizeof(self.HyperCube0):
                        resize(self.HyperCube0,sizeof(c_ubyte)*(NCubesPerEdge**3)*NPixels)
                self._NCubesPerEdge[0]=NCubesPerEdge    
                print('Allocated memory for {0} cubes per edge for magnification level i'.format(NCubesPerEdge))
                allocated+=1
                break;
            except:
                print('Not enough continuous memory to allocate {0} cubes per edge for magnification level i'.format(NCubesPerEdge))

        if not allocated==1:
            print('Error: Could not allocate memory for magnification level i.')
            self._NCubesPerEdge[0]=0
                
        for NCubesPerEdge in range(self._NCubesPerEdge[1],0,-1):        
            try:
                if not hasattr(self,'HyperCube1'):
                    self.HyperCube1=RawArray(c_ubyte,(NCubesPerEdge**3)*NPixels)
                else:
                    if sizeof(c_ubyte)*(NCubesPerEdge**3)*NPixels>sizeof(self.HyperCube1):
                        resize(self.HyperCube1,sizeof(c_ubyte)*(NCubesPerEdge**3)*NPixels)
                self._NCubesPerEdge[1]=NCubesPerEdge    
                print('Allocated memory for {0} cubes per edge for magnification level i+1'.format(NCubesPerEdge))
                allocated+=1
                break;
            except:
                print('Not enough continuous memory to allocate {0} cubes per edge for magnification level i+1'.format(NCubesPerEdge))
                2
        if not allocated==2:
            print('Error: Could not allocate memory for magnification level i+1.')
            self._NCubesPerEdge[1]=0

        for NCubesPerEdge in range(self._NCubesPerEdge[2],0,-1):        
            try:
                if not hasattr(self,'HyperCube2'):
                    self.HyperCube2=RawArray(c_ubyte,(NCubesPerEdge**3)*NPixels)
                else:
                    if sizeof(c_ubyte)*(NCubesPerEdge**3)*NPixels>sizeof(self.HyperCube2):
                        resize(self.HyperCube2,sizeof(c_ubyte)*(NCubesPerEdge**3)*NPixels)
                self._NCubesPerEdge[2]=NCubesPerEdge    
                print('Allocated memory for {0} cubes per edge for magnification level i-1'.format(NCubesPerEdge))
                allocated+=1
                break;
            except:
                print('Not enough continuous memory to allocate {0} cubes per edge for magnification level i-1'.format(NCubesPerEdge))

        if not allocated==3:
            print('Error: Could not allocate memory for magnification level i-1.')
            self._NCubesPerEdge[2]=0
            
            
        self.NCubesPerEdge=RawArray(c_int,self._NCubesPerEdge[:])
            
        totalNCubes=0;
        for imag in range(NMags):
            totalNCubes+=self._NumberofCubes[imag*3+0]*self._NumberofCubes[imag*3+1]*self._NumberofCubes[imag*3+2]
            
        self.AllCubes=RawArray(c_short,totalNCubes); 
    
    def UpdatePosition(self,position=None):
        if not invalidvector(position):
            if position.__class__.__name__=='list' or position.__class__.__name__=='set':
                self.Position[0]=position[0].copy()
                self.Position[1]=position[1].copy()
                self.Position[2]=position[2].copy()
            else:
                self.Position[0]=position[0]
                self.Position[1]=position[1]
                self.Position[2]=position[2]
        if self.LoaderState[0]>0:
            self.LoaderState[0]=2;
        if (not self.multiprocessing) and self.LoaderState[0]>0:
            print "Update postion and load cubes"
            loaderlib.load_cubes(None)
            self.LoaderState[0]=1;

    def LoaderLoop(self,Position,LoaderState,HyperCube0,HyperCube1,HyperCube2,\
        AllCubes,DataScale,CubeSize,NMags,Magnification,NumberofCubes,\
        BaseName,BaseExt,BasePath,NCubesPerEdge,FileType,BaseURL,UserName,Password,ServerFormat,\
        NStreamingChannels,WorkingOffline,NCubes2Load): 
#        for iloop in range(100):
        print "LoaderState: ",  LoaderState[0]
        
        if LoaderState[0]==0:
            print "Initialize Loader..."   
            loaderlib.init_loader(Position,HyperCube0,HyperCube1,HyperCube2,AllCubes,\
                NCubesPerEdge,BasePath,BaseName,BaseExt,NMags,DataScale,CubeSize,NumberofCubes,Magnification,\
                FileType,BaseURL,UserName,Password,ServerFormat,NStreamingChannels,\
                WorkingOffline,NCubes2Load,LoaderState)
        LoaderState[0]=5
        step=0;
        while LoaderState[0]==5 and step<100:
            print "waiting for PyKnossos"
            step+=1;
            time.sleep(0.1)
        print "LoaderState: ",  LoaderState[0]
        while LoaderState[0]>0:
    #        print "in loop, LoaderState: %i..." % (LoaderState[0])
            if LoaderState[0]==2:
                LoaderState[0]=3
#            print "Pos:({0},{1},{2})".format(Position[0],Position[1],Position[2])
            loaderlib.load_cubes(None)
            if LoaderState[0]==3:
                LoaderState[0]=1
            else:
                1
            while LoaderState[0]==1:
                time.sleep(0.01)

        print "Loader loop quited..."        
        loaderlib.release_loader(None)
#        del BasePath
#        del BaseName
#        del AllCubes
        LoaderState[0]=-1
        print "Loader stopped..."   
        

    def StartROI_Extraction(self):
        print "start ROI extraction"
#        if self.LoaderState[0]==0:
#            self.LoadDataset()
        step=0
        while (self.LoaderState[0]<1 and step<20):
            step+=1
            print "sleep"
            time.sleep(0.1)
        if self.LoaderState[0]<1:
            return

        if self.ROIState.value==0:
            extractROIlib.init_ROI(self.HyperCube0,self.HyperCube1,self.HyperCube2,\
            self.AllCubes,self.NMags,self.DataScale,self.CubeSize,self.NCubesPerEdge,self.NumberofCubes)
        elif self.ROIState.value>0:
            extractROIlib.release_ROI(None)   
            extractROIlib.init_ROI(self.HyperCube0,self.HyperCube1,self.HyperCube2,\
            self.AllCubes,self.NMags,self.DataScale,self.CubeSize,self.NCubesPerEdge,self.NumberofCubes)
        self.ROIState.value=1


class BoundingBox:
    def __init__(self):
        self.BoundingBox = vtk.vtkCubeSource()
        self.BoundingBoxEdges = vtk.vtkExtractEdges()
        self.BoundingBoxEdges.SetInputConnection(self.BoundingBox.GetOutputPort())
        self.BoundingBoxMapper = vtk.vtkPolyDataMapper()
        self.BoundingBoxMapper.SetInputConnection(self.BoundingBoxEdges.GetOutputPort())
#        self.BoundingBoxMapper.SetInputConnection(self.BoundingBox.GetOutputPort())
        self.BoundingBoxActor = vtk.vtkActor()
        self.BoundingBoxActor.SetMapper(self.BoundingBoxMapper)
        self.BoundingBoxActor.GetProperty().LightingOff()
        self.BoundingBoxActor.GetProperty().ShadingOff()
        self.BoundingBoxActor.GetProperty().SetColor(0.6,0.6,0.6)
#        self.BoundingBoxActor.GetProperty().SetColor(1.0,0.0,0.0)
        self.BoundingBoxActor.GetProperty().SetLineWidth(2)

    def AddBoundingBox(self,viewport):
        viewport.AddActor(self.BoundingBoxActor)
        
    def RemoveBoundingBox(self,viewport):
        viewport.RemoveViewProp(self.BoundingBoxActor)    

    def UpdateBounds(self,bounds):        
        self.BoundingBox.SetBounds(bounds[0],bounds[1],bounds[2],bounds[3],bounds[4],bounds[5]);
        self.BoundingBox.Modified()    

class timer(QtCore.QTimer):
    IdleThres=2.5*60 #in sec
    lastAction=0
    recordWindow=5*60 #in sec
        
    def __init__(self,ariadne,qt_running_time=None,qt_working_time=None,qt_idle_time=None):
        QtCore.QTimer.__init__(self)
        self.ariadne=ariadne
        self.qt_running_time=qt_running_time
        self.qt_working_time=qt_working_time
        self.qt_idle_time=qt_idle_time
        self.speedRecord=[]
        self.speedTime=[]
        
        self.reset()

    def reset(self,timerOffset=0,cumWorkingTime=0,cumIdleTime=0):
        self.timerOffset=timerOffset
        self.idleTime=cumIdleTime
        self.workTime=cumWorkingTime

        self.speedRecord=[]
        self.speedTime=[]

        self.startTime=time.time()
        self.lastAction=self.startTime
        self.lastUpdate=self.startTime
        self.lastSave=self.startTime
        self.changesSaved=1
                
        self.idleMode=0
        try:
            self.timeout.disconnect(self.updateTime)
        except:
            1
        try:
           self.timeout.connect(self.updateTime)
        except:
            1
        
        self.start(1000)
        
    def submitSpeed(self,dlength,dtime):
        self.speedRecord.append((dlength,dtime))
        self.speedTime.append(time.time())
        self.updateSpeed()
        
    def action(self):
        newTime=time.time()
        dtime=newTime-self.lastAction
        self.lastAction=newTime
        timeStamp=self.lastAction-self.startTime+self.timerOffset
        if self.changesSaved:
            self.changesSaved=0
            self.ariadne.UpdateWindowTitle()
        else:
            self.changesSaved=0
        return timeStamp,dtime
        
    def autoSave(self):
        if self.changesSaved:
            return            
        self.ariadne.Save()

    def updateSpeed(self):

        SpeedCorr=3600/(1000.0*1000.0)
        SpeedUnits='h/mm'
        SpeedTarget=7.0
        SpeedTol=3.0
        SpeedType='inverse'
        
        currJob=self.ariadne.job
        if not (not currJob):
            currTask=self.ariadne.job.get_current_task()
            if not (not currTask):
                SpeedType=currTask._SpeedType
                SpeedUnits=currTask._SpeedUnits
                SpeedTarget=currTask._SpeedTarget
                SpeedTol=currTask._SpeedTol
                SpeedCorr=currTask._SpeedCorr

        if self.speedRecord.__len__()==0:
            meanSpeed=0.0
        else:
            refTime=self.speedTime[self.speedRecord.__len__()-1]
            while self.speedRecord.__len__()>1 and (refTime-self.speedTime[0])>self.recordWindow:
                del self.speedTime[0]
                del self.speedRecord[0]
            meanSpeed=sum([pair[0] for pair in self.speedRecord])/sum([pair[1] for pair in self.speedRecord]) #nm/sec
        if meanSpeed>0.0:
            meanSpeed=meanSpeed*SpeedCorr
            
            if SpeedType=='inverse':
                meanSpeed=1.0/meanSpeed

            self.ariadne.timer_speed.setText("Speed: %02.2f %s" % (meanSpeed, SpeedUnits))
            
            redfract=(meanSpeed-SpeedTarget)/SpeedTol
            redfract=min(1.0,max(0.0,redfract))
            color=colormap[int(redfract*63.0)]
            
            self.ariadne.Job.setStyleSheet("QDockWidget::title{background:rgb("+\
                unicode(255*color[0]) + ',' + unicode(255*color[1]) + ',' + unicode(255*color[2])+')}')
            
        else:
            self.ariadne.timer_speed.setText("Speed: --.-- %s" % (SpeedUnits))
            self.ariadne.Job.setStyleSheet('QDockWidget::title{background:rgb(128,0, 0)}')
        
    def updateTime(self):
        curTime=time.time()        
        elapsedTime=1
        self.lastUpdate=curTime
        
        if self.ariadne.comboBox_AutoSave.currentIndex()>0:
            if (curTime-self.lastSave)> self.ariadne.SpinBoxAutosave.value()*60.0:
                self.lastSave=curTime
                self.autoSave()
                        
        if self.idleMode:
            self.idleTime+=elapsedTime
        else:            
            self.workTime+=elapsedTime

        cur_idlTime=curTime-self.lastAction
        if cur_idlTime>self.IdleThres:
            if not self.idleMode:
                self.idleMode=1                
                self.idleTime+=self.IdleThres
                self.workTime-=self.IdleThres
                if self.workTime<0:
                    self.workTime=0
        else:
            self.idleMode=0
        
        if not (not self.qt_running_time):
            runTime=self.workTime+self.idleTime
            m, s = divmod(runTime, 60)
            h, m = divmod(m, 60)        
            self.qt_running_time.setText("Running: %02d:%02d:%02d" % (h, m, s))
           
        if not (not self.qt_working_time):
            m, s = divmod(self.workTime, 60)
            h, m = divmod(m, 60)
            if self.idleMode:
                self.qt_working_time.setText("Working: %02d:%02d:%02d" % (h, m, s))
            else:
                self.qt_working_time.setText("<font color='green'>Working: %02d:%02d:%02d</font>" % (h, m, s))            
#                self.qt_working_time.setText("<b><font color='green'>Working: %02d:%02d:%02d</font></b>" % (h, m, s))            

        if not (not self.qt_idle_time):
            m, s = divmod(self.idleTime, 60)
            h, m = divmod(m, 60)
            if self.idleMode:
                self.qt_idle_time.setText("<b><font color='red'>Idle: %02d:%02d:%02d</font></b>" % (h, m, s))
            else:    
                self.qt_idle_time.setText("Idle: %02d:%02d:%02d" % (h, m, s))

class config(ConfigObj):  
    def LoadConfig(self,obj,SectionName=None):
        if SectionName==None:
            SectionName=obj.__class__.__name__
        if not self.has_key(SectionName):
            return 0
        allAttributes=self[SectionName]
        for key in allAttributes:
            if not hasattr(obj,key):
                continue
            attribute=self.Cast(allAttributes[key],getattr(obj,key))
            setattr(obj,key,attribute)
#            print SectionName, key, attribute
        return 1
    
    def Cast(self,attr,template):
        templClass=template.__class__.__name__
        if templClass=='list':
            return list(self.ParseListTupleString(attr))
        elif templClass=='tuple':
            return tuple(self.ParseListTupleString(attr))
        elif templClass=='ndarray':
            return np.array(self.ParseListTupleString(attr),dtype=template.dtype)
        elif templClass=='int':
            return int(float(attr))
        elif templClass=='float':
            return float(attr)
        elif templClass=='bool':           
            return self._bools[attr.lower()]
        elif templClass=='unicode':           
            return unicode(attr)
        elif templClass=='str':           
            return str(attr)
        else:
            return attr
    
   
    def ParseListTupleString(self,string):
        if string.__class__.__name__=="tuple":
            return tuple(self.ParseListTupleString(element) for element in string)
        if string.__class__.__name__=="list":
            return list(self.ParseListTupleString(element) for element in string)
        if (string.startswith(u"[") and string.endswith(u"]")):
            string=string[1:(string.__len__()-1)]
            string=string.split(u",")
            return [self.ParseListTupleString(element) for element in string ]
        elif (string.startswith("[") and string.endswith("]")):
            string=string[1:(string.__len__()-1)]
            string=string.split(",")
            return [self.ParseListTupleString(element) for element in string ]
        elif string.startswith(u"(") and string.endswith(u")"):
            string=string[1:(string.__len__()-1)]
            string=string.split(u",")
            return tuple(self.ParseListTupleString(element) for element in string ) 
        elif string.startswith("(") and string.endswith(")"):
            string=string[1:(string.__len__()-1)]
            string=string.split(",")
            return tuple(self.ParseListTupleString(element) for element in string ) 
            
        else:
            if (string.startswith(u" u''") and string.endswith(u"''")):
                string=string[4:(string.__len__()-2)]
            elif (string.startswith(" u''") and string.endswith("''")):
                string=string[4:(string.__len__()-2)]
            elif (string.startswith(u" ''") and string.endswith(u"''")):
                string=string[3:(string.__len__()-2)]
            elif (string.startswith(" ''") and string.endswith("''")):
                string=string[3:(string.__len__()-2)]
            elif (string.startswith(u"''") and string.endswith(u"''")):
                string=string[2:(string.__len__()-2)]
            elif (string.startswith("''") and string.endswith("''")):
                string=string[2:(string.__len__()-2)]
            elif (string.startswith(u" u'") and string.endswith(u"'")):
                string=string[3:(string.__len__()-1)]
            elif (string.startswith(u" '") and string.endswith(u"'")):
                string=string[2:(string.__len__()-1)]
            elif (string.startswith(" u'") and string.endswith("'")):
                string=string[3:(string.__len__()-1)]
            elif (string.startswith(" '") and string.endswith("'")):
                string=string[2:(string.__len__()-1)]
            if (string.startswith(u' u"') and string.endswith(u'"')):
                string=string[3:(string.__len__()-1)]
            elif (string.startswith(u' "') and string.endswith(u'"')):
                string=string[2:(string.__len__()-1)]
            elif (string.startswith(' u"') and string.endswith('"')):
                string=string[3:(string.__len__()-1)]
            elif (string.startswith(' "') and string.endswith('"')):
                string=string[2:(string.__len__()-1)]
            if (string.startswith(u"u'") and string.endswith(u"'")):
                string=string[2:(string.__len__()-1)]
            elif (string.startswith(u"'") and string.endswith(u"'")):
                string=string[1:(string.__len__()-1)]
            elif (string.startswith("u'") and string.endswith("'")):
                string=string[2:(string.__len__()-1)]
            elif (string.startswith("'") and string.endswith("'")):
                string=string[1:(string.__len__()-1)]
            if (string.startswith(u'u"') and string.endswith(u'"')):
                string=string[2:(string.__len__()-1)]
            elif (string.startswith(u'"') and string.endswith(u'"')):
                string=string[1:(string.__len__()-1)]
            elif (string.startswith('u"') and string.endswith('"')):
                string=string[2:(string.__len__()-1)]
            elif (string.startswith('"') and string.endswith('"')):
                string=string[1:(string.__len__()-1)]
            if string.startswith(u' '):
                string=string[1:string.__len__()]
            elif string.startswith(' '):
                string=string[1:string.__len__()]
            if string.__class__.__name__=="str" or string.__class__.__name__=="unicode":                
                if isnumeric(string):
                    if string.isdigit():
                        string=int(string)
                    else:
                        string=float(string)
                elif u',' in string:
                    string=string.split(u",")
                    return list(self.ParseListTupleString(element) for element in string)
                elif ',' in string:
                    string=string.split(",")
                    return list(self.ParseListTupleString(element) for element in string)
            return string
            
    def SaveConfig(self,obj,SectionName=None):
        attributes2exclude=['_inspec','_original_configspec','_created'] +  dir(config)

        if SectionName==None:
            SectionName=obj.__class__.__name__

        allAttributes=list(set(dir(obj))-set(attributes2exclude))        

        for key in allAttributes:
            if key.__len__()<2:
                continue
            if (not key[0]=="_") or key[1]=="_":
                continue
            attribute=getattr(obj,key)
            if callable(attribute):
                continue
            attr_class=attribute.__class__.__name__
            if attr_class[0]=="Q": #very likely it is a QT GUI object; somewhat dirty hack
                continue
            if not self.has_key(SectionName):
                self[SectionName]={}
            if attr_class=="ndarray":
                attribute=[item for item in attribute]
            self[SectionName][key]=attribute
#            print SectionName, key, attribute
            
class job(config):
    _Dataset=''
    def __init__(self,ariadne,jobfile=None):
        config.__init__(self,jobfile, encoding='UTF8')
        self.ariadne=ariadne
        self.tasks=[]
        self._taskIdx=-1
        self.DoneTasks=0
        
#        self.jobtype=self.__class__.__name__
#        self._job_done=False
#        self._datafile=""
#        self._idleTime=0
#        self._workTime=0
            
                
    def load_job(self):        
        self.LoadConfig(self,"job")
        itask=0
        taskname="task{0}".format(itask)
        self.DoneTasks=0;
        while self.has_key(taskname):
            tasktype=self[taskname]["_tasktype"]
            if tasktype=="synapse_detection":
                temptask=synapse_detection(self.ariadne)
            elif tasktype=="tracing":
                temptask=tracing(self.ariadne)
            else:
                temptask=task(self.ariadne)
            self.LoadConfig(temptask,taskname)    
            self.tasks.append(temptask)
            if temptask._done:
                self.DoneTasks+=1
            itask+=1            
            taskname="task{0}".format(itask)
            
        if (not self._taskIdx):
            self._taskIdx=0
        NTasks=self.tasks.__len__()
        self.ariadne.done_tasks.setText("Done tasks: %d/%d" % (self.DoneTasks,NTasks))
        
#        if NTasks<1:
#            return
#        self.goto_task(self._taskIdx)
        
        
    
    def save_job(self):
        self.SaveConfig(self,"job")
        for itask in range(self.tasks.__len__()):
            self.SaveConfig(self.tasks[itask],"task{0}".format(itask))
        self.write()
        

    def get_next_task(self,checkstate=False):
        Ntasks=self.tasks.__len__()
        if Ntasks==0:
            return -1
        if self._taskIdx==None:
            taskIdx=-1
        else:
            taskIdx=self._taskIdx
        for step in range(Ntasks):
            if taskIdx>=(Ntasks-1):
                taskIdx=0
            else:
                taskIdx+=1
            if checkstate:
                if self.tasks[taskIdx]._done:
                    continue
            return taskIdx

    def get_previous_task(self,checkstate=False):
        Ntasks=self.tasks.__len__()
        if Ntasks==0:
            return -1
        if self._taskIdx==None:
            taskIdx=-1
        else:
            taskIdx=self._taskIdx
        for step in range(Ntasks):
            if taskIdx<=0:
                taskIdx=Ntasks-1
            else:
                taskIdx-=1
            if checkstate:
                if self.tasks[taskIdx]._done:
                    continue
            return taskIdx
        
    def goto_next_task(self,checkstate=False):
        taskIdx=self.get_next_task(checkstate)
        self.goto_task(taskIdx)
            
    def goto_previous_task(self,checkstate=False):
        taskIdx=self.get_previous_task(checkstate)
        self.goto_task(taskIdx)

    def goto_task(self,taskIdx):
        if taskIdx==None:
            return
        if taskIdx<0:
            return
        if taskIdx>(self.tasks.__len__()-1):
            return
        self._taskIdx=taskIdx
        self.tasks[taskIdx].show_task()

    def get_current_task(self):
        if self._taskIdx==None:
            return None
        Ntasks=self.tasks.__len__()
        if self._taskIdx>(Ntasks-1):
            return None
        if self._taskIdx<0:
            return None
        return self.tasks[self._taskIdx]

class task():
    _task_description="no description"
    _SpeedCorr=3600/(1000.0*1000.0)
    _SpeedUnits='h/mm'
    _SpeedTarget=7.0
    _SpeedTol=3.0
    _SpeedType='inverse'
    
    def __init__(self,ariadne):
        self._tasktype=self.__class__.__name__
        self._done=False
        self.ariadne=ariadne
        self.Source=None

#        self._Attributes=[]
#        self._AttributesId=[]
        
        self._currNodeId=-1
        self._neuronId=None

        self._enabled_workmodes=["TagMode","SynMode","TracingMode","BrowsingMode"]
        self._curr_workmode="BrowsingMode"

        self.init_task()
        
    def show_task(self):
        self.ariadne.currentTaskDescription.setText(self._task_description)
        self.ariadne._ckbx_TaskDone.setChecked(self._done)
        if self in self.ariadne.job.tasks:
            taskIdx=self.ariadne.job.tasks.index(self)
            self.ariadne.CurrentTaskGroup.setTitle(r'Current Task: #{0}'.format(taskIdx))
                
        self.update_workmodes()
        self.custom_show_task()
    
    def custom_show_task(self):
        1
    
    def load_source(self):
        1

    def update_workmodes(self):
        #enable/disable workmodes:
        if not (self.ariadne._UserMode=="Expert" or usermode==0):
            self.ariadne.radioBtn_Browsing.setEnabled("BrowsingMode" in self._enabled_workmodes)
            self.ariadne.radioBtn_Tracing.setEnabled("TracingMode" in self._enabled_workmodes)
            self.ariadne.radioBtn_Tagging.setEnabled("TagMode" in self._enabled_workmodes)
            self.ariadne.radioBtn_Synapses.setEnabled("SynMode" in self._enabled_workmodes)
        
        if self._curr_workmode=="BrowsingMode":
            self.ariadne.radioBtn_Browsing.setChecked(True)
        elif self._curr_workmode=="TracingMode":
            self.ariadne.radioBtn_Tracing.setChecked(True)
        elif self._curr_workmode=="TagMode":
            self.ariadne.radioBtn_Tagging.setChecked(True)
        elif self._curr_workmode=="SynMode":
            self.ariadne.radioBtn_Synapses.setChecked(True)

class synapse_detection(task):
    _task_description="Synapse detection. Follow the branch using the scroll wheel or up/down keys. At each synaptic site mark the post-synaptic density with a first click. If the post-synaptic process has been traced, mark the closest node of this process with a second click. Else just mark the center of the post-synaptic process by the second click."
    _ReferenceData=""

    def init_task(self):
        self._sequence=[]
        self._sequenceType=""
        self._enabled_workmodes=["SynMode"]
        self._curr_workmode="SynMode"

        self._currPathId=-1        
        self.cDir=None
        self.vDir=None
        self.hDir=None
        self.pathPt=None
        self.locator=None
                
    def current_nodeId(self):
        point, nodeId =self.pathId2nodeId(self._currPathId)     
        return nodeId

    def current_ptId(self):
        point, nodeId =self.pathId2nodeId(self._currPathId)      
        pointIdx=self.Source.nodeId2pointIdx(nodeId)
        if pointIdx<0:
            return None
        return pointIdx
        
        
    def load_source(self):
        self._neuronId=float(self._neuronId)
        if not self.ariadne.Neurons.has_key(self._neuronId):
            print "Source {0} not found.".format(self._neuronId)
            return
        self.Source=self.ariadne.Neurons[self._neuronId].children["skeleton"]
        self.RMF()

    def pathId2nodeId(self,pathId):
        if not self.Source:
            return None, -1, -1
        if not self.isvalid_pathId(pathId):
            return None, -1, -1            
        point,nodeId=self.Source.get_closest_point(self.pathPt[pathId])
        if nodeId==None or nodeId<0:
            return tuple([None,None,None]), -1
        return tuple(point), nodeId

    def nodeId2pathId(self,nodeId):
        if nodeId==None:
            return None
        if not self.Source:
            self.load_source()
            if not self.Source:
                return -1
        pointIdx=self.Source.nodeId2pointIdx(nodeId)
        if pointIdx<0:
            return -1
        point=self.Source.data.GetPoint(pointIdx)
        pathId=self.locator.FindClosestPoint(point)
        return self.pathPt[pathId], pathId

    def isvalid_pathId(self,pathId):
        if pathId==None:
            return 0
        if pathId<0:
            return 0
        if pathId>self.pathPt.__len__()-1:
            return 0
        return 1

    def custom_show_task(self):
        if not hasattr(self,'_ReferenceData'):    
            self.ariadne.btn_loadReferenceFile.setEnabled(False)
        else:
            self.ariadne.btn_loadReferenceFile.setEnabled(True)
            if not (not self._ReferenceData):
                SelObj=self.ariadne.QRWin.SelObj
                filename=self._ReferenceData
                if not (filename in self.ariadne.DemDriFiles):                    
                    filename=self.ariadne.LoadReferenceFile(filename)
                    if not (not filename):
                        self._ReferenceData=filename
                    if not (not SelObj):
                        self.ariadne.QRWin.SetActiveObj(SelObj[0],SelObj[1],SelObj[2])
        if invalidvector(self.cDir):
            self.load_source()
            if invalidvector(self.cDir):
                return
        pathId=self._currPathId
        if not self.isvalid_pathId(pathId):
            pathPoint,pathId=self.nodeId2pathId(self._currNodeId)
        self.GotoPathId(pathId)

    def GotoNodeId(self,nodeId):    
        if nodeId==None:
            return None
        if not self.Source:
            self.load_source()
            if not self.Source:
                return
        pointIdx= self.Source.nodeId2pointIdx(nodeId)
        if pointIdx<0:
            return
        self._currNodeId=nodeId

        SelObj=self.ariadne.QRWin.SelObj
        if SelObj[0]=="synapse" and SelObj[1]==self._neuronId:
            NeuronID=SelObj[1]
            syn_nodeId=SelObj[2]
            obj=self.ariadne.Neurons[NeuronID].children["synapse"]  
            syn_tagIdx=obj.nodeId2tagIdx(syn_nodeId)
            nodeIds=obj.tagIdx2nodeId(syn_tagIdx)
            nodeId0=nodeIds[0]
            jneuron=obj.comments.get(nodeId0,"partner")
            if not (jneuron=="None" or jneuron==None):#->syn partner already assigned
                self.ariadne.QRWin.SetActiveObj("skeleton",self._neuronId,nodeId)
        else:
            self.ariadne.QRWin.SetActiveObj("skeleton",self._neuronId,nodeId)

        point=self.Source.data.GetPoint(pointIdx)
        self.ariadne.JumpToPoint(np.array(point,dtype=np.float))

    def GotoPathId(self,pathId):
        if not self.isvalid_pathId(pathId):
            return -1,0
        point, nodeId = self.pathId2nodeId(pathId)

        SelObj=self.ariadne.QRWin.SelObj
        if SelObj[0]=="synapse" and SelObj[1]==self._neuronId:
            NeuronID=SelObj[1]
            syn_nodeId=SelObj[2]
            obj=self.ariadne.Neurons[NeuronID].children["synapse"]  
            syn_tagIdx=obj.nodeId2tagIdx(syn_nodeId)
            if not (syn_tagIdx==None):
                nodeIds=obj.tagIdx2nodeId(syn_tagIdx)
                nodeId0=nodeIds[0]
                jneuron=obj.comments.get(nodeId0,"partner")
                if not (jneuron=="None" or jneuron==None):#->syn partner already assigned
                    self.ariadne.QRWin.SetActiveObj("skeleton",self._neuronId,nodeId)
        else:
            self.ariadne.QRWin.SetActiveObj("skeleton",self._neuronId,nodeId)
        self._currPathId=pathId
        self._currNodeId=nodeId
        
        actiontime,dt=self.ariadne.Timer.action()
        self.Source.comments.set(nodeId,"time",actiontime)
        
        self.ariadne.JumpToPoint(self.pathPt[pathId],self.cDir[pathId],self.vDir[pathId],self.hDir[pathId])
        return pathId,dt    

    def MoveAlongPath(self,dstep=0):
        if not self.isvalid_pathId(self._currPathId):
            return
        pathId=self._currPathId
        pathId+=dstep
        success,dt=self.GotoPathId(pathId)

        if success>-1:
            dlength=self.ariadne.DataScale[0]*dstep
            if dt>0.0 and dlength>=0.0: #we only care about forward movement
                self.ariadne.Timer.submitSpeed(dlength,dt)                    
            
        #the end of the path is reached the path/ task is considered to be finished
        if pathId>=self.pathPt.__len__()-1:
            self.ariadne.ChangeTaskState(True)
   

    #calculate the rotation minimizing frames      
    def RMF(self):
        tempData=vtk.vtkPolyData()
        tempData.SetPoints(self.Source.data.GetPoints())

        line=list()
        if self._sequenceType=="branches":
            for idx, ibranch in enumerate(self._sequence):
                tempCell=vtk.vtkCellArray()
                tempCell.InsertNextCell(self.Source.data.GetCell(ibranch))
                tempLine=vtk_to_numpy(tempCell.GetData())
                #first element of tempLine is the number of points of the cell
                if idx>0:
                    if line[-1]==tempLine[1]:
                        line.extend(tempLine[2:])
                    elif line[-1]==tempLine[-1]:
                        line.extend(tempLine[-2:0:-1]) #append the reversed sense cell
                    elif idx==1:
                        if line[0]==tempLine[1]:
                            line.reverse()
                            line.extend(tempLine[2:])
                        elif line[0]==tempLine[-1]:
                            line.reverse()
                            line.extend(tempLine[-2:0:-1]) #append the reversed sense cell
                        else:
                            #non-contiuous lines
                            print "branch {0} is non-continuous to branch {1}".format(ibranch,self._sequence[idx-1])
                            line.extend(tempLine[1:])                    
                    else:
                        #non-contiuous lines
                        print "branch {0} is non-continuous to branch {1}".format(ibranch,self._sequence[idx-1])
                        line.extend(tempLine[1:])
                else:
                    line.extend(tempLine[1:])
        elif self._sequenceType=="nodeIds":
            #should test if the returned pointIdx is valid, i.e.>-1
            line=[self.Source.nodeId2pointIdx(nodeId) for nodeId in self._sequence]
        fullline=[line.__len__()]
        fullline.extend(line)
        tempCell=vtk.vtkCellArray()
        tempCell.SetCells(1,numpy_to_vtk(fullline, deep=1, array_type=vtk.VTK_ID_TYPE))
        
        tempData.SetLines(tempCell)

        vtkSplineFilter = vtk.vtkSplineFilter()
        vtkSplineFilter.SetSubdivideToLength()
        vtkSplineFilter.SetLength(200)
        vtkSplineFilter.SetInput(tempData)
    
        vtkSmoothPolyDataFilter=vtk.vtkSmoothPolyDataFilter()
        vtkSmoothPolyDataFilter.FeatureEdgeSmoothingOn()
        vtkSmoothPolyDataFilter.SetNumberOfIterations(5)
        vtkSmoothPolyDataFilter.SetInputConnection(vtkSplineFilter.GetOutputPort())
        
        vtkSplineFilter = vtk.vtkSplineFilter()
        vtkSplineFilter.SetSubdivideToLength()
        vtkSplineFilter.SetLength(self.ariadne.DataScale[0])
        vtkSplineFilter.SetInputConnection(vtkSmoothPolyDataFilter.GetOutputPort())
        vtkSplineFilter.Update()

        Path=vtkSplineFilter.GetOutput()
        
        self.locator=vtk.vtkPointLocator()
        self.locator.SetDataSet(Path)
        self.locator.BuildLocator()

#        LineMapper= vtk.vtkDataSetMapper()
#        LineMapper.SetInputConnection(Path.GetProducerPort()) 
#
#        LineActor =vtk.vtkActor()
#        LineActor.GetProperty().SetLineWidth(2)
#        LineActor.SetMapper(LineMapper)      
#        LineActor.GetProperty().SetDiffuseColor(0, 0, 0)         
#        self.ariadne.QRWin.viewports["skeleton_viewport"].AddActor(LineActor)
        
        NPts=Path.GetNumberOfPoints()
        if NPts <1:
            return

        self.pathPt= vtk_to_numpy(Path.GetPoints().GetData())

        if NPts == 1: #return arbitrary
            self.cDir=np.array((0.0,0.0,1.0),dtype=np.float)
            self.vDir=np.array((0.0,1.0,0.0),dtype=np.float)
            self.hDir=np.array((1.0,0.0,0.0),dtype=np.float)
            return
        
        self.cDir = np.zeros([NPts,3],dtype=np.float)
        self.vDir = np.zeros([NPts,3],dtype=np.float)
        self.hDir = np.zeros([NPts,3],dtype=np.float)
            
        #Compute first normal. All "new" normals try to point in the same 
        #direction.
        idx=0 #first point
        p=self.pathPt[idx+0]    
        pNext=self.pathPt[idx+1]  
        sPrev=pNext-p
        sNext=sPrev.copy()
        tNext=sNext.copy()
        length=vtk.vtkMath.Normalize(tNext)
        if length<1.0e-5:
            print "Coincident points in polyline...can't compute normals"
            return 0
        #the following logic will produce a normal orthogonal
        #to the first line segment. If we have three points
        #we use special logic to select a normal orthogonal
        #to the first two line segments
        foundNormal=0

        if NPts > 2:
            #Look at the line segments (0,1), (ipt-1, ipt)
            #until a pair which meets the following criteria
            #is found: ||(0,1)x(ipt-1,ipt)|| > 1.0E-3.
            #This is used to eliminate nearly parallel cases.
            for ipt in range(2,NPts):
                ftmp=self.pathPt[ipt]-self.pathPt[ipt-1]
                
                length=vtk.vtkMath.Normalize(ftmp)
                if length<1.0e-5:
                    continue                
                #now the starting normal should simply be the cross product
                #in the follvtk.vtkStripper()owing if statement we check for the case where
                #the two segments are parallel 
                normal=np.cross(tNext,ftmp)
                length=vtk.vtkMath.Normalize(normal)
                if length>0.001:
                    foundNormal = 1;
                    break;
        if (NPts<=2 or (not foundNormal)):
            print "Normal not found..."
            normal=np.array([0,0,0],np.float)
            for i in range(0,3):
                if sNext[i] != 0.0:
                    normal[(i+2)%3] = 0.0;
                    normal[(i+1)%3] = 1.0;
                    normal[i] = -sNext[(i+1)%3]/sNext[i];
                    break
            length=vtk.vtkMath.Normalize(normal)
        hDir=np.cross(tNext,normal)
        length=vtk.vtkMath.Normalize(hDir)

        todelete=[]
        self.vDir[0]=normal
        self.cDir[0]=tNext
        self.hDir[0]=hDir
        start_idx=1
    
        for idx in range(start_idx,NPts-1): 
            #inbetween points
            #Generate normals for new point by projecting previous normal
            p=pNext.copy()
            pNext=self.pathPt[idx+1]  
            if all(p==pNext):
                todelete.append(idx)
            else:
                tPrev=tNext.copy()
                sPrev=sNext.copy()        
                sNext=pNext-p
    
                tNext=sNext+sPrev        
                length=vtk.vtkMath.Normalize(tNext)
                if length<1.0e-5:
                    tNext=sNext.copy()
                    vtk.vtkMath.Normalize(tNext)
                c1=vtk.vtkMath.Dot(sNext,sNext)
                normalL=normal-2.0/c1*vtk.vtkMath.Dot(sNext,normal)*sNext    
                tPrevL=tPrev-2.0/c1*vtk.vtkMath.Dot(sNext,tPrev)*sNext
                v2=tNext-tPrevL
                c2=vtk.vtkMath.Dot(v2,v2)
                normal=normalL-2.0/c2*vtk.vtkMath.Dot(v2,normalL)*v2
                vtk.vtkMath.Normalize(normal)
                hDir=np.cross(tNext,normal)
                vtk.vtkMath.Normalize(hDir)
            self.vDir[idx]=normal.copy()
            self.cDir[idx]=tNext.copy()
            self.hDir[idx]=hDir.copy()
            
        idx=NPts-1 #last point; just insert previous
        self.vDir[idx]=normal.copy()
        self.cDir[idx]=tNext.copy()
        self.hDir[idx]=hDir.copy()
        
        self.vDir=np.delete(self.vDir,todelete,0)
        self.cDir=np.delete(self.cDir,todelete,0)
        self.hDir=np.delete(self.hDir,todelete,0)
        self.pathPt=np.delete(self.pathPt,todelete,0)

class tracing(task):
    _task_description="Tracing. Follow untraced branches and put nodes in the center of the process."
    _ReferenceData=""
    
    def init_task(self):
        self._enabled_workmodes=["BrowsingMode","TracingMode"]
        self._curr_workmode="TracingMode"

    def current_nodeId(self):
        return self._currNodeId

    def current_ptId(self):
        pointIdx= self.Source.nodeId2pointIdx(self._currNodeId)
        if pointIdx<0:
            return None
        return pointIdx

    def load_source(self):
        self._neuronId=float(self._neuronId)
        if not self._neuronId in self.ariadne.Neurons:
            print "Neuron id ", self._neuronId , "not found."
            return
        self.Source=self.ariadne.Neurons[self._neuronId].children["skeleton"]

    def custom_show_task(self):
        if not hasattr(self,'_ReferenceData'):    
            self.ariadne.btn_loadReferenceFile.setEnabled(False)
        else:
            self.ariadne.btn_loadReferenceFile.setEnabled(True)
            if not (not self._ReferenceData):
                SelObj=self.ariadne.QRWin.SelObj

                filename=self.ariadne.LoadReferenceFile(self._ReferenceData)
                if not (not filename):                    
                    self._ReferenceData=filename
                if not (not SelObj):
                    self.ariadne.QRWin.SetActiveObj(SelObj[0],SelObj[1],SelObj[2])
        if not self.Source:
            self.load_source()
        self.GotoNodeId(self._currNodeId)

    def GotoNodeId(self,nodeId):    
        if nodeId==None:
            return None
        if not self.Source:
            self.load_source()
            if not self.Source:
                return
        pointIdx= self.Source.nodeId2pointIdx(nodeId)
        if pointIdx<0:
            return
        self._currNodeId=nodeId
        self.ariadne.QRWin.SetActiveObj("skeleton",self._neuronId,pointIdx)
        point=self.Source.data.GetPoint(pointIdx)
        self.ariadne.JumpToPoint(np.array(point,dtype=np.float))

class planeROI():        
    ariadne=QtGui.QMainWindow
    _Orientation="orthogonal"
    ROISize=np.array([256,256],np.int) #[vertical scale,horizontal scale]
    _ROIScale=np.array([1.0,1.0],np.float) #[vertical scale,horizontal scale]
    ROIRes=[1.0,1.0,1.0]
    _OutlineColor=(1,0,0)
    _ForceLoaderFlag=0;
    ScaleBarLength=10000 #nm
    ScaleBarWidth=100 #nm
    _ScaleBarColor=(1.0,0.0,0.0)
    ClippingPlaneTol=50.0;

    def __init__(self):        
        self.ROI=RawArray(c_ubyte,255*np.ones(self.ROISize[0]*self.ROISize[1],dtype=np.uint))
        self.cCenter=RawArray(c_float,np.array([0.0,0.0,0.0],np.float))
        self.cvDir=RawArray(c_float,np.array([0.0,1.0,0.0],np.float))
        self.chDir=RawArray(c_float,np.array([1.0,0.0,0.0],np.float))
        
        self.cROISize=RawArray(c_int,self.ROISize) #[vertical size,horizontal size]

        self.Image= vtk.vtkImageImport()
        self.Image.SetDataScalarTypeToUnsignedChar()
        self.Image.SetNumberOfScalarComponents(1)
        self.complete=0
                
        useLookupTable=1
        if useLookupTable:
            # Create a greyscale lookup table
            self.Table = vtk.vtkLookupTable()
            self.Table.SetRange(0, 255) # image intensity range
#            self.Table.SetRange(50, 205) # image intensity range
            self.Table.SetValueRange(0.0, 1.0) # from black to white
#            self.Table.SetSaturationRange(1.0, 1.0) # no color saturation
            self.Table.SetSaturationRange(0.0, 0.0) # no color saturation
#            self.Table.SetHueRange(0.33,0.33)
#            self.Table.SetHueRange(0.0,0.0)
            self.Table.SetRampToLinear()
            self.Table.Build()
    
            # Map the image through the lookup table
            self.Image2ColorMap = vtk.vtkImageMapToColors()
            self.Image2ColorMap.SetLookupTable(self.Table)
            self.Image2ColorMap.SetInputConnection(self.Image.GetOutputPort())

        self.Texture = vtk.vtkTexture()
        self.Texture.InterpolateOff()
        self.Texture.RepeatOff()
        self.Texture.EdgeClampOff()
        if useLookupTable:
            self.Texture.SetInputConnection(self.Image2ColorMap.GetOutputPort())
        else:
            self.Texture.MapColorScalarsThroughLookupTableOff()
            self.Texture.SetInputConnection(self.Image.GetOutputPort())

        self.PlaneSource = vtk.vtkPlaneSource()
        self.TextureMapper = vtk.vtkDataSetMapper()
        self.TextureMapper.SetInputConnection(self.PlaneSource.GetOutputPort())

        self.PlaneActor = vtk.vtkActor();
        self.PlaneActor.GetProperty().LightingOff()
        self.PlaneActor.GetProperty().ShadingOff()
        self.PlaneActor.PickableOff()        
        #in order to resolve coincident topology issues, eg. between points/lines in plane and the image plane, we set the opacity of the plane below 1.0
        #this is probably not necessary anymore, because we set the global "ResolveCoincidentTopology" parameters for the mappers at the beginning.
        if self._Orientation=="orthogonal":    
            1
#            self.PlaneActor.GetProperty().SetOpacity(0.9999)
        self.PlaneActor.SetMapper(self.TextureMapper);
        self.PlaneActor.SetTexture(self.Texture);
                
        # Create plane outline (frame)
        tempPoints = vtk.vtkPoints()
        tempPoints.SetNumberOfPoints(4)
        tempPoints.InsertPoint(0, 0, 0, 0)
        tempPoints.InsertPoint(1, 1, 0, 0)
        tempPoints.InsertPoint(2, 1, 1, 0)
        tempPoints.InsertPoint(3, 0, 1, 0)
        tempLines = vtk.vtkPolyLine()
        tempLines.GetPointIds().SetNumberOfIds(5)
        tempLines.GetPointIds().SetId(0,0)
        tempLines.GetPointIds().SetId(1,1)
        tempLines.GetPointIds().SetId(2,2)
        tempLines.GetPointIds().SetId(3,3)
        tempLines.GetPointIds().SetId(4,0)
        self.Outline=vtk.vtkPolyData()
        self.Outline.Allocate(1,1)
        self.Outline.SetPoints(tempPoints)
        self.Outline.InsertNextCell(tempLines.GetCellType(),tempLines.GetPointIds())
        self.OutlineMapper=vtk.vtkDataSetMapper()
#        self.OutlineMapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.OutlineMapper.SetInput(self.Outline)
        self.OutlineActor = vtk.vtkActor()
        self.OutlineActor.PickableOff()
        self.OutlineActor.SetMapper(self.OutlineMapper)
        self.OutlineActor.GetProperty().SetDiffuseColor(1, 0, 0)
        self.OutlineActor.GetProperty().SetLineWidth(1.0)
        
        tempPoints = vtk.vtkPoints()
        tempPoints.SetNumberOfPoints(3*4)
        tempPoints.InsertPoint(0, 0.5, 0, 0)
        tempPoints.InsertPoint(1, 1.0, 0, 0)
        tempPoints.InsertPoint(2, 1.0, 0.5, 0)

        tempPoints.InsertPoint(3, 1.0, 0.5, 0)
        tempPoints.InsertPoint(4, 1.0, 1.0, 0)
        tempPoints.InsertPoint(5, 0.5, 1.0, 0)

        tempPoints.InsertPoint(6, 0.5, 1.0, 0)
        tempPoints.InsertPoint(7, 0.0, 1.0, 0)
        tempPoints.InsertPoint(8, 0.0, 0.5, 0)

        tempPoints.InsertPoint(9, 0.0, 0.5, 0)
        tempPoints.InsertPoint(10, 0.0, 0.0, 0)
        tempPoints.InsertPoint(11, 0.5, 0.0, 0)
        tempPoints.Modified()
        
        self.ScaleBar=vtk.vtkPolyData()
        self.ScaleBar.Allocate(1,1)
        self.ScaleBar.SetPoints(tempPoints)
        for icorner in range(4):
            tempLines = vtk.vtkPolyLine()
            tempLines.GetPointIds().SetNumberOfIds(3)
            for ipoint in range(3):
                tempLines.GetPointIds().SetId(ipoint,icorner*3+ipoint)
            self.ScaleBar.InsertNextCell(tempLines.GetCellType(),tempLines.GetPointIds())
        
        self.ScaleBarTube = vtk.vtkTubeFilter()
        self.ScaleBarTube.SetNumberOfSides(32)
        self.ScaleBarTube.SetRadius(self.ScaleBarWidth/2.0)
        self.ScaleBarTube.SetInputConnection(self.ScaleBar.GetProducerPort())
        self.ScaleBarMapper=vtk.vtkDataSetMapper()
        self.ScaleBarMapper.SetInputConnection(self.ScaleBarTube.GetOutputPort())
        self.ScaleBarActor = vtk.vtkActor()
        self.ScaleBarActor.PickableOff()
        self.ScaleBarActor.SetMapper(self.ScaleBarMapper)
        self.ScaleBarActor.GetProperty().SetDiffuseColor(0, 0, 0)
        self.ScaleBarActor.GetProperty().SetLineWidth(3.0)
        self.ScaleBarVisible=False
        
        self.UpdateScaleBars()
        self.SetImageSource()

        self.ClippingPlane=[vtk.vtkPlane(),vtk.vtkPlane()]

    def UpdateScaleBars(self,ScaleBarLength=None,ScaleBarWidth=None,ScaleBarColor=None):
        if not self.ScaleBarVisible:
            return
        cornerPoints=self.Outline.GetPoints()
        if ScaleBarLength==None:
            ScaleBarLength=self.ScaleBarLength;
        else:
            self.ScaleBarLength=ScaleBarLength
        if ScaleBarWidth==None:
            ScaleBarWidth=self.ScaleBarWidth;
        else:
            self.ScaleBarWidth=ScaleBarWidth
            
        self.ScaleBarTube.SetRadius(ScaleBarWidth/2.0)
        self.ScaleBarTube.Modified()
        
        if ScaleBarColor==None:
            ScaleBarColor=self._ScaleBarColor;
        else:
            self._ScaleBarColor=ScaleBarColor;
        self.ScaleBarActor.GetProperty().SetColor(ScaleBarColor)
        vDir=np.array(cornerPoints.GetPoint(3))-np.array(cornerPoints.GetPoint(0))
        vtk.vtkMath.Normalize(vDir)
        hDir=np.array(cornerPoints.GetPoint(1))-np.array(cornerPoints.GetPoint(0))
        vtk.vtkMath.Normalize(hDir)
        
        tempPoints = self.ScaleBar.GetPoints()
        
        cornerPoint=np.array(cornerPoints.GetPoint(0))
        tempPoint=cornerPoint+vDir*ScaleBarLength;
        tempPoints.SetPoint(0,tempPoint)
        tempPoint=cornerPoint;
        tempPoints.SetPoint(1,tempPoint)
        tempPoint=cornerPoint+hDir*ScaleBarLength;
        tempPoints.SetPoint(2,tempPoint)
                
        cornerPoint=np.array(cornerPoints.GetPoint(1))
        tempPoint=cornerPoint+vDir*ScaleBarLength;
        tempPoints.SetPoint(3,tempPoint)
        tempPoint=cornerPoint;
        tempPoints.SetPoint(4,tempPoint)
        tempPoint=cornerPoint-hDir*ScaleBarLength;
        tempPoints.SetPoint(5,tempPoint)

        cornerPoint=np.array(cornerPoints.GetPoint(2))
        tempPoint=cornerPoint-vDir*ScaleBarLength;
        tempPoints.SetPoint(6,tempPoint)
        tempPoint=cornerPoint;
        tempPoints.SetPoint(7,tempPoint)
        tempPoint=cornerPoint-hDir*ScaleBarLength;
        tempPoints.SetPoint(8,tempPoint)

        cornerPoint=np.array(cornerPoints.GetPoint(3))
        tempPoint=cornerPoint-vDir*ScaleBarLength;
        tempPoints.SetPoint(9,tempPoint)
        tempPoint=cornerPoint;
        tempPoints.SetPoint(10,tempPoint)
        tempPoint=cornerPoint+hDir*ScaleBarLength;
        tempPoints.SetPoint(11,tempPoint)
        tempPoints.Modified()

    def UpdateImage(self,Center=None,hDir=None,vDir=None,waitflag=False):
#        print "update image"
#        print self._Orientation
        if not invalidvector(Center): #should here test hDir and vDir as well
            hDir=hDir*self.ROIRes[1]*self._ROIScale[1]
            vDir=vDir*self.ROIRes[0]*self._ROIScale[0]
            for i in range(3):
                self.cCenter[i]=Center[i].copy()
                self.chDir[i]=hDir[i].copy()
                self.cvDir[i]=vDir[i].copy()
        self.complete=0;
        if CubeLoader.ROIState.value==0:
            return    
                        
#        maxROIEdge=np.max([[plane.cROISize[0]*plane.ROIRes[0]*plane._ROIScale[0],plane.cROISize[1]*plane.ROIRes[1]*plane._ROIScale[1]] for key,plane in window1.planeROIs.iteritems()])
        
        maxROIEdge=np.min([self.ROIRes[0]*self._ROIScale[0],self.ROIRes[1]*self._ROIScale[1]])
 
        Magnification=c_int(CubeLoader.CheckMagnification(maxROIEdge));
        
#        if waitflag:
#            complete=0; step=0
#            while (not complete) and step<5:
#                complete=extractROIlib.interp_ROI(self.cCenter,self.cvDir,self.chDir,self.cROISize,self.ROI,Magnification);
#                step+=1
#                
#        else:
        if not CubeLoader.LoaderProcess==None:
            if not CubeLoader.LoaderProcess.is_alive():
                print  "Loader process is dead"
                return
#        print "Orientation: ", self._Orientation, "Center", self.cCenter[0],self.cCenter[1],self.cCenter[2], \
#            "vDir", self.cvDir[0],self.cvDir[1],self.cvDir[2], \
#            "hDir", self.chDir[0],self.chDir[1],self.chDir[2], \
#            "ROISize", self.cROISize[0],self.cROISize[1], "Magnification", Magnification
        complete=extractROIlib.interp_ROI(self.cCenter,self.cvDir,self.chDir,self.cROISize,self.ROI,Magnification,\
            c_int(self._ForceLoaderFlag),CubeLoader.Position,CubeLoader.Magnification,CubeLoader.LoaderState);
        self.complete=complete;
        
#        print complete
#        self.Image.SetDataSpacing(self.ROIRes[0]*self._ROIScale[0],self.ROIRes[1]*self._ROIScale[1],self.ROIRes[0]*self._ROIScale[0])
        self.Image.Modified()
        if not complete:
            if not CubeLoader.LoaderProcess==None:
                if not CubeLoader.LoaderProcess.is_alive():
                    print  "Loader process is dead"
                    return
            if not self.ariadne.QRWin.Timer.isActive():
                self.ariadne.QRWin.Timer.start(1000)
            
#        self.Image.Update()
        
    def SetImageSource(self,ROISize=None):
        if not ROISize:
            ROISize=self.ROISize
        else:
            self.ROISize[0]=ROISize[0]
            self.ROISize[1]=ROISize[1]
        self.OutlineActor.GetProperty().SetDiffuseColor(self._OutlineColor)
        self.ROI=RawArray(c_ubyte,255*np.ones([self.ROISize[0]*self.ROISize[1]],dtype=np.uint))
        self.cROISize[0]=self.ROISize[0]
        self.cROISize[1]=self.ROISize[1]
        self.Image.SetImportVoidPointer(self.ROI)        
        self.Image.SetWholeExtent(0,self.ROISize[0]-1, 0,self.ROISize[1]-1, 0, 0)
        self.Image.SetDataExtent(0,self.ROISize[0]-1, 0,self.ROISize[1]-1, 0, 0)
        self.Image.SetDataSpacing(self.ROIRes[0]*self._ROIScale[0],self.ROIRes[1]*self._ROIScale[1],self.ROIRes[0]*self._ROIScale[0])
        self.Image.Update()
    
    def JumpToPoint(self,NewPoint,cDirRef=None,vDirRef=None,hDirRef=None):
        useRefFlag=1
        if invalidvector(cDirRef):
            cDirRef=np.array(self.PlaneSource.GetNormal(),dtype=np.float)
            useRefFlag=0
        if invalidvector(vDirRef):
            pt0=np.array(self.PlaneSource.GetOrigin(),dtype=np.float)
            pt1=np.array(self.PlaneSource.GetPoint1(),dtype=np.float)
            vDirRef=pt1-pt0
            vtk.vtkMath.Normalize(vDirRef)
            useRefFlag=0
        if invalidvector(hDirRef) and not (invalidvector(cDirRef) or invalidvector(vDirRef)):
            hDirRef=np.cross(cDirRef,vDirRef)
            vtk.vtkMath.Normalize(hDirRef)            
        elif invalidvector(hDirRef):
            pt0=np.array(self.PlaneSource.GetOrigin(),dtype=np.float)
            pt2=np.array(self.PlaneSource.GetPoint2(),dtype=np.float)
            hDirRef=pt2-pt0
            vtk.vtkMath.Normalize(hDirRef)
            useRefFlag=0
#        print "plane: {0}".format(self._Orientation)
#        print "cDir{0},vDir{1}".format(cDir,vDir)
        corrfact=0.5
        corrsign=1.0
        if self._Orientation=="orthogonal":
            corrsign=-1.0
            cDir=cDirRef
            vDir=vDirRef
            hDir=hDirRef
        elif self._Orientation=="ZX":
            if self.ariadne.radioButton_orthRef.isChecked():
                corrsign=-1.0
                if not useRefFlag:
                    cDir=cDirRef
                    vDir=vDirRef
                    hDir=hDirRef
                else:
                    cDir=np.array(vDirRef,dtype=np.float)
                    vDir=np.array(-cDirRef,dtype=np.float)
                    hDir=np.array(hDirRef,dtype=np.float);                    
            else:
                cDir=np.array([0.0,1.0,0.0],dtype=np.float)
                vDir=np.array([0.0,0.0,1.0],dtype=np.float)
                hDir=np.array([1.0,0.0,0.0],dtype=np.float)
        elif self._Orientation=="YZ":
            if self.ariadne.radioButton_orthRef.isChecked():
                corrsign=-1.0
                if not useRefFlag:
                    cDir=cDirRef
                    vDir=vDirRef
                    hDir=hDirRef
                else:
                    cDir=np.cross(vDirRef,cDirRef);
                    vDir=np.array(vDirRef,dtype=np.float)
                    hDir=np.array(cDirRef,dtype=np.float)
            else:
                corrsign=1.0
                cDir=np.array([1.0,0.0,0.0],dtype=np.float)
                vDir=np.array([0.0,1.0,0.0],dtype=np.float)
                hDir=np.array([0.0,0.0,1.0],dtype=np.float)
        elif self._Orientation=="YX":
            corrsign=-1.0
            cDir=np.array([0.0,0.0,1.0],dtype=np.float)
            vDir=np.array([0.0,1.0,0.0],dtype=np.float)
            hDir=np.array([1.0,0.0,0.0],dtype=np.float)
        else:
            cDir=cDirRef
            vDir=vDirRef
            hDir=hDirRef

        distance=vtk.vtkMath.Distance2BetweenPoints(self.PlaneSource.GetCenter(),NewPoint)
                    
#        print "distance: " , np.sqrt(distance)
#        print self._Orientation, ': ', NewPoint, hDir, vDir
        #for wide jumps we wait until all cubes have been loaded.
        vlength=self.ROISize[0]*self.ROIRes[0]*self._ROIScale[0]
        hlength=self.ROISize[1]*self.ROIRes[1]*self._ROIScale[1]
        if distance>(hlength*vlength):
            self.UpdateImage(NewPoint,hDir,vDir,True)
        else:
            self.UpdateImage(NewPoint,hDir,vDir,False);
        
        hax=0.5*hDir*hlength
        vax=0.5*vDir*vlength

        pt0=NewPoint-vax-hax
        pt3=NewPoint+hax+vax
        pt1=NewPoint+vax-hax
        pt2=NewPoint-vax+hax

        self.Outline.GetPoints().SetPoint(0,pt0)
        self.Outline.GetPoints().SetPoint(1,pt1)
        self.Outline.GetPoints().SetPoint(2,pt3)
        self.Outline.GetPoints().SetPoint(3,pt2)
        self.Outline.Modified()
        
        self.UpdateScaleBars()
    
#        self.PlaneSource.SetCenter(pt0)
        self.PlaneSource.SetNormal(cDir)
        self.PlaneSource.SetOrigin(pt0-corrsign*corrfact*cDir)
        self.PlaneSource.SetPoint1(pt1-corrsign*corrfact*cDir)
        self.PlaneSource.SetPoint2(pt2-corrsign*corrfact*cDir)
        self.PlaneSource.Modified()        
        origin=NewPoint-vax-hax
        for idim in range(3):           
            origin[idim]=origin[idim]+corrsign*corrfact*cDir[idim]

        Offset=self.ClippingPlaneTol;
        self.ClippingPlane[0].SetOrigin(origin-Offset*cDir)
        self.ClippingPlane[0].SetNormal(cDir)
        self.ClippingPlane[0].Modified()
        self.ClippingPlane[1].SetOrigin(origin+Offset*cDir)
        self.ClippingPlane[1].SetNormal(-1.0*cDir)
        self.ClippingPlane[1].Modified()

class viewport(vtk.vtkRenderer):
    #interaction mode switches
    _VpId=0
    _Orientation="orthogonal"
    _BorderColor=np.array([1,0,0],dtype=np.float)
    _BorderWidth=2.0
    _AspectRatio=1
    _EnableRotating = 1
    _AllowPanning = 1
    _RestrictZooming=0
    _AllowTagging = 0
    _AllowTracing = 0
    _FollowFocalPoint=1
    _ViewportPlane=""
    _PickingMode="slopy" #"precise"
    _ClippingRange=0.0
    _ClipHulls=True
    _Visible=1
    Magnification=1;
    Maximized=0;
    
    Rotating = 0
    Contrast = 0
    Zooming = 0
    Panning = 0
    MoveTag = None
    
    def __init__(self,ariadne):
        self.LinkedPlaneROIs={}
        self.Intersections=[]
        self._LinkedPlanes=[]
        self._LinkedPlaneOutlines=[]
               
        self.border=vtk.vtkRenderer()
        self.border.SetInteractive(0)
        if not (not ariadne) and self._Visible:
            ariadne.QRWin.RenderWindow.AddRenderer(self.border)           
            ariadne.QRWin.RenderWindow.AddRenderer(self)                 
            self.Iren=ariadne.QRWin.Iren
            self.ariadne=ariadne                
        self.Camera=vtk.vtkCamera()
                
        self.SetActiveCamera(self.Camera)
        self.SetBackground(1,1,1)
        self.ViewportPlane=[];
        
        self.CenterCross=CenterGlyph(self)
               
    def GetPoint(self,x,y):
        FPoint = self.Camera.GetFocalPoint()       
        self.SetWorldPoint(FPoint[0], FPoint[1], FPoint[2], 1.0)
        self.WorldToDisplay()

        DPoint = self.GetDisplayPoint()        
        self.SetDisplayPoint(x,y, DPoint[2])
        self.DisplayToWorld()
        CurrPos0,CurrPos1,CurrPos2,CurrPos3 = self.GetWorldPoint()

        if CurrPos3 != 0.0:
            CurrPos0 = CurrPos0/CurrPos3
            CurrPos1 = CurrPos1/CurrPos3
            CurrPos2 = CurrPos2/CurrPos3
        return np.array([CurrPos0,CurrPos1,CurrPos2],dtype=np.float)
         
    def Resize(self,Geometry):
        BorderWidth=self._BorderWidth
        self.border.SetBackground(self._BorderColor)
#        self.border.SetBackground(np.random.uniform(),np.random.uniform(),np.random.uniform())
        RenderWidth=float(self.ariadne.QRWin.width())
        RenderHeight=float(self.ariadne.QRWin.height())
        self.border.SetViewport(Geometry)
        if self._Visible:
            if BorderWidth==0.0:
                if self.ariadne.QRWin.RenderWindow.HasRenderer(self.border):
                    self.ariadne.QRWin.RenderWindow.RemoveRenderer(self.border)
            else:
                if not self.ariadne.QRWin.RenderWindow.HasRenderer(self.border):
                    self.ariadne.QRWin.RenderWindow.AddRenderer(self.border)       
            if not self.ariadne.QRWin.RenderWindow.HasRenderer(self):
                self.ariadne.QRWin.RenderWindow.AddRenderer(self)
        else:
            if self.ariadne.QRWin.RenderWindow.HasRenderer(self):
                self.ariadne.QRWin.RenderWindow.RemoveRenderer(self)
            if self.ariadne.QRWin.RenderWindow.HasRenderer(self.border):
                self.ariadne.QRWin.RenderWindow.RemoveRenderer(self.border)
        insetGeometry=Geometry+np.array([BorderWidth/RenderWidth,BorderWidth/RenderHeight,-BorderWidth/RenderWidth,-BorderWidth/RenderHeight],dtype=np.double)
        self.SetViewport(insetGeometry);
        self.CenterCross.updatePosition()
        self.Modified()
        size=self.GetSize()
        origin=self.GetOrigin()
        for obj in [region,soma]:
            for ilabel in range(obj.VisibleLabels.GetNumberOfItems()):
                VisibleLabel=obj.VisibleLabels.GetItemAsObject(ilabel)
                if VisibleLabel.GetRenderer()==self:
                    VisibleLabel.SetSelection(origin[0],origin[0]+size[0],origin[1],origin[1]+size[1])                
                    VisibleLabel.Modified()

    def Rotate(self,x, y, lastX, lastY):    
        self.Camera.Azimuth(lastX-x)
        self.Camera.Elevation(lastY-y)
        self.Camera.OrthogonalizeViewUp()
        self.ResetCameraClippingRange()   
        self.CenterCross.updatePosition()
#        if hasattr(self,'ScaleBar'):
#            self.ScaleBar.update()

    # Pan translates x-y motion into translation of the focal point and
    # position.
    def Pan(self,x, y, RPoint):
        if self._AllowPanning==0:
            return
        
        FPoint = np.array(self.Camera.GetFocalPoint(),dtype=np.float)   
        self.SetWorldPoint(FPoint[0], FPoint[1], FPoint[2], 1.0)
        self.WorldToDisplay()
        DPoint = self.GetDisplayPoint()
    
        self.SetDisplayPoint(x, y, DPoint[2])
        self.DisplayToWorld()
        CPoint = np.array(self.GetWorldPoint(),dtype=np.float)        
        if CPoint[3] != 0.0:
            CPoint = CPoint/CPoint[3]

        LateralShift=(RPoint-CPoint)
    
        NewFocalPoint=LateralShift[0:3]+FPoint

        if self._FollowFocalPoint:
            CubeLoader.UpdatePosition(NewFocalPoint)
            self.ariadne.JumpToPoint(NewFocalPoint);
        else:
            self.JumpToPoint(NewFocalPoint)
            self.ariadne.QRWin.Render_Intersect()
        
    def JumpToPoint(self,NewPoint,cDirRef=None,vDirRef=None):
        useRefFlag=1
        if invalidvector(cDirRef):
            FPoint=np.array(self.Camera.GetFocalPoint(),dtype=np.float)
            PPoint=np.array(self.Camera.GetPosition(),dtype=np.float)
            cDirRef=(FPoint-PPoint)
            vtk.vtkMath.Normalize(cDirRef)
            useRefFlag=0
        if invalidvector(vDirRef):
            vDirRef=np.array(self.Camera.GetViewUp(),dtype=np.float)
            useRefFlag=0

        if self._Orientation=="YX":
            cDir=np.array([0.0,0.0,1.0],dtype=np.float)
            vDir=np.array([0.0,-1.0,0.0],dtype=np.float)
        elif self._Orientation=="YZ":
            if self.ariadne.radioButton_orthRef.isChecked():
                if not useRefFlag:
                    cDir=cDirRef;
                    vDir=vDirRef;
                else:
                    cDir=np.cross(vDirRef,cDirRef);
                    vDir=np.array(vDirRef,dtype=np.float)
            else:
                cDir=np.array([-1.0,0.0,0.0],dtype=np.float)
                vDir=np.array([0.0,-1.0,0.0],dtype=np.float)
        elif self._Orientation=="ZX":
            if self.ariadne.radioButton_orthRef.isChecked():
                if not useRefFlag:
                    cDir=cDirRef;
                    vDir=vDirRef;
                else:
                    cDir=np.array(vDirRef,dtype=np.float)
                    vDir=np.array(-cDirRef,dtype=np.float)
            else:
                cDir=np.array([0.0,-1.0,0.0],dtype=np.float)
                vDir=np.array([0.0,0.0,-1.0],dtype=np.float)
        elif self._Orientation=="orthogonal":
            cDir=cDirRef;
            vDir=vDirRef;
        else:
            FPoint=np.array(self.Camera.GetFocalPoint(),dtype=np.float)
            PPoint=np.array(self.Camera.GetPosition(),dtype=np.float)
            cDir=(FPoint-PPoint)
            vtk.vtkMath.Normalize(cDir)
            vDir=np.array(self.Camera.GetViewUp(),dtype=np.float)
                    
        Distance = np.array(self.Camera.GetDistance(),dtype=np.float)
        self.Camera.SetFocalPoint(NewPoint)
        self.Camera.SetPosition(NewPoint-cDir*Distance)
        self.Camera.SetViewUp(vDir)                      
        self.CenterCross.updatePosition()
         
    def AdjustContrast(self,x,y,lastX,lastY):
        dollyFactor =y-lastY
        dollxFactor =x-lastX
        lower=self.ariadne.span_contrast.lowerPosition+dollxFactor
        lower=max(lower,0)
        lower=min(lower,255)
        upper=self.ariadne.span_contrast.upperPosition+dollyFactor
        upper=max(upper,0)
        upper=min(upper,255)

        self.ariadne.span_contrast.setLowerPosition(lower)
        self.ariadne.span_contrast.setUpperPosition(upper)
        self.ariadne.ChangeContrast()

    def ResetViewport(self):
        FocalPoint=np.array([CubeLoader.Position[0],CubeLoader.Position[1],CubeLoader.Position[2]],dtype=np.float)
        self.JumpToPoint(FocalPoint)
        self.Zoom(0)    
        if not (not self.ViewportPlane):
            self.ViewportPlane.JumpToPoint(FocalPoint)
        
            
    # Dolly converts y-motion into a camera dolly commands.
    def Dolly(self,y,lastY):
        dollyFactor = pow(1.02,(0.5*(y-lastY)))
        if "skeleton_viewport" in self.ariadne.QRWin.viewports:
            if (self==self.ariadne.QRWin.viewports["skeleton_viewport"]):
                self.Zoom(dollyFactor)
                return;
            
        if self.ariadne.QRWin.SynZoom>0:
            self.ariadne.SynchronizedZoom(dollyFactor)
        else:
            self.Zoom(dollyFactor)
        
            
    def Zoom(self,dollyFactor):
        ImageChanged=False;

        viewangle=30.0   

        minScale=np.min(CubeLoader._DataScale)
        maxScale=np.max(CubeLoader._DataScale)

        min_distance=10.0*minScale
        default_distance=500.0*minScale
        max_distance=1000000.0*minScale
        
        default_edgelength=500.0 *minScale
        
        if not self.ViewportPlane:
            Zoom0EdgeLength=default_edgelength
        else:
            default_distance= np.sqrt(3.0)*self.ViewportPlane.cROISize[0]*minScale/2.0
            Zoom0EdgeLength=max([self.ViewportPlane.cROISize[0]*minScale,\
                self.ViewportPlane.cROISize[1]*minScale])
            ZoomMaxEdgeLength=max([self.ViewportPlane.cROISize[0]*maxScale,\
                self.ViewportPlane.cROISize[1]*maxScale])
            zoommax_distance=ZoomMaxEdgeLength*0.5/np.tan(viewangle/180.0*np.pi*0.5)

                
        zoom0_distance=Zoom0EdgeLength*0.5/np.tan(viewangle/180.0*np.pi*0.5)
        
        old_distance=self.Camera.GetDistance()
        if not self.ViewportPlane:
            if dollyFactor==0: #reset
                new_distance=default_distance
            else:
                new_distance=old_distance/dollyFactor

            new_distance=max(min_distance,new_distance)
            new_distance=min(max_distance,new_distance)
            dollyFactor=old_distance/new_distance
        else:
            if dollyFactor==0.0: #reset
                new_distance=zoom0_distance
            else:
                new_distance=old_distance/dollyFactor

            new_distance=max(min_distance,new_distance)
            new_distance=min(zoommax_distance,new_distance)
            
            dollyFactor=old_distance/new_distance
                
            NewScale=new_distance/zoom0_distance;
            if NewScale>1.0:
                self.ViewportPlane._ROIScale[0]=NewScale
                self.ViewportPlane._ROIScale[1]=NewScale  
                ImageChanged=True
            else:
                if self.ViewportPlane._ROIScale[0]>=1.0:
                    ImageChanged=True;
                if self.ViewportPlane._ROIScale[1]>=1.0:
                    ImageChanged=True;
                self.ViewportPlane._ROIScale[0]=1.0    
                self.ViewportPlane._ROIScale[1]=1.0    
#                ImageChanged=True

        if self.Camera.GetParallelProjection():
            self.Camera.SetParallelScale(self.Camera.GetParallelScale()/dollyFactor)
        else:
            self.Camera.Dolly(dollyFactor)

            ClippingRange=self._ClippingRange*min(self.ariadne.DataScale)
            
            if self._ClippingRange<1:
                self.ResetCameraClippingRange()    
            else:
#                self.Camera.SetClippingRange(max(min_distance-1,new_distance-ClippingRange),new_distance+1.0)   
                self.Camera.SetClippingRange(max(min_distance-1.0,new_distance-ClippingRange),min(max_distance+1.0,new_distance+ClippingRange))   
        
#        print "Orientation: ",  self._Orientation, "CamPos:", self.Camera.GetPosition(), \
#            "FocalPoint:", self.Camera.GetFocalPoint(), "ClippingRange: ", self.Camera.GetClippingRange(), \
#            "_self._ClippingRange:", self._ClippingRange, \
#            "ClippingRange:",ClippingRange
        for key,intersection in self.ariadne.intersections.iteritems():
            intersection.Intersect()

        self.CenterCross.updatePosition()
        
        if ImageChanged:
            FocalPoint=np.array(self.Camera.GetFocalPoint(),dtype=np.float)
            if not (not self.ViewportPlane):
                self.ViewportPlane.JumpToPoint(FocalPoint)
            for key,linkedplane in self.LinkedPlaneROIs.iteritems():
                linkedplane.JumpToPoint(FocalPoint)                       
        
    def Move(self,dstep,Direction):
        FocalPoint=np.array(self.Camera.GetFocalPoint(),dtype=np.float)
        if Direction=="Z":
            FocalPoint[2]+=dstep*self.ariadne.DataScale[2]
        elif Direction=="Y":
            FocalPoint[1]+=dstep*self.ariadne.DataScale[2]
        elif Direction=="X":
            FocalPoint[0]+=dstep*self.ariadne.DataScale[2]
        elif Direction=="orth":
            PPoint=np.array(self.Camera.GetPosition(),dtype=np.float)
            cDir=(FocalPoint-PPoint)
            vtk.vtkMath.Normalize(cDir)
            FocalPoint+=dstep*self.ariadne.DataScale[2]*cDir
            
        #CubeLoader.UpdatePosition(FocalPoint)
            
        if self._FollowFocalPoint:
            self.ariadne.JumpToPoint(FocalPoint)
        else:
            if not (not self.ViewportPlane):
                self.ViewportPlane.JumpToPoint(FocalPoint)
            for key,linkedplane in self.LinkedPlaneROIs.iteritems():
                linkedplane.JumpToPoint(FocalPoint)
            self.JumpToPoint(FocalPoint)
        
        
class doubleclickFilter(QtCore.QObject):
    state=False
    #delkeyPressed = pyqtSignal()

    def eventFilter(self,  obj,  event):
        if (event.type() == QtCore.QEvent.MouseButtonDblClick) and (event.button()==1): 
            #would have to use QEvent.NonClientAreaMouseButtonDblClick here, but it's not available
            if obj.ariadne.ckbx_MaximizeJobTab.isChecked():
                obj.ariadne.ckbx_MaximizeJobTab.setChecked(False)
            else:
                obj.ariadne.ckbx_MaximizeJobTab.setChecked(True)
            return True
            
        return False
        

class QRenWin(QtGui.QWidget):
    Timer=None
    
    """ Based on QVTKRenderWindowInteractor for Python and Qt. Use
    GetRenderWindow() to get the vtkRenderWindow.  Create with the
    keyword stereo=1 in order to generate a stereo-capable window.
    """

    # Map between VTK and Qt cursors.
    CURSOR_MAP = {
        0:  QtCore.Qt.ArrowCursor,          # VTK_CURSOR_DEFAULT
        1:  QtCore.Qt.ArrowCursor,          # VTK_CURSOR_ARROW
        2:  QtCore.Qt.SizeBDiagCursor,      # VTK_CURSOR_SIZENE
        3:  QtCore.Qt.SizeFDiagCursor,      # VTK_CURSOR_SIZENWSE
        4:  QtCore.Qt.SizeBDiagCursor,      # VTK_CURSOR_SIZESW
        5:  QtCore.Qt.SizeFDiagCursor,      # VTK_CURSOR_SIZESE
        6:  QtCore.Qt.SizeVerCursor,        # VTK_CURSOR_SIZENS
        7:  QtCore.Qt.SizeHorCursor,        # VTK_CURSOR_SIZEWE
        8:  QtCore.Qt.SizeAllCursor,        # VTK_CURSOR_SIZEALL
        9:  QtCore.Qt.PointingHandCursor,   # VTK_CURSOR_HAND
        10: QtCore.Qt.CrossCursor,          # VTK_CURSOR_CROSSHAIR
    }
    
    def __init__(self,parent=None, wflags=QtCore.Qt.WindowFlags(),**kw):
        self.activeRenderer=[]
        
        self.viewports={}

        self.TagMode=0
        self.SynMode=0
        self.TracingMode=0
        self.SynZoom=0

        self.SelObj=["None",-1,-1]
                      
        # the current button
        self.ActiveButton = QtCore.Qt.NoButton

        # private attributes
        setattr(self,'__oldFocus',None)
        setattr(self,'__saveX',0)
        setattr(self,'__saveY',0)
        setattr(self,'__saveModifiers',QtCore.Qt.NoModifier)
        setattr(self,'__saveButtons',QtCore.Qt.NoButton)        
        
        self._ViewportLayout=[]

        # create qt-level widget
        QtGui.QWidget.__init__(self, parent, wflags|QtCore.Qt.MSWindowsOwnDC)



        self.RenderWindow = vtk.vtkRenderWindow()
 
        self.RenderWindow.SetWindowInfo(unicode(int(self.winId())))

        #else the points are ploted as squares.
#        self.RenderWindow.SetPointSmoothing(1)
        self.RenderWindow.OpenGLInit()

        self.Iren = vtk.vtkGenericRenderWindowInteractor()
        self.Iren.SetRenderWindow(self.RenderWindow)
        self.Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self.Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        self.Iren.GetRenderWindow().AddObserver('CursorChangedEvent',
                                                 self.CursorChangedEvent)

        # do all the necessary qt setup
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.setAttribute(QtCore.Qt.WA_PaintOnScreen)
        self.setMouseTracking(True) # get all mouse events
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))


        self.cellPicker=vtk.vtkCellPicker()
#        self.cellPicker.SetTolerance(1e-6)
        self.cellPicker.SetTolerance(5e-3)
        self.cellPicker.SetVolumeOpacityIsovalue(0.1)

        self.InitTimer()
        self.InitCubeQueueTimer()

    def InitCubeQueueTimer(self):
        self.CubeQueueTimer = QtCore.QTimer()
        self.CubeQueueTimer.timeout.connect(self.UpdateCubeQueue)
        self.CubeQueueTimer.init_time=None
        self.CubeQueueTimer.start(200)
        
    def InitTimer(self):
        self.Timer = QtCore.QTimer(self)
        self.connect(self.Timer, QtCore.SIGNAL('timeout()'), self.TimerEvent)
        
        self.Timer.timeout.connect(self.UpdateTimer)
        self.Timer.init_time=None
        self.Timer.start(1000)
        print "Timer initialized"
        
    def UpdateCubeQueue(self):
        self.ariadne.CubeQueue.setTitle("{0} cubes in the queue".format(CubeLoader.NCubes2Load[0]));
        
    def UpdateTimer(self):
#        print "UpdateTimer"
        if not self.Timer.init_time:
            print "timer on"
            self.Timer.init_time=time.time()
            if not CubeLoader.LoaderProcess==None:
                if not CubeLoader.LoaderProcess.is_alive():
                    print  "Loader process is dead"
                    self.Timer.stop()
                    print "timer off. Run time: ",  time.time()-self.Timer.init_time
                    self.Timer.init_time=None
                    
        restart=0
        for key, iplane in self.ariadne.planeROIs.iteritems():
            if not iplane.complete:
                iplane.UpdateImage()
                restart+=1
        self.Render()
        if restart>0: 
            if CubeLoader.LoaderState[0]>0:
                CubeLoader.LoaderState[0]=2
#                self.Timer.start(10)
                return
        self.Timer.stop()
        print "timer off. Run time: ",  time.time()-self.Timer.init_time
        self.Timer.init_time=None

    def HideFocalpoint(self,state):
        if self.ariadne.ckbx_HideFocalpoint.isChecked():
            visibility=0
        else:
            visibility=1
        for key, iviewport in self.viewports.iteritems():
                iviewport.CenterCross.actor.SetVisibility(visibility)
        self.RenderWindow.Render()

    def HideCrosshairs(self,state):
        if self.ariadne.ckbx_HideCrossHairs.isChecked():
            visibility=0
        else:
            visibility=1
        for key, iviewport in self.viewports.iteritems():
            if key=="skeleton_viewport":
                continue
            for intersection in iviewport.Intersections:
                intersection.SetVisibility(visibility)
        if visibility==0:
            self.RenderWindow.Render()
        else:
            self.Render_Intersect()
    
    def SetActiveObj(self,ObjType=None,neuronID=None,nodeId=-1):
        #Active object convention:
        #[objtype,neuronID,nodeID]
#        print "SetActiveObj: ", ObjType, neuronID, nodeId
        if neuronID==None:
            return
        if not self.ariadne.Neurons.has_key(neuronID):
            return

#        spinboxID=np.int(neuronID)
        spinboxID=neuronID
        self.ariadne._SpinBoxNeuronId.setValue(spinboxID)
        self.ariadne._SpinBoxNeuronId_2.setValue(spinboxID)
        self.ariadne._SpinBoxNeuronId_3.setValue(spinboxID)

        if ObjType=="neuron" or ObjType=="area":
            self.unselect()
            self.SelObj=[ObjType,neuronID,None]
            self.ariadne.Neurons[neuronID].select()
        else:
            if not self.ariadne.Neurons[neuronID].children.has_key(ObjType):
                return
            obj=self.ariadne.Neurons[neuronID].children[ObjType]
            
            if 'l' in obj.flags:
                self.ariadne._text_Comment.setEnabled(0)
                self.ariadne._text_Comment_2.setEnabled(0)
                self.ariadne._text_Comment_3.setEnabled(0)
                self.ariadne._text_Comment_4.setEnabled(0)
                self.ariadne.btn_DelNeuron_2.setEnabled(0)
                self.ariadne.btn_DelNeuron_3.setEnabled(0)
                self.ariadne.btn_DelNeuron.setEnabled(0)
            else:
                self.ariadne._text_Comment.setEnabled(1)
                self.ariadne._text_Comment_2.setEnabled(1)
                self.ariadne._text_Comment_3.setEnabled(1)
                self.ariadne._text_Comment_4.setEnabled(1)
                self.ariadne.btn_DelNeuron_2.setEnabled(1)
                self.ariadne.btn_DelNeuron_3.setEnabled(1)
                self.ariadne.btn_DelNeuron.setEnabled(1)
                self.ariadne._CertaintyLevel.setEnabled(1)
                self.ariadne._SynClassTable.setEnabled(1)
                self.ariadne.btn_Syn_assign.setEnabled(1)
                self.ariadne.btn_delete_syn.setEnabled(1)
                self.ariadne.btn_SplitConComp.setEnabled(1)
                self.ariadne.btn_DelConComp.setEnabled(1)
                self.ariadne.btn_MergeConComp.setEnabled(1)
                self.ariadne.btn_MergeNeuron.setEnabled(1)
                self.ariadne.btn_DelNode.setEnabled(1)
                self.ariadne.btn_DelNode_2.setEnabled(1)
                self.ariadne.btn_DelNode_3.setEnabled(1)
            
            if (nodeId==None) or (nodeId<0):
                nodeId=obj.pointIdx2nodeId(0) #if no node is selected, try to select the first node
                if nodeId<0:
                    return
                    
            if ObjType=="region" or ObjType=="soma":
                self.unselect()
                self.SelObj=[ObjType,neuronID,nodeId]
                obj.select()
                
            elif ObjType=="tag":
                self.unselect()
                self.SelObj=[ObjType,neuronID,nodeId]
                tagIdx=obj.nodeId2tagIdx(nodeId)
                obj.select_tag(tagIdx)
            elif ObjType=="synapse":
                tagIdx=obj.nodeId2tagIdx(nodeId)

                if 'l' in obj.flags:
                    self.ariadne._CertaintyLevel.setEnabled(0)
                    self.ariadne._SynClassTable.setEnabled(0)
                    self.ariadne.btn_Syn_assign.setEnabled(0)
                    self.ariadne.btn_delete_syn.setEnabled(0)
               
                if self.SelObj[0]==ObjType and self.SelObj[1]==neuronID:
                    prevtagIdx=obj.nodeId2tagIdx(self.SelObj[2])
                    if prevtagIdx==tagIdx:
                        self.SelObj=[ObjType,neuronID,nodeId]                        
                        self.ariadne.ShowComments(self.SelObj[1],self.SelObj[0],self.SelObj[2])
                        return
    
                self.unselect()
                self.SelObj=[ObjType,neuronID,nodeId]
                obj.select_tag(tagIdx)
                
            elif ObjType=="skeleton":
                self.unselect()
                self.SelObj=[ObjType,neuronID,nodeId]
                obj.select_node(nodeId)            
                if self.TracingMode and not (not self.ariadne.job):
                    currTask=self.ariadne.job.get_current_task()
                    if not (not currTask):
                        if currTask._tasktype=="tracing" and currTask._neuronId==neuronID:
                            currTask._currNodeId=nodeId
                
                if 'l' in obj.flags:
                    self.ariadne.btn_SplitConComp.setEnabled(0)
                    self.ariadne.btn_DelConComp.setEnabled(0)
                    self.ariadne.btn_MergeConComp.setEnabled(0)
                    self.ariadne.btn_MergeNeuron.setEnabled(0)
                    self.ariadne.btn_DelNode.setEnabled(0)
                    self.ariadne.btn_DelNode_2.setEnabled(0)                    
                    self.ariadne.btn_DelNode_3.setEnabled(0)                    
            else:
                self.unselect()
                self.ariadne.Neurons[neuronID].children[ObjType].select()
                self.SelObj=[ObjType,neuronID,None]
        
            self.ariadne._SpinBoxNodeId.setValue(np.int(nodeId))
            self.ariadne._SpinBoxNodeId_2.setValue(np.int(nodeId))
            self.ariadne._SpinBoxNodeId_3.setValue(np.int(nodeId))
        self.ariadne.ShowComments(self.SelObj[1],self.SelObj[0],self.SelObj[2])

    def GotoActiveObj(self,keepframe=False):
        if self.SelObj.__len__()<2:
            return

        ObjType=self.SelObj[0]
        NeuronID=self.SelObj[1]
        nodeId=self.SelObj[2]
       
        if not (NeuronID in self.ariadne.Neurons):
            self.SelObj=[None,None,None]
            return
        if ObjType=="neuron" or ObjType=="area":
            return
        if not ObjType in self.ariadne.Neurons[NeuronID].children:
            self.SelObj=[None,None,None]
            return
            
        obj=self.ariadne.Neurons[NeuronID].children[ObjType]
        Point=None
        if ObjType=="soma" or ObjType=="region":
            cellCenters=vtk.vtkCellCenters()
            cellCenters.SetInputConnection(obj.allDataInput.GetOutputPort())
            cellCenters.Update()
            FaceCenters=cellCenters.GetOutput()
            if not (not FaceCenters):
                FaceCenterPoints=FaceCenters.GetPoints()
                if not (not FaceCenterPoints):
                    FaceCenterPoints=FaceCenterPoints.GetData()
                if not (not FaceCenterPoints):
                    Point=np.mean(vtk_to_numpy(FaceCenterPoints),0)
        elif ObjType=="skeleton":
            pointIdx=obj.nodeId2pointIdx(nodeId)
            if  pointIdx>-1:
                Point=obj.data.GetPoint(pointIdx)
                if self.TracingMode and not (not self.ariadne.job):
                    currTask=self.ariadne.job.get_current_task()
                    if not (not currTask):
                        if currTask._tasktype=="tracing" and currTask._neuronId==NeuronID:
                            currTask._currNodeId=nodeId

        elif ObjType=="synapse":
            tagIdx=obj.nodeId2tagIdx(nodeId)
            if tagIdx==None:
                return
            nodeIds=obj.tagIdx2nodeId(tagIdx)
            if nodeIds==None:
                return
            nodeId=nodeIds[0]

            Point=str2array(obj.comments.get(nodeIds[0],"FPoint"))

            if Point==None:
                pointIdx=obj.nodeId2pointIdx(nodeIds[0])
                if pointIdx>-1:
                    Point=obj.data.GetPoint(pointIdx)    

            if self.SynMode:
                if not (not self.ariadne.job):
                    currTask=self.ariadne.job.get_current_task()
                    if not (not currTask):
                        if currTask._tasktype=="synapse_detection":
                            if not currTask.Source:
                                currTask.load_source()
                            if not (not currTask.Source):
                                skel_Point,skel_nodeId=currTask.Source.get_closest_point(Point)
                                path_Point,pathId=currTask.nodeId2pathId(skel_nodeId)
                                currTask._currPathId=pathId

        elif ObjType=="tag":
            pointIdx=obj.nodeId2pointIdx(nodeId)
            Point=obj.data.GetPoint(pointIdx)
        else:
            return
            
            
        cDir=str2array(obj.comments.get(nodeId,"cDir"))
        vDir=str2array(obj.comments.get(nodeId,"vDir"))
        hDir=None
          
        if (not keepframe) and invalidvector(cDir) and ObjType=="skeleton":
            frame=obj.RMF(nodeId)      
            if frame is None:
                cDir=None
                vDir=None
                hDir=None
            else:
                cDir,vDir,hDir=frame
        if Point==None:
            self.SelObj=[None,None,None]
            return
        self.ariadne.JumpToPoint(np.array(Point),np.array(cDir),np.array(vDir),np.array(hDir))
    
    def DeleteActiveObj(self):
        if not( self.SelObj[1] in self.ariadne.Neurons):
            self.SelObj=[None,None,None]
            return
        neuron_obj=self.ariadne.Neurons[self.SelObj[1]]
        if not (self.SelObj[0] in neuron_obj.children):
            if self.SelObj[0]=="neuron" and not self.SynMode:
                neuron_obj.unselect()
                neuron_obj.delete()            
            self.SelObj=[None,None,None]
            return
        obj=neuron_obj.children[self.SelObj[0]];
        
        if (self.SelObj[0]=="synapse" and self.SynMode) or (self.SelObj[0]=="tag" and self.TagMode):
            self.activeRenderer.MoveTag=None                
            tagIdx=obj.nodeId2tagIdx(self.SelObj[2])
            obj.delete_tag(tagIdx)
            self.SelObj=[None,None,None]
            self.ariadne.Timer.action()
        elif (self.SelObj[0]=="skeleton" and (self.TracingMode>0)):
            NeuronID=self.SelObj[1]
            nodeId=self.SelObj[2]
            pointIdx=obj.nodeId2pointIdx(nodeId)
            if pointIdx>-1:
                Point=obj.data.GetPoints().GetPoint(pointIdx)
                                
            obj.delete_node(nodeId)
            self.ariadne.Timer.action()
            
            Point,nodeId=obj.get_closest_point(np.array(Point,dtype=np.float))
            if nodeId==None or nodeId==-1:
                self.SelObj=[None,None,None]
            else:
                self.SetActiveObj("skeleton",NeuronID,nodeId)
        self.ariadne.ShowComments()
#           print "Deleted NeuronID: {0}, point id: {2}".format(NeuronID,pointIdx)
        self.RenderWindow.Render() 

        
    def unselect(self):
        if not self.SelObj:
            return        
        neuronID=self.SelObj[1]
        if not self.ariadne.Neurons.has_key(neuronID):
            return
        ObjType=self.SelObj[0]    
        if ObjType=="neuron":
            self.ariadne.Neurons[neuronID].unselect()
            return
        if not self.ariadne.Neurons[neuronID].children.has_key(ObjType):
            return

        obj=self.ariadne.Neurons[neuronID].children[ObjType]
        nodeId=self.SelObj[2]
        
        if ObjType=="tag":
            #unselect tag object
            tagIdx=obj.nodeId2tagIdx(nodeId)
            if tagIdx==None:
                return
            obj.unselect_tag(tagIdx)
        elif ObjType=="synapse":
            #unselect synapse object
            tagIdx=obj.nodeId2tagIdx(nodeId)
            if tagIdx==None:
                return
            obj.unselect_tag(tagIdx)
        elif ObjType=="skeleton":
            #unselect skeleton node
            obj.unselect_node(nodeId)
        else:
            self.ariadne.Neurons[neuronID].children[ObjType].unselect()

    def DistributeViewports(self,Layout=None):
        if not Layout:                
            if not self._ViewportLayout:
                return
            Layout=self._ViewportLayout
        else:
            self._ViewportLayout=Layout

        WinHeight=float(self.height())
        WinWidth=float(self.width())
        
        for key,iviewport in self.viewports.iteritems():
            iviewport._Visible=0;
            
        if (Layout.__len__()==1):
            if Layout[0].__len__()==1:
                Height=1.0
                Width=1.0
                X=0.0
                Y=0.0
                Position=np.array([X,Y,min(1.0,X+Width),min(1.0,Y+Height)],dtype=np.double)
                self.viewports[Layout[0][0]]._Visible=1
                self.viewports[Layout[0][0]].Resize(Position)
        else:
            #in order to have proper alignment of the crosshairs, we need an odd number of pixels for the viewport size
            X=0.0
            for col in Layout:
                Y=0.0
                NRows=float(col.__len__())  
                Height=1.0/NRows
                HeightPx=np.floor(Height*WinHeight)
                if np.floor(HeightPx/2.0)*2.0==HeightPx: #even number of pixels
                    HeightPx-=1 #turn it into an odd number
                Height=HeightPx/WinHeight
                newX=1.0*X
                for row in col:
                    if (row=='NaN') or (row==''):
                        Y+=Height
                        continue
                    WidthPx=np.floor(HeightPx*self.viewports[row]._AspectRatio)
                    Width=WidthPx/WinWidth
                    Position=np.array([X,Y,min(1.0,X+Width),min(1.0,Y+Height)],dtype=np.double)
    #                print (Position[3]-Position[1])*WinHeight,(Position[2]-Position[0])*WinWidth
                    newX=max(newX,Position[2].copy())               
                    Y=Position[3].copy()
                    self.viewports[row]._Visible=1
                    self.viewports[row].Resize(Position)
                X=1.0*newX
        
        for key,iviewport in self.viewports.iteritems():
            if not (iviewport._Visible==1):
                for obj in [region,soma]:
                    if not (iviewport in obj.viewports):
                        continue;
                    obj.hide_actors(iviewport,"labelactor");

                if self.RenderWindow.HasRenderer(iviewport.border):
                    self.RenderWindow.RemoveRenderer(iviewport.border)           
                if self.RenderWindow.HasRenderer(iviewport):
                    self.RenderWindow.RemoveRenderer(iviewport)         
            else:
                for obj in [region,soma]:
                    if not (iviewport in obj.viewports):
                        continue;
                    obj.show_actors(iviewport,"labelactor");
#        print "------------"
        return (1.0-X)*WinWidth
       
    def CreateTimer(self, obj, evt):
        self.Timer.start(1000)
        
    def DestroyTimer(self, obj, evt):
        self.Timer.stop()
        return 1

    def TimerEvent(self):
#        print "timer event"
        self.Iren.TimerEvent()

    def CursorChangedEvent(self, obj, evt):
        """Called when the CursorChangedEvent fires on the render window."""
        # This indirection is needed since when the event fires, the current
        # cursor is not yet set so we defer this by which time the current
        # cursor should have been set.
        QtCore.QTimer.singleShot(0, self.ShowCursor)

    def HideCursor(self):
        """Hides the cursor."""
        self.setCursor(QtCore.Qt.BlankCursor)

    def ShowCursor(self):
        """Shows the cursor."""
        vtk_cursor = self.Iren.GetRenderWindow().GetCurrentCursor()
        qt_cursor = self.CURSOR_MAP.get(vtk_cursor, QtCore.Qt.ArrowCursor)
        self.setCursor(qt_cursor)

    def sizeHint(self):
        return self.parent().size()

    def paintEngine(self):        
        return None

    def paintEvent(self, ev):    
        self.RenderWindow.Render()
          
    def ParentResizeEvent(self, ev=None):
        w = self.parent().width()
        h = self.parent().height()
        
        self.resize(w,h)
        vtk.vtkRenderWindow.SetSize(self.RenderWindow,w,h)
        #self.RenderWindow.SetSize(w, h) does not work. For some reason, the sizes are not accepted/ do not match and then the X Y coordinates do not match
        self.Iren.SetSize(w, h)
        self.DistributeViewports()
        
        self.Iren.ConfigureEvent()
        self.update()
        
    def GetCtrlShift(self, ev):
        ctrl = shift = False

        if hasattr(ev, 'modifiers'):
            if ev.modifiers() & QtCore.Qt.ShiftModifier:
                shift = True
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                ctrl = True
        else:
            if getattr(self,'__saveModifiers') & QtCore.Qt.ShiftModifier:
                shift = True
            if getattr(self,'__saveModifiers') & QtCore.Qt.ControlModifier:
                ctrl = True

        return ctrl, shift

    def enterEvent(self, ev):
        if not self.hasFocus():
            setattr(self,'__oldFocus',self.focusWidget())
            self.setFocus()

        ctrl, shift = self.GetCtrlShift(ev)
        self.Iren.SetEventInformationFlipY(getattr(self,'__saveX'), getattr(self,'__saveY'),
                                            ctrl, shift, chr(0), 0, None)
        self.Iren.EnterEvent()

    def leaveEvent(self, ev):
        if getattr(self,'__saveButtons') == QtCore.Qt.NoButton and getattr(self,'__oldFocus'):
            getattr(self,'__oldFocus').setFocus()
            setattr(self,'__oldFocus',None)

        ctrl, shift = self.GetCtrlShift(ev)
        self.Iren.SetEventInformationFlipY(getattr(self,'__saveX'), getattr(self,'__saveY'),
                                            ctrl, shift, chr(0), 0, None)
        self.Iren.LeaveEvent()

    def mousePressEvent(self, ev):
        ctrl, shift = self.GetCtrlShift(ev)
        repeat = 0
        if ev.type() == QtCore.QEvent.MouseButtonDblClick:
            print "double click"
            repeat = 1
        self.Iren.SetEventInformationFlipY(ev.x(), ev.y(),
                                            ctrl, shift, chr(0), repeat, None)
        #for some reason single clicks can result in double click events
#        if repeat>0:
#            return
            
        self.ActiveButton = ev.button()

        x,y = self.Iren.GetEventPosition()

        activeRenderer=self.Iren.FindPokedRenderer(x,y)
        self.activeRenderer=activeRenderer
        
        if not activeRenderer:
            return
            
        inMag=CubeLoader.Magnification[0]
        inVp=activeRenderer._VpId
        #we do not allow clicks outside of viewport
        if not activeRenderer.IsInViewport(x,y):
            return

        if self.ActiveButton == QtCore.Qt.LeftButton:
            if ctrl:
                activeRenderer.Rotating = 1
                return
            else:
                activeRenderer.Rotating = 0

            if repeat: #double click, maximize corresponding viewport
                for key,iviewport in self.viewports.iteritems():
                    if inVp==iviewport._VpId:
                        if iviewport.Maximized:
                            Layout=[[u'ZX_viewport', u'YX_viewport'], [u'Orth_viewport', u'YZ_viewport'], [u'skeleton_viewport']]
                            self.DistributeViewports(Layout)
                            iviewport.Maximized=0
                        else:
                            self.DistributeViewports([[key]])
                            iviewport.Maximized=1
                self.Render()
                return
                
            actors2switchback=[]
            if (not activeRenderer.MoveTag) and (self.SynMode or self.TagMode):
                #switch of picking of skeletons to prioritise synapses and tags
                skel_picking=0
                        
                lowpriority_children=[soma,skeleton]
                for child in lowpriority_children:
                    if hasattr(child,'clippedactor'):
                        if not (not child.clippedactor):
                            if child.clippedactor.GetClassName()=='vtkOpenGLActor':
                                actor=child.clippedactor
                                if actor.GetPickable():
                                    actor.PickableOff()
                                    actor.Modified()
                                    actors2switchback.append(actor)
                            elif child.clippedactor.GetClassName()=='vtkActorCollection':
                                for iactor in range(child.clippedactor.GetNumberOfItems()):
                                    actor=child.clippedactor.GetItemAsObject(iactor)
                                    if actor.GetPickable():
                                        actor.PickableOff()
                                        actor.Modified()
                                        actors2switchback.append(actor)
                        
                    if not child.actor:
                        continue
                    if child.actor.GetClassName()=='vtkOpenGLActor':
                        actor=child.actor
                        if actor.GetPickable():
                            actor.PickableOff()
                            actor.Modified()
                            actors2switchback.append(actor)
                    elif child.actor.GetClassName()=='vtkActorCollection':
                        for iactor in range(child.actor.GetNumberOfItems()):
                            actor=child.actor.GetItemAsObject(iactor)
                            if actor.GetPickable():
                                actor.PickableOff()
                                actor.Modified()
                                actors2switchback.append(actor)
                    
            else:
                skel_picking=1
                
            self.cellPicker.Modified()
            self.cellPicker.Pick(x, y, 0,activeRenderer)

            CellID=self.cellPicker.GetCellId()

            #switch pickability of low priority actors back on
            for actor in actors2switchback:
                actor.PickableOn()
                actor.Modified()

            DataSet=None
            if CellID==-1 and skel_picking==0:
                    self.cellPicker.Modified()
                    self.cellPicker.Pick(x, y, 0,activeRenderer)
                    CellID=self.cellPicker.GetCellId()
            if CellID==-1:
                #clicked in empty space, so we allow panning
                FPoint = np.array(activeRenderer.Camera.GetFocalPoint(),dtype=np.float)   
                activeRenderer.SetWorldPoint(FPoint[0], FPoint[1], FPoint[2], 1.0)
                activeRenderer.WorldToDisplay()
                DPoint = activeRenderer.GetDisplayPoint()
                activeRenderer.SetDisplayPoint(x, y, DPoint[2])
                activeRenderer.DisplayToWorld()
                activeRenderer.PanningReference=np.array(activeRenderer.GetWorldPoint(),dtype=np.float)    
                activeRenderer.Panning = 1
                return

            DataSet=self.cellPicker.GetDataSet()
            PointID=self.cellPicker.GetPointId()
            PickedPoint=np.array(self.cellPicker.GetPickPosition(),dtype=np.float);

            ObjType=DataSet.GetFieldData().GetAbstractArray("ObjType")
            if ObjType==None:
                return
            else:
                ObjType=ObjType.GetValue(0)
            
            tempArray=DataSet.GetPointData().GetArray("NeuronID")
            if tempArray==None:
                return
            NeuronID=float(np.round(tempArray.GetValue(PointID),3))
            if NeuronID==None:
                return
                
            if ObjType in ["tag","synapse"]: #clicked on tag/ synapse
                if not (not activeRenderer.MoveTag):
                    return
                #self.ariadne.Timer.action()

                obj=self.ariadne.Neurons[NeuronID].children[ObjType]
                Point,nodeId,tagIdx=obj.get_closest_point(PickedPoint)
                self.SetActiveObj(ObjType,NeuronID,nodeId)

                activeRenderer.MoveTag=None
                if activeRenderer._AllowTagging:
                    if self.TagMode and ObjType=="tag":
                        activeRenderer.MoveTag=[ObjType,NeuronID,nodeId]
                    elif self.SynMode and ObjType=="synapse":
                        activeRenderer.MoveTag=[ObjType,NeuronID,nodeId]
                self.RenderWindow.Render()

            elif ObjType in ["soma","region"]: #clicked on soma
                obj=self.ariadne.Neurons[NeuronID].children[ObjType]
                nodeId=obj.pointIdx2nodeId(0)
#                Point,nodeId=obj.get_closest_point(PickedPoint)
                self.SetActiveObj(ObjType,NeuronID,nodeId)
#                print "soma of neuron:{0}".format(NeuronID)
                return
            elif ObjType=="skeleton": #clicked on skeleton    
                if self.TagMode and activeRenderer._AllowTagging:
                    return
                obj=self.ariadne.Neurons[NeuronID].children[ObjType]
                Point,nodeId=obj.get_closest_point(PickedPoint)
                self.SetActiveObj(ObjType,NeuronID,nodeId)
                print "neuronID: {0}, nodeId: {1} ".format(NeuronID,nodeId)
                #we are likely in the skeleton viewport
                if self.TracingMode>0:
                    if shift:
                        if activeRenderer._AllowTracing:
                            activeRenderer.MoveTag=[ObjType,NeuronID,nodeId]
                        return

                    #jump to selected skeleton node    
                    if activeRenderer._Orientation=="orthogonal": 
                        cDir=None
                        vDir=None
                        hDir=None
                    else:
                        frame=obj.RMF(nodeId)      
                        if frame is None:
                            cDir=None
                            vDir=None
                            hDir=None
                        else:
                            cDir,vDir,hDir=frame

                    self.ariadne.JumpToPoint(Point,cDir,vDir,hDir)
                else:
                    self.RenderWindow.Render()
                    return
                    
        elif self.ActiveButton == QtCore.Qt.RightButton:
            if ctrl:
                activeRenderer.Zooming = 1
                activeRenderer.Contrast = 0
                return
            elif shift:
                activeRenderer.Contrast= 1
                activeRenderer.Zooming = 0
                return

            activeRenderer.Zooming = 0
            activeRenderer.Contrast = 0

            self.cellPicker.Modified()
            self.cellPicker.Pick(x, y, 0,activeRenderer)
            CellID=self.cellPicker.GetCellId()
            DataSet=None
            PointID=-1
            if not CellID==-1:
                DataSet=self.cellPicker.GetDataSet()
                PointID=self.cellPicker.GetPointId()
                PickedPoint=np.array(self.cellPicker.GetPickPosition(),dtype=np.float);

#            print "picked point (CellID: {0}): {1}; pos: {2}".format(CellID,PickedPoint)
            if activeRenderer._AllowTagging and (not activeRenderer.MoveTag):
                if self.SynMode: #drop new synapse or assign syn partner 
                    ObjType=self.SelObj[0]
                    if ObjType=="synapse":
                        NeuronID=self.SelObj[1]
                        nodeId=self.SelObj[2]
                        obj=self.ariadne.Neurons[NeuronID].children[ObjType]  
                        
                        if not ('l' in obj.flags): #locked obj
                            tagIdx=obj.nodeId2tagIdx(nodeId)    
                            nodeIds=obj.tagIdx2nodeId(tagIdx)
                            nodeId0=nodeIds[0]
                            nodeId2=nodeIds[2]
                            jneuron=obj.comments.get(nodeId0,"partner")
#                            print jneuron
                            if jneuron=="None" or jneuron==None:#->assign syn partner
                                jneuron="Unknown"
                                Point2 = None
                                if not DataSet==None:
                                    tempArray=DataSet.GetPointData().GetArray("NeuronID")
                                    if tempArray==None:
                                        PartnerNeuronID=None
                                    else:
                                        PartnerNeuronID=float(np.round(tempArray.GetValue(PointID),3))
                                    if (not PartnerNeuronID==None) and (not DataSet.GetFieldData().GetAbstractArray("ObjType")==None):
                                        ObjType2=DataSet.GetFieldData().GetAbstractArray("ObjType").GetValue(0)
                                        obj2=self.ariadne.Neurons[PartnerNeuronID].children[ObjType2]
                                        if ObjType2=="skeleton":
                                            Point2,Obj2nodeId=obj2.get_closest_point(PickedPoint)
                                            jneuron=PartnerNeuronID #do we have to convert this to int?
                                        elif ObjType2=="soma":
                                            jneuron=PartnerNeuronID #do we have to convert this to int?
                                            
                                if Point2==None:
                                    if not (activeRenderer.ViewportPlane):
                                        return
                                    picker = vtk.vtkCellPicker()
                                    picker.SetPickFromList(1)                
                                    picker.AddPickList(activeRenderer.ViewportPlane.PlaneActor);
                                    picker.Pick(x,y,0,activeRenderer)
                                    Point2 = picker.GetPickPosition()
    #                                print "Picked point: {0}".format(Point2)
                                actiontime,dt=self.ariadne.Timer.action()  
                                obj.modify_tag(tagIdx,2,Point2)
                                obj.comments.set(nodeId0,"partner",jneuron)
                                obj.comments.set(nodeId2,"time",actiontime)
                                obj.comments.set(nodeId2,"inMag",inMag)
                                obj.comments.set(nodeId2,"inVp",inVp)
                            
                                self.SetActiveObj(ObjType,NeuronID,self.SelObj[2])
                                self.RenderWindow.Render()
                                return
                                
                    #drop new synapse   
                    ObjType="synapse"
                    Point0=None
                    NeuronID=None
                    if not (not self.ariadne.job):
                        currTask=self.ariadne.job.get_current_task()
                        if not currTask==None:
                            if currTask._tasktype=="synapse_detection":
                                NeuronID= float(np.round(currTask._neuronId,3))
                                Point0 = currTask.pathPt[currTask._currPathId]
                    if self.SelObj[0]=="skeleton":
                        if NeuronID==None:
                            NeuronID=self.SelObj[1]
                        obj=self.ariadne.Neurons[NeuronID].children["skeleton"]
                        if NeuronID==self.SelObj[1]:
                            pointIdx=obj.nodeId2pointIdx(self.SelObj[2])
                            if pointIdx>-1:
                                Point0=obj.data.GetPoint(pointIdx)

                    if Point0==None:
                        return

                        
                    if not (activeRenderer.ViewportPlane):
                        return
                    picker = vtk.vtkCellPicker()
                    picker.SetPickFromList(1)                
                    picker.AddPickList(activeRenderer.ViewportPlane.PlaneActor);
                    picker.Pick(x,y,0,activeRenderer)
                    Point1 = picker.GetPickPosition()
#                    print "Picked point: {0}".format(Point1)
                    

                    if ObjType in self.ariadne.Neurons[NeuronID].children:
                        obj=self.ariadne.Neurons[NeuronID].children[ObjType]
                    else:
                        color=self.ariadne.Neurons[NeuronID].LUT.GetTableValue(self.ariadne.Neurons[NeuronID].colorIdx)
                        obj=synapse(self.ariadne.Neurons[NeuronID].item,NeuronID,color)
                        self.ariadne.Neurons[NeuronID].children[ObjType]=obj
                        obj.start_VisEngine(self.ariadne)

                    nodeId=-1
                    if not ('l' in obj.flags): #locked obj
                        actiontime,dt=self.ariadne.Timer.action()  
                        nodeId,tagIdx =obj.add_tag(Point0,Point1,[])
                        nodeIds=obj.tagIdx2nodeId(tagIdx)
                        nodeId1=nodeIds[1]
                        obj.comments.set(nodeId1,"time",actiontime)
                        obj.comments.set(nodeId1,"inMag",inMag)
                        obj.comments.set(nodeId1,"inVp",inVp)
                            
                    if "Orth_viewport" in self.viewports:
                        FPoint=self.viewports["Orth_viewport"].Camera.GetFocalPoint()
                        CPoint=self.viewports["Orth_viewport"].Camera.GetPosition()
                        cDir=np.array(FPoint,dtype=np.float)-np.array(CPoint,dtype=np.float)
                        vtk.vtkMath.Normalize(cDir)
                        vDir=np.array(self.viewports["Orth_viewport"].Camera.GetViewUp(),dtype=np.float)  
                        if not ('l' in obj.flags): #locked obj
                            obj.comments.set(nodeId,"FPoint",FPoint)
                            obj.comments.set(nodeId,"cDir",tuple(cDir))
                            obj.comments.set(nodeId,"vDir",tuple(vDir))

                    if nodeId>-1:
                        self.SetActiveObj(ObjType,NeuronID,nodeId)
#                        print "new synapse: {0}".format(self.SelObj)

                    self.RenderWindow.Render()
                    return
                if self.TagMode:
                    #drop new tag            
                    actiontime,dt=self.ariadne.Timer.action()

                    if self.SelObj[0] in ["tag","skeleton","synapse"]:
                        NeuronID=self.SelObj[1]
                    else:
                        print "No parent neuron selected."
                        #new neuron
                        NeuronID=self.ariadne.NewNeuron()

                    picker = vtk.vtkWorldPointPicker()
                    picker.Pick(x,y,0,activeRenderer)
                    PickedPoint = picker.GetPickPosition()

#                    print "tagging at position: {0}".format(PickedPoint)
                    
                    ObjType="tag"
                    if self.ariadne.Neurons[NeuronID].children.has_key(ObjType):
                        obj=self.ariadne.Neurons[NeuronID].children[ObjType]
                    else:
                        color=self.ariadne.Neurons[NeuronID].LUT.GetTableValue(self.ariadne.Neurons[NeuronID].colorIdx)
                        obj=tag(self.ariadne.Neurons[NeuronID].item,NeuronID,color)
                        self.ariadne.Neurons[NeuronID].children[ObjType]=obj
                        obj.start_VisEngine(self.ariadne)
                    
                    if not ('l' in obj.flags): #locked obj
                        nodeId=obj.add_tag(np.array(PickedPoint,dtype=np.float))
                        obj.comments.set(nodeId,"time",actiontime)
                        obj.comments.set(nodeId,"inMag",inMag)
                        obj.comments.set(nodeId,"inVp",inVp)
                        self.SetActiveObj(ObjType,NeuronID,nodeId)

                    self.RenderWindow.Render()       
                    return

            if self.TracingMode>0 and activeRenderer._AllowTracing:

                if not (activeRenderer.ViewportPlane):
                    return
                picker = vtk.vtkCellPicker()
                picker.SetPickFromList(1)                
                picker.AddPickList(activeRenderer.ViewportPlane.PlaneActor);
                picker.Pick(x,y,0,activeRenderer)
                PickedPoint = picker.GetPickPosition()
#                print "Picked point: {0}".format(PickedPoint)

                #add new node, if necessary create new neuron
                actiontime,dt=self.ariadne.Timer.action()
                
                NeuronID=self.SelObj[1]                

                ObjType=self.SelObj[0]
                cDir=None
                vDir=None
                hDir=None
                if not (NeuronID in self.ariadne.Neurons):
                    #new neuron
                    NeuronID=self.ariadne.NewNeuron()

                if "skeleton" in self.ariadne.Neurons[NeuronID].children:  
                    obj=self.ariadne.Neurons[NeuronID].children["skeleton"]
                else:
                    color=self.ariadne.Neurons[NeuronID].LUT.GetTableValue(self.ariadne.Neurons[NeuronID].colorIdx)
                    obj=skeleton(self.ariadne.Neurons[NeuronID].item,NeuronID,color)
                    self.ariadne.Neurons[NeuronID].children["skeleton"]=obj
                    obj.start_VisEngine(self.ariadne)
                
                if ('l' in obj.flags): #locked obj
                    return

                nodeId,pointIdx=obj.add_node(PickedPoint)
                obj.comments.set(nodeId,"time",actiontime)
                obj.comments.set(nodeId,"inMag",inMag)
                obj.comments.set(nodeId,"inVp",inVp)
                
                
                #TracingMode==2: single nodes; TracingMode==1: connected nodes
                if ObjType=="skeleton" and (self.TracingMode==1):
                    dlength=obj.add_edge(self.SelObj[2],nodeId)  
                    if dt>0.0 and dlength>=0.0:
                        self.ariadne.Timer.submitSpeed(dlength,dt)
                    
                    if self.ariadne.planeROIs.has_key("Orth_planeROI"):
                        plane=self.ariadne.planeROIs["Orth_planeROI"]
                        cDir=np.array(plane.PlaneSource.GetNormal(),dtype=np.float)
                        pt0=np.array(plane.PlaneSource.GetOrigin(),dtype=np.float)
                        pt1=np.array(plane.PlaneSource.GetPoint1(),dtype=np.float)
                        vDir=pt1-pt0
                        vtk.vtkMath.Normalize(vDir)
                    frame=obj.RMF(self.SelObj[2],nodeId,cDir,vDir)
                    if not frame==None:
                        cDir,vDir,hDir=frame

                self.SetActiveObj("skeleton",NeuronID,nodeId)
                    
                FPoint=activeRenderer.Camera.GetFocalPoint()
                MoveDir=np.array((PickedPoint[0]-FPoint[0],PickedPoint[1]-FPoint[1],PickedPoint[2]-FPoint[2]),dtype=np.float)
                vtk.vtkMath.Normalize(MoveDir)
                newPoint=np.array(FPoint,dtype=np.float)
                
                if self.ariadne.ckbx_Recenter.isChecked():
                    MovementTime=0.3#sec for whole distance one call to ariadne.JumpToPoint takes about 20ms  
                    TimePerCall=0.085#sec, estimated time per call to JumpToPoint
                    min_stepsize=8*np.min(CubeLoader._DataScale)#9.252*3
                    startDistance=np.sqrt(vtk.vtkMath.Distance2BetweenPoints(newPoint,PickedPoint))
                    stepsize=np.max([min_stepsize,startDistance/MovementTime*TimePerCall])
                    Nsteps=np.floor(startDistance/stepsize)
                    startTime=time.time()
                    istep=0
                    while (istep<Nsteps) and ((time.time()-startTime)<MovementTime):
    #                (vtk.vtkMath.Distance2BetweenPoints(newPoint,PickedPoint)>stepsize*stepsize):
                        newPoint=newPoint+stepsize*MoveDir
                        self.ariadne.JumpToPoint(newPoint,cDir,vDir,hDir)
                        istep+=1
    
                    self.ariadne.JumpToPoint(np.array(PickedPoint,dtype=np.float),cDir,vDir,hDir)
                else:
                    self.Render()
                return

            InfoStr="Local information:\n"
            if not CellID==-1:      
                InfoStr+= extractObjInfo(DataSet.GetCellData(),CellID)
                
            if not PointID==-1:      
                InfoStr+=extractObjInfo(DataSet.GetPointData(),PointID)
            
            if not InfoStr=="Local information:\n":
                print InfoStr

        elif self.ActiveButton == QtCore.Qt.MidButton:
            if not (ctrl or shift):
                FPoint = np.array(activeRenderer.Camera.GetFocalPoint(),dtype=np.float)   
                activeRenderer.SetWorldPoint(FPoint[0], FPoint[1], FPoint[2], 1.0)
                activeRenderer.WorldToDisplay()
                DPoint = activeRenderer.GetDisplayPoint()
                activeRenderer.SetDisplayPoint(x, y, DPoint[2])
                activeRenderer.DisplayToWorld()
                activeRenderer.PanningReference=np.array(activeRenderer.GetWorldPoint(),dtype=np.float)    
                activeRenderer.Panning = 1
                return;
            if ctrl:
                activeRenderer.Panning = 0

                if not (self.TracingMode>0):
                    return
                    
                if not self.SelObj[0]=="skeleton":
                    return

                        
                self.cellPicker.Modified()
                self.cellPicker.Pick(x, y, 0,activeRenderer)
                CellID=self.cellPicker.GetCellId()
                if CellID==-1:
                    return
#                    self.pointPicker.Pick(x, y, 0,activeRenderer)
#                    PointID=self.pointPicker.GetPointId()
#                    if PointID==-1:
#                        return
#                    DataSet=self.pointPicker.GetDataSet()
#                    PickedPoint=self.pointPicker.GetPickPosition()
                else:                    
                    DataSet=self.cellPicker.GetDataSet()
                    PointID=self.cellPicker.GetPointId()
                    PickedPoint=np.array(self.cellPicker.GetPickPosition(),dtype=np.float);
    
                if DataSet.GetFieldData().GetAbstractArray("ObjType")==None:
                    return

                tempArray=DataSet.GetPointData().GetArray("NeuronID")
                if tempArray==None:
                    return
                NeuronID=self.SelObj[1]
                newNeuronID=float(np.round(tempArray.GetValue(PointID),3))
                if not (NeuronID==newNeuronID):
                    return

                obj=self.ariadne.Neurons[NeuronID].children["skeleton"]
                
                if ('l' in obj.flags): #locked obj
                    return
                    
                Point,nodeId=obj.get_closest_point(PickedPoint) 
                
                if self.SelObj[2]==nodeId:
                    return        
                
                actiontime,dt=self.ariadne.Timer.action()
                
                if obj.isconnected(self.SelObj[2],nodeId):
                    obj.delete_edge(self.SelObj[2],nodeId)
                    print "Deleted edge between nodeId {0} - nodeId {1}".format(self.SelObj[2],nodeId)
                else:
                    obj.add_edge(self.SelObj[2],nodeId)
                    print "Added edge between nodeId {0} - nodeId {1}".format(self.SelObj[2],nodeId)
                self.Render()

    def mouseReleaseEvent(self, ev):
        ctrl, shift = self.GetCtrlShift(ev)
        self.Iren.SetEventInformationFlipY(ev.x(), ev.y(),
                                            ctrl, shift, chr(0), 0, None)

        x,y = self.Iren.GetEventPosition()       

        if not self.activeRenderer:
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)            
        elif not (self.activeRenderer.Rotating or self.activeRenderer.Zooming \
        or self.activeRenderer.Panning or self.activeRenderer.Contrast \
        or (not (not self.activeRenderer.MoveTag))):
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)

        inMag=CubeLoader.Magnification[0]
        inVp=self.activeRenderer._VpId

        self.activeRenderer.Rotating = 0
        self.activeRenderer.Panning = 0
        self.activeRenderer.Zooming = 0
        self.activeRenderer.Contrast = 0

        if self.ActiveButton == QtCore.Qt.LeftButton:
            if not (not self.activeRenderer.MoveTag):
                
                ObjType,NeuronID,nodeId=self.activeRenderer.MoveTag    
                self.activeRenderer.MoveTag=None    
                if not( ObjType in self.ariadne.Neurons[NeuronID].children):
                    return                
                obj=self.ariadne.Neurons[NeuronID].children[ObjType]

                if ('l' in obj.flags): #locked obj
                    return

                Point2Selected=False                
                if ObjType=="synapse":
                    tagIdx=obj.nodeId2tagIdx(nodeId)
                    nodeIds=obj.tagIdx2nodeId(tagIdx)
                    if nodeIds[2]==nodeId:
                        Point2Selected=True
                        
                if Point2Selected:
                    obj.set_pickable(0) #to avoid that the moved synapse it self is picked
                    #we are moving the 3rd node of a synapse
                    self.cellPicker.Modified()
                    self.cellPicker.Pick(x, y, 0,self.activeRenderer)
                    CellID=self.cellPicker.GetCellId()
                    DataSet=None
                    PickedPoint=np.array(self.cellPicker.GetPickPosition(),dtype=np.float);
                    if not CellID==-1:
                        DataSet=self.cellPicker.GetDataSet()
                        PointID=self.cellPicker.GetPointId()
                    
                    jneuron="Unknown"
                    Point2 = PickedPoint                                 
                    if not DataSet==None:
                        tempArray=DataSet.GetPointData().GetArray("NeuronID")
                        if tempArray==None:
                            NeuronID=None
                        else:
                            NeuronID=float(np.round(tempArray.GetValue(PointID),3))
                        if not (NeuronID==None):
                            ObjType2=DataSet.GetFieldData().GetAbstractArray("ObjType").GetValue(0)
                            if ObjType2=="skeleton":
                                jneuron=NeuronID
                                Point2,Obj2nodeId=\
                                    self.ariadne.Neurons[jneuron].children["skeleton"].get_closest_point(PickedPoint)
                            elif ObjType2=="soma":
                                jneuron=NeuronID
                                
                    
                    obj.set_pickable(1) #was turned off above
                    
#                    print jneuron
                    actiontime,dt=self.ariadne.Timer.action() 
                    obj.modify_tag(tagIdx,2,Point2)
                    nodeId2=nodeIds[2]
                    obj.comments.set(nodeId2,"time",actiontime)                    
                    obj.comments.set(nodeId2,"inMag",inMag)
                    obj.comments.set(nodeId2,"inVp",inVp)
                    nodeId0=nodeIds[0]
                    obj.comments.set(nodeId0,"partner",jneuron)
                    self.ariadne.ShowComments()
                    
                    self.RenderWindow.Render()
                    return

        elif self.ActiveButton == QtCore.Qt.RightButton:
            1
        elif self.ActiveButton == QtCore.Qt.MidButton:
            1
        self.activeRenderer=0

    def mouseMoveEvent(self, ev):
        
        setattr(self,'__saveModifiers',ev.modifiers())
        setattr(self,'__saveButtons',ev.buttons())
        setattr(self,'__saveX',ev.x())
        setattr(self,'__saveY',ev.y())

        ctrl, shift = self.GetCtrlShift(ev)
        self.Iren.SetEventInformationFlipY(ev.x(), ev.y(),
                                            ctrl, shift, chr(0), 0, None)
 
        x,y = self.Iren.GetEventPosition()   
#        print "x:{0},y:{1}".format(x,y)
              
        if not self.activeRenderer:
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)
        elif not (self.activeRenderer.Rotating or self.activeRenderer.Zooming \
        or self.activeRenderer.Panning or self.activeRenderer.Contrast\
        or not (not self.activeRenderer.MoveTag)):
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)

        inMag=CubeLoader.Magnification[0]
        inVp=self.activeRenderer._VpId
 
        if not (not self.activeRenderer.MoveTag):
            CurrPosition=None

            ObjType,NeuronID,nodeId=self.activeRenderer.MoveTag    
            if not( ObjType in self.ariadne.Neurons[NeuronID].children):
                self.activeRenderer.MoveTag=[]
                return
            obj=self.ariadne.Neurons[NeuronID].children[ObjType]

            if ('l' in obj.flags): #locked obj
                return
                
            Point2Selected=False
            if ObjType=="synapse":
                tagIdx=obj.nodeId2tagIdx(nodeId)
                nodeIds=obj.tagIdx2nodeId(tagIdx)
                if nodeIds[2]==nodeId:
                    Point2Selected=True
                    
            if Point2Selected:
                #we are moving the 3rd node of a synapse
                DataSet=None
                self.cellPicker.Modified()
                self.cellPicker.Pick(x, y, 0,self.activeRenderer)
                CellID=self.cellPicker.GetCellId()
                if not CellID==-1:
                    DataSet=self.cellPicker.GetDataSet()
                    PickedPoint=self.cellPicker.GetPickPosition()     
                    PointID=self.cellPicker.GetPointId()
                
                if not DataSet==None:
                    tempArray=DataSet.GetPointData().GetArray("NeuronID")
                    if tempArray==None:
                        NeuronID=None
                    else:
                        NeuronID=float(np.round(tempArray.GetValue(PointID),3))

                    if not (NeuronID==None):
                        ObjType=DataSet.GetFieldData().GetAbstractArray("ObjType").GetValue(0)
                        if ObjType=="skeleton":
                            jneuron=NeuronID
                            CurrPosition,nodeIdpartner=self.ariadne.Neurons[jneuron].children["skeleton"].get_closest_point(PickedPoint)

            if CurrPosition==None:
                CurrPosition=self.activeRenderer.GetPoint(x,y)
            pointIdx=obj.nodeId2pointIdx(nodeId)
            if pointIdx<0:
                return
            
            actiontime,dt=self.ariadne.Timer.action()    
            obj.comments.set(nodeId,"time",actiontime)
            obj.comments.set(nodeId,"inMag",inMag)
            obj.comments.set(nodeId,"inVp",inVp)

            obj.data.GetPoints().SetPoint(pointIdx,CurrPosition)            
            obj.data.Modified()
            self.RenderWindow.Render()       
            return
                                          
        lastX,lastY = self.Iren.GetLastEventPosition()
        if  self.activeRenderer.Rotating and self.activeRenderer._EnableRotating:
            self.activeRenderer.Rotate(x, y, lastX, lastY)
            self.RenderWindow.Render()       
        elif  self.activeRenderer.Panning:
            centerX,centerY= self.activeRenderer.GetCenter()
            RefPoint = self.activeRenderer.PanningReference
            self.activeRenderer.Pan(x, y,RefPoint)
        elif  self.activeRenderer.Zooming:
            self.activeRenderer.Dolly(y,lastY)
            self.RenderWindow.Render()       
        elif self.activeRenderer.Contrast:
            self.activeRenderer.AdjustContrast(x,y,lastX,lastY)
        else:
            return

    def keyPressEvent(self, ev):
        syn_class_keys=[QtCore.Qt.Key_F1,QtCore.Qt.Key_F2,QtCore.Qt.Key_F3,\
            QtCore.Qt.Key_F4,QtCore.Qt.Key_F5,QtCore.Qt.Key_F6,QtCore.Qt.Key_F7,\
            QtCore.Qt.Key_F8,QtCore.Qt.Key_F9,QtCore.Qt.Key_F10,QtCore.Qt.Key_F11,\
            QtCore.Qt.Key_F12]
        syn_certainty_keys=[QtCore.Qt.Key_1,QtCore.Qt.Key_2,QtCore.Qt.Key_3,\
            QtCore.Qt.Key_4,QtCore.Qt.Key_5,QtCore.Qt.Key_6,QtCore.Qt.Key_7,\
            QtCore.Qt.Key_8,QtCore.Qt.Key_9,QtCore.Qt.Key_0]
        comment_shortcut_keys=[QtCore.Qt.Key_F1,QtCore.Qt.Key_F2,QtCore.Qt.Key_F3,\
            QtCore.Qt.Key_F4,QtCore.Qt.Key_F5]
        ctrl, shift = self.GetCtrlShift(ev)
        if ev.key() < 256:
            key = unicode(ev.text())
            if key.__len__()>1:
                key=key[0]
        else:
            key = chr(0)

        self.Iren.SetEventInformationFlipY(getattr(self,'__saveX'), getattr(self,'__saveY'),
                                           ctrl, shift, key, 0, None)
 
        x,y = self.Iren.GetEventPosition()       

        if not self.activeRenderer:
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)
        elif not (self.activeRenderer.Rotating or self.activeRenderer.Zooming or self.activeRenderer.Contrast or self.activeRenderer.Panning):
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)

        key=ev.key()     
        if key == QtCore.Qt.Key_S and ctrl and shift:
            self.ariadne.SaveSeparately()
        elif key == QtCore.Qt.Key_S and ctrl:
            self.ariadne.Save()
        elif key == QtCore.Qt.Key_Space:
            self.ariadne.SynchronizedZoom(0.0)
            self.Render()
        elif key == QtCore.Qt.Key_Plus:
            dollyFactor = 1.1
            self.ariadne.SynchronizedZoom(dollyFactor)
            self.Render()
        elif key == QtCore.Qt.Key_Minus:
            dollyFactor = 0.9
            self.ariadne.SynchronizedZoom(dollyFactor)
            self.Render()
        elif shift:
            if key == QtCore.Qt.Key_Up:
                self.activeRenderer.AdjustContrast(0,1.0,0,0)
            elif key == QtCore.Qt.Key_Down:
                self.activeRenderer.AdjustContrast(0,-1.0,0,0)
            elif key == QtCore.Qt.Key_Left:
                self.activeRenderer.AdjustContrast(-1.0,0,0,0)
            elif key == QtCore.Qt.Key_Right:
                self.activeRenderer.AdjustContrast(1.0,0,0,0)
            return
        elif key == QtCore.Qt.Key_Up:
            if self.activeRenderer._Orientation=="orthogonal":
                self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth")                
            elif self.activeRenderer._Orientation=="YX":
                self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Z")
            elif self.activeRenderer._Orientation=="YZ":
                if self.ariadne.radioButton_orthRef.isChecked():
                    self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                else:
                    self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"X")
            elif self.activeRenderer._Orientation=="ZX":
                if self.ariadne.radioButton_orthRef.isChecked():
                    self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                else:
                    self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Y")
            else:
                return
        elif key==QtCore.Qt.Key_F:
            if self.SynMode:
                if self.activeRenderer._Orientation=="arbitrary":
                    return
                if (not self.ariadne.job):
                    return
                currTask=self.ariadne.job.get_current_task()
                if currTask==None:
                    return
                if currTask._tasktype=="synapse_detection":
                    currTask.MoveAlongPath(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0])
            else:
                if self.activeRenderer._Orientation=="orthogonal":
                    self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth")                
                elif self.activeRenderer._Orientation=="YX":
                    self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Z")
                elif self.activeRenderer._Orientation=="YZ":
                    if self.ariadne.radioButton_orthRef.isChecked():
                        self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                    else:
                        self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"X")
                elif self.activeRenderer._Orientation=="ZX":
                    if self.ariadne.radioButton_orthRef.isChecked():
                        self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                    else:
                        self.activeRenderer.Move(1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Y")
                else:
                    return
                
        elif key == QtCore.Qt.Key_Down:
            if self.activeRenderer._Orientation=="orthogonal":
                self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth")                
            elif self.activeRenderer._Orientation=="YX":
                self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Z")
            elif self.activeRenderer._Orientation=="YZ":
                if self.ariadne.radioButton_orthRef.isChecked():
                    self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                else:
                    self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"X")
            elif self.activeRenderer._Orientation=="ZX":
                if self.ariadne.radioButton_orthRef.isChecked():
                    self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                else:
                    self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Y")
            else:
                return
        elif key==QtCore.Qt.Key_D:
            if self.SynMode:
                if self.activeRenderer._Orientation=="arbitrary":
                    return
                if (not self.ariadne.job):
                    return
                currTask=self.ariadne.job.get_current_task()
                if currTask==None:
                    return
                if currTask._tasktype=="synapse_detection":
                    currTask.MoveAlongPath(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0])
            else:
                if self.activeRenderer._Orientation=="orthogonal":
                    self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth")                
                elif self.activeRenderer._Orientation=="YX":
                    self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Z")
                elif self.activeRenderer._Orientation=="YZ":
                    if self.ariadne.radioButton_orthRef.isChecked():
                        self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                    else:
                        self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"X")
                elif self.activeRenderer._Orientation=="ZX":
                    if self.ariadne.radioButton_orthRef.isChecked():
                        self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"orth") 
                    else:
                        self.activeRenderer.Move(-1*self.ariadne.SpinBoxSpeed.value()*CubeLoader.Magnification[0],"Y")
                else:
                    return

        elif key == QtCore.Qt.Key_Right:
#            if self.SynMode:
#                currTask=self.ariadne.job.get_current_task()
#                if not (not currTask):
#                    if currTask._tasktype=="synapse_detection":
                self.ariadne.synapse_browser.search_synapse("forward")
        elif key == QtCore.Qt.Key_Left:
#            if self.SynMode:
#                currTask=self.ariadne.job.get_current_task()
#                if not (not currTask):
#                    if currTask._tasktype=="synapse_detection":
            self.ariadne.synapse_browser.search_synapse("backward")
        elif key == QtCore.Qt.Key_I:
            if self.TracingMode>0:
                self.ariadne.NewNeuron()   
        elif key == QtCore.Qt.Key_M:
            if self.ariadne.Job.isVisible() or self.ariadne.Job.isVisible():
                self.ariadne.Job.setVisible(0)
                self.ariadne.Settings.setVisible(0)
            else:
                self.ariadne.Job.setVisible(1)
                self.ariadne.Settings.setVisible(1)
        elif key == QtCore.Qt.Key_C:
            if usermode==1:
                return
            if self.ariadne.radioBtn_showconcomp.isChecked():
                self.ariadne.radioBtn_showall.setChecked(1)
            else:
                self.ariadne.radioBtn_showconcomp.setChecked(1)
#            self.ariadne.Capture()
        elif key == QtCore.Qt.Key_Q:
            return
        elif key == QtCore.Qt.Key_J:
            self.viewports["skeleton_viewport"].JumpToPoint(np.array([CubeLoader.Position[0],CubeLoader.Position[1],CubeLoader.Position[2]],dtype=np.float))
            self.Render()
#        elif key == QtCore.Qt.Key_E:
#            self.ariadne.close()
        elif key == QtCore.Qt.Key_H:
            if self.ariadne.radioBtn_showall.isChecked():
                self.ariadne.radioBtn_hideall.setChecked(1)
            else:
                self.ariadne.radioBtn_showall.setChecked(1)
        elif key == QtCore.Qt.Key_R:
            if self.ariadne.ckbx_hideReference.isChecked():
                self.ariadne.ckbx_hideReference.setChecked(0)
            else:
                self.ariadne.ckbx_hideReference.setChecked(1)
        elif key == QtCore.Qt.Key_X:
            self.ariadne.ckbx_HideFocalpoint.click()
        elif key == QtCore.Qt.Key_P:
            self.ariadne.SearchKeys("backward")
        elif key == QtCore.Qt.Key_N:
            self.ariadne.SearchKeys("forward")
        elif key == QtCore.Qt.Key_PageUp:
            if self.ariadne.job==None:
                return
            checkstate=self.ariadne.ckbx_skipDoneTasks.isChecked()
            self.ariadne.job.goto_next_task(checkstate)
        elif key == QtCore.Qt.Key_PageDown:
            if self.ariadne.job==None:
                return

            checkstate=self.ariadne.ckbx_skipDoneTasks.isChecked()
            self.ariadne.job.goto_previous_task(checkstate)
        elif key == QtCore.Qt.Key_S:
            self.GotoActiveObj(keepframe=True)
        elif key == QtCore.Qt.Key_Delete:
            self.DeleteActiveObj()
        elif key in comment_shortcut_keys and self.ariadne._comboBox_Shortcuts.isEnabled():
            if not (self.SynMode or (self.TracingMode>0) or self.TagMode):
                return
            index=comment_shortcut_keys.index(key)
            self.ariadne._comboBox_Shortcuts.setCurrentIndex(index)
            text=self.ariadne._comboBox_Shortcuts.itemText(index)
            if text==None:
                return
            text=unicode(text)
            self.ariadne.ChangeComment("SelObj","comment",text)
            self.ariadne.ShowComments()
        elif key in syn_class_keys:
            if not self.SynMode:
                return
            classNo=syn_class_keys.index(key)+1
            key="class{0}".format(classNo)   
            if self.ariadne.synapse_browser.classes.has_key(key):
                item=self.ariadne.synapse_browser.classes[key].item[0]
                self.ariadne.synapse_browser.imagelist.setCurrentItem(item)
        elif key in syn_certainty_keys:
            if not self.SynMode:
                return
            certainty=(syn_certainty_keys.index(key)+1)*10
            self.ariadne._CertaintyLevel.setValue(certainty)
        elif key==QtCore.Qt.Key_Return or key==QtCore.Qt.Key_Enter:
            if not self.SynMode:
                return
            self.ariadne.btn_Syn_assign.click()

#    def keyReleaseEvent(self, ev):
#        #no command yet


    def wheelEvent(self, ev):
        x,y = self.Iren.GetEventPosition()     

        if not self.activeRenderer:
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)
        elif not (self.activeRenderer.Rotating or self.activeRenderer.Zooming or self.activeRenderer.Contrast or self.activeRenderer.Panning):
            self.activeRenderer=self.Iren.FindPokedRenderer(x,y)

        if self.activeRenderer._Orientation=="orthogonal":
            self.activeRenderer.Move(-1*ev.delta()/120*CubeLoader.Magnification[0],"orth")                
        elif self.activeRenderer._Orientation=="YX":
            self.activeRenderer.Move(-1*ev.delta()/120*CubeLoader.Magnification[0],"Z")
        elif self.activeRenderer._Orientation=="YZ":
            if self.ariadne.radioButton_orthRef.isChecked():
                self.activeRenderer.Move(-1*ev.delta()/120*CubeLoader.Magnification[0],"orth") 
            else:
                self.activeRenderer.Move(-1*ev.delta()/120*CubeLoader.Magnification[0],"X")
        elif self.activeRenderer._Orientation=="ZX":
            if self.ariadne.radioButton_orthRef.isChecked():
                self.activeRenderer.Move(-1*ev.delta()/120*CubeLoader.Magnification[0],"orth") 
            else:
                self.activeRenderer.Move(ev.delta()/120*CubeLoader.Magnification[0],"Y")
        else:
            return

    def GetRenderWindow(self):
        return self.RenderWindow

    def Render(self):
        self.RenderWindow.Render()
#        self.update()

    def Render_Intersect(self):
        for key,intersection in self.ariadne.intersections.iteritems():
            intersection.Intersect()
        self.Render()                    

class CenterGlyph():
    _SourceStyle='Cross'
    _Mode='default'
    def __init__(self,parentviewport,SourceStyle=None):
        self.parent=parentviewport

        if SourceStyle==None:
            SourceStyle=self._SourceStyle
        else:
            self._SourceStyle=SourceStyle
        if SourceStyle=='Cross':
            self.data=vtk.vtkPolyData()
            self.data.Allocate()
            Points=vtk.vtkPoints()
            Points.InsertNextPoint(-20.0,0.0,0.0)
            Points.InsertNextPoint(20.0,0.0,0.0)
            Points.InsertNextPoint(0.0,-20.0,0.0)
            Points.InsertNextPoint(0.0,20.0,0.0)
            self.data.SetPoints(Points)
            for i in range(2):
                edge=vtk.vtkIdList()
                edge.InsertNextId(0+i*2)
                edge.InsertNextId(1+i*2)
                self.data.InsertNextCell(vtk.VTK_POLY_LINE,edge)

        self.mapper= vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection(self.data.GetProducerPort()) 

        self.actor =vtk.vtkActor()
        self.actor.SetMapper(self.mapper)       
        self.actor.GetProperty().SetDiffuseColor(1.0,0.0,0.0)
        self.actor.PickableOff()
        self.parent.AddActor(self.actor)


    def getPoint(self,dx,dy):      
        viewportPos=self.parent.GetViewport()
        x=vtk.mutable((viewportPos[0]+viewportPos[2])/2.0)
        y=vtk.mutable((viewportPos[1]+viewportPos[3])/2.0)
        self.parent.NormalizedDisplayToDisplay(x,y)
        x=vtk.vtkMath.Floor(x.get()+dx)+0.5
        y=vtk.vtkMath.Floor(y.get()+dy)+0.5
        
        self.parent.SetDisplayPoint(x,y,0.1)
        self.parent.DisplayToWorld()
        CurrPos0,CurrPos1,CurrPos2,CurrPos3 = self.parent.GetWorldPoint()

        if CurrPos3 != 0.0:
            CurrPos0 = CurrPos0/CurrPos3
            CurrPos1 = CurrPos1/CurrPos3
            CurrPos2 = CurrPos2/CurrPos3

        return CurrPos0, CurrPos1, CurrPos2
        
    def updatePosition(self):      
        self.data.GetPoints().SetPoint(0,self.getPoint(-9,0))
        self.data.GetPoints().SetPoint(1,self.getPoint(9,0))
        self.data.GetPoints().SetPoint(2,self.getPoint(0,-9))
        self.data.GetPoints().SetPoint(3,self.getPoint(0,9))
        self.data.Modified()       

#class ScaleBar():
#    _Length=5.0;
#    _Position=[0,0,0]
#    def __init__(self,parentviewport):
#        self.parent=parentviewport
#
#        self.data=vtk.vtkPolyData()
#        self.data.Allocate()
#        Points=vtk.vtkPoints()
#
#        origin=np.array(self._Position,dtype=np.float)
#        length=self._Length;
#        Points.InsertNextPoint(origin)
#        Points.InsertNextPoint(origin+
#            np.array([length,0.0,0.0],dtype=np.float))
#        self.data.SetPoints(Points)
#
#        edge=vtk.vtkIdList()
#        edge.InsertNextId(0)
#        edge.InsertNextId(1)
#        self.data.InsertNextCell(vtk.VTK_POLY_LINE,edge)
#
#        self.mapper= vtk.vtkDataSetMapper()
#        self.mapper.SetInputConnection(self.data.GetProducerPort()) 
#
#        self.actor =vtk.vtkActor()
#        self.actor.SetMapper(self.mapper)       
#        self.actor.GetProperty().SetDiffuseColor(0.0,0.0,0.0)
#        self.actor.GetProperty().SetLineWidth(5)
#        self.actor.PickableOff()
#        self.parent.AddActor(self.actor)
#        
#    def update(self,length=None,origin=None):
#        if length==None:
#            length=self._Length;
#        else:
#            self._Length=length
#        if origin==None:
#            origin=np.array(self._Position,dtype=np.float)
#        else:
#            self._Position=origin
#        
#        cDir=np.array(self.parent.Camera.GetDirectionOfProjection(),dtype=np.float)
#        vDir=np.array(self.parent.Camera.GetViewUp(),dtype=np.float)
#        direction=np.cross(cDir,vDir);
#        
#        Points=self.data.GetPoints()
#        Points.SetPoint(0,origin);
#        Points.SetPoint(1,origin+length*direction);
#        Points.Modified()       
    

class Comments():
    parent=None
    def __init__(self,parent):
        self._Comments=OrderedDict()
        self.parent=parent

    def ids(self):
        return [key for key in self._Comments.keys() if (self._Comments[key].__class__.__name__=='OrderedDict' or self._Comments[key].__class__.__name__=='dict')]
        
    def delete(self,Id,key=None):
        if Id==self.parent:  
            if not key:
                return
            try:
                #remove attribute
                del self._Comments[key]
            except:
                1
            return            

        Id=unicode(Id)
        if not key:
            try:
                #remove id
                del self._Comments[Id] 
            except:
                1
            return
        try:
            #remove attribute
            del self._Comments[Id][key]
        except:
            1
        return
        
    def set(self,Id=None,key=None,value=None):
        if Id==None:
            return
        if key==None:
            return
        if Id==self.parent:  
            if value==None:
                if key.__class__.__name__=='OrderedDict' or \
                    key.__class__.__name__=='dict':
                        self._Comments.update(key)
                        return
                try:
                    #remove attribute
                    del self._Comments[key]
                except:
                    1
                return 
            self._Comments[key]=value
            return
        Id=unicode(Id)
        if value==None:
            if key.__class__.__name__=='OrderedDict' or \
                key.__class__.__name__=='dict':
                    self._Comments[Id]=key
                    return
            try:
                #remove attribute
                del self._Comments[Id][key]
            except:
                1
            return
        if not self._Comments.has_key(Id):
            self._Comments[Id]={}        
        self._Comments[Id][key]=value

    def get(self,Id=None,key=None):
        if Id==None or (Id=='all'):
            if key==None:
                return None
            comments=OrderedDict()
            for iid in self._Comments:
                if key in self._Comments[iid]:
                    comments[iid]=OrderedDict()
                    comments[iid][key]=self._Comments[iid][key]
            return comments
        if Id==self.parent:
            if key==None:
                try: 
                    return self._Comments
                except:
                    return None
            try:
                return self._Comments[key]
            except:
                return None
            return
            
        Id=unicode(Id)
        if key==None:
            try: 
                return self._Comments[Id]
            except:
                return None
        try:
            return self._Comments[Id][key]
        except:
            return None

class objs():
    #object flags:
    #'d': demand-driven object
    #'x': exclude object when saving
    #'l': locked object (cannot edit, add, remove nodes, comments)
    #'r': reference object flag
    maxNofColors=[2000]
    ariadne=None
    VisEngine_started=[False]
    visible=1;
    viewports=[];
    activeInstances=[];
    instancecount=[0]
    deleted=False
    filename='';
    
    def __init__(self):
        self.objtype=""
        self.name=""    
        self.reset()
        self.instancecount[0]+=1


    def reset(self):
        self.NeuronID=None
        self.data=None
        self.item=None
        self.visible=1;
        self.colorIdx=-1
        self.children=OrderedDict()
        self.comments=Comments(self)
        self.flags='';
        
    def inherit_method(self,method):
        return lambda *args, **kw:  method(self, *args, **kw)

    def addflags(self,flags):
        for key,obj in self.children.iteritems():
            obj.addflags(flags)
        self.flags="".join(set(self.flags+flags)) #creates a string with unique flags        

    def isvalid_nodeId(self,nodeId):
        if nodeId==None:
            return False
        if not nodeId.__class__.__name__=='int':
            nodeId=np.int(nodeId)
        if not hasattr(self,'validData'):
            return False
        if not self.validData:
            return False
        self.validData.Update()
        PointData=self.validData.GetOutput().GetPointData()
        if not PointData:
            return False
        NodeID=PointData.GetArray("NodeID")
        if not NodeID:
            return False
        found=NodeID.LookupValue(nodeId)
        if found==-1:
            return False
        return True
        
    def set_new_neuronId(self,neuronId):
        oldNeuronID=self.NeuronID
        self.NeuronID=neuronId
        if hasattr(self,'data'):
            if not (not self.data):    
                NNodes=self.data.GetNumberOfPoints()
                neuronID=self.NeuronID*np.ones([NNodes,1],dtype=np.float32)
                NeuronID=self.data.GetPointData().GetArray("NeuronID")
                NeuronID.DeepCopy(numpy_to_vtk(neuronID, deep=1, array_type=vtk.VTK_FLOAT))
                NeuronID.Modified()

        if hasattr(self,'item'):
            if hasattr(self.item,'neuronId'):
                self.item.neuronId=neuronId
            for irow in range(self.item.rowCount()):
                child=self.item.child(irow)
                if not child:
                    continue
                if hasattr(child,'neuronId'):
                    child.neuronId=neuronId
            
            self.updateItem()

        for key, child in self.children.iteritems():
            child.set_new_neuronId(neuronId)
        if not hasattr(self.ariadne,'Neurons'):
            return            
        if oldNeuronID in self.ariadne.Neurons:
            if (self == self.ariadne.Neurons[oldNeuronID]) and (not neuronId in self.ariadne.Neurons):
                self.ariadne.Neurons[neuronId]=self
                self.ariadne.Neurons.pop(oldNeuronID, None)
        
    def get_prev_obj(self,start_Idx=None,warparound=True):
        NObjs=self.data.GetNumberOfCells()
        if start_Idx==None:
            newIdx=NObjs-1
        else:
            newIdx=start_Idx-1
        if newIdx<0:
            if warparound:
                newIdx=NObjs-1 #wrap around
            else:
                newIdx=None
#        print newIdx
        return newIdx

    def get_next_obj(self,start_Idx=None,warparound=True):
        NObjs=self.data.GetNumberOfCells()
        if start_Idx==None:
            newIdx=0
        else:
            newIdx=start_Idx+1
        if newIdx>(NObjs-1):
            if warparound:
                newIdx=0 #wrap around
            else:
                newIdx=None
#        print newIdx
        return newIdx

    def search_child(self,objtype,start_nodeId=None,direction=None,warparound=True):
        if not objtype in self.children:
            return None
        obj=self.children[objtype]
        start_idx=obj.nodeId2tagIdx(start_nodeId)
        if direction=="forward":
            return obj.get_next_obj(start_idx,warparound)
        elif direction=="backward":
            return obj.get_prev_obj(start_idx,warparound)
        return start_idx
                
    def search_comment(self,keys,values,start_id=None,direction="forward",search_children=False):
        #!!!NOTE: this might not work for nested child-structures (eg. children of children)
        allIds=self.comments.ids()
#        print "allIds: " + "{0}".format(allIds)
        
        if direction=="backward":
            allIds.reverse()
        if not  (start_id=='None' or start_id==None or start_id<0):   
            start_id=unicode(start_id)
            if start_id in allIds:
                startIdx=allIds.index(start_id)
            else:
                startIdx=-1
            #standard forward search
            for idx in [idx for idx in range(startIdx+1,allIds.__len__())]:
                found=True
                for key,value in zip(keys,values):
                    result=self.comments.get(allIds[idx],key)
                    if result==None:
                        found=False
                        break
                    if result.__class__.__name__=='str' or result.__class__.__name__=='unicode':
                        if not (value in result):
                            found=False
                            break
                    else:
                        if not (result==value):
                            found=False
                            break
                if found:
                    return (self.objtype,self.NeuronID,int(allIds[idx]))
        else:
            for iid in allIds:
                found=True
                for key,value in zip(keys,values):
                    result=self.comments.get(iid,key)
                    if result==None:
                        found=False
                        break
                    if result.__class__.__name__=='str' or result.__class__.__name__=='unicode':
                        if not (value in result):
                            found=False
                            break
                    else:
                        if not (result==value):
                            found=False
                            break
                if found:
                    return (self.objtype,self.NeuronID,int(iid))

        if search_children:
            for key, child in self.children.iteritems():
                found=child.search_comment(keys,values,None,direction,search_children)
                if not (not found):
                    return found

        return None

    def extract_metadata(self,UniqueLookupTables):
        metaData=vtk.vtkFieldData()
        type=vtk.vtkStringArray()
        type.SetName("type")
        type.SetNumberOfValues(1)
        type.SetValue(0,self.objtype)
        metaData.AddArray(type)
        text=vtk.vtkStringArray()
        text.SetName("text")
        text.SetNumberOfValues(1)
        text.SetValue(0,unicode(self.item.text()))
        metaData.AddArray(text)
        id=vtk.vtkStringArray()
        id.SetName("id")
        id.SetNumberOfValues(1)
        id.SetValue(0,unicode(self.NeuronID))
        metaData.AddArray(id)
        
        #unfortunately XMLWriter does not yet support vtkVariant data type.
        #Therefore we have to save the comment tuples (id,key,value) as strings
        comments=vtk.vtkStringArray()
        comments.SetName("comments")
        comments.SetNumberOfComponents(3)
        comments.SetComponentName(0,"id")
        comments.SetComponentName(1,"key")
        comments.SetComponentName(2,"value")
        for id in self.comments.ids():
            for key in self.comments.get(id).keys():
                value=self.comments.get(id,key)
                if value==None:
                    continue
                dtype=value.__class__.__name__    
                if dtype=='unicode':
                    dtype='string'
                value=dtype[0]+unicode(value)
                if dtype=='ndarray' or dtype=='tuple' or dtype=='list':
                    value=value.replace(' ',',')
                    value=value.replace('\n',',')
                    newvalue=value
                    value=''
                    while not newvalue==value:
                        value=newvalue
                        newvalue=value.replace(',,',',')
                    value=value.replace('[,','[')
                    value=value.replace(',]',']')
                comments.InsertNextValue(unicode(id))
                comments.InsertNextValue(unicode(key))
                comments.InsertNextValue(value)
        if comments.GetNumberOfValues()>0:
            metaData.AddArray(comments)
        else:
            metaData.RemoveArray("comments")
            
        #Process color
        if self.LUT not in UniqueLookupTables:
            UniqueLookupTables.append(self.LUT)
        tableIdx=UniqueLookupTables.index(self.LUT)
        
        coloridx=vtk.vtkIntArray()
        coloridx.SetName("coloridx")
        coloridx.SetNumberOfValues(2)
        coloridx.SetNumberOfComponents(2)        
        coloridx.SetValue(0,tableIdx)
        coloridx.SetValue(1,self.colorIdx)
        metaData.AddArray(coloridx)
        
        return metaData


    def get_cleanedup_data(self):
        if hasattr(self,'validData'):
            GeometryFilter=vtk.vtkGeometryFilter()
            GeometryFilter.SetInputConnection(self.validData.GetOutputPort())
            GeometryFilter.Update()
            tempData=vtk.vtkPolyData()
            tempData.DeepCopy(GeometryFilter.GetOutput())
        else:
            tempData=self.data

        tempData.GetPointData().SetActivePedigreeIds(None)
        tempData.GetPointData().RemoveArray("PointColor")
        tempData.GetPointData().RemoveArray("NeuronID")
        tempData.GetPointData().RemoveArray("DeletedNodes")
        tempData.GetPointData().RemoveArray("VisibleNodes")
        tempData.GetPointData().RemoveArray("Radius")
        tempData.GetPointData().RemoveArray("vtkOriginalPointIds")
        tempData.GetCellData().RemoveArray("vtkOriginalCellIds")
        tempData.Modified()
        tempData.Update()
        return tempData

    def setup_color(self,color):
        NColors=self.LUT.GetNumberOfTableValues()
        
        if NColors+1>self.maxNofColors[0]:
            self.maxNofColors[0]*=2
            newLUT=vtk.vtkLookupTable()
            newLUT.Allocate(self.maxNofColors[0])
            for icolor in range(NColors):
                newLUT.SetTableValue(icolor,self.LUT.GetTableValue(icolor))
            self.LUT.DeepCopy(newLUT)

        self.LUT.SetNumberOfTableValues(NColors+1)
        self.LUT.SetTableValue(NColors,color)
        self.LUT.SetTableRange(0,NColors)       
        self.LUT.Modified()
        self.colorIdx=NColors
#        print "setup_color", self.NeuronID, self.objtype, self.colorIdx
#        oldcolor=self.LUT.GetTableValue(self.colorIdx)
#        print "old color:", oldcolor, "new color", color
        return self.colorIdx
                

    def change_color(self,color=None):
        if (color==None):
            return      
#        print self.NeuronID, self.objtype, self.colorIdx, color
#        oldcolor=self.LUT.GetTableValue(self.colorIdx)
        self.LUT.SetTableValue(self.colorIdx,color)
        self.LUT.Modified()
        for key, child in self.children.iteritems():            
            child.change_color(color)
        self.item.setData(QtGui.QColor(color[0]*255.0,color[1]*255.0,color[2]*255.0),QtCore.Qt.BackgroundRole)
#        print "change_color", self.NeuronID, self.objtype, self.colorIdx
#        print "old color:", oldcolor, "new color", color
    
    def add_toactive(self):
        if not hasattr(self,'allDataInput'):
            return
        if hasattr(self,'activeData'):
            if self in self.activeInstances:
                return
            if hasattr(self,'allData'):
#                print "Remove ", self.objtype, " of neuron ", self.NeuronID, " from all data."
                self.allData.RemoveInputConnection(0,self.allDataInput.GetOutputPort())
                self.allData.Modified()
                if self.allData.GetNumberOfInputConnections(0)==0:
                    for viewport in self.viewports:
                        self.hide_actors(viewport,'actor')
#            print "Set ", self.objtype, " of neuron ", self.NeuronID, " as active."
            self.activeData.AddInputConnection(0,self.allDataInput.GetOutputPort())
            self.activeData.Modified()
            self.activeInstances.append(self);
            for viewport in self.viewports:
                self.show_actors(viewport,'activeactor')
        
    def select(self):
        self.select_item()

    def select_item(self):
#        print "select item"
        if not hasattr(self,'item'):
            return
        if not (not self.item):
            parent=self.item.parent()
            if not (not parent):
                self.ariadne.ObjectBrowser.setExpanded(parent.index(),1)            
        if not (not self.item):
            self.ariadne.ObjectBrowser.setCurrentIndex(self.item.index())
            
    def new_singleactive(self,newactive):
        if newactive in self.activeInstances:
            return;            
        while self.activeInstances.__len__()>0:
            self.activeInstances[0].remove_fromactive()
        newactive.add_toactive()
    
    def remove_fromactive(self):
        if not hasattr(self,'allDataInput'):
            return
        if not hasattr(self,'activeInstances'):
            return
        if not (self in self.activeInstances):
            return
        if not hasattr(self,'activeData'):
            return
#        print "Remove ", self.objtype, " of neuron ", self.NeuronID, " from active."
        self.activeData.RemoveInputConnection(0,self.allDataInput.GetOutputPort())
        if self.activeData.GetNumberOfInputConnections(0)==0:
            for viewport in self.viewports:
                self.hide_actors(viewport,'activeactor')
        self.activeInstances.remove(self)
        if hasattr(self,'allData'):
#            print "Add ", self.objtype, " of neuron ", self.NeuronID, " to all data."
            self.allData.AddInputConnection(0,self.allDataInput.GetOutputPort())
            self.allData.Modified()
            for viewport in self.viewports:
                self.show_actors(viewport,'actor')
        
    def unselect(self):
        self.unselect_item()
        
    def unselect_item(self):
        if not hasattr(self,'item'):
            return
        if not (not self.item):
            parent= self.item.parent()
            if not (not parent):
                self.ariadne.ObjectBrowser.setExpanded(parent.index(),0)
    
    def set_alpha(self,alpha):
        for key, child in self.children.iteritems():    
            child.set_alpha(alpha)
        color=np.array(self.LUT.GetTableValue(self.colorIdx))
        newcolor=np.array([color[0]*1.0,color[1]*1.0,color[2]*1.0,alpha*1.0,])
        self.change_color(newcolor)

    @classmethod
    def add_toviewport(self,viewport):
        if viewport in self.viewports:
            return #the actor has already been added to the viewport

#        print "add ", self.objtype, " to viewport id", viewport._VpId

        ClippingClasses=['soma','region']
        for attr in ["activeactor","actor"]:
            if hasattr(self,attr):  
                actor=getattr(self,attr)
                if not (not actor):
                    if ((self.objtype in ClippingClasses) and not viewport._ClipHulls) or \
                        (not (self.objtype in ClippingClasses)):

                        if attr=='activeactor':
                            dataattr='activeData';
                            if hasattr(self,dataattr):  
                                data=getattr(self,dataattr)
                            else:
                                data=None
                        elif attr=='actor':
                            dataattr='allData';
                            if hasattr(self,dataattr):  
                                data=getattr(self,dataattr)
                            else:
                                data=None
                        else:
                            continue;
                        if not data:
                            continue;
                        if data.GetNumberOfInputConnections(0)==0:
                            continue                        
                        data.Update()

                        if actor.GetClassName()=='vtkOpenGLActor':
                            if not viewport.HasViewProp(actor):
                                viewport.AddActor(actor)                     
                        elif actor.GetClassName()=='vtkActorCollection':
                            for iactor in range(actor.GetNumberOfItems()):
                                childactor=actor.GetItemAsObject(iactor)
                                if not viewport.HasViewProp(childactor):
                                    viewport.AddActor(childactor)

        #load viewport plane, if any (orthogonal to camera direction)
        planeKeys = [viewport._ViewportPlane] + viewport._LinkedPlanes
        for attr in ["clippedactor"]:
            if  hasattr(self,attr) and (self.objtype in ClippingClasses):  
                if attr=='clippedactor':
                    dataattr='allData';
                    if hasattr(self,dataattr):  
                        data=getattr(self,dataattr)
                    else:
                        data=None;
                else:
                    continue;
                if not data:
                    continue;
                if data.GetNumberOfInputConnections(0)==0:
                    continue
                data.Update()
                
                for planeKey in planeKeys:
                    if not planeKey:
                        continue
                    if not self.ariadne.planeROIs.has_key(planeKey):
                        continue
                    if self.ariadne.planeROIs[planeKey].ClippingPlane.__len__()<1:
                        continue

                    FirstClippingPlane=self.ariadne.planeROIs[planeKey].ClippingPlane[0]
                    if not (FirstClippingPlane in self.ClippingPlanes):
                        convert2PolyData=vtk.vtkGeometryFilter()
                        convert2PolyData.SetInputConnection(data.GetOutputPort())
                    
                        passArray=vtk.vtkPassArrays()
                        passArray.AddArray(vtk.VTK_POINTS, "PointColor");
                        passArray.AddArray(vtk.VTK_POINTS, "NeuronID");
                        passArray.AddArray(vtk.VTK_POINTS, "Labels");
                        passArray.SetInputConnection(convert2PolyData.GetOutputPort())
                    
                        planeCollection=vtk.vtkPlaneCollection()
                        for ClippingPlane in self.ariadne.planeROIs[planeKey].ClippingPlane:
                            planeCollection.AddItem(ClippingPlane)
        
                        Clipper = vtk.vtkClipClosedSurface()
                        Clipper.SetTolerance(1e-1)
                        Clipper.SetGenerateFaces(1)
                        Clipper.PassPointDataOn()
                        Clipper.SetClippingPlanes(planeCollection)
                        Clipper.SetInputConnection(passArray.GetOutputPort())
                        
                        PlaneClipperMapper= vtk.vtkDataSetMapper()
                        PlaneClipperMapper.SelectColorArray("PointColor"); 
                        PlaneClipperMapper.SetLookupTable(self.LUT);  
                        PlaneClipperMapper.SetUseLookupTableScalarRange(1);
                        PlaneClipperMapper.SetInputConnection(Clipper.GetOutputPort())
                        PlaneClipperActor = vtk.vtkActor();
                        PlaneClipperActor.SetMapper(PlaneClipperMapper);
                        self.clippedmapper.AddItem(PlaneClipperMapper)
                        self.clippedactor.AddItem(PlaneClipperActor)
                        self.ClippingPlanes.append(FirstClippingPlane)
                        self.ClippingActors.append(PlaneClipperActor)
    
                    if viewport._ClipHulls:
                        ind=self.ClippingPlanes.index(FirstClippingPlane)
                        PlaneClipperActor=self.ClippingActors[ind]
                        if not viewport.HasViewProp(PlaneClipperActor):
                            viewport.AddActor(PlaneClipperActor)

        for attr in ["labelactor"]:
            if hasattr(self,attr):  
                if attr=='labelactor':
                    dataattr='allData';
                    if hasattr(self,dataattr):  
                        data=getattr(self,dataattr)
                else:
                    continue;
                if not data:
                    continue;
                if data.GetNumberOfInputConnections(0)==0:
                    continue
                if self.allLabels.GetNumberOfInputConnections(0)==0:
                    continue
                VisibleLabels=vtk.vtkSelectVisiblePoints()
                VisibleLabels.viewport=viewport
                if viewport._Visible==1:
                    VisibleLabels.SetInputConnection(self.allLabels.GetOutputPort())
                    VisibleLabels.SetRenderer(viewport)
                else:
                    dummyLabel=vtk.vtkPolyData()
                    VisibleLabels.SetRenderer(None)
                    VisibleLabels.SetInputConnection(dummyLabel.GetProducerPort())
#                    print viewport._Orientation, " dummy placed"
                VisibleLabels.SelectionWindowOn()
                VisibleLabels.SetTolerance(0)
                size=viewport.GetSize()
                origin=viewport.GetOrigin()
                VisibleLabels.SetSelection(origin[0],origin[0]+size[0],origin[1],origin[1]+size[1])
                self.VisibleLabels.AddItem(VisibleLabels)

                labelmapper=vtk.vtkLabeledDataMapper()
                labelmapper.SetInputConnection(VisibleLabels.GetOutputPort())
                labelmapper.SetFieldDataName("Labels")
                labelmapper.SetLabelModeToLabelFieldData()
                labelmapper.SetFieldDataArray(0)
                labelmapper.GetLabelTextProperty().SetFontSize(15)
                labelmapper.GetLabelTextProperty().SetColor(0,0,0)
    
                self.labelmapper.AddItem(labelmapper)
                labelactor=vtk.vtkActor2D()
                labelactor.SetMapper(labelmapper)
                self.labelactor.AddItem(labelactor)
                if (viewport._Visible==1):
                    viewport.AddActor2D(labelactor)

        self.viewports.append(viewport)
    
    @classmethod
    def show_actors(self,viewport,whichactors=['activeactor','actor','clippedactor','labelactor']):
        if not (viewport in self.viewports):
            return
        if not (whichactors.__class__.__name__=='list'):
            whichactors=[whichactors]

#        print "show actors (", whichactors, ") for ", self.objtype

        ClippingClasses=['soma','region']
        for attr in ["activeactor","actor"]:
            if (attr in whichactors) and hasattr(self,attr):  
                if attr=='activeactor':
                    dataattr='activeData';
                    if hasattr(self,dataattr):  
                        data=getattr(self,dataattr)
                    else:
                        data=None
                elif attr=='actor':
                    dataattr='allData';
                    if hasattr(self,dataattr):  
                        data=getattr(self,dataattr)
                    else:
                        data=None
                else:
                    continue;
                if not data:
                    continue;
                if data.GetNumberOfInputConnections(0)==0:
                    continue
#                data.Update() #do we need this? excluded on 20150826

                actor=getattr(self,attr)
                if not (not actor):
                    if ((self.objtype in ClippingClasses) and not viewport._ClipHulls) or \
                        (not (self.objtype in ClippingClasses)):
                        if actor.GetClassName()=='vtkOpenGLActor':
                            if not viewport.HasViewProp(actor):
                                viewport.AddActor(actor)                     
                        elif actor.GetClassName()=='vtkActorCollection':
                            changed=0
                            for iactor in range(actor.GetNumberOfItems()):
                                childactor=actor.GetItemAsObject(iactor)
                                if not viewport.HasViewProp(childactor):
                                    viewport.AddActor(childactor)
                                    changed=1
                            if changed:
                                viewport.Modified()

        #load viewport plane, if any (orthogonal to camera direction)
        planeKeys = [viewport._ViewportPlane] + viewport._LinkedPlanes
        for attr in ["clippedactor"]:
            if (attr in whichactors) and hasattr(self,attr) and (self.objtype in ClippingClasses):  
                for planeKey in planeKeys:
                    if not planeKey:
                        continue
                    if not self.ariadne.planeROIs.has_key(planeKey):
                        continue
                    if self.ariadne.planeROIs[planeKey].ClippingPlane.__len__()<1:
                        continue
                    FirstClippingPlane=self.ariadne.planeROIs[planeKey].ClippingPlane[0]
                    if (FirstClippingPlane in self.ClippingPlanes):
                        if viewport._ClipHulls:
                            ind=self.ClippingPlanes.index(FirstClippingPlane)
                            PlaneClipperActor=self.ClippingActors[ind]
                            if not viewport.HasViewProp(PlaneClipperActor):
                                viewport.AddActor(PlaneClipperActor)
                                viewport.Modified()

#        if hasattr(self,"labelactor"):
#            for ilabel in range(self.VisibleLabels.GetNumberOfItems()):
#                VisibleLabel=self.VisibleLabels.GetItemAsObject(ilabel)
#                VisibleLabel.SetInputConnection(self.allLabels.GetOutputPort())
                
        for attr in ["labelactor"]:
            if not (viewport._Visible==1):
                continue
            if not ((attr in whichactors) and hasattr(self,attr)):
                continue
            if self.allLabels.GetNumberOfInputConnections(0)==0:
                continue
            change=False
            for ilabel in range(self.VisibleLabels.GetNumberOfItems()):
                VisibleLabel=self.VisibleLabels.GetItemAsObject(ilabel)
                if VisibleLabel.viewport==viewport:
                    VisibleLabel.SetRenderer(viewport)
                    VisibleLabel.SetInputConnection(self.allLabels.GetOutputPort())
#                    print viewport._Orientation, " dummy removed"
                    VisibleLabel.Modified()
                    change=True
            if change:
                self.VisibleLabels.Modified()
            actor=getattr(self,attr)
            if (not actor):
                continue
            if actor.GetClassName()=='vtkActor2D':
                if not viewport.HasViewProp(actor):
                    viewport.AddActor2D(actor)
                    viewport.Modified()
            elif actor.GetClassName()=='vtkActor2DCollection':
                changed=0
                for iitem in range(actor.GetNumberOfItems()):
                    labelactor=actor.GetItemAsObject(iitem);
                    if not viewport.HasViewProp(labelactor):
                        viewport.AddActor2D(labelactor)
                        changed=1
                if changed:
                    viewport.Modified()         

    @classmethod
    def hide_actors(self,viewport,whichactors=['activeactor','actor','clippedactor','labelactor']):
        if not (viewport in self.viewports):
            return
        if not (whichactors.__class__.__name__=='list'):
            whichactors=[whichactors]

#        print "hide actors (", whichactors, ") for ", self.objtype

        for attr in ["activeactor","actor","clippedactor"]:
            if (attr in whichactors) and hasattr(self,attr):  
                actor=getattr(self,attr)
                if not (not actor):
                    if actor.GetClassName()=='vtkOpenGLActor':
                        if viewport.HasViewProp(actor):
                            viewport.RemoveViewProp(actor)
                            viewport.Modified()
                    elif actor.GetClassName()=='vtkActorCollection':
                        changed=0
                        for iitem in range(actor.GetNumberOfItems()):
                            childactor=actor.GetItemAsObject(iitem);                        
                            if viewport.HasViewProp(childactor):
                                viewport.RemoveViewProp(childactor)
                                changed=1
                        if changed:
                            viewport.Modified()

        if ('labelactor' in whichactors) and hasattr(self,"labelactor"):
            dummyLabel=vtk.vtkPolyData()
            change=False;        
            for ilabel in range(self.VisibleLabels.GetNumberOfItems()):
                VisibleLabel=self.VisibleLabels.GetItemAsObject(ilabel)
                if VisibleLabel.viewport==viewport:
                    VisibleLabel.RemoveInputConnection(0,self.allLabels.GetOutputPort())
                    VisibleLabel.SetRenderer(None)
                    VisibleLabel.SetInputConnection(dummyLabel.GetProducerPort())
#                    print viewport._Orientation, " dummy placed"
                    VisibleLabel.Modified()
                    change=True;
            if change:
                self.VisibleLabels.Modified()
            if not (not self.labelactor):
                if self.labelactor.GetClassName()=='vtkActor2D':
                    viewport.RemoveViewProp(self.labelactor)
                    viewport.Modified()
                elif self.labelactor.GetClassName()=='vtkActor2DCollection':
                    for iitem in range(self.labelactor.GetNumberOfItems()):
                        labelactor=self.labelactor.GetItemAsObject(iitem);
                        viewport.RemoveViewProp(labelactor)                        
                    viewport.Modified()
            viewport.Modified()
#            for ilabel in range(self.VisibleLabels.GetNumberOfItems()):
#                VisibleLabel=self.VisibleLabels.GetItemAsObject(ilabel)
#                VisibleLabel.RemoveInputConnection(0,self.allLabels.GetOutputPort())
#                VisibleLabel.Modified()

    @classmethod
    def remove_fromviewport(self,viewport):
        if not viewport in self.viewports:
            return
        self.hide_actors(viewport,['activeactor','actor','clippedactor','labelactor'])

        if hasattr(self,"labelactor"):
            if not (not self.labelactor):
                if self.labelactor.GetClassName()=='vtkActor2D':
                    labelmapper=self.labelactor.GetMapper()                    
                    VisibleLabel=labelmapper.GetInputConnection(0,0).GetProducer()
                    if VisibleLabel.viewport==viewport:
                        self.VisibleLabels.RemoveItem(VisibleLabel)
                        del self.labelactor
                        del self.labelmapper
                        self.labelactor=None
                        self.labelmapper=None
                elif self.labelactor.GetClassName()=='vtkActor2DCollection':
                    for iitem in range(self.labelactor.GetNumberOfItems()):
                        labelactor=self.labelactor.GetItemAsObject(iitem);
                        if not labelactor:
                            continue
                        labelmapper=labelactor.GetMapper()
                        VisibleLabel=labelmapper.GetInputConnection(0,0).GetProducer()
                        if not (VisibleLabel.viewport==viewport):
                            continue
                        self.VisibleLabels.RemoveItem(VisibleLabel)
                        self.labelmapper.RemoveItem(labelmapper)
                        self.labelactor.RemoveItem(labelactor)
                if self.allLabels.GetNumberOfInputConnections(0)==0:
                    for iviewport in self.viewports:
                        self.hide_actors(iviewport,['labelactor'])
        self.viewports.remove(viewport)

    @classmethod
    def start_VisEngine(self,ariadne=None):
        if ariadne==None:
            ariadne=self.ariadne
        else:
            if not self.ariadne==ariadne:
                self.ariadne=ariadne;
        if not ariadne:
            return
        if not self.VisEngine_started[0]:            
            anynonemptydata=0
            if hasattr(self,'activeData'):
                if self.activeData.GetNumberOfInputConnections(0)>0:
                    anynonemptydata=1;
            if hasattr(self,'allData'):
                if self.allData.GetNumberOfInputConnections(0)>0:
                    anynonemptydata=1;
            if not anynonemptydata:
                return
            for key,iviewport in ariadne.QRWin.viewports.iteritems():
                self.add_toviewport(iviewport)
            self.VisEngine_started[0]=True

    def set_visibility(self,vis):
        for key, child in self.children.iteritems():            
            child.set_visibility(vis)

        if 'r' in self.flags:
#            print "neuron " , self.NeuronID, "has the reference flag."
            if not (not self.ariadne):
                if self.ariadne.ckbx_hideReference.isChecked():
#                    print "hide reference is checked."
                    vis=0                   

        if not (self.visible==vis  and (vis==0 or vis==1)):

            self.visible=vis
            if hasattr(self,'data'):
                if self.data==None:
                    return
                if self.data.GetPointData()==None:
                    return
                VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
                if VisibleNodes==None:
                    return
                NNodes=self.data.GetNumberOfPoints()
                if NNodes==0:
                    return
                if vis:
                    visibleNodes=np.ones([NNodes,1],dtype=np.uint)
                else:
                    visibleNodes=np.zeros([NNodes,1],dtype=np.uint)
                
                VisibleNodes.DeepCopy(numpy_to_vtk(visibleNodes,deep=1, array_type=vtk.VTK_UNSIGNED_INT))
                VisibleNodes.Modified()

    def set_pickable(self,pickable):
        for key, child in self.children.iteritems():            
            child.set_pickable(pickable)

        if hasattr(self,'actor'):            
            if self.actor:
                if self.actor.GetClassName()=='vtkOpenGLActor':
                    actor=self.actor
                    if not actor.GetPickable()==pickable:
                        actor.SetPickable(pickable)
                        actor.Modified()
                elif self.actor.GetClassName()=='vtkActorCollection':
                    for iactor in range(self.actor.GetNumberOfItems()):
                        actor=self.actor.GetItemAsObject(iactor);
                        if not actor.GetPickable()==pickable:
                            actor.SetPickable(pickable)
                            actor.Modified()

        if hasattr(self,'clippedactor'):            
            if self.clippedactor:
                if self.clippedactor.GetClassName()=='vtkOpenGLActor':
                    actor=self.clippedactor
                    if not actor.GetPickable()==pickable:
                        actor.SetPickable(pickable)
                        actor.Modified()
                elif self.clippedactor.GetClassName()=='vtkActorCollection':
                    for iactor in range(self.clippedactor.GetNumberOfItems()):
                        actor=self.clippedactor.GetItemAsObject(iactor);
                        if not actor.GetPickable()==pickable:
                            actor.SetPickable(pickable)
                            actor.Modified()
        if hasattr(self,'labelactor'):            
            if not (not self.labelactor):
                if self.labelactor.GetClassName()=='vtkActor2D':
                    actor=self.labelactor
                    if not actor.GetPickable()==pickable:
                        actor.SetPickable(pickable)
                        actor.Modified()
                elif self.labelactor.GetClassName()=='vtkActor2DCollection':
                    for iitem in range(self.labelactor.GetNumberOfItems()):
                        actor=self.labelactor.GetItemAsObject(iitem);
                        if not actor.GetPickable()==pickable:
                            actor.SetPickable(pickable)
                            actor.Modified()

    def internal_delete(self):
        1

    def delete(self):
        if self.deleted:
            return
        self.deleted=True
        self.remove_fromactive()
#        print "delete ", self.objtype, " of neuron ", self.NeuronID 
        for key, child in self.children.iteritems():            
            child.delete()
            
        if hasattr(self,'nodeSelection'):
            self.nodeSelection.remove()
        anynonemptydata=0
        if hasattr(self,'activeData'):
            if self.activeData.GetNumberOfInputConnections(0)>0:
                anynonemptydata=1;

        if hasattr(self,'allData'):
            if hasattr(self,'allDataInput'):               
                self.allData.RemoveInputConnection(0,self.allDataInput.GetOutputPort())   
                self.allData.Modified()             
            elif hasattr(self,'data'):               
                self.allData.RemoveInputConnection(0,self.data.GetProducerPort())
                self.allData.Modified()             
            if self.allData.GetNumberOfInputConnections(0)>0:
                anynonemptydata=1;
            else:
                for viewport in self.viewports:
                    self.hide_actors(viewport,['actor','clippedactor','labelactor'])

        if hasattr(self,'allLabels'):
            if hasattr(self,'label'):               
                self.allLabels.RemoveInputConnection(0,self.label.GetProducerPort())                   
                self.allLabels.Modified()
            if self.allLabels.GetNumberOfInputConnections(0)==0:
                for viewport in self.viewports:
                    self.hide_actors(viewport,['labelactor'])        
        self.instancecount[0]-=1
        if self.instancecount[0]<1:
            if self.objtype=='synapse':
                self.LUT.SetNumberOfTableValues(2+12)    
            else:
                self.LUT.SetNumberOfTableValues(2)   
            self.LUT.Modified()
#            print "Reset LUT for {0}".format(self.objtype)
        
        if not anynonemptydata:
            
            while not (not self.viewports):
                self.remove_fromviewport(self.viewports[0])
            
            if hasattr(self,'clippedmapper'):
                self.clippedmapper.RemoveAllItems()
                self.clippedmapper.Modified()
            if hasattr(self,'clippedactor'):
                self.clippedactor.RemoveAllItems()
                self.clippedactor.Modified()
            if hasattr(self,'ClippingActors'):            
                while self.ClippingActors.__len__()>0:
                    self.ClippingActors.remove(self.ClippingActors[0])
            if hasattr(self,'ClippingPlanes'):            
                while self.ClippingPlanes.__len__()>0:
                    self.ClippingPlanes.remove(self.ClippingPlanes[0])

            self.VisEngine_started[0]=False

        self.internal_delete()

        if hasattr(self,"data"):
            del self.data
        if hasattr(self,"item"):
            if not (not self.item):
                self.item.removeRows(0,self.item.rowCount());            
                if self.item.parent().__class__.__name__==  "NoneType":
                    self.item.model().removeRow(self.item.index().row())
            del self.item
        
    def appendItem(self,parentItem):
        if not self.item:
            self.item=QtGui.QStandardItem("")
            
            self.item.objtype=self.objtype
            self.item.neuronId=self.NeuronID
            self.item.obj=self
        if not (not parentItem):
            parentItem.appendRow(self.item)
        self.updateItem()

    def updateItem(self):
        if not self.item:
            return
        comment=self.comments.get(self,"comment")
        if not comment:
            comment==""
            
        itemstr="{0}".format(self.objtype);
        
        if not self.NeuronID==None:
            itemstr+= " {0}".format(self.NeuronID)
            
        if not (not comment):
            itemstr+= ": {0}".format(comment)

        self.item.setText(itemstr)
        
    def get_obj_info(self,ipoint,icell):
        info_str=extractObjInfo(self.data.GetCellData(),icell)
        info_str+=extractObjInfo(self.data.GetPointData(),ipoint)

    def setup_VisEngine(self):
        1

    def pointIdx2nodeId(self,pointIdx):
        if not hasattr(self,'data'):
            return -1
        if not self.data:
            return -1
        if pointIdx==None:
            return -1
        if pointIdx<0 or pointIdx>(self.data.GetNumberOfPoints()-1):
            print "Warning: pointIdx out of range: ", pointIdx
            return -1
        NodeID=self.data.GetPointData().GetArray("NodeID")
        if not NodeID:
            return -1
        return NodeID.GetValue(pointIdx)


    def nodeId2pointIdx(self,nodeId):
        if not hasattr(self,'data'):
            return -1
        if not self.data:
            return -1
        if nodeId==None:
            return -1
        if not nodeId.__class__.__name__=='int':
            nodeId=np.int(nodeId)
        NodeID=self.data.GetPointData().GetArray("NodeID")
        if not NodeID:
            return -1
            
        pointIdx=NodeID.LookupValue(nodeId)
        if pointIdx==-1:
            NodeID.ClearLookup()
            pointIdx=NodeID.LookupValue(nodeId)
            if not pointIdx==-1:
                print "Error: The lookup for node id was not up-to-date. This should not happen!"
        return pointIdx

class DemandDrivenFile(config):
    archive=None
    BasePath='';
    _Dataset='E085L01';
    _FileExt='.enml'
    _BaseFileNames=[]
    _CubeSize=[1500.0,1500.0,1500.0]
    _CubePattern="_x%04.0f_y%04.0f_z%04.0f"
    _LoadingMode='FromArchive'
    _ComplementaryLoading=1
    _minDist2Border=500.0 #nm
    filelist=None
    
    def __init__(self,ariadne=None,filename=None):
        if not ariadne:
            return
        if not filename:
            return
        if not os.path.isfile(filename):
            return
        self.ariadne=ariadne
        self.archive = zipfile.ZipFile(filename, 'r');
        self.filelist=set(self.archive.namelist())
        self.CurrentCentralCube=None
        self.LoadedCubes=dict()
        self.Neurons=set()
#        self.pool = Pool() #defaults to number of available CPU's    

        
        ddconfigfile=[x for x in self.filelist if x.endswith('.ddf')]        
        if ddconfigfile.__len__()==1:
            ddconfigfile=ddconfigfile[0]
            ddconfigfileobj = cStringIO.StringIO(self.archive.read(ddconfigfile))
            config.__init__(self,ddconfigfileobj, encoding='UTF8');
        else:
            print "Error: invalid demand-driven config file: ", ddconfigfile
            return
        
        self.LoadConfig(self,"DemandDrivenFile")
        if self._LoadingMode=='FromArchive':
            self.BasePath='';
        else:
            self.BasePath=os.path.dirname(filename)
    
    def delete(self):
        oldMode=self.ariadne.ObjectBrowser.selectionMode();
        self.ariadne.ObjectBrowser.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        for neuron_obj in self.Neurons:
            neuronId=neuron_obj.NeuronID
            neuron_obj.delete()
            if neuronId in self.ariadne.Neurons:
                del self.ariadne.Neurons[neuronId]
        self.ariadne.ObjectBrowser.setSelectionMode(oldMode)

        self.CurrentCentralCube=None
        self.LoadedCubes=dict()
        self.Neurons=set()
        if not self.archive:
            return
        self.archive.close()

class brain(objs):
    objtype="brain"
    
class neuron(objs):
    objtype="neuron"
    maxNofColors=[2000]
    initialized=[False]
    VisEngine_started=[False]

    LUT=vtk.vtkLookupTable()
    instancecount=[0]

    def __init__(self,parentItem,NeuronID,color):
        self.reset()
        self.NeuronID=NeuronID
        self.instancecount[0]+=1
        self.init_VisPipeline()

        self.setup_color(color)
        
        self.appendItem(parentItem)
        self.activity=[]

    def init_VisPipeline(self):
        if self.initialized[0]:
            return

        self.LUT.Allocate(self.maxNofColors[0])
        self.LUT.SetNumberOfTableValues(2)
        self.LUT.SetTableRange(0,1)
        self.LUT.SetTableValue(deleteColor,[0.0,0.0,0.0,0.0])
        self.LUT.SetTableValue(selectColor,[1.0,0.0,0.0,1.0])
        
        self.initialized[0]=True

    def activity2alpha(self,activity,\
        alpha=5.0,minAlpha=0.0,offset=0.0,amplitude=6.0):
        newAlpha= np.min([1.0,np.max([minAlpha,amplitude*(1.0+np.sign(activity)*(\
        (np.exp(alpha*np.min([1.0,np.abs(activity)+offset]))-1.0)/(np.exp(alpha)-1.0))-1.0)])]);        
        if newAlpha<=minAlpha:
            self.set_visibility(0)
        else:
            self.set_visibility(1)
            self.set_alpha(newAlpha);

class area(neuron):
    objtype="area"
    instancecount=[0]
    def __init__(self,parentItem,NeuronID,color):
        self.reset()
        self.NeuronID=NeuronID
        self.instancecount[0]+=1
        self.init_VisPipeline()

        self.setup_color(color)
        
        self.appendItem(parentItem)
        self.activity=[]

class soma(objs):
    objtype="soma"
    maxNofColors=[2000]
    DefaultAlpha=[100]
    activityFlag=0

    initialized = [False]
    VisEngine_started=[False]
    
    viewports=list()
    
    allData= vtk.vtkAppendFilter()    
    
    mapper=vtk.vtkMapperCollection()
    actor=vtk.vtkActorCollection()

    ClippingPlanes=[] 
    ClippingActors=[]
    clippedmapper=vtk.vtkMapperCollection()
    clippedactor=vtk.vtkActorCollection()    
    
    allLabels= vtk.vtkAppendFilter()    
    labelmapper=vtk.vtkCollection()
    labelactor=vtk.vtkActor2DCollection()
    VisibleLabels = vtk.vtkCollection()
#    labelmapper=vtk.vtkLabeledDataMapper()
#    labelactor=vtk.vtkActor2D()

    LUT=vtk.vtkLookupTable()
    instancecount=[0]
    
    def __init__(self,parentItem,parentID,color):
        self.reset()
        self.NeuronID=parentID
        self.instancecount[0]+=1

        self.init_VisPipeline()
        
        self.setup_color(color)

        self.setup_VisEngine()

        self.appendItem(parentItem)

    def get_closest_point(self,point):
        if not self.PointLocator.GetDataSet():
            self.visibleData.Update()
            DataSet=self.visibleData.GetOutput()
            if DataSet.GetNumberOfPoints()>0:
                self.PointLocator.SetDataSet(DataSet)
                self.PointLocator.BuildLocator()
            else:
                return -1,-1
        DataSet=self.PointLocator.GetDataSet()
        DataSet.Update()
        pointIdx=self.PointLocator.FindClosestPoint(point)
        if pointIdx==-1 or pointIdx==None:
            return -1,-1

        nodeId=DataSet.GetPointData().GetArray("NodeID").GetValue(pointIdx)
        return np.array(DataSet.GetPoint(pointIdx),dtype=np.float), nodeId

    def update_label(self):
        cellCenters=vtk.vtkCellCenters()
        cellCenters.SetInputConnection(self.validData.GetOutputPort())
        cellCenters.Update()
        FaceCenters=cellCenters.GetOutput()
        if not (not FaceCenters):
            if not (not FaceCenters.GetPoints()):
                FaceCenterPoints=FaceCenters.GetPoints().GetData()
                if not (not FaceCenterPoints):
                    Point=np.mean(vtk_to_numpy(FaceCenterPoints),0)
                    Points=self.label.GetPoints()
                    if not Points:
                        Points=vtk.vtkPoints()
                        Points.SetNumberOfPoints(1)
                        self.label.SetPoints(Points)
                        Points.SetPoint(0,Point)
                        Points.Modified()
        
        nodeId=self.pointIdx2nodeId(0)
        comments=self.comments.get(nodeId)
        if not not comments:
            labelstr=None
            if comments.has_key("comment"):
                labelstr=comments["comment"]
            print "label: ", labelstr
            if not (not labelstr):    
                Labels=self.label.GetPointData().GetAbstractArray("Labels")
                if not (not Labels):
                    Labels.SetValue(0,labelstr)
                    Labels.Modified()
        self.label.Modified()

    def set_nodes(self,points,NodeID=None):
        self.data.SetPoints(points)        

        NNodes=points.GetNumberOfPoints()
       
        colors=self.colorIdx*np.ones([NNodes,1],dtype=np.float)
        PointColor=self.data.GetPointData().GetArray("PointColor")
        PointColor.DeepCopy(numpy_to_vtk(colors, deep=1, array_type=vtk.VTK_FLOAT))
        PointColor.Modified()
        
        neuronID=self.NeuronID*np.ones([NNodes,1],dtype=np.float32)
        NeuronID=self.data.GetPointData().GetArray("NeuronID")
        NeuronID.DeepCopy(numpy_to_vtk(neuronID, deep=1, array_type=vtk.VTK_FLOAT))
        NeuronID.Modified()
        
        NodeIDArray=self.data.GetPointData().GetArray("NodeID") 
        if NodeID==None:
            NodeID=np.array(range(NNodes),dtype=np.int)      
        
        if not (NodeID.__class__.__name__=='vtkIdTypeArray' or NodeID.__class__.__name__=='vtkIntArray' or NodeID.__class__.__name__=='vtkLongArray' or NodeID.__class__.__name__=='vtkLongLongArray'): 
            if not NodeID.__class__.__name__=='ndarray':
                NodeID=np.array([NodeID],dtype=np.int)
            
            NodeID=numpy_to_vtk(NodeID, deep=1, array_type=vtk.VTK_ID_TYPE)
        else:
            if NodeID.GetNumberOfTuples()>0:
                NodeID=numpy_to_vtk(vtk_to_numpy(NodeID), deep=1, array_type=vtk.VTK_ID_TYPE)
            else:
                1
        NodeIDArray.DeepCopy(NodeID)
        NodeIDArray.Modified()
        
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        DeletedNodes.DeepCopy(numpy_to_vtk(np.zeros([NNodes,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        DeletedNodes.Modified()

        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
        VisibleNodes.DeepCopy(numpy_to_vtk(np.ones([NNodes,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        VisibleNodes.Modified()
        
        vertex=np.array([np.ones(NNodes),range(NNodes)],dtype=np.int).reshape(2,NNodes).transpose().reshape(2*NNodes,)
        vertex=numpy_to_vtk(vertex, deep=1, array_type=vtk.VTK_ID_TYPE)
        Vertices=vtk.vtkCellArray()
        Vertices.SetCells(NNodes,vertex)
        self.data.SetVerts(Vertices)
      
        self.data.Modified()
        self.data.BuildCells()
        self.data.BuildLinks()
        self.data.Modified()
        
        self.update_label()
        
        if NodeID.GetNumberOfTuples()==0:
            return -1,-1
        if NNodes==1:
            return NodeID.GetValue(0),0
        NodeID=vtk_to_numpy(NodeID)
        NodeID=NodeID.tolist()
        return NodeID,range(NNodes)  #last nodeId and last pointIdx
      
    @classmethod
    def update_VisEngine(self, alpha=None):
        if self.activityFlag:
            return
        if alpha==None:
            alpha=self.DefaultAlpha[0]        
        if hasattr(self,'actor'):
            for iactor in range(self.actor.GetNumberOfItems()):
                Modified=False;
                actor=self.actor.GetItemAsObject(iactor)
                if alpha==0:
                    if actor.GetVisibility():
                        actor.VisibilityOff()
                        Modified=True;
                    if actor.GetPickable():
                        actor.PickableOff()
                        Modified=True;
                else:
                    if not actor.GetVisibility():
                        actor.VisibilityOn()
                        Modified=True
                    if not actor.GetPickable():
                        actor.PickableOn()
                        Modified=True
                    if not (actor.GetProperty().GetOpacity()==alpha/100.0):
                        actor.GetProperty().SetOpacity(alpha/100.0)
                        Modified=True
                if Modified:
                    actor.Modified()
                    
        if hasattr(self,'clippedactor'):
            for iactor in range(self.clippedactor.GetNumberOfItems()):
                Modified=False;
                actor=self.clippedactor.GetItemAsObject(iactor)
                if alpha==0:
                    if actor.GetVisibility():
                        actor.VisibilityOff()
                        Modified=True;
                    if actor.GetPickable():
                        actor.PickableOff()
                        Modified=True;
                else:
                    if not actor.GetVisibility():
                        actor.VisibilityOn()
                        Modified=True
                    if not actor.GetPickable():
                        actor.PickableOn()
                        Modified=True
                    if not (actor.GetProperty().GetOpacity()==alpha/100.0):
                        actor.GetProperty().SetOpacity(alpha/100.0)
                        Modified=True
                if Modified:
                    actor.Modified()        
                    
    def setup_VisEngine(self):
        self.data = vtk.vtkPolyData()
        self.data.Allocate()

        self.label= vtk.vtkPolyData()
        labelstr=vtk.vtkStringArray();
        labelstr.SetName("Labels");
        labelstr.SetNumberOfValues(1);
        labelstr.SetValue(0,'')   
        self.label.GetPointData().AddArray(labelstr)
        
        ObjType = vtk.vtkStringArray()
        ObjType.SetName("ObjType")
        ObjType.SetNumberOfValues(1)
        ObjType.SetValue(0,self.objtype)        
        self.data.GetFieldData().AddArray(ObjType)
         
        PointColor = vtk.vtkFloatArray()
        PointColor.SetName("PointColor")
        PointColor.SetNumberOfComponents(1)
        self.data.GetPointData().SetScalars(PointColor)

        NeuronID=vtk.vtkFloatArray()
        NeuronID.SetName("NeuronID")
        NeuronID.SetNumberOfComponents(1)
        self.data.GetPointData().AddArray(NeuronID)

        NodeID = vtk.vtkIdTypeArray()
        NodeID.SetName("NodeID")
        self.data.GetPointData().AddArray(NodeID)

        DeletedNodes=vtk.vtkUnsignedIntArray()
        DeletedNodes.SetName("DeletedNodes")
        self.data.GetPointData().AddArray(DeletedNodes)

        VisibleNodes=vtk.vtkUnsignedIntArray()
        VisibleNodes.SetName("VisibleNodes")
        self.data.GetPointData().AddArray(VisibleNodes)

        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("DeletedNodes")
        selection.AddThreshold(0,0)
        selection.Update()
        
        self.validData =vtk.vtkExtractSelection()    
        self.validData.SetInputConnection(0,self.data.GetProducerPort());
        self.validData.SetInputConnection(1,selection.GetOutputPort());
        
        #to avoid problems with degenerated points/triangles we randomly 
        #displace each input point by at max 1px
        randDisplacement=vtk.vtkBrownianPoints()
        randDisplacement.SetMinimumSpeed(0.0)
        randDisplacement.SetMaximumSpeed(1.0)        
        randDisplacement.SetInputConnection(self.validData.GetOutputPort())
        warp = vtk.vtkWarpVector()
        warp.SetInputConnection(randDisplacement.GetOutputPort())
        self.SomaDelaunay3D = vtk.vtkDelaunay3D()
        self.SomaDelaunay3D.SetTolerance(0.0)
        self.SomaDelaunay3D.BoundingTriangulationOff()
        self.SomaDelaunay3D.SetInputConnection(warp.GetOutputPort())

        self.SomaSurface=vtk.vtkDataSetSurfaceFilter()
        self.SomaSurface.SetInputConnection(self.SomaDelaunay3D.GetOutputPort())
        
        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("VisibleNodes")
        selection.AddThreshold(1,1)
        selection.Update()        
        self.visibleData =vtk.vtkExtractSelection()
        self.visibleData.SetInputConnection(0,self.SomaSurface.GetOutputPort());
        self.visibleData.SetInputConnection(1,selection.GetOutputPort());
        
        self.allDataInput = self.visibleData

        randDisplacement.ReleaseDataFlagOn()
        warp.ReleaseDataFlagOn()
        self.SomaDelaunay3D.ReleaseDataFlagOn()
        self.SomaSurface.ReleaseDataFlagOn()
        self.visibleData.ReleaseDataFlagOn()
        self.validData.ReleaseDataFlagOn()
        self.allDataInput.ReleaseDataFlagOn()

        self.allData.AddInputConnection(0,self.allDataInput.GetOutputPort())
        self.allData.Modified()
        
        self.allLabels.AddInputConnection(0,self.label.GetProducerPort())
        self.allLabels.Modified()

        self.PointLocator=vtk.vtkPointLocator()

    def init_VisPipeline(self):
        if self.initialized[0]:
            return

        self.LUT.Allocate(self.maxNofColors[0])
        self.LUT.SetNumberOfTableValues(2)
        self.LUT.SetTableRange(0,1)
        self.LUT.SetTableValue(deleteColor,[0.0,0.0,0.0,0.0])
        self.LUT.SetTableValue(selectColor,[1.0,0.0,0.0,1.0])

        SomaMapper= vtk.vtkDataSetMapper()
        SomaMapper.SetLookupTable(self.LUT);    
        SomaMapper.SetUseLookupTableScalarRange(1);
#        self.mapper.SetColorModeToMapScalars(); 
        SomaMapper.SetInputConnection(self.allData.GetOutputPort()) 

#        SomaMapper.GlobalImmediateModeRenderingOn()
        
        SomaActor=vtk.vtkActor()
        SomaActor.SetMapper(SomaMapper)       
        SomaActor.GetProperty().SetSpecular(.3)
        SomaActor.GetProperty().SetSpecularPower(30)

        self.mapper.AddItem(SomaMapper)
        self.actor.AddItem(SomaActor)

#        self.labelmapper.SetInputConnection(self.allLabels.GetOutputPort()) 
#        self.labelmapper.SetFieldDataName("Labels")
#        self.labelmapper.SetLabelModeToLabelFieldData()
#        self.labelmapper.SetFieldDataArray(0)
#        self.labelactor.SetMapper(self.labelmapper);
                
        self.initialized[0]=True

class region(soma):
    objtype="region"
    maxNofColors=[2000]
    DefaultAlpha=[100]
    activityFlag=0

    initialized = [False]
    VisEngine_started=[False]
    
    viewports=list()
    
    allData= vtk.vtkAppendFilter()    
    
    mapper=vtk.vtkMapperCollection()
    actor=vtk.vtkActorCollection()

    ClippingPlanes=[] 
    ClippingActors=[]
    clippedmapper=vtk.vtkMapperCollection()
    clippedactor=vtk.vtkActorCollection()    
    
    allLabels= vtk.vtkAppendFilter()    
    labelmapper=vtk.vtkCollection()
    labelactor=vtk.vtkActor2DCollection()
    VisibleLabels = vtk.vtkCollection()
#    labelmapper=vtk.vtkLabeledDataMapper()
#    labelactor=vtk.vtkActor2D()

    LUT=vtk.vtkLookupTable()
    instancecount=[0]
 
class NodeSelection():
    initialized=[False]
    DefaultLineStyle=["lines"]
    DefaultPointStyle=["points"]
    Defaultvis3D=["off"]
    DefaultshowNodes=[1]
    DefaultPointSize=[5+6]
    DefaultLineWidth=[1+6]
    
    allSelection=vtk.vtkAppendFilter()
    vertices=vtk.vtkVertexGlyphFilter()
    glyphSource = vtk.vtkSphereSource()

    glyph= vtk.vtkGlyph3D()
    tube = vtk.vtkTubeFilter()
    mapper= vtk.vtkDataSetMapper()
    linemapper= vtk.vtkDataSetMapper()
    actor= vtk.vtkActor()
    lineactor= vtk.vtkActor()
    
    def __init__(self,ParentSource,ParentActors):   
        if not (ParentActors.__class__.__name__=='list'):
            ParentActors=[ParentActors]
        if self.initialized[0]:
            return

       
        self.vertices.SetInputConnection(self.allSelection.GetOutputPort())
        self.glyph.SetScaleFactor(1.33)
        self.glyph.SetScaleModeToScaleByScalar()
        self.glyph.SetInputConnection(self.vertices.GetOutputPort())
        self.glyph.SetSourceConnection(ParentSource.GetOutputPort())
        
        GeometryFilter=vtk.vtkGeometryFilter()
        GeometryFilter.MergingOff()
        GeometryFilter.SetInputConnection(self.allSelection.GetOutputPort())        
        self.tube.SetNumberOfSides(6)
        self.tube.SetRadius(25)
        self.tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.tube.SetInputConnection(GeometryFilter.GetOutputPort())        
        
        self.mapper.SetInputConnection(self.glyph.GetOutputPort()) 
        self.mapper.ScalarVisibilityOff()

        self.linemapper.SetInputConnection(self.allSelection.GetOutputPort()) 
#        self.linemapper.SetInputConnection(self.tube.GetOutputPort()) 
        self.linemapper.ScalarVisibilityOff()

        self.actor.SetMapper(self.mapper)       
        self.actor.GetProperty().SetColor(1.0,0.0, 0.0);
        self.actor.GetProperty().SetSpecular(0.3)
        self.actor.GetProperty().SetSpecularPower(30)
        self.actor.GetProperty().SetPointSize(self.DefaultPointSize[0]) 
        self.actor.GetProperty().SetOpacity(0.3)
        self.actor.PickableOff()

        self.lineactor.SetMapper(self.linemapper)       
        self.lineactor.GetProperty().SetColor(1.0,0.0, 0.0);
        self.lineactor.GetProperty().SetSpecular(0.3)
        self.lineactor.GetProperty().SetSpecularPower(30)
#        self.actor.GetProperty().SetDiffuse(1.0); 
        self.lineactor.GetProperty().SetOpacity(0.3)
        self.lineactor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
        self.lineactor.PickableOff()
        
        for parent in ParentActors:
            parent.AddItem(self.actor)
            parent.AddItem(self.lineactor)

        self.update_VisEngine()
        self.initialized[0]=True

    def add(self,data,\
        FieldType=vtk.vtkSelectionNode.POINT,ContentType=vtk.vtkSelectionNode.PEDIGREEIDS,ContainsCells=False):
        ids =vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1);
 
        self.node =vtk.vtkSelectionNode()
        self.node.SetFieldType(FieldType);
        self.node.SetContentType(ContentType);        
        self.node.SetSelectionList(ids)
        if ContainsCells:
            self.node.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(),1);
 
        self.selection =vtk.vtkSelection()
        self.selection.AddNode(self.node);
        
        self.extract=vtk.vtkExtractSelection();
        self.extract.SetInputConnection(1,self.selection.GetProducerPort());
        self.extract.SetInputConnection(0,data.GetProducerPort());
        
        self.allSelection.AddInputConnection(self.extract.GetOutputPort())
        self.allSelection.Modified()

    def remove(self):
        self.allSelection.RemoveInputConnection(0,self.extract.GetOutputPort())
        self.allSelection.Modified()
        
    @classmethod
    def update_VisEngine(self, LineStyle=None, PointStyle=None, vis3D=None, showNodes=None):
        if not hasattr(self,'actor'):
            return
        if LineStyle==None:
            LineStyle=self.DefaultLineStyle[0]
        if PointStyle==None:
            PointStyle=self.DefaultPointStyle[0]
        if vis3D==None:
            vis3D=self.Defaultvis3D[0]
        if showNodes==None:
            showNodes=self.DefaultshowNodes[0]

        if LineStyle=='lines':
            if not (self.lineactor.GetProperty().GetLineWidth()==\
                self.DefaultLineWidth[0]):
                self.lineactor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
                self.lineactor.Modified()

        if PointStyle=='points':
            if self.mapper.GetNumberOfInputConnections(0)>0:
                if not (self.mapper.GetInputConnection(0,0)==\
                    self.vertices.GetOutputPort()):
                    self.mapper.SetInputConnection(\
                        self.vertices.GetOutputPort())
                    self.mapper.Modified()
            if not (self.actor.GetProperty().GetPointSize()==self.DefaultPointSize[0]):
                self.actor.GetProperty().SetPointSize(self.DefaultPointSize[0])
                self.actor.Modified()
        
        if PointStyle=='spheres':
            if self.mapper.GetNumberOfInputConnections(0)>0:
                if not (self.mapper.GetInputConnection(0,0)==\
                    self.glyph.GetOutputPort()):
                    self.mapper.SetInputConnection(\
                        self.glyph.GetOutputPort())
                    self.mapper.Modified()

class skeleton(objs):
    objtype="skeleton"
    maxNofColors=[2000]
    DefaultLineStyle=["lines"]
    DefaultPointStyle=["points"]
    Defaultvis3D=["off"]
    DefaultshowNodes=[1]
    DefaultRadius=[25]
    DefaultLineWidth=[1]
    DefaultPointSize=[5]
    doRestrictVOI=[False]
    
    initialized=[False]
    VisEngine_started=[False]

    viewports=list()

    NodeSource=vtk.vtkSphereSource()    
    LUT=vtk.vtkLookupTable()

    VOIExtent=[0.0,1000.0,0.0,1000.0,0.0,1000.0,0.0,1000.0]

    allData= vtk.vtkAppendFilter()
    VOIFilter=vtk.vtkUnstructuredGridGeometryFilter()
    NodeGlyphs= vtk.vtkGlyph3D()
    GeometryFilter=vtk.vtkGeometryFilter()
    Stripper=vtk.vtkStripper()
    BranchTubeFilter = vtk.vtkTubeFilter()
    mapper=vtk.vtkMapperCollection()
    actor=vtk.vtkActorCollection()

    activeData= vtk.vtkAppendFilter()
    activeVOIFilter=vtk.vtkUnstructuredGridGeometryFilter()
    activeNodeGlyphs= vtk.vtkGlyph3D()
    activeGeometryFilter=vtk.vtkGeometryFilter()
    activeStripper=vtk.vtkStripper()
    activeBranchTubeFilter = vtk.vtkTubeFilter()
    activemapper=vtk.vtkMapperCollection()
    activeactor=vtk.vtkActorCollection()
    instancecount=[0]

    def __init__(self,parentItem,parentID,color):
        self.reset()
        self.NeuronID=parentID
        self.instancecount[0]+=1
        
        self.init_VisPipeline()
        
        self.setup_color(color)
        
        self.setup_VisEngine()
        
        self.nodeSelection=NodeSelection(self.NodeSource,[self.activeactor,self.actor])
        self.nodeSelection.add(self.data)
        
        self.appendItem(parentItem)

    def extract_nodeId_seeded_region(self,seedlist):
        #the returned list contains the original node ids from the array NodeID
        if not (seedlist.__class__.__name__=='list'): 
            seedlist=[seedlist]
            
        connfilter=vtk.vtkConnectivityFilter()
        connfilter.SetExtractionModeToPointSeededRegions()
        connfilter.SetInputConnection(self.data.GetProducerPort())
        for seedId in seedlist:
            pointIdx=self.nodeId2pointIdx(seedId)
            if pointIdx<0:
                continue;
            connfilter.AddSeed(pointIdx)
        connfilter.Update()
        return connfilter.GetOutput().GetPointData().GetArray("NodeID")
        
        
    def delete_edge(self,nodeId1,nodeId2):
        if 'l' in self.flags: #locked object
            return
        self.data.BuildCells()
        
        pointIdx1=self.nodeId2pointIdx(nodeId1)
        if pointIdx1<0:
            return
        pointIdx2=self.nodeId2pointIdx(nodeId2)
        if pointIdx2<0:
            return
        
        CellList=vtk.vtkIdList()
        self.data.GetPointCells(pointIdx1,CellList)

        CellList2=vtk.vtkIdList()
        self.data.GetPointCells(pointIdx2,CellList2)
        
        CellList.IntersectWith(CellList2)

        NCells=CellList.GetNumberOfIds()
        if NCells<1: #no edge between pointIdx1 and pointIdx2
            return
        Branches2Add=list()
        Cells2Remove=list()
        for icell in range(NCells):
            cellIdx=CellList.GetId(icell)
            CellPointIds=self.data.GetCell(cellIdx).GetPointIds()
            NPoints=CellPointIds.GetNumberOfIds()
            if NPoints<=1:
                continue
            branch0=vtk.vtkIdList()
            branch1=vtk.vtkIdList()
            tempbranch=branch0
            tempPointId1=CellPointIds.GetId(0)               
            tempbranch.InsertNextId(tempPointId1)
            splitflag=0
            for inode in range(1,NPoints):
                tempPointId2=CellPointIds.GetId(inode)
                if (tempPointId1==pointIdx1 and tempPointId2==pointIdx2) or \
                    (tempPointId1==pointIdx2 and tempPointId2==pointIdx1):
                    tempbranch=branch1
                    splitflag=1
                tempPointId1=tempPointId2
                tempbranch.InsertNextId(tempPointId1)
            if splitflag:
                Cells2Remove.append(cellIdx)
                if branch0.GetNumberOfIds()>1:
                    Branches2Add.append(branch0)
                if branch1.GetNumberOfIds()>1:
                    Branches2Add.append(branch1)

        #Remove the marked cell.
        for cellIdx in Cells2Remove:
            #Mark a cell as deleted.
            self.data.DeleteCell(cellIdx);

        self.data.RemoveDeletedCells();

        self.data.BuildCells() 
        self.data.BuildLinks()
        self.data.Modified()

        for newbranch in Branches2Add:
            self.add_branch(newbranch)
    
    def isconnected(self,nodeId1,nodeId2):
        pointIdx1=self.nodeId2pointIdx(nodeId1)
        if pointIdx1<0:
            return 0
        pointIdx2=self.nodeId2pointIdx(nodeId2)
        if pointIdx2<0:
            return 0
        
        NPoints=self.data.GetNumberOfPoints()
        if pointIdx1>=NPoints or pointIdx2>=NPoints or pointIdx1<0 or pointIdx2<0:
            return 0
        CellList=vtk.vtkIdList()
        self.data.GetPointCells(pointIdx1,CellList)
        CellList2=vtk.vtkIdList()
        self.data.GetPointCells(pointIdx2,CellList2)        
        CellList.IntersectWith(CellList2)
        for icell in range(CellList.GetNumberOfIds()):
            branchID=CellList.GetId(icell)
            CellPointIds=self.data.GetCell(branchID).GetPointIds()
            NPoints=CellPointIds.GetNumberOfIds()
            if NPoints<=1:
                continue
            temp_pointIdx=CellPointIds.GetId(0)
            for inode in range(1,NPoints):
                if temp_pointIdx==pointIdx1:
                    temp_pointIdx=CellPointIds.GetId(inode)
                    if temp_pointIdx==pointIdx2:
                        return 1
                elif temp_pointIdx==pointIdx2:
                    temp_pointIdx=CellPointIds.GetId(inode)
                    if temp_pointIdx==pointIdx1:
                        return 1
                else:
                    temp_pointIdx=CellPointIds.GetId(inode)
        return 0
        
    def add_edge(self,nodeId1,nodeId2):
        if 'l' in self.flags: #locked object
            return
        self.new_singleactive(self)
        #check first, if there is already an edge between the two points
        if self.isconnected(nodeId1,nodeId2):
            return -1
        pointIdx1=self.nodeId2pointIdx(nodeId1)
        if pointIdx1<0:
            return -1
        pointIdx2=self.nodeId2pointIdx(nodeId2)
        if pointIdx2<0:
            return -1

        edge=vtk.vtkIdList()
        edge.InsertNextId(pointIdx1)
        edge.InsertNextId(pointIdx2)
        self.data.InsertNextCell(vtk.VTK_LINE,edge)

        self.data.BuildCells()
        self.data.BuildLinks()
        self.data.Modified()
        
        distance=np.sqrt(vtk.vtkMath.Distance2BetweenPoints(\
            self.data.GetPoint(pointIdx1),\
            self.data.GetPoint(pointIdx2)))
        return distance

    def get_closest_point(self,point):
        if not self.PointLocator.GetDataSet():
            self.visibleData.Update()
            DataSet=self.visibleData.GetOutput()
            if DataSet.GetNumberOfPoints()>0:
                self.PointLocator.SetDataSet(DataSet)
                self.PointLocator.BuildLocator()
            else:
                return -1,-1
        DataSet=self.PointLocator.GetDataSet()
        DataSet.Update()
        pointIdx=self.PointLocator.FindClosestPoint(point)
        if pointIdx==-1 or pointIdx==None:
            return -1,-1

        nodeId=DataSet.GetPointData().GetArray("NodeID").GetValue(pointIdx)
        return np.array(DataSet.GetPoint(pointIdx),dtype=np.float), nodeId

    def delete_node(self,nodeIds):   
        if 'l' in self.flags: #locked object
            return
        self.new_singleactive(self)
        if nodeIds==None:
            return None
        if nodeIds.__class__.__name__.startswith('vtk'):
            nodeIds=vtk_to_numpy(nodeIds)
            nodeIds=nodeIds.tolist()
        if not nodeIds.__class__.__name__=='set':
            if not nodeIds.__class__.__name__=='list':
                nodeIds=[nodeIds]
            nodeIds=set(nodeIds)
        allCells=vtk.vtkIdList()
        allPointIdxs=list()
        
        self.data.BuildCells() 
        self.data.BuildLinks()        
        
        self.unselect_node(nodeIds)
        for nodeId in nodeIds:
            self.comments.delete(nodeId)
        
            pointIdx=self.nodeId2pointIdx(nodeId)
            if pointIdx<0:
                continue;
            allPointIdxs.append(pointIdx)
            
            if self.data.GetNumberOfCells()==0:
                continue
            CellList=vtk.vtkIdList()
            self.data.GetPointCells(pointIdx,CellList)
            NCells=CellList.GetNumberOfIds()
            for icell in range(NCells):
                allCells.InsertUniqueId(CellList.GetId(icell))

        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
#        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")

        for pointIdx in allPointIdxs:
            DeletedNodes.SetValue(pointIdx,1);
#            VisibleNodes.SetValue(pointIdx,0);

        NCells=allCells.GetNumberOfIds()                
        Branches2Add=list()
        for icell in range(NCells):
            cellIdx=allCells.GetId(icell)
            
            tempCell=self.data.GetCell(cellIdx)
            tempCellType=tempCell.GetCellType()

            #Mark the cell as deleted.
            self.data.DeleteCell(cellIdx);

            if tempCellType<3:
                # VTK_EMPTY_CELL = 0, VTK_VERTEX = 1, VTK_POLY_VERTEX = 2, VTK_LINE = 3, VTK_POLY_LINE = 4
                continue;

            CellpointIdxs=tempCell.GetPointIds()
            NPoints=CellpointIdxs.GetNumberOfIds()
            tempbranch=vtk.vtkIdList()
            for inode in range(NPoints):
                tempPointIdx=CellpointIdxs.GetId(inode)
                if (tempPointIdx in allPointIdxs):
                    if tempbranch.GetNumberOfIds()>0:
                        Branches2Add.append(tempbranch)
                        tempbranch=vtk.vtkIdList()
                else:
                    tempbranch.InsertNextId(tempPointIdx)
            if tempbranch.GetNumberOfIds()>0:
                Branches2Add.append(tempbranch)

        #Remove the marked cells.
        self.data.RemoveDeletedCells();
        self.data.BuildCells() 	

        DeletedNodes.Modified()
#        VisibleNodes.Modified()

        self.data.BuildLinks()
        self.data.Modified()
                
        for newbranch in Branches2Add:
            self.add_branch(newbranch)

    def SetRadius(self,radius,overwrite):
        NNodes=self.data.GetNumberOfPoints()
        Radius=self.data.GetPointData().GetArray("Radius")
        newRadius=radius*np.ones([NNodes,1],dtype=np.float)
        if not overwrite:
            NodeID=self.data.GetPointData().GetArray("NodeID")
            if not (not NodeID):               
                for inode in range(NNodes):
                    nodeId=NodeID.GetValue(inode)
                    tempRadius=self.comments.get(nodeId,"radius")
                    if not tempRadius:
                        continue
                    if tempRadius.__class__.__name__=='str':
                        if not isnumeric(tempRadius):
                            continue
                        if ("none" in tempRadius) or ("None" in tempRadius) or ("nan" in tempRadius) or ("NaN" in tempRadius):
                            continue
                        newRadius[inode]=max(str2num('float',tempRadius),radius)
#                        print tempRadius
                    else:
                        if isnan(tempRadius):
                            continue
                        
                        newRadius[inode]=max(float(tempRadius),radius)
#                        print tempRadius
        Radius.DeepCopy(numpy_to_vtk(newRadius, deep=1, array_type=vtk.VTK_FLOAT))

        Radius.Modified()
        self.data.Modified()

    def setupNodeSource(self):
        Sphere = self.NodeSource
        Sphere.SetPhiResolution(6)
        Sphere.SetThetaResolution(6)
        Sphere.SetRadius(1.33)#9.252*1.5)
        Sphere.Update()
        
#        NSides=6
#        Circle=vtk.vtkRegularPolygonSource()
#        Circle.SetRadius(30);
#        Circle.SetNumberOfSides(NSides)
#        Circle.SetCenter(0,0,0);
#        Circle.GeneratePolygonOn();
#        Circle.SetNormal(0.0,0.0,-1.0)
#        Circle.Update()
#        
#        self.NodeSource=vtk.vtkPolyData()
#        self.NodeSource.Allocate()
#        Points1=vtk.vtkPoints()
#        Points1.DeepCopy(Circle.GetOutput().GetPoints())
#        self.NodeSource.SetPoints(Points1)
#        IdList1=vtk.vtkIdList()
#        IdList1.SetNumberOfIds(NSides)
#        IdList2=vtk.vtkIdList()
#        IdList2.SetNumberOfIds(NSides)
#        for iid in range(0,NSides):
#            IdList1.SetId(iid,NSides-1-iid)    
#            IdList2.SetId(iid,iid)    
#        self.NodeSource.InsertNextCell(vtk.VTK_POLYGON,IdList1);
#        self.NodeSource.InsertNextCell(vtk.VTK_POLYGON,IdList2);
#        self.NodeSource.Update()

    def setup_VisEngine(self):
        self.data = vtk.vtkPolyData()
        self.data.Allocate()

        ObjType = vtk.vtkStringArray()
        ObjType.SetName("ObjType")
        ObjType.SetNumberOfValues(1)
        ObjType.SetValue(0,self.objtype)        
        self.data.GetFieldData().AddArray(ObjType)
        
        PointColor = vtk.vtkFloatArray()
        PointColor.SetName("PointColor")
        PointColor.SetNumberOfComponents(1)
        self.data.GetPointData().AddArray(PointColor)
        
        NeuronID=vtk.vtkFloatArray()
        NeuronID.SetName("NeuronID")
        NeuronID.SetNumberOfComponents(1)
        self.data.GetPointData().AddArray(NeuronID)

        NodeID = vtk.vtkIdTypeArray()
        NodeID.SetName("NodeID")
        self.data.GetPointData().AddArray(NodeID)
        
        DeletedNodes=vtk.vtkUnsignedIntArray()
        DeletedNodes.SetName("DeletedNodes")
        self.data.GetPointData().AddArray(DeletedNodes)

        VisibleNodes=vtk.vtkUnsignedIntArray()
        VisibleNodes.SetName("VisibleNodes")
        self.data.GetPointData().AddArray(VisibleNodes)

        Radius=vtk.vtkFloatArray()
        Radius.SetName("Radius")
        self.data.GetPointData().AddArray(Radius)

        self.data.GetPointData().SetActivePedigreeIds("NodeID")
        self.data.GetPointData().SetScalars(Radius)
        self.data.Modified()

        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("DeletedNodes")
        selection.AddThreshold(0,0)
        selection.Update()
        
        self.validData =vtk.vtkExtractSelection()    
        self.validData.SetInputConnection(0,self.data.GetProducerPort());
        self.validData.SetInputConnection(1,selection.GetOutputPort());

        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("VisibleNodes")
        selection.AddThreshold(1,1)
        selection.Update()        
        self.visibleData =vtk.vtkExtractSelection()
        self.visibleData.SetInputConnection(0,self.validData.GetOutputPort());
        self.visibleData.SetInputConnection(1,selection.GetOutputPort());

        PolyData2UnstructuredGrid=vtk.vtkGeometryFilter()
        PolyData2UnstructuredGrid.SetInputConnection(self.visibleData.GetOutputPort());
        PolyLines2Edges=vtk.vtkTriangleFilter()
        PolyLines2Edges.SetInputConnection(PolyData2UnstructuredGrid.GetOutputPort());
        self.allDataInput=PolyLines2Edges

        PolyData2UnstructuredGrid.ReleaseDataFlagOn()
        self.visibleData.ReleaseDataFlagOn()
        self.validData.ReleaseDataFlagOn()
        self.allDataInput.ReleaseDataFlagOn()
        
        self.allData.AddInputConnection(0,self.allDataInput.GetOutputPort())
        self.allData.Modified()

        self.PointLocator=vtk.vtkPointLocator()


    @classmethod
    def update_VisEngine(self, LineStyle=None, PointStyle=None, vis3D=None, showNodes=None):
        self.restrictVOI(self.doRestrictVOI[0])
        if not hasattr(self,'actor'):
            return
        if self.actor.GetNumberOfItems()==0:
            return
            
        if LineStyle==None:
            LineStyle=self.DefaultLineStyle[0]
        if PointStyle==None:
            PointStyle=self.DefaultPointStyle[0]
        if vis3D==None:
            vis3D=self.Defaultvis3D[0]
        if showNodes==None:
            showNodes=self.DefaultshowNodes[0]
            
        LineActors=[self.activeactor.GetItemAsObject(0),self.actor.GetItemAsObject(0)]            
        SphereActors=[self.activeactor.GetItemAsObject(1),self.actor.GetItemAsObject(1)]            
        TubeActors=[self.activeactor.GetItemAsObject(2),self.actor.GetItemAsObject(2)]            
            
        if LineStyle=='lines':
            for actor in LineActors:            
                if not (actor.GetProperty().GetLineWidth()==\
                    self.DefaultLineWidth[0]):
                    actor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
                    actor.Modified()
                if not actor.GetVisibility():
                    actor.VisibilityOn()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOn()
                    actor.Modified()
            for actor in TubeActors:
                if actor.GetVisibility():
                    actor.VisibilityOff()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOff()
                    actor.Modified()
        if LineStyle=='tubes':
            for actor in TubeActors:
                if vis3D=="on": 
                    if not actor.GetProperty().GetShading():
                        actor.GetProperty().ShadingOn()
                        actor.Modified()
                    if not actor.GetProperty().GetLighting():
                        actor.GetProperty().LightingOn()
                        actor.Modified()
                else:               
                    if actor.GetProperty().GetShading():
                        actor.GetProperty().ShadingOff()
                        actor.Modified()
                    if actor.GetProperty().GetLighting():
                        actor.GetProperty().LightingOff()
                        actor.Modified()
                if not actor.GetVisibility():
                    actor.VisibilityOn()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOn()
                    actor.Modified()

            for actor in LineActors:
                if actor.GetVisibility():
                    actor.VisibilityOff()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOff()
                    actor.Modified()
                
        if PointStyle=='points':
            for actor in LineActors:
                if showNodes:
                    if not actor.GetVisibility():
                        actor.VisibilityOn()
                        actor.Modified()
                    if not actor.GetPickable():
                        actor.PickableOn()
                        actor.Modified()
                    if not actor.GetProperty().GetPointSize()==self.DefaultPointSize[0]:
                        actor.GetProperty().SetPointSize(self.DefaultPointSize[0])
                        actor.Modified()
                else:
                    if not actor.GetProperty().GetPointSize()==0:
                        actor.GetProperty().SetPointSize(0)
                        actor.Modified()

            for actor in SphereActors:
                if actor.GetVisibility():
                    actor.VisibilityOff()
                    actor.Modified()
                if actor.GetPickable():
                    actor.PickableOff()
                    actor.Modified()
            
        if PointStyle=='spheres':
            for actor in SphereActors:
                if vis3D=="on": 
                    if not actor.GetProperty().GetShading():
                        actor.GetProperty().ShadingOn()
                        actor.Modified()
                    if not actor.GetProperty().GetLighting():
                        actor.GetProperty().LightingOn()
                        actor.Modified()
                else:               
                    if actor.GetProperty().GetShading():
                        actor.GetProperty().ShadingOff()
                        actor.Modified()
                    if actor.GetProperty().GetLighting():
                        actor.GetProperty().LightingOff()
                        actor.Modified()
    
                if showNodes:
                    if not actor.GetVisibility():
                        actor.VisibilityOn()
                        actor.Modified()
                    if not actor.GetPickable():
                        actor.PickableOn()
                        actor.Modified()
                else:
                    if actor.GetVisibility():
                        actor.VisibilityOff()
                        actor.Modified()
                    if not actor.GetPickable():
                        actor.PickableOff()
                        actor.Modified()
            
            for actor in LineActors:
                if not actor.GetProperty().GetPointSize()==0:
                    actor.GetProperty().SetPointSize(0)
                    actor.Modified()

    @classmethod
    def restrictVOI(self,dorestrict=None):
        if dorestrict==None:
            dorestrict=self.doRestrictVOI[0]
        if dorestrict:
            if self.initialized[0]:
                self.VOIFilter.SetInputConnection(self.allData.GetOutputPort())
                self.activeVOIFilter.SetInputConnection(self.activeData.GetOutputPort())
                self.NodeGlyphs.SetInputConnection(self.VOIFilter.GetOutputPort())
                self.activeNodeGlyphs.SetInputConnection(self.activeVOIFilter.GetOutputPort())
                self.GeometryFilter.SetInputConnection(self.VOIFilter.GetOutputPort())        
                self.activeGeometryFilter.SetInputConnection(self.activeVOIFilter.GetOutputPort())    
            self.doRestrictVOI[0]=True
        else:
            if self.initialized[0]:
                self.VOIFilter.RemoveInputConnection(0,self.allData.GetOutputPort())
                self.activeVOIFilter.RemoveInputConnection(0,self.activeData.GetOutputPort())
                self.NodeGlyphs.SetInputConnection(self.allData.GetOutputPort())
                self.activeNodeGlyphs.SetInputConnection(self.activeData.GetOutputPort())        
                self.GeometryFilter.SetInputConnection(self.allData.GetOutputPort())        
                self.activeGeometryFilter.SetInputConnection(self.activeData.GetOutputPort())        
            self.doRestrictVOI[0]=False
        if self.initialized[0]:
                self.NodeGlyphs.Modified()
                self.GeometryFilter.Modified()
                self.VOIFilter.Modified()
                self.activeNodeGlyphs.Modified()
                self.activeGeometryFilter.Modified()
                self.activeVOIFilter.Modified()

    def init_VisPipeline(self):
        if self.initialized[0]:
            return
            
        self.setupNodeSource()
        self.LUT.Allocate(self.maxNofColors[0])
        self.LUT.SetNumberOfTableValues(2)
        self.LUT.SetTableRange(0,1)
        self.LUT.SetTableValue(deleteColor,[0.0,0.0,0.0,0.0])
        self.LUT.SetTableValue(selectColor,[1.0,0.0,0.0,1.0])
        
        self.GeometryFilter.ReleaseDataFlagOn()
        self.activeGeometryFilter.ReleaseDataFlagOn()  
        self.NodeGlyphs.ReleaseDataFlagOn()
        self.activeNodeGlyphs.ReleaseDataFlagOn()
        self.Stripper.ReleaseDataFlagOn()
        self.activeStripper.ReleaseDataFlagOn()
        self.BranchTubeFilter.ReleaseDataFlagOn()
        self.activeBranchTubeFilter.ReleaseDataFlagOn()       
        
        self.VOIFilter.ExtentClippingOn()
        self.VOIFilter.MergingOff()
#        self.VOIFilter.PassThroughCellIdsOn()

        self.activeVOIFilter.ExtentClippingOn() 	        
        self.activeVOIFilter.MergingOff()
#        self.activeVOIFilter.PassThroughCellIdsOn()
        
        if self.doRestrictVOI[0]:
            self.VOIFilter.SetInputConnection(self.allData.GetOutputPort())
            self.activeVOIFilter.SetInputConnection(self.activeData.GetOutputPort())
            self.NodeGlyphs.SetInputConnection(self.VOIFilter.GetOutputPort())
            self.activeNodeGlyphs.SetInputConnection(self.activeVOIFilter.GetOutputPort())
            self.GeometryFilter.SetInputConnection(self.VOIFilter.GetOutputPort())        
            self.activeGeometryFilter.SetInputConnection(self.activeVOIFilter.GetOutputPort())    
        else:
            self.VOIFilter.RemoveInputConnection(0,self.allData.GetOutputPort())
            self.VOIFilter.Modified()
            self.activeVOIFilter.RemoveInputConnection(0,self.activeData.GetOutputPort())
            self.activeVOIFilter.Modified()
            self.NodeGlyphs.SetInputConnection(self.allData.GetOutputPort())
            self.activeNodeGlyphs.SetInputConnection(self.activeData.GetOutputPort())        
            self.GeometryFilter.SetInputConnection(self.allData.GetOutputPort())        
            self.activeGeometryFilter.SetInputConnection(self.activeData.GetOutputPort())        

#        self.NodeGlyphs.SetInputConnection(self.allData.GetOutputPort())
#        self.NodeGlyphs.SetInputConnection(self.VOIFilter.GetOutputPort())
        self.NodeGlyphs.SetSourceConnection(self.NodeSource.GetOutputPort())
        self.NodeGlyphs.SetScaleFactor(1.0)
        self.NodeGlyphs.SetScaleModeToScaleByScalar()
        self.NodeGlyphs.GeneratePointIdsOn()        

#        self.activeNodeGlyphs.SetInputConnection(self.activeData.GetOutputPort())
#        self.activeNodeGlyphs.SetInputConnection(self.activeVOIFilter.GetOutputPort())
        self.activeNodeGlyphs.SetSourceConnection(self.NodeSource.GetOutputPort())
        self.activeNodeGlyphs.SetScaleFactor(1.0)
        self.activeNodeGlyphs.SetScaleModeToScaleByScalar()
        self.activeNodeGlyphs.GeneratePointIdsOn()

        self.Stripper.SetInputConnection(self.GeometryFilter.GetOutputPort())
        self.activeStripper.SetInputConnection(self.activeGeometryFilter.GetOutputPort())
       
        self.BranchTubeFilter.SetNumberOfSides(6)
        self.BranchTubeFilter.CappingOn()
        self.BranchTubeFilter.SetRadius(self.DefaultRadius[0])
        self.BranchTubeFilter.SetRadiusFactor(1.0)
        self.BranchTubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.BranchTubeFilter.SetInputConnection(self.Stripper.GetOutputPort())        

        self.activeBranchTubeFilter.SetNumberOfSides(6)
        self.activeBranchTubeFilter.CappingOn()
        self.activeBranchTubeFilter.SetRadius(self.DefaultRadius[0])
        self.activeBranchTubeFilter.SetRadiusFactor(1.0)
        self.activeBranchTubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.activeBranchTubeFilter.SetInputConnection(self.activeStripper.GetOutputPort())        
        
        LineMapper= vtk.vtkDataSetMapper()
        LineMapper.SetUseLookupTableScalarRange(1);
        LineMapper.SetColorModeToMapScalars()
        LineMapper.SetScalarModeToUsePointFieldData()
        LineMapper.SetInputConnection(self.Stripper.GetOutputPort()) 
        LineMapper.SelectColorArray("PointColor"); 
        LineMapper.SetLookupTable(self.LUT);    

        activeLineMapper= vtk.vtkDataSetMapper()
        activeLineMapper.SetUseLookupTableScalarRange(1);
        activeLineMapper.SetColorModeToMapScalars()
        activeLineMapper.SetScalarModeToUsePointFieldData()
        activeLineMapper.SetInputConnection(self.activeStripper.GetOutputPort()) 
        activeLineMapper.SelectColorArray("PointColor"); 
        activeLineMapper.SetLookupTable(self.LUT);    

        GlyphMapper= vtk.vtkDataSetMapper()
        GlyphMapper.SetUseLookupTableScalarRange(1);
        GlyphMapper.SetColorModeToMapScalars()
        GlyphMapper.SetScalarModeToUsePointFieldData()
        GlyphMapper.SetInputConnection(self.NodeGlyphs.GetOutputPort()) 
        GlyphMapper.SelectColorArray("PointColor"); 
        GlyphMapper.SetLookupTable(self.LUT);    

        activeGlyphMapper= vtk.vtkDataSetMapper()
        activeGlyphMapper.SetUseLookupTableScalarRange(1);
        activeGlyphMapper.SetColorModeToMapScalars()
        activeGlyphMapper.SetScalarModeToUsePointFieldData()
        activeGlyphMapper.SetInputConnection(self.activeNodeGlyphs.GetOutputPort()) 
        activeGlyphMapper.SelectColorArray("PointColor"); 
        activeGlyphMapper.SetLookupTable(self.LUT);    

        TubeMapper= vtk.vtkDataSetMapper()
        TubeMapper.SetUseLookupTableScalarRange(1);
        TubeMapper.SetColorModeToMapScalars()
        TubeMapper.SetScalarModeToUsePointFieldData()
        TubeMapper.SetInputConnection(self.BranchTubeFilter.GetOutputPort()) 
        TubeMapper.SelectColorArray("PointColor"); 
        TubeMapper.SetLookupTable(self.LUT);    

        activeTubeMapper= vtk.vtkDataSetMapper()
        activeTubeMapper.SetUseLookupTableScalarRange(1);
        activeTubeMapper.SetColorModeToMapScalars()
        activeTubeMapper.SetScalarModeToUsePointFieldData()
        activeTubeMapper.SetInputConnection(self.activeBranchTubeFilter.GetOutputPort()) 
        activeTubeMapper.SelectColorArray("PointColor"); 
        activeTubeMapper.SetLookupTable(self.LUT);    

        self.mapper.AddItem(LineMapper)
        self.mapper.AddItem(GlyphMapper)
        self.mapper.AddItem(TubeMapper)

        self.activemapper.AddItem(activeLineMapper)
        self.activemapper.AddItem(activeGlyphMapper)
        self.activemapper.AddItem(activeTubeMapper)

        LineActor =vtk.vtkActor()
        LineActor.Type="LineActor"
        LineActor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
        LineActor.GetProperty().SetPointSize(self.DefaultPointSize[0])
        LineActor.SetMapper(LineMapper)       

        activeLineActor =vtk.vtkActor()
        activeLineActor.Type="LineActor"
        activeLineActor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
        activeLineActor.GetProperty().SetPointSize(self.DefaultPointSize[0])
        activeLineActor.SetMapper(activeLineMapper)       

        GlyphActor =vtk.vtkActor()
        GlyphActor.Type="GlyphActor"
        GlyphActor.SetMapper(GlyphMapper)       
        GlyphActor.GetProperty().SetSpecular(.3)
        GlyphActor.GetProperty().SetSpecularPower(30)

        activeGlyphActor =vtk.vtkActor()
        activeGlyphActor.Type="GlyphActor"
        activeGlyphActor.SetMapper(activeGlyphMapper)       
        activeGlyphActor.GetProperty().SetSpecular(.3)
        activeGlyphActor.GetProperty().SetSpecularPower(30)

        TubeActor =vtk.vtkActor()
        TubeActor.Type="TubeActor"
        TubeActor.SetMapper(TubeMapper)       
        TubeActor.GetProperty().SetSpecular(.3)
        TubeActor.GetProperty().SetSpecularPower(30)

        activeTubeActor =vtk.vtkActor()
        activeTubeActor.Type="TubeActor"
        activeTubeActor.SetMapper(activeTubeMapper)       
        activeTubeActor.GetProperty().SetSpecular(.3)
        activeTubeActor.GetProperty().SetSpecularPower(30)
        
        self.actor.AddItem(LineActor)
        self.actor.AddItem(GlyphActor)
        self.actor.AddItem(TubeActor)

        self.activeactor.AddItem(activeLineActor)
        self.activeactor.AddItem(activeGlyphActor)
        self.activeactor.AddItem(activeTubeActor)
        
        self.update_VisEngine()
        self.initialized[0]=True

    
    def select_node(self,nodeId):
        self.select()
        if nodeId==None:
            return None
        nodeId=int(nodeId)
        selectionlist=self.nodeSelection.node.GetSelectionList()
        selIdx=selectionlist.LookupValue(nodeId)
        if selIdx==-1:
            selectionlist.InsertNextValue(nodeId);
            selectionlist.DataChanged() #somehow this is not done internally
        self.nodeSelection.node.Modified()
            
    def unselect_node(self,nodeIds):
        self.unselect()
        if not nodeIds.__class__.__name__=='set':
            if not nodeIds.__class__.__name__=='list':
                nodeIds=[nodeIds]
            nodeIds=set(nodeIds)
        selectionlist=self.nodeSelection.node.GetSelectionList()
        if selectionlist==None or selectionlist.GetNumberOfTuples()==0:
            return
        selectionlist=list(vtk_to_numpy(selectionlist))
        changed=False
        for nodeId in nodeIds:
            if nodeId==None:
                continue           
            nodeId=int(nodeId)
            if nodeId in selectionlist:
                selectionlist.remove(nodeId)
                changed=True
        if changed:
            selectionlist=numpy_to_vtk(selectionlist, deep=1, array_type=vtk.VTK_ID_TYPE)
            self.nodeSelection.node.SetSelectionList(selectionlist)
            self.nodeSelection.node.Modified()

    def set_nodes(self,points,NodeID=None):
        self.data.SetPoints(points)        

        NNodes=points.GetNumberOfPoints()

        colors=self.colorIdx*np.ones([NNodes,1],dtype=np.float)
        PointColor=self.data.GetPointData().GetArray("PointColor")
        PointColor.DeepCopy(numpy_to_vtk(colors, deep=1, array_type=vtk.VTK_FLOAT))
        PointColor.Modified()

        radius=self.DefaultRadius[0]*np.ones([NNodes,1],dtype=np.float)
        Radius=self.data.GetPointData().GetArray("Radius")
        Radius.DeepCopy(numpy_to_vtk(radius, deep=1, array_type=vtk.VTK_FLOAT))
        Radius.Modified()
        
        neuronID=self.NeuronID*np.ones([NNodes,1],dtype=np.float32)
        NeuronID=self.data.GetPointData().GetArray("NeuronID")
        NeuronID.DeepCopy(numpy_to_vtk(neuronID, deep=1, array_type=vtk.VTK_FLOAT))
        NeuronID.Modified()
        
        NodeIDArray=self.data.GetPointData().GetArray("NodeID") 
        if NodeID==None:
            NodeID=np.array(range(NNodes),dtype=np.int)      
        
        if not NodeID.__class__.__name__=='vtkIdTypeArray':
            if not (NodeID.__class__.__name__=='vtkIntArray' or NodeID.__class__.__name__=='vtkLongArray' or NodeID.__class__.__name__=='vtkLongLongArray'): 
                if not NodeID.__class__.__name__=='ndarray':
                    NodeID=np.array([NodeID],dtype=np.int)
                
                NodeID=numpy_to_vtk(NodeID, deep=1, array_type=vtk.VTK_ID_TYPE)
            elif NodeID.GetNumberOfTuples()>0:
                NodeID=numpy_to_vtk(vtk_to_numpy(NodeID), deep=1, array_type=vtk.VTK_ID_TYPE)
        NodeIDArray.DeepCopy(NodeID)
        NodeIDArray.Modified()
        
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        DeletedNodes.DeepCopy(numpy_to_vtk(np.zeros([NNodes,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        DeletedNodes.Modified()

        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
        VisibleNodes.DeepCopy(numpy_to_vtk(np.ones([NNodes,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        VisibleNodes.Modified()
        
        vertex=np.array([np.ones(NNodes),range(NNodes)],dtype=np.int).reshape(2,NNodes).transpose().reshape(2*NNodes,)
        vertex=numpy_to_vtk(vertex, deep=1, array_type=vtk.VTK_ID_TYPE)
        Vertices=vtk.vtkCellArray()
        Vertices.SetCells(NNodes,vertex)
        self.data.SetVerts(Vertices)
        
        self.data.BuildCells()
        self.data.Update()
        self.data.BuildLinks()
        self.data.Modified()
                
        if NodeID.GetNumberOfTuples()==0:
            return -1,-1
        if NNodes==1:
            return NodeID.GetValue(0),0
        NodeID=vtk_to_numpy(NodeID)
        NodeID=NodeID.tolist()
        return NodeID,range(NNodes)  #last nodeId and last pointIdx

    def add_node(self,newpoint):
        if 'l' in self.flags: #locked object
            return None,None
        self.new_singleactive(self)
        classname=newpoint.__class__.__name__
        if not classname=='vtkPoints':
            points=vtk.vtkPoints()
            points.InsertNextPoint(newpoint)
        else:
            points=newpoint
        if not self.data.GetPoints():
            nodeId,pointIdx=self.set_nodes(points,None)
            return nodeId,pointIdx                
        elif self.data.GetNumberOfPoints()==0:
            nodeId,pointIdx=self.set_nodes(points,None)
            return nodeId,pointIdx                

        NPoints2Add=points.GetNumberOfPoints()
        allPoints=self.data.GetPoints()
        NeuronID=self.data.GetPointData().GetArray("NeuronID")
        PointColor=self.data.GetPointData().GetArray("PointColor")
        Radius=self.data.GetPointData().GetArray("Radius")
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
        NodeID=self.data.GetPointData().GetArray("NodeID")
        if NodeID.GetNumberOfTuples()>0:
            nodeId=np.int(NodeID.GetMaxNorm())
        else:
            nodeId=-1
        
        insertedPointIdx=list()
        insertedNodeIds=list()
        for ipoint in range(NPoints2Add):
            pointIdx=allPoints.InsertNextPoint(points.GetPoint(ipoint))
            NeuronID.InsertNextValue(np.float32(self.NeuronID))
            PointColor.InsertNextValue(self.colorIdx)
            Radius.InsertNextValue(self.DefaultRadius[0])
            DeletedNodes.InsertNextValue(0)
            VisibleNodes.InsertNextValue(1)
            nodeId+=1
            NodeID.InsertNextValue(nodeId)

            vertex=vtk.vtkIdList()
            vertex.InsertNextId(pointIdx)
            self.data.InsertNextCell(vtk.VTK_VERTEX,vertex)

            insertedPointIdx.append(pointIdx)
            insertedNodeIds.append(nodeId)

        self.data.Modified()    
        self.data.BuildCells()
        self.data.Update()
        self.data.BuildLinks()
        self.data.Modified()    
        if NPoints2Add>1:
            return insertedNodeIds,insertedPointIdx
        else:
            return nodeId,pointIdx

    def add_branch(self,newbranch,type='vtkIdList',updateflag=1):
#        if 'l' in self.flags: #locked object
#            return
#        self.new_singleactive(self)
        if type=='vtkIdList':
            if newbranch.GetNumberOfIds()>2:
                ibranch=self.data.InsertNextCell(vtk.VTK_POLY_LINE,newbranch)
            elif newbranch.GetNumberOfIds()==2:
                ibranch=self.data.InsertNextCell(vtk.VTK_LINE,newbranch)
            elif newbranch.GetNumberOfIds()==1 :
                ibranch=self.data.InsertNextCell(vtk.VTK_VERTEX,newbranch)
            else:
                return;
        elif type=='Verts':
            self.data.SetVerts(newbranch)
            ibranch=self.data.GetNumberOfCells()-1
        elif type=='Lines':
            self.data.SetLines(newbranch)
            ibranch=self.data.GetNumberOfCells()-1
        else:
            return
                
                                       
        self.data.Modified()
        if updateflag:
            self.data.BuildCells()
            self.data.Update()
            self.data.BuildLinks();
            self.data.Modified()
        return ibranch

    #calculate the rotation minimizing frames      
    def RMF(self,nodeId=-1,nextNodeId=-1,cDir=None,vDir=None):
        pIdx=self.nodeId2pointIdx(nodeId)
        pNextIdx=self.nodeId2pointIdx(nextNodeId)
        NPts=self.data.GetNumberOfPoints()
        if not (pIdx>-1 and pIdx<NPts):
            hDir=None
            return cDir, vDir, hDir
            
        def find_path(pId0,pId1,depth):
            if depth<1:
                return [[pId0]]
            #finds all connected pId-1 not equal pId1
            candidates=list()        
            CellList=vtk.vtkIdList()
            self.data.GetPointCells(pId0,CellList)
            nodeDegree=CellList.GetNumberOfIds()
            for icell in range(nodeDegree):
                cellIdx=CellList.GetId(icell)
                CellPointIds=self.data.GetCell(cellIdx).GetPointIds()
                pCellPointIdx=CellPointIds.IsId(pId0)
                if pCellPointIdx>0:
                    candidateIdx=CellPointIds.GetId(pCellPointIdx-1)
                    if not candidateIdx==pId1:
                        path=find_path(candidateIdx,pId0,depth-1)
                        candidates.extend(path)
                if pCellPointIdx<(CellPointIds.GetNumberOfIds()-1):
                    candidateIdx=CellPointIds.GetId(pCellPointIdx+1)
                    if not candidateIdx==pId1:
                        path=find_path(candidateIdx,pId0,depth-1)
                        candidates.extend(path)
                    
            if not candidates:
                return [[pId0]]
            else:
                for candidate in candidates:
                    candidate.append(pId0)
            return candidates
        
        if (pNextIdx>-1 and pNextIdx<NPts):
            paths=find_path(pIdx,pNextIdx,1)
            for path in paths:
                path.append(pNextIdx)
                
        else:
            paths=find_path(pIdx,pNextIdx,2)
        
        if not paths:
            return
            
        angles=list()
        for path in paths:
            NPts=path.__len__()
            if NPts==3:
                v0=[0,0,0]
                v1=[0,0,0]
                vtk.vtkMath.Subtract(self.data.GetPoint(path[1]),self.data.GetPoint(path[0]),v0)
                vtk.vtkMath.Subtract(self.data.GetPoint(path[2]),self.data.GetPoint(path[1]),v1)
                vtk.vtkMath.Normalize(v0)
                vtk.vtkMath.Normalize(v1)
                angles.append(vtk.vtkMath.Dot(v0,v1))
            elif NPts==2:
                angles.append(-1.1)
            else:
                angles.append(-1.2)
        
        whichPath=np.argmax(angles)   
        NPts=paths[whichPath].__len__()
        if NPts>2:
            pathPt=np.array([self.data.GetPoint(paths[whichPath][0]),self.data.GetPoint(paths[whichPath][1]),self.data.GetPoint(paths[whichPath][2])],dtype=np.float)         
        elif NPts>1:
            pathPt=np.array([self.data.GetPoint(paths[whichPath][0]),self.data.GetPoint(paths[whichPath][1])],dtype=np.float)         
        elif NPts>0:
#            pathPt=np.array([self.data.GetPoint(paths[whichPath][0])],dtype=np.float)         
            if (invalidvector(cDir) or invalidvector(vDir)):
                hDir=None
            else:
                hDir=np.cross(cDir,vDir)
            return cDir, vDir, hDir
        else:
            return            
        #Compute first normal. All "new" normals try to point in the same 
        #direction.
        p=pathPt[0]    
        pNext=pathPt[1]  
                
        sPrev=pNext-p
        sNext=sPrev.copy()        
        if invalidvector(cDir):
            tNext=sNext.copy()
        else:
            tNext=cDir.copy()
            
        length=vtk.vtkMath.Normalize(tNext)
        if length<1.0e-5:
            print "Coincident points in polyline...can't compute normals"
            if (invalidvector(cDir) or invalidvector(vDir)):
                hDir=None
            else:
                hDir=np.cross(cDir,vDir)
            return cDir, vDir, hDir

        
        if invalidvector(vDir):
            #the following logic will produce a normal orthogonal
            #to the first line segment. If we have three points
            #we use special logic to select a normal orthogonal
            #to the first two line segments
            foundNormal=0    
            if NPts > 2:
                #Look at the line segments (0,1), (ipt-1, ipt)
                #until a pair which meets the following criteria
                #is found: ||(0,1)x(ipt-1,ipt)|| > 1.0E-3.
                #This is used to eliminate nearly parallel cases.
                ftmp=pathPt[2]-pathPt[1]
                
                length=vtk.vtkMath.Normalize(ftmp)
                if length>1.0e-5:
                    #now the starting normal should simply be the cross product
                    #in the follvtk.vtkStripper()owing if statement we check for the case where
                    #the two segments are parallel 
                    normal=np.cross(tNext,ftmp)
                    length=vtk.vtkMath.Normalize(normal)
                    if length>0.001:
                        foundNormal = 1;
            if (NPts<=2 or (not foundNormal)):
                print "Normal not found..."
                normal=np.array([0,0,0],np.float)
                for i in range(0,3):
                    if sNext[i] != 0.0:
                        normal[(i+2)%3] = 0.0;
                        normal[(i+1)%3] = 1.0;
                        normal[i] = -sNext[(i+1)%3]/sNext[i];
                        break
                length=vtk.vtkMath.Normalize(normal)
            cDir=tNext.copy()
            vDir=normal.copy()
            hDir=np.cross(cDir,vDir)
            length=vtk.vtkMath.Normalize(hDir)
        else:
            normal=vDir
            
#        if NPts<3:
#            hDir=np.cross(cDir,vDir)
#            length=vtk.vtkMath.Normalize(hDir)            
#            return cDir, vDir, hDir

        #Generate normals for new point by projecting previous normal
        tPrev=tNext.copy()
        sPrev=sNext.copy()        
        if NPts>2:
            p=pNext.copy()
            pNext=pathPt[2]  
            if all(p==pNext):
                hDir=np.cross(cDir,vDir)
                length=vtk.vtkMath.Normalize(hDir)            
                return cDir, vDir, hDir    
            sNext=pNext-p
        else:
            1 #do not update sNext

        tNext=sNext+sPrev        
        length=vtk.vtkMath.Normalize(tNext)
        if length<1.0e-5:
            tNext=sNext.copy()
            vtk.vtkMath.Normalize(tNext)
        c1=vtk.vtkMath.Dot(sNext,sNext)
        if c1>0:
            normalL=normal-2.0/c1*vtk.vtkMath.Dot(sNext,normal)*sNext  
            tPrevL=tPrev-2.0/c1*vtk.vtkMath.Dot(sNext,tPrev)*sNext
        else:
            normalL=normal;
            tPrevL=tPrev;            
        v2=tNext-tPrevL
        c2=vtk.vtkMath.Dot(v2,v2)
        if c2>0:
            normal=normalL-2.0/c2*vtk.vtkMath.Dot(v2,normalL)*v2
        else:
            normal=normalL
        vtk.vtkMath.Normalize(normal)

        cDir=tNext.copy()
        vDir=normal.copy()
        hDir=np.cross(cDir,vDir)
        length=vtk.vtkMath.Normalize(hDir)
        
        return cDir, vDir, hDir
        
class synapse(objs):
    allclassnames=[]
    objtype="synapse"
    maxNofColors=[2000]
    DefaultLineStyle=["lines"]
    DefaultPointStyle=["points"]
    Defaultvis3D=["off"]
    DefaultshowNodes=[1]
    DefaultRadius=[25]
    DefaultLineWidth=[1]
    DefaultPointSize=[5]
    DefaultStartNodeOnly=[0]
    
    LUTinitialized=[False] #LUT should be set up by the synapse browser
    initialized=[False]
    VisEngine_started=[False]
    
    viewports=list()

    allData= vtk.vtkAppendFilter()
    TagGlyphs= vtk.vtkGlyph3D()
    GeometryFilter=vtk.vtkGeometryFilter()
    Stripper=vtk.vtkStripper()
    TagTubeFilter = vtk.vtkTubeFilter()
    mapper=vtk.vtkMapperCollection()
    actor=vtk.vtkActorCollection()
    LUT=vtk.vtkLookupTable()

    TagSource = vtk.vtkSphereSource()
    instancecount=[0]

    def __init__(self,parentItem,parentID,color):
        self.reset()
        self.NeuronID=parentID
        self.instancecount[0]+=1

        self.init_VisPipeline()
        
        self.setup_color(color)

        self.setup_VisEngine()
        
        self.nodeSelection=NodeSelection(self.TagSource,self.actor)
        self.nodeSelection.add(self.data,vtk.vtkSelectionNode.POINT,\
            vtk.vtkSelectionNode.INDICES,True)

        self.appendItem(parentItem)

    def change_color(self,color=None):
        return #color of synapses is controled by the synapse browser
        
    def setupTagSource(self):
        Sphere = self.TagSource
        Sphere.SetPhiResolution(6)
        Sphere.SetThetaResolution(6)
        Sphere.SetRadius(1.33)
        Sphere.Update()
                
    def init_VisPipeline(self):
        if self.initialized[0]:
            return
            
        self.setupTagSource()
        
        if not self.LUTinitialized[0]:
            self.LUT.Allocate(self.maxNofColors[0])
            #we allow maximally 12 classes, might want to change this later
            #the default color will be idx=14 (starting from 0) after calling setup_color
            self.LUT.SetNumberOfTableValues(2+12)
            self.LUT.SetTableRange(0,1)
            self.LUT.Build()
            self.LUTinitialized[0]=True

        self.LUT.SetTableValue(deleteColor,[0.0,0.0,0.0,0.0])
        self.LUT.SetTableValue(selectColor,[1.0,0.0,0.0,1.0])
                
        self.TagGlyphs.SetInputConnection(self.allData.GetOutputPort())
        self.TagGlyphs.SetSourceConnection(self.TagSource.GetOutputPort())
        self.TagGlyphs.SetScaleFactor(1.0)
        self.TagGlyphs.SetScaleModeToScaleByScalar()
        self.TagGlyphs.GeneratePointIdsOn()
        
        self.GeometryFilter.SetInputConnection(self.allData.GetOutputPort())
        self.Stripper.SetInputConnection(self.GeometryFilter.GetOutputPort())

        self.TagTubeFilter.SetNumberOfSides(6)
        self.TagTubeFilter.SetRadius(self.DefaultRadius[0])
        self.TagTubeFilter.SetRadiusFactor(1.0)
        self.TagTubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.TagTubeFilter.SetInputConnection(self.Stripper.GetOutputPort())

        self.TagGlyphs.ReleaseDataFlagOn()
        self.GeometryFilter.ReleaseDataFlagOn()
        self.Stripper.ReleaseDataFlagOn()
        self.TagTubeFilter.ReleaseDataFlagOn()

        LineMapper= vtk.vtkDataSetMapper()
        LineMapper.SetScalarModeToUsePointFieldData()
        LineMapper.SetInputConnection(self.allData.GetOutputPort()) 
        LineMapper.SelectColorArray("PointColor"); 
        LineMapper.SetLookupTable(self.LUT);    
        LineMapper.SetUseLookupTableScalarRange(1);

        GlyphMapper= vtk.vtkDataSetMapper()
        GlyphMapper.SetScalarModeToUsePointFieldData()
        GlyphMapper.SetInputConnection(self.TagGlyphs.GetOutputPort()) 
        GlyphMapper.SelectColorArray("PointColor"); 
        GlyphMapper.SetLookupTable(self.LUT);    
        GlyphMapper.SetUseLookupTableScalarRange(1);

        TubeMapper= vtk.vtkDataSetMapper()
        TubeMapper.SetScalarModeToUsePointFieldData()
        TubeMapper.SetInputConnection(self.TagTubeFilter.GetOutputPort()) 
        TubeMapper.SelectColorArray("PointColor"); 
        TubeMapper.SetLookupTable(self.LUT);    
        TubeMapper.SetUseLookupTableScalarRange(1);

        self.mapper.AddItem(LineMapper)
        self.mapper.AddItem(GlyphMapper)
        self.mapper.AddItem(TubeMapper)
        
        LineActor =vtk.vtkActor()
        LineActor.Type="LineActor"
        LineActor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
        LineActor.GetProperty().SetPointSize(self.DefaultPointSize[0])
        LineActor.SetMapper(LineMapper)       

        GlyphActor =vtk.vtkActor()
        GlyphActor.SetMapper(GlyphMapper)       
        GlyphActor.GetProperty().SetSpecular(.3)
        GlyphActor.GetProperty().SetSpecularPower(30)

        TubeActor =vtk.vtkActor()
        TubeActor.SetMapper(TubeMapper)       
        TubeActor.GetProperty().SetSpecular(.3)
        TubeActor.GetProperty().SetSpecularPower(30)
        
        self.actor.AddItem(LineActor)
        self.actor.AddItem(GlyphActor)
        self.actor.AddItem(TubeActor)
        
        self.update_VisEngine()
        self.initialized[0]=True

    def setup_VisEngine(self):
        self.data = vtk.vtkPolyData()
        self.data.Allocate()

        ObjType = vtk.vtkStringArray()
        ObjType.SetName("ObjType")
        ObjType.SetNumberOfValues(1)
        ObjType.SetValue(0,self.objtype)        
        self.data.GetFieldData().AddArray(ObjType)

        PointColor = vtk.vtkFloatArray()
        PointColor.SetName("PointColor")
        PointColor.SetNumberOfComponents(1)
        self.data.GetPointData().AddArray(PointColor)

        NeuronID=vtk.vtkFloatArray()
        NeuronID.SetName("NeuronID")
        NeuronID.SetNumberOfComponents(1)
        self.data.GetPointData().AddArray(NeuronID)

        NodeID = vtk.vtkIdTypeArray()
        NodeID.SetName("NodeID")
        self.data.GetPointData().AddArray(NodeID)
 
        DeletedNodes=vtk.vtkUnsignedIntArray()
        DeletedNodes.SetName("DeletedNodes")
        self.data.GetPointData().AddArray(DeletedNodes)

        VisibleNodes=vtk.vtkUnsignedIntArray()
        VisibleNodes.SetName("VisibleNodes")
        self.data.GetPointData().AddArray(VisibleNodes)

        Radius=vtk.vtkFloatArray()
        Radius.SetName("Radius")
        self.data.GetPointData().AddArray(Radius)

        self.data.GetPointData().SetActivePedigreeIds("NodeID")
        self.data.GetPointData().SetScalars(Radius)
        self.data.SetPoints(vtk.vtkPoints())                 
        self.data.Modified()

        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("DeletedNodes")
        selection.AddThreshold(0,0)
        selection.Update()
        
        self.validData =vtk.vtkExtractSelection()    
        self.validData.SetInputConnection(0,self.data.GetProducerPort());
        self.validData.SetInputConnection(1,selection.GetOutputPort());

        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("VisibleNodes")
        selection.AddThreshold(1,1)
        selection.Update()        
        self.visibleData =vtk.vtkExtractSelection()
        self.visibleData.SetInputConnection(0,self.validData.GetOutputPort());
        self.visibleData.SetInputConnection(1,selection.GetOutputPort());
        self.allDataInput=self.visibleData

        self.visibleData.ReleaseDataFlagOn()
        self.validData.ReleaseDataFlagOn()
        self.allDataInput.ReleaseDataFlagOn()

        self.allData.AddInputConnection(0,self.allDataInput.GetOutputPort())
        self.allData.Modified()

        self.PointLocator=vtk.vtkPointLocator()

    def nodeId2tagIdx(self,nodeId):
        if nodeId==None:
            return None
        pointIdx=self.nodeId2pointIdx(nodeId)    
        if pointIdx<0:
            return None
        return self.pointIdx2tagIdx(pointIdx)
            
    def pointIdx2tagIdx(self,pointIdx):
        if pointIdx==None:
            return None
        self.data.Update()
        if pointIdx<0 or pointIdx>(self.data.GetNumberOfPoints()-1):
            print "Error: pointIdx out of range: ", pointIdx
            return None
        CellList=vtk.vtkIdList()
        self.data.GetPointCells(pointIdx,CellList)
        for icell in range(CellList.GetNumberOfIds()):
            cellIdx=CellList.GetId(icell)
            #the tagIdx corresponds to the cellIdx of the one cell containing the connectivity, which should be the only one having VTK_POLY_LINE type.
            #other cells should have type VTK_VERTEX
            if self.data.GetCell(cellIdx).GetCellType()==vtk.VTK_POLY_LINE:
                return cellIdx
        return None
        
    def tagIdx2nodeId(self,tagIdx):
        if tagIdx==None:
            return None
        self.data.Update()
        if tagIdx<0 or tagIdx>(self.data.GetNumberOfCells()-1):
            print "Error: tagIdx out of range: ", tagIdx
            return None

        PointIds=self.data.GetCell(tagIdx).GetPointIds()
        return [self.pointIdx2nodeId(PointIds.GetId(pIdx)) for pIdx in range(3)]

    def SetRadius(self,radius):
        NNodes=self.data.GetNumberOfPoints()
        Radius=self.data.GetPointData().GetArray("Radius")
        Radius.DeepCopy(numpy_to_vtk(radius*np.ones([NNodes,1],dtype=np.float), deep=1, array_type=vtk.VTK_FLOAT))
        Radius.Modified()
        self.data.Modified()

    def set_tags(self,Points,Connections,NodeID=None):

        self.data.SetPoints(Points)         
        NPoints=Points.GetNumberOfPoints()

        colors=self.colorIdx*np.ones([NPoints,1],dtype=np.float)

        PointColor=self.data.GetPointData().GetArray("PointColor")
        PointColor.DeepCopy(numpy_to_vtk(colors, deep=1, array_type=vtk.VTK_FLOAT))
        PointColor.Modified()

        radius=self.DefaultRadius[0]*np.ones([NPoints,1],dtype=np.float)
        Radius=self.data.GetPointData().GetArray("Radius")
        Radius.DeepCopy(numpy_to_vtk(radius, deep=1, array_type=vtk.VTK_FLOAT))
        Radius.Modified()
        
        neuronID=self.NeuronID*np.ones([NPoints,1],dtype=np.float32)
        NeuronID=self.data.GetPointData().GetArray("NeuronID")
        NeuronID.DeepCopy(numpy_to_vtk(neuronID, deep=1, array_type=vtk.VTK_FLOAT))
        NeuronID.Modified()
        
        NodeIDArray=self.data.GetPointData().GetArray("NodeID") 
        if NodeID==None:
            NodeID=np.array(range(NPoints),dtype=np.int)      
        
        if not (NodeID.__class__.__name__=='vtkIdTypeArray' or NodeID.__class__.__name__=='vtkIntArray' or NodeID.__class__.__name__=='vtkLongArray' or NodeID.__class__.__name__=='vtkLongLongArray'): 
            if not NodeID.__class__.__name__=='ndarray':
                NodeID=np.array([NodeID],dtype=np.int)
            
            NodeID=numpy_to_vtk(NodeID, deep=1, array_type=vtk.VTK_ID_TYPE)
        else:
            if NodeID.GetNumberOfTuples()>0:
                NodeID=numpy_to_vtk(vtk_to_numpy(NodeID), deep=1, array_type=vtk.VTK_ID_TYPE)
            else:
                1
        NodeIDArray.DeepCopy(NodeID)
        NodeIDArray.Modified()
        
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        DeletedNodes.DeepCopy(numpy_to_vtk(np.zeros([NPoints,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        DeletedNodes.Modified()

        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
        VisibleNodes.DeepCopy(numpy_to_vtk(np.ones([NPoints,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        VisibleNodes.Modified()
        
        
        vertex=np.array([np.ones(NPoints),range(NPoints)],dtype=np.int).reshape(2,NPoints).transpose().reshape(2*NPoints,)
        vertex=numpy_to_vtk(vertex, deep=1, array_type=vtk.VTK_ID_TYPE)
        Vertices=vtk.vtkCellArray()
        Vertices.SetCells(NPoints,vertex)
        self.data.SetVerts(Vertices)
        
        self.data.SetLines(Connections)
        tagIdx=self.data.GetNumberOfCells()-1
        
        NSynapses=Connections.GetNumberOfCells()
        for isyn in range(NSynapses):
            if not self.item.__class__()==[]:
                nodeId=NodeIDArray.GetValue(isyn*3)
                tagItem=QtGui.QStandardItem("{0} {1}".format(self.objtype,int(nodeId/3.0)))            
                tagItem.nodeId=nodeId
                tagItem.objtype=self.objtype
                tagItem.neuronId=self.NeuronID
                
                self.item.appendRow(tagItem)

        self.data.BuildCells()
        self.data.Update()
        self.data.BuildLinks()
        self.data.Modified()    
        
        self.assign_classcolor()


        if NSynapses==0:
            tagIdx=-1
            nodeId=None
        else:
            nodeId=NodeID.GetValue(NPoints-3)
        return nodeId,tagIdx   

    def add_tag(self,Point0,Point1,Point2):
        if 'l' in self.flags: #locked object
            return
        if Point0==[]:
            return -1, -1
        if Point1==[]:
            return -1, -1
        if vtk.vtkMath.Distance2BetweenPoints(Point0,Point1)<1.0e-5:
            return -1, -1
            
        NodeID=self.data.GetPointData().GetArray("NodeID")

        if NodeID.GetNumberOfTuples()>0:
            maxNodeId=np.int(NodeID.GetMaxNorm())
            nodeIds=[maxNodeId+1,maxNodeId+2,maxNodeId+3]
        else:
            nodeIds=[0,1,2]
        
        Points=self.data.GetPoints()
        
        Point0Idx=Points.InsertNextPoint(Point0)
        NodeID.InsertNextValue(nodeIds[0])
        nodeId0=nodeIds[0]
        
        Point1Idx=Points.InsertNextPoint(Point1)
        NodeID.InsertNextValue(nodeIds[1])

        if Point2==[]:
            Point2Idx=Points.InsertNextPoint(Point0)
        else:
            Point2Idx=Points.InsertNextPoint(Point2)
        NodeID.InsertNextValue(nodeIds[2])
 
        PointColor=self.data.GetPointData().GetArray("PointColor")
        Radius=self.data.GetPointData().GetArray("Radius")
        NeuronID=self.data.GetPointData().GetArray("NeuronID")
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
        for ii in range(3):
            PointColor.InsertNextValue(self.colorIdx)
            Radius.InsertNextValue(self.DefaultRadius[0])
            NeuronID.InsertNextValue(np.float32(self.NeuronID))
            DeletedNodes.InsertNextValue(0)
            VisibleNodes.InsertNextValue(1)
       
        tempBranch=vtk.vtkIdList()
        tempBranch.InsertNextId(Point0Idx)
        tempBranch.InsertNextId(Point1Idx)
        tempBranch.InsertNextId(Point2Idx)

        vertex=vtk.vtkIdList()
        vertex.InsertNextId(Point0Idx)
        self.data.InsertNextCell(vtk.VTK_VERTEX,vertex)
        vertex=vtk.vtkIdList()
        vertex.InsertNextId(Point1Idx)
        self.data.InsertNextCell(vtk.VTK_VERTEX,vertex)
        vertex=vtk.vtkIdList()
        vertex.InsertNextId(Point2Idx)
        self.data.InsertNextCell(vtk.VTK_VERTEX,vertex)

        tagIdx=self.data.InsertNextCell(vtk.VTK_POLY_LINE,tempBranch)        
                
        if not self.item.__class__()==[]:
            tagItem=QtGui.QStandardItem("{0} {1}".format(self.objtype,int(nodeIds[0]/3.0)))
            tagItem.nodeId=nodeIds[0]
            tagItem.objtype=self.objtype
            tagItem.neuronId=self.NeuronID
            self.item.appendRow(tagItem)

        self.data.BuildCells()
        self.data.Update()
        self.data.BuildLinks()
        self.data.Modified()    
        return nodeId0,tagIdx

    def modify_tag(self,tagIdx,ptIdx,Point):
        if 'l' in self.flags: #locked object
            return None
        #ptIdx=0,1,2
        pointIds=self.data.GetCell(tagIdx).GetPointIds()
        if ptIdx>0:
            if vtk.vtkMath.Distance2BetweenPoints(Point,self.data.GetPoints().GetPoint(pointIds.GetId(ptIdx-1)))<1.0e-4:
                print "Coincident points"
                return None
        if ptIdx<(pointIds.GetNumberOfIds()-1):
            if vtk.vtkMath.Distance2BetweenPoints(Point,self.data.GetPoints().GetPoint(pointIds.GetId(ptIdx+1)))<1.0e-4:
                print "Coincident points"
                return None
        pointIdx=pointIds.GetId(ptIdx)
        self.data.GetPoints().SetPoint(pointIdx,Point)        
        nodeId=self.pointIdx2nodeId(pointIdx)
        self.data.Modified()    
        return nodeId

    def delete_tag(self,tagIdx):
        if 'l' in self.flags: #locked object
            return
        self.unselect_tag(tagIdx)
        
        for nodeId in self.tagIdx2nodeId(tagIdx):
            self.comments.delete(nodeId)

        PointIds=self.data.GetCell(tagIdx).GetPointIds()
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
#        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")

        tagCells=set([tagIdx])
        for ipoint in range(PointIds.GetNumberOfIds()):
            tempPtId=PointIds.GetId(ipoint)
            DeletedNodes.SetValue(tempPtId,1);
#            VisibleNodes.SetValue(tempPtId,0);
            #we have to consider all the vertex cells for deletion
            tempCells=vtk.vtkIdList()
            self.data.GetPointCells(tempPtId,tempCells)
            for icell in range(tempCells.GetNumberOfIds()):
                tagCells.add(tempCells.GetId(icell))
            
        DeletedNodes.Modified()
#        VisibleNodes.Modified()

        #Mark cells for deletion
        for cellId in tagCells:
            self.data.DeleteCell(cellId);
            
        if not self.item.__class__()==[]:
            self.item.removeRow(tagIdx)

        self.data.RemoveDeletedCells();
        self.data.Modified()
        self.data.Update()
        self.data.BuildCells()
        self.data.BuildLinks()
        self.data.Modified()
        
    def assign_classcolor(self,tagIdx=None):
        PointColor=self.data.GetPointData().GetArray("PointColor")
        if tagIdx==None:
            tagNodes=self.comments.get('all','class')
            for nodeId,node in tagNodes.iteritems():
                classname=node['class']
                if not classname in self.allclassnames:
                    continue
                classNo=self.allclassnames.index(classname)+1
                tagIdx=self.nodeId2tagIdx(nodeId)
                PointIds=self.data.GetCell(tagIdx).GetPointIds()
                for ipt in range(PointIds.GetNumberOfIds()):
                    pointIdx=PointIds.GetId(ipt)
                    PointColor.SetValue(pointIdx,classNo-1+2)
        else:
            if not tagIdx.__class__.__name__=='list':
                tagIdx=[tagIdx]
            NTags=tagIdx.__len__()
            for itag in range(NTags):
                nodeIds=self.tagIdx2nodeId(tagIdx[itag])
                if not nodeIds:
                    continue
                classname=self.comments.get(nodeIds[0],"class")

                if not classname in self.allclassnames:
                    continue
                classNo=self.allclassnames.index(classname)+1
                PointIds=self.data.GetCell(tagIdx[itag]).GetPointIds()
                for ipt in range(PointIds.GetNumberOfIds()):
                    pointIdx=PointIds.GetId(ipt)
                    PointColor.SetValue(pointIdx,classNo-1+2)
        PointColor.Modified()

    def select_tag(self,tagIdx):
        if tagIdx==None:
            return None
        selectionlist=self.nodeSelection.node.GetSelectionList()
        PointIds=self.data.GetCell(tagIdx).GetPointIds()
        for ipt in range(PointIds.GetNumberOfIds()):
            pointId=PointIds.GetId(ipt)
            selIdx=selectionlist.LookupValue(pointId)
            if selIdx==-1:
                selectionlist.InsertNextValue(pointId);
        selectionlist.DataChanged() #somehow this is not done internally
        self.nodeSelection.node.Modified()

    def unselect_tag(self,tagIndices):
        if not tagIndices.__class__.__name__=='set':
            if not tagIndices.__class__.__name__=='list':
                tagIndices=[tagIndices]
            tagIndices=set(tagIndices)

        selectionlist=self.nodeSelection.node.GetSelectionList()
        if selectionlist==None  or selectionlist.GetNumberOfTuples()==0:
            return
        selectionlist=list(vtk_to_numpy(selectionlist))
        changed=False
        for tagIdx in tagIndices:
            if tagIdx==None:
                continue
            PointIds=self.data.GetCell(tagIdx).GetPointIds()
            for ipt in range(PointIds.GetNumberOfIds()):
                pointId=PointIds.GetId(ipt)  
                if pointId in selectionlist:
                    selectionlist.remove(pointId)
                    changed=True
        if changed:
            selectionlist=numpy_to_vtk(selectionlist, deep=1, array_type=vtk.VTK_ID_TYPE)
            self.nodeSelection.node.SetSelectionList(selectionlist)
            self.nodeSelection.node.Modified()
        
    def get_closest_point(self,point):
        if not self.PointLocator.GetDataSet():
            self.visibleData.Update()
            DataSet=self.visibleData.GetOutput()
            if DataSet.GetNumberOfPoints()>0:
                self.PointLocator.SetDataSet(DataSet)
                self.PointLocator.BuildLocator()
            else:
                return
        DataSet=self.PointLocator.GetDataSet()
        DataSet.Update()
        pointIdx=self.PointLocator.FindClosestPoint(point)
        if pointIdx==-1 or pointIdx==None:
            return -1,-1,-1

        nodeId=DataSet.GetPointData().GetArray("NodeID").GetValue(pointIdx)
        tagIdx=self.nodeId2tagIdx(nodeId)
        return np.array(DataSet.GetPoint(pointIdx),dtype=np.float), nodeId, tagIdx


    def get_prev_obj(self,start_Idx=None,warparound=True):
        NCells=self.data.GetNumberOfCells()
        if start_Idx==None:
            start_Idx=NCells-1
        else:
            start_Idx-=1
        celllist=range(start_Idx,-1,-1)
        if warparound:
            celllist.extend(range(NCells-1,start_Idx,-1))
        for icell in celllist:
            if self.data.GetCell(icell).GetCellType()==vtk.VTK_POLY_LINE:
#                print icell
                return icell
        return None

    def get_next_obj(self,start_Idx=None,warparound=True):
        NCells=self.data.GetNumberOfCells()
        if start_Idx==None:
            start_Idx=0
        else:
            start_Idx+=1
        celllist=range(start_Idx,NCells)
        if warparound:
            celllist.extend(range(0,start_Idx))
        for icell in celllist:
            if self.data.GetCell(icell).GetCellType()==vtk.VTK_POLY_LINE:
#                print icell
                return icell
        return None

    @classmethod
    def update_VisEngine(self, LineStyle=None, PointStyle=None, vis3D=None, showNodes=None,startNodeOnly=None):
        if not hasattr(self,'actor'):
            return
        if self.actor.GetNumberOfItems()==0:
            return
        if LineStyle==None:
            LineStyle=self.DefaultLineStyle[0]
        if PointStyle==None:
            PointStyle=self.DefaultPointStyle[0]
        if vis3D==None:
            vis3D=self.Defaultvis3D[0]
        if showNodes==None:
            showNodes=self.DefaultshowNodes[0]
        if startNodeOnly==None:
            startNodeOnly=self.DefaultStartNodeOnly[0]
            
        if startNodeOnly:
            1
            
        if LineStyle=='lines':
            actor=self.actor.GetItemAsObject(0)
            if not (actor.GetProperty().GetLineWidth()==\
                self.DefaultLineWidth[0]):
                actor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
                actor.Modified()
            if not actor.GetVisibility():
                actor.VisibilityOn()
                actor.Modified()
            if not actor.GetPickable():
                actor.PickableOn()
                actor.Modified()
            actor=self.actor.GetItemAsObject(2)
            if actor.GetVisibility():
                actor.VisibilityOff()
                actor.Modified()
            if not actor.GetPickable():
                actor.PickableOff()
                actor.Modified()

        if LineStyle=='tubes':
            actor=self.actor.GetItemAsObject(2)
            if vis3D=="on": 
                if not actor.GetProperty().GetShading():
                    actor.GetProperty().ShadingOn()
                    actor.Modified()
                if not actor.GetProperty().GetLighting():
                    actor.GetProperty().LightingOn()
                    actor.Modified()
            else:               
                if actor.GetProperty().GetShading():
                    actor.GetProperty().ShadingOff()
                    actor.Modified()
                if actor.GetProperty().GetLighting():
                    actor.GetProperty().LightingOff()
                    actor.Modified()
            if not actor.GetVisibility():
                actor.VisibilityOn()
                actor.Modified()
            if not actor.GetPickable():
                actor.PickableOn()
                actor.Modified()

            actor=self.actor.GetItemAsObject(0)
            if actor.GetVisibility():
                actor.VisibilityOff()
                actor.Modified()
            if not actor.GetPickable():
                actor.PickableOff()
                actor.Modified()

        if PointStyle=='points':
            actor=self.actor.GetItemAsObject(0)
            if showNodes:
                if not actor.GetVisibility():
                    actor.VisibilityOn()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOn()
                    actor.Modified()
                if not actor.GetProperty().GetPointSize()==self.DefaultPointSize[0]:
                    actor.GetProperty().SetPointSize(self.DefaultPointSize[0])
                    actor.Modified()
            else:
                if not actor.GetProperty().GetPointSize()==0:
                    actor.GetProperty().SetPointSize(0)
                    actor.Modified()
            
            actor=self.actor.GetItemAsObject(1)
            if actor.GetVisibility():
                actor.VisibilityOff()
                actor.Modified()
            if actor.GetPickable():
                actor.PickableOff()
                actor.Modified()
            
        if PointStyle=='spheres':
            actor=self.actor.GetItemAsObject(1)
            if vis3D=="on": 
                if not actor.GetProperty().GetShading():
                    actor.GetProperty().ShadingOn()
                    actor.Modified()
                if not actor.GetProperty().GetLighting():
                    actor.GetProperty().LightingOn()
                    actor.Modified()
            else:               
                if actor.GetProperty().GetShading():
                    actor.GetProperty().ShadingOff()
                    actor.Modified()
                if actor.GetProperty().GetLighting():
                    actor.GetProperty().LightingOff()
                    actor.Modified()

            if showNodes:
                if not actor.GetVisibility():
                    actor.VisibilityOn()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOn()
                    actor.Modified()
            else:
                if actor.GetVisibility():
                    actor.VisibilityOff()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOff()
                    actor.Modified()

            actor=self.actor.GetItemAsObject(0)
            if not actor.GetProperty().GetPointSize()==0:
                actor.GetProperty().SetPointSize(0)
                actor.Modified()
        
class tag(objs):
    objtype="tag"
    maxNofColors=[2000]
    DefaultLineStyle=["lines"]
    DefaultPointStyle=["points"]
    Defaultvis3D=["off"]
    DefaultshowNodes=[1]
    DefaultRadius=[25]
    DefaultLineWidth=[1]
    DefaultPointSize=[5]
    
    initialized=[False]
    VisEngine_started=[False]
    
    viewports=list()

    allData= vtk.vtkAppendFilter()
    TagGlyphs= vtk.vtkGlyph3D()
    VertexGlyphs=vtk.vtkVertexGlyphFilter()
    mapper=vtk.vtkMapperCollection()
    actor=vtk.vtkActorCollection()
    LUT=vtk.vtkLookupTable()

    TagSource = vtk.vtkSphereSource()
    instancecount=[0]

    def __init__(self,parentItem,parentID,color):
        self.reset()
        self.NeuronID=parentID
        self.instancecount[0]+=1
        
        self.init_VisPipeline()
        
        self.setup_color(color)

        self.setup_VisEngine()
        
        self.nodeSelection=NodeSelection(self.TagSource,self.actor)
        self.nodeSelection.add(self.data,vtk.vtkSelectionNode.POINT,\
            vtk.vtkSelectionNode.INDICES,False)

        self.appendItem(parentItem)
                
    def setupTagSource(self):
        Sphere = self.TagSource
        Sphere.SetPhiResolution(6)
        Sphere.SetThetaResolution(6)
        Sphere.SetRadius(1.33)
        Sphere.Update()

    def init_VisPipeline(self):
        if self.initialized[0]:
            return
            
        self.setupTagSource()
        self.LUT.Allocate(self.maxNofColors[0])
        self.LUT.SetNumberOfTableValues(2)
        self.LUT.SetTableRange(0,1)
        self.LUT.SetTableValue(deleteColor,[0.0,0.0,0.0,0.0])
        self.LUT.SetTableValue(selectColor,[1.0,0.0,0.0,1.0])
                
        self.TagGlyphs.SetInputConnection(self.allData.GetOutputPort())
        self.TagGlyphs.SetSourceConnection(self.TagSource.GetOutputPort())
        self.TagGlyphs.SetScaleFactor(1.0)
        self.TagGlyphs.SetScaleModeToScaleByScalar()
        self.TagGlyphs.GeneratePointIdsOn()
 
        self.TagGlyphs.ReleaseDataFlagOn()
       
        LineMapper= vtk.vtkDataSetMapper()
        LineMapper.SetScalarModeToUsePointFieldData()
        LineMapper.SetInputConnection(self.allData.GetOutputPort()) 
        LineMapper.SelectColorArray("PointColor"); 
        LineMapper.SetLookupTable(self.LUT);    
        LineMapper.SetUseLookupTableScalarRange(1);

        GlyphMapper= vtk.vtkDataSetMapper()
        GlyphMapper.SetScalarModeToUsePointFieldData()
        GlyphMapper.SetInputConnection(self.TagGlyphs.GetOutputPort()) 
        GlyphMapper.SelectColorArray("PointColor"); 
        GlyphMapper.SetLookupTable(self.LUT);    
        GlyphMapper.SetUseLookupTableScalarRange(1);

        self.mapper.AddItem(LineMapper)
        self.mapper.AddItem(GlyphMapper)
        
        LineActor =vtk.vtkActor()
        LineActor.Type="LineActor"
        LineActor.GetProperty().SetLineWidth(self.DefaultLineWidth[0])
        LineActor.GetProperty().SetPointSize(self.DefaultPointSize[0])
        LineActor.SetMapper(LineMapper)       

        GlyphActor =vtk.vtkActor()
        GlyphActor.SetMapper(GlyphMapper)       
        GlyphActor.GetProperty().SetSpecular(.3)
        GlyphActor.GetProperty().SetSpecularPower(30)
        
        self.actor.AddItem(LineActor)
        self.actor.AddItem(GlyphActor)
        
        self.update_VisEngine()
        self.initialized[0]=True
        
    def setup_VisEngine(self):
        self.data = vtk.vtkPolyData()
        self.data.Allocate()

        ObjType = vtk.vtkStringArray()
        ObjType.SetName("ObjType")
        ObjType.SetNumberOfValues(1)
        ObjType.SetValue(0,self.objtype)        
        self.data.GetFieldData().AddArray(ObjType)

        PointColor = vtk.vtkFloatArray()
        PointColor.SetName("PointColor")
        PointColor.SetNumberOfComponents(1)
        self.data.GetPointData().AddArray(PointColor)

        NeuronID=vtk.vtkFloatArray()
        NeuronID.SetName("NeuronID")
        NeuronID.SetNumberOfComponents(1)
        self.data.GetPointData().AddArray(NeuronID)

        NodeID = vtk.vtkIdTypeArray()
        NodeID.SetName("NodeID")
        self.data.GetPointData().AddArray(NodeID)
 
        DeletedNodes=vtk.vtkUnsignedIntArray()
        DeletedNodes.SetName("DeletedNodes")
        self.data.GetPointData().AddArray(DeletedNodes)

        VisibleNodes=vtk.vtkUnsignedIntArray()
        VisibleNodes.SetName("VisibleNodes")
        self.data.GetPointData().AddArray(VisibleNodes)

        Radius=vtk.vtkFloatArray()
        Radius.SetName("Radius")
        self.data.GetPointData().AddArray(Radius)

        self.data.GetPointData().SetActivePedigreeIds("NodeID")
        self.data.GetPointData().SetScalars(Radius)
        self.data.SetPoints(vtk.vtkPoints())                 
        self.data.Modified()

        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("DeletedNodes")
        selection.AddThreshold(0,0)
        selection.Update()
        
        self.validData =vtk.vtkExtractSelection()    
        self.validData.SetInputConnection(0,self.data.GetProducerPort());
        self.validData.SetInputConnection(1,selection.GetOutputPort());

        selection = vtk.vtkSelectionSource()
        selection.SetContentType(7) # vtkSelection::THRESHOLDS
        selection.SetFieldType(1) # vtkSelection::POINTS
        selection.SetArrayName("VisibleNodes")
        selection.AddThreshold(1,1)
        selection.Update()        
        self.visibleData =vtk.vtkExtractSelection()
        self.visibleData.SetInputConnection(0,self.validData.GetOutputPort());
        self.visibleData.SetInputConnection(1,selection.GetOutputPort());
        self.allDataInput=self.visibleData

        self.visibleData.ReleaseDataFlagOn()
        self.validData.ReleaseDataFlagOn()
        self.allDataInput.ReleaseDataFlagOn()

        self.allData.AddInputConnection(0,self.allDataInput.GetOutputPort())
        self.allData.Modified()

        self.PointLocator=vtk.vtkPointLocator()

    def nodeId2tagIdx(self,nodeId):
        if nodeId==None:
            return None
        pointIdx=self.nodeId2pointIdx(nodeId)   
        if pointIdx<0:
            return None
        return self.pointIdx2tagIdx(pointIdx)
            
    def pointIdx2tagIdx(self,pointIdx):
        if pointIdx==None:
            return None
        self.data.Update()
        if pointIdx<0 or pointIdx>(self.data.GetNumberOfPoints()-1):
            print "Error: pointIdx out of range: ", pointIdx
            return None
        CellList=vtk.vtkIdList()
        self.data.GetPointCells(pointIdx,CellList)
        if CellList.GetNumberOfIds()<1:
            return None
        return CellList.GetId(0)
        
    def tagIdx2pointIdx(self,tagIdx):
        if tagIdx==None:
            return None
        self.data.Update()
        if tagIdx<0 or tagIdx>(self.data.GetNumberOfCells()-1):
            print "Error: tagIdx out of range: ", tagIdx
            return None
        PointList=vtk.vtkIdList()
        self.data.GetCellPoints(tagIdx,PointList)
        if PointList.GetNumberOfIds()<1:
            return None
        return PointList.GetId(0)        

    def tagIdx2nodeId(self,tagIdx):
        return self.pointIdx2nodeId(self.tagIdx2pointIdx(tagIdx))

    def SetRadius(self,radius):
        NNodes=self.data.GetNumberOfPoints()
        Radius=self.data.GetPointData().GetArray("Radius")
        Radius.DeepCopy(numpy_to_vtk(radius*np.ones([NNodes,1],dtype=np.float), deep=1, array_type=vtk.VTK_FLOAT))
        Radius.Modified()
        self.data.Modified()

    def set_tags(self,Points,NodeID=None):

        self.data.SetPoints(Points)        
        NPoints=Points.GetNumberOfPoints()

        colors=self.colorIdx*np.ones([NPoints,1],dtype=np.float)

        PointColor=self.data.GetPointData().GetArray("PointColor")
        PointColor.DeepCopy(numpy_to_vtk(colors, deep=1, array_type=vtk.VTK_FLOAT))
        PointColor.Modified()

        radius=self.DefaultRadius[0]*np.ones([NPoints,1],dtype=np.float)
        Radius=self.data.GetPointData().GetArray("Radius")
        Radius.DeepCopy(numpy_to_vtk(radius, deep=1, array_type=vtk.VTK_FLOAT))
        Radius.Modified()
        
        neuronID=self.NeuronID*np.ones([NPoints,1],dtype=np.float32)
        NeuronID=self.data.GetPointData().GetArray("NeuronID")
        NeuronID.DeepCopy(numpy_to_vtk(neuronID, deep=1, array_type=vtk.VTK_FLOAT))
        NeuronID.Modified()
        
        NodeIDArray=self.data.GetPointData().GetArray("NodeID") 
        if NodeID==None:
            NodeID=np.array(range(NPoints),dtype=np.int)      
        
        if not (NodeID.__class__.__name__=='vtkIdTypeArray' or NodeID.__class__.__name__=='vtkIntArray' or NodeID.__class__.__name__=='vtkLongArray' or NodeID.__class__.__name__=='vtkLongLongArray'): 
            if not NodeID.__class__.__name__=='ndarray':
                NodeID=np.array([NodeID],dtype=np.int)
            
            NodeID=numpy_to_vtk(NodeID, deep=1, array_type=vtk.VTK_ID_TYPE)
        else:
            if NodeID.GetNumberOfTuples()>0:
                NodeID=numpy_to_vtk(vtk_to_numpy(NodeID), deep=1, array_type=vtk.VTK_ID_TYPE)
            else:
                1
        NodeIDArray.DeepCopy(NodeID)
        NodeIDArray.Modified()
        
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        DeletedNodes.DeepCopy(numpy_to_vtk(np.zeros([NPoints,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        DeletedNodes.Modified()

        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
        VisibleNodes.DeepCopy(numpy_to_vtk(np.ones([NPoints,1],dtype=np.uint), deep=1, array_type=vtk.VTK_UNSIGNED_INT))
        VisibleNodes.Modified()
        
        
        vertex=np.array([np.ones(NPoints),range(NPoints)],dtype=np.int).reshape(2,NPoints).transpose().reshape(2*NPoints,)
        vertex=numpy_to_vtk(vertex, deep=1, array_type=vtk.VTK_ID_TYPE)
        Vertices=vtk.vtkCellArray()
        Vertices.SetCells(NPoints,vertex)
        self.data.SetVerts(Vertices)

        for pointIdx in range(NPoints):
            if not (not self.item):
                tagId=NodeIDArray.GetValue(pointIdx)
                tagItem=QtGui.QStandardItem("tag  {0}".format(tagId))            
                tagItem.nodeId=tagId
                tagItem.objtype=self.objtype
                tagItem.neuronId=self.NeuronID
                self.item.appendRow(tagItem)
    
        self.data.BuildCells()
        self.data.Update()
        self.data.BuildLinks()
        self.data.Modified()    
        if NPoints>0:          
            nodeId=NodeID.GetValue(NPoints-1)  #last nodeId 
        else:
            nodeId=-1
        return nodeId

    def add_tag(self,Points):
        if 'l' in self.flags: #locked object
            return
        classStr=Points.__class__.__name__
        pointIdx=-1
        if classStr=="NoneType":
            return -1
            
        NodeID=self.data.GetPointData().GetArray("NodeID")

        if NodeID.GetNumberOfTuples()>0:
            nodeId=np.int(NodeID.GetMaxNorm())
        else:
            nodeId=-1
            
        PointColor=self.data.GetPointData().GetArray("PointColor")
        Radius=self.data.GetPointData().GetArray("Radius")
        NeuronID=self.data.GetPointData().GetArray("NeuronID")
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
        
        if classStr=="vtkPoints":
            NPoints=[Points.GetNumberOfPoints()]
        else:
            if Points.ndim==1:
                NPoints=[1]
                Points=[Points]
            else:
                NPoints=Points.shape
        PointArray=self.data.GetPoints()
        for ipoint in range(NPoints[0]):
            if classStr=="vtkPoints":
                pointIdx=PointArray.InsertNextPoint(Points.GetPoint(ipoint))
            else:
                pointIdx=PointArray.InsertNextPoint(Points[ipoint][0],\
                    Points[ipoint][1],Points[ipoint][2])                

            nodeId+=1
            NodeID.InsertNextValue(nodeId)
 
            PointColor.InsertNextValue(self.colorIdx)
            Radius.InsertNextValue(self.DefaultRadius[0])
            NeuronID.InsertNextValue(np.float32(self.NeuronID))
            DeletedNodes.InsertNextValue(0)
            VisibleNodes.InsertNextValue(1)


            vertex=vtk.vtkIdList()
            vertex.InsertNextId(pointIdx)
            self.data.InsertNextCell(vtk.VTK_VERTEX,vertex)

            if not (not self.item):
                tagItem=QtGui.QStandardItem("tag {0}".format(nodeId))
                tagItem.nodeId=nodeId
                tagItem.objtype=self.objtype
                tagItem.neuronId=self.NeuronID
                self.item.appendRow(tagItem)    

        self.data.BuildCells()
        self.data.Update()
        self.data.BuildLinks()
        self.data.Modified()    
        return nodeId

    def delete_tag(self,tagIdx):        
        self.unselect_tag(tagIdx)

        nodeId =self.tagIdx2nodeId(tagIdx)
        self.comments.delete(nodeId)

        pointIdx=self.tagIdx2pointIdx(tagIdx)
        DeletedNodes=self.data.GetPointData().GetArray("DeletedNodes")
        DeletedNodes.SetValue(pointIdx,1);
        DeletedNodes.Modified()
#        VisibleNodes=self.data.GetPointData().GetArray("VisibleNodes")
#        VisibleNodes.SetValue(pointIdx,0);
#        VisibleNodes.Modified()

        #Mark the cell as deleted.
        self.data.DeleteCell(tagIdx);

        if not self.item.__class__()==[]:
            self.item.removeRow(tagIdx)

        #Remove the marked cell.
        self.data.RemoveDeletedCells();

        self.data.Modified()
        self.data.Update()
        self.data.BuildCells()
        self.data.BuildLinks()
        self.data.Modified()

    def select_tag(self,tagIdx):
        if tagIdx==None:
            return None
        selectionlist=self.nodeSelection.node.GetSelectionList()
        pointIdx =self.tagIdx2pointIdx(tagIdx)
        selIdx=selectionlist.LookupValue(pointIdx)
        if selIdx==-1:
            selectionlist.InsertNextValue(pointIdx);
        selectionlist.DataChanged() #somehow this is not done internally
        self.nodeSelection.node.Modified()

    def unselect_tag(self,tagIndices):
        if not tagIndices.__class__.__name__=='set':
            if not tagIndices.__class__.__name__=='list':
                tagIndices=[tagIndices]
            tagIndices=set(tagIndices)

        selectionlist=self.nodeSelection.node.GetSelectionList()
        if selectionlist==None  or selectionlist.GetNumberOfTuples()==0:
            return
        selectionlist=list(vtk_to_numpy(selectionlist))
        changed=False
        for tagIdx in tagIndices:
            if tagIdx==None:
                continue
            pointIdx =self.tagIdx2pointIdx(tagIdx)
            if pointIdx in selectionlist:
                selectionlist.remove(pointIdx)
                changed=True
        if changed:
            selectionlist=numpy_to_vtk(selectionlist, deep=1, array_type=vtk.VTK_ID_TYPE)
            self.nodeSelection.node.SetSelectionList(selectionlist)
            self.nodeSelection.node.Modified()
        
    def get_closest_point(self,point):
        if not self.PointLocator.GetDataSet():
            self.visibleData.Update()
            DataSet=self.visibleData.GetOutput()
            if DataSet.GetNumberOfPoints()>0:
                self.PointLocator.SetDataSet(DataSet)
                self.PointLocator.BuildLocator()
            else:
                return
        DataSet=self.PointLocator.GetDataSet()
        DataSet.Update()
        pointIdx=self.PointLocator.FindClosestPoint(point)
        if pointIdx==-1 or pointIdx==None:
            return -1,-1,-1

        nodeId=DataSet.GetPointData().GetArray("NodeID").GetValue(pointIdx)
        tagIdx=self.nodeId2tagIdx(nodeId)

        return np.array(DataSet.GetPoint(pointIdx),dtype=np.float), nodeId, tagIdx

    @classmethod
    def update_VisEngine(self, LineStyle=None, PointStyle=None, vis3D=None, showNodes=None):
        if not hasattr(self,'actor'):
            return
        if self.actor.GetNumberOfItems()==0:
            return
        if PointStyle==None:
            PointStyle=self.DefaultPointStyle[0]
        if vis3D==None:
            vis3D=self.Defaultvis3D[0]
        if showNodes==None:
            showNodes=self.DefaultshowNodes[0]
        if PointStyle=='points':
            actor=self.actor.GetItemAsObject(0)
            if showNodes:
                if not actor.GetVisibility():
                    actor.VisibilityOn()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOn()
                    actor.Modified()
                if not actor.GetProperty().GetPointSize()==self.DefaultPointSize[0]:
                    actor.GetProperty().SetPointSize(self.DefaultPointSize[0])
                    actor.Modified()
            else:
                if not actor.GetProperty().GetPointSize()==0:
                    actor.GetProperty().SetPointSize(0)
                    actor.Modified()
            
            actor=self.actor.GetItemAsObject(1)
            if actor.GetVisibility():
                actor.VisibilityOff()
                actor.Modified()
            if actor.GetPickable():
                actor.PickableOff()
                actor.Modified()
            
        if PointStyle=='spheres':
            actor=self.actor.GetItemAsObject(1)
            if vis3D=="on": 
                if not actor.GetProperty().GetShading():
                    actor.GetProperty().ShadingOn()
                    actor.Modified()
                if not actor.GetProperty().GetLighting():
                    actor.GetProperty().LightingOn()
                    actor.Modified()
            else:               
                if actor.GetProperty().GetShading():
                    actor.GetProperty().ShadingOff()
                    actor.Modified()
                if actor.GetProperty().GetLighting():
                    actor.GetProperty().LightingOff()
                    actor.Modified()

            if showNodes:
                if not actor.GetVisibility():
                    actor.VisibilityOn()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOn()
                    actor.Modified()
            else:
                if actor.GetVisibility():
                    actor.VisibilityOff()
                    actor.Modified()
                if not actor.GetPickable():
                    actor.PickableOff()
                    actor.Modified()

            actor=self.actor.GetItemAsObject(0)
            if not actor.GetProperty().GetPointSize()==0:
                actor.GetProperty().SetPointSize(0)
                actor.Modified()

class menuPlugin(QtGui.QAction):
    _Name=""
    _File=""
    def __init__(self,parent,**kw):
        QtGui.QAction.__init__(self,parent,**kw)
                
class menuDataset(QtGui.QAction):
    _Name=""
    _File=""
    def __init__(self,parent,**kw):
        QtGui.QAction.__init__(self,parent,**kw)

class ARIADNE(QtGui.QMainWindow):
    version=PyKNOSSOS_VERSION
    Datasets=[]
    CurrentFile=""
    CurrentDataset=['','']
    _DefaultDataPath=os.path.join(application_path,'data')
    DataScale=[9.252,9.252,25]
    _CommentShortcuts=["","","","",""]
    _StartPosition=[]
    _WorkingOffline=0;
    DemDriFiles={}; #demand-driven files
    Plugins=OrderedDict()
    console=None
    
    def __init__(self,uifile):
        QtGui.QMainWindow.__init__(self)
                
        uic.loadUi(uifile, self)
        self._UserMode="User"
        self.QRWin=[]
        self.planeROIs={}
        self.intersections={}
        self.Neurons=OrderedDict()
        self.Plugins=OrderedDict()
        self.job=job(self)
        #gui elements whose value/state is saved in the configuration file.
        #gui elements whose name starts with '_' are excluded.
        self.GUIElements=[QtGui.QSlider,QtGui.QRadioButton,QtGui.QCheckBox,QtGui.QDoubleSpinBox,QtGui.QSpinBox,QtGui.QComboBox,QtGui.QLineEdit]
        
        self.setWindowState(QtCore.Qt.WindowMaximized)

        self.Job.ariadne=self
        self.Settings.ariadne=self
        
        filter = doubleclickFilter(self.Job)
        self.Job.installEventFilter(filter)

        self.ckbx_MaximizeJobTab.setVisible(False)
        self.ckbx_MaximizeJobTab.setEnabled(False)
        
        self.Job.setFloating(1)
        self.Settings.setFloating(1)

        self.QRWin=QRenWin(self.centralwidget)
        self.QRWin.parent().resizeEvent=self.QRWin.ParentResizeEvent
                
        self.QRWin.ariadne=self
                
        self.ObjectBrowser.__class__=ObjectBrowser
        self.ObjectBrowser.__init__(self.ObjectBrowser.parent())

        myconfig.LoadConfig(self,"ARIADNE")

        NTableValues=256;
        self.tempLUT=vtk.vtkLookupTable()
        self.tempLUT.SetRampToLinear()
        self.tempLUT.SetNumberOfTableValues(NTableValues)
        self.tempLUT.Build()

        self.Console.setVisible(0)

        self.comboBox_TaskType.addItem("Tracing")
        
        self.comboBox_AutoSave.addItem("Disabled");
        self.comboBox_AutoSave.addItem("Auto save (*.nmx)");
        if usermode==0:
            self.comboBox_AutoSave.addItem("Auto save (*.nml)");
            self.comboBox_AutoSave.addItem("Auto save scaled (*.nml)");

        self.comboBox_Coloring.addItem("None");
        self.comboBox_Coloring.addItem("Auto");
        if usermode==0:
            self.comboBox_Coloring.addItem("axis(experimental)");
        

        for iitem in range(5):
            self._comboBox_Shortcuts.addItem(unicode(self._CommentShortcuts[iitem]))
            icon=QtGui.QIcon(os.path.join('imageformats',"F{0}-Key.png".format(iitem+1)))
            self._comboBox_Shortcuts.setItemIcon(iitem,icon)
            
        self.Timer=timer(self,self.timer_running,self.timer_working,self.timer_idle)
                
        self.synapse_browser=synapse_browser(self)
        myconfig.LoadConfig(self.synapse_browser,'SynapseBrowser')

        self.ConnectGUISlots()

        self.Filelist=list()
        self.File.addSeparator()
        for ifile in range(5):
            recentFile=menuDataset(self, visible=False,triggered=self.OpenRecentFile)
            self.Filelist.append(recentFile)
            self.File.addAction(recentFile)
                
        for ifile in range(self.Filelist.__len__()):
            recentFile=self.Filelist[ifile]
            if not myconfig.LoadConfig(recentFile,"RecentFile{0}".format(ifile)):
                break
            if recentFile._Name=='':
                break
            text = "&%d %s" % (ifile + 1, recentFile._Name)
            recentFile.setText(text)
            recentFile.setVisible(True)

        self.menuDatasets=list()
        for idataset in range(5):
            dataset=menuDataset(self, visible=False,triggered=self.LoadRecentCubeDataset)
            self.menuDatasets.append(dataset)
            self.menuDataset.insertAction(self.ActionSetDefaultDataPath,dataset)
            self.menuDataset.insertAction(self.ActionLoadNewDataset,dataset)
        self.menuDataset.insertSeparator(self.ActionLoadNewDataset)
        
        self.menuDatasets.reverse()

        for idataset in range(self.menuDatasets.__len__()-1,-1,-1):
            dataset=self.menuDatasets[idataset]
            if not myconfig.LoadConfig(dataset,"Dataset{0}".format(self.menuDatasets.__len__()-1-idataset)):
                break
            if dataset._Name=='':
                break
            text = "&%d %s" % (self.menuDatasets.__len__()-1-idataset + 1, dataset._Name)
            dataset.setText(text)
            dataset.setVisible(True)
        
        self.menuPluginList=list()        
        for iplugin in range(5):
            mplugin=menuPlugin(self, visible=False,triggered=self.RunPlugin)
            self.menuPluginList.append(mplugin)
            self.menuPlugins.insertAction(self.ActionLoadPlugin,mplugin)
        self.menuPlugins.insertSeparator(self.ActionLoadPlugin)
        
        self.menuPluginList.reverse()
        
        for iplugin in range(self.menuPluginList.__len__()-1,-1,-1):
            mplugin=self.menuPluginList[iplugin]
            if not myconfig.LoadConfig(mplugin,"Plugin{0}".format(self.menuPluginList.__len__()-1-iplugin)):
                break
            if mplugin._Name=='':
                break
            text = "&%d %s" % (self.menuPluginList.__len__()-1-iplugin + 1, mplugin._Name)
            mplugin.setText(text)
            mplugin.setVisible(True)
            
        if hasattr(self,'_StartPosition'):
            if self._StartPosition.__len__()>3:
                self.setWindowState(QtCore.Qt.WindowNoState)
                self.setGeometry(self._StartPosition[0],self._StartPosition[1],self._StartPosition[2],self._StartPosition[3])
        self.show()
        
        #load viewports
        myconfig.LoadConfig(self.QRWin,"QRWin")
        viewports2show= list(itertools.chain(*self.QRWin._ViewportLayout))
        viewports2load = set(["Orth_viewport","YX_viewport","YZ_viewport","ZX_viewport","skeleton_viewport"]+\
           viewports2show)
        for viewportname in viewports2load:
            if not viewportname:
                continue
            if not myconfig.has_key(viewportname):
                continue
            tempViewport=viewport(self);
            myconfig.LoadConfig(tempViewport,viewportname)
            self.QRWin.viewports[viewportname]=tempViewport
            if not (viewportname in viewports2show):
               tempViewport._Visible=0
               self.QRWin.RenderWindow.RemoveRenderer(tempViewport.border)           
               self.QRWin.RenderWindow.RemoveRenderer(tempViewport)                 


        self.QRWin.DistributeViewports()
        
        planeROI.ariadne=self

#        ROISize=2.0*np.floor((CubeLoader._NCubesPerEdge[0]-1)*128.0/np.sqrt(2.0)/2.0)-1
#        InterPolFactor=CubeLoader.InterPolFactor;
#        if ROISize>361:

        ROISize=361

        InterPolFactor=(CubeLoader._NCubesPerEdge[0]-1)*CubeLoader._CubeSize[0]/np.sqrt(2.0)/361;
        CubeLoader.InterPolFactor=min(2.0,max(1.0,InterPolFactor));
        if InterPolFactor>2.0:
            ROISize=2.0*np.floor(361*InterPolFactor/4.0)+1.0

        print "ROISize: ", ROISize

        for planeKey in ["Orth_planeROI","YX_planeROI","YZ_planeROI","ZX_planeROI"]:
            if not myconfig.has_key(planeKey):
                continue
            if not self.planeROIs.has_key(planeKey):
                tempPlane=planeROI()
                tempPlane.ROIRes=self.DataScale
                myconfig.LoadConfig(tempPlane,planeKey)
                tempPlane.SetImageSource([ROISize,ROISize])
                self.planeROIs[planeKey]=tempPlane
            
        #load planes and link them with associated viewports
        for key, iviewport in self.QRWin.viewports.iteritems():
            for planeKey in iviewport._LinkedPlanes:
                if not self.planeROIs.has_key(planeKey):
                    tempPlane=planeROI()
                    tempPlane.ROIRes=self.DataScale
                    myconfig.LoadConfig(tempPlane,planeKey)
                    tempPlane.SetImageSource([ROISize,ROISize])
                    self.planeROIs[planeKey]=tempPlane
                tempPlane=self.planeROIs[planeKey]
                iviewport.LinkedPlaneROIs[planeKey]=tempPlane
                iviewport.AddViewProp(tempPlane.PlaneActor)
                
            #load viewport plane, if any (orthogonal to camera direction)
            planeKey = iviewport._ViewportPlane
            if not (not planeKey):
                if not self.planeROIs.has_key(planeKey):
                    tempPlane=planeROI()
                    tempPlane.ROIRes=self.DataScale
                    myconfig.LoadConfig(tempPlane,planeKey)
                    tempPlane.SetImageSource([ROISize,ROISize])
                    self.planeROIs[planeKey]=tempPlane
                tempPlane=self.planeROIs[planeKey]
                iviewport.ViewportPlane=tempPlane
                iviewport.AddViewProp(tempPlane.PlaneActor)
                
        for key, iviewport in self.QRWin.viewports.iteritems():
            for planeKey in iviewport._LinkedPlaneOutlines:   
                if not self.planeROIs.has_key(planeKey):
                    tempPlane=planeROI()
                    tempPlane.ROIRes=self.DataScale
                    myconfig.LoadConfig(tempPlane,planeKey)
                    tempPlane.SetImageSource([ROISize,ROISize])
                    self.planeROIs[planeKey]=tempPlane
                iviewport.AddViewProp(self.planeROIs[planeKey].OutlineActor)
                iviewport.AddViewProp(self.planeROIs[planeKey].ScaleBarActor)
                
        #TODO: wrap up!
        #define intersections
        if ("Orth_planeROI" in self.planeROIs) and ("YX_planeROI" in self.planeROIs):
            self.intersections["Orth_YX"]=PlaneIntersection(self.planeROIs["Orth_planeROI"],self.planeROIs["YX_planeROI"])
        if ("Orth_planeROI" in self.planeROIs) and ("YZ_planeROI" in self.planeROIs):
            self.intersections["Orth_YZ"]=PlaneIntersection(self.planeROIs["Orth_planeROI"],self.planeROIs["YZ_planeROI"])
        if ("Orth_planeROI" in self.planeROIs) and ("ZX_planeROI" in self.planeROIs):
            self.intersections["Orth_ZX"]=PlaneIntersection(self.planeROIs["Orth_planeROI"],self.planeROIs["ZX_planeROI"])
        if ("YX_planeROI" in self.planeROIs) and ("YZ_planeROI" in self.planeROIs):
            self.intersections["YX_YZ"]=PlaneIntersection(self.planeROIs["YX_planeROI"],self.planeROIs["YZ_planeROI"])
        if ("YX_planeROI" in self.planeROIs) and ("ZX_planeROI" in self.planeROIs):
            self.intersections["YX_ZX"]=PlaneIntersection(self.planeROIs["YX_planeROI"],self.planeROIs["ZX_planeROI"])
        if ("YZ_planeROI" in self.planeROIs) and ("ZX_planeROI" in self.planeROIs):
            self.intersections["YZ_ZX"]=PlaneIntersection(self.planeROIs["YZ_planeROI"],self.planeROIs["ZX_planeROI"])

        if "Orth_planeROI" in self.planeROIs:
            orthColor=self.planeROIs["Orth_planeROI"]._OutlineColor
        if "YX_planeROI" in self.planeROIs:
            YXColor=self.planeROIs["YX_planeROI"]._OutlineColor
        if "YZ_planeROI" in self.planeROIs:
            YZColor=self.planeROIs["YZ_planeROI"]._OutlineColor
        if "ZX_planeROI" in self.planeROIs:
            ZXColor=self.planeROIs["ZX_planeROI"]._OutlineColor
        
        if self.QRWin.viewports.has_key("Orth_viewport"):
            if "Orth_YX" in self.intersections:
                self.intersections["Orth_YX"].AddIntersection(self.QRWin.viewports["Orth_viewport"],YXColor)
            if "Orth_YZ" in self.intersections:
                self.intersections["Orth_YZ"].AddIntersection(self.QRWin.viewports["Orth_viewport"],YZColor)
            if "Orth_ZX" in self.intersections:
                self.intersections["Orth_ZX"].AddIntersection(self.QRWin.viewports["Orth_viewport"],ZXColor)

        if self.QRWin.viewports.has_key("YX_viewport"):
            if "Orth_YX" in self.intersections:
                self.intersections["Orth_YX"].AddIntersection(self.QRWin.viewports["YX_viewport"],orthColor)
            if "YX_YZ" in self.intersections:
                self.intersections["YX_YZ"].AddIntersection(self.QRWin.viewports["YX_viewport"],YZColor)
            if "YX_ZX" in self.intersections:
                self.intersections["YX_ZX"].AddIntersection(self.QRWin.viewports["YX_viewport"],ZXColor)

        if self.QRWin.viewports.has_key("YZ_viewport"):
            if "Orth_YZ" in self.intersections:
                self.intersections["Orth_YZ"].AddIntersection(self.QRWin.viewports["YZ_viewport"],orthColor)
            if "YX_YZ" in self.intersections:
                self.intersections["YX_YZ"].AddIntersection(self.QRWin.viewports["YZ_viewport"],YXColor)
            if "YZ_ZX" in self.intersections:
                self.intersections["YZ_ZX"].AddIntersection(self.QRWin.viewports["YZ_viewport"],ZXColor)

        if self.QRWin.viewports.has_key("ZX_viewport"):
            if "Orth_ZX" in self.intersections:
                self.intersections["Orth_ZX"].AddIntersection(self.QRWin.viewports["ZX_viewport"],orthColor)
            if "YX_ZX" in self.intersections:
                self.intersections["YX_ZX"].AddIntersection(self.QRWin.viewports["ZX_viewport"],YXColor)
            if "YZ_ZX" in self.intersections:
                self.intersections["YZ_ZX"].AddIntersection(self.QRWin.viewports["ZX_viewport"],YZColor)
        
        if self.QRWin.viewports.has_key("skeleton_viewport"):
            iviewport=self.QRWin.viewports["skeleton_viewport"]
            if "YX_planeROI" in (iviewport._LinkedPlaneOutlines + iviewport._LinkedPlanes):
                if "Orth_YX" in self.intersections:
                    self.intersections["Orth_YX"].AddIntersection(iviewport,YXColor)
                if "YX_YZ" in self.intersections:
                    self.intersections["YX_YZ"].AddIntersection(iviewport,YXColor)
                if "YX_ZX" in self.intersections:
                    self.intersections["YX_ZX"].AddIntersection(iviewport,YXColor)
            if "YZ_planeROI" in (iviewport._LinkedPlaneOutlines + iviewport._LinkedPlanes):
                if "Orth_YZ" in self.intersections:
                    self.intersections["Orth_YZ"].AddIntersection(iviewport,YZColor)
                if "YX_YZ" in self.intersections:
                    self.intersections["YX_YZ"].AddIntersection(iviewport,YZColor)
                if "YZ_ZX" in self.intersections:
                    self.intersections["YZ_ZX"].AddIntersection(iviewport,YZColor)
            if "orth_planeROI" in (iviewport._LinkedPlaneOutlines + iviewport._LinkedPlanes):
                if "Orth_YX" in self.intersections:
                    self.intersections["Orth_YX"].AddIntersection(iviewport,orthColor)
                if "Orth_YZ" in self.intersections:
                    self.intersections["Orth_YZ"].AddIntersection(iviewport,orthColor)
                if "Orth_ZX" in self.intersections:
                    self.intersections["Orth_ZX"].AddIntersection(iviewport,orthColor)
            if "ZX_planeROI" in (iviewport._LinkedPlaneOutlines + iviewport._LinkedPlanes):
                if "Orth_ZX" in self.intersections:
                    self.intersections["Orth_ZX"].AddIntersection(iviewport,ZXColor)
                if "YX_ZX" in self.intersections:
                    self.intersections["YX_ZX"].AddIntersection(iviewport,ZXColor)
                if "YZ_ZX" in self.intersections:
                    self.intersections["YZ_ZX"].AddIntersection(iviewport,ZXColor)

        self.Job.setFloating(0)
        self.Settings.setFloating(0)
                      
        
        self.Console.setFloating(0)
            
        if usermode>0:
            self.File.removeAction(self.ActionExportSynapses)
            self.File.removeAction(self.ActionSaveSeparately)
            self.File.removeAction(self.ActionSaveActiveNeuron)
        if usermode==1:
            self.SettingsTab.removeTab(self.SettingsTab.indexOf(self.ExtractGIFTab))
            self.SettingsTab.removeTab(self.SettingsTab.indexOf(self.CaptureTab))

        if not experimental:
            self.SettingsTab.removeTab(self.SettingsTab.indexOf(self.DatasetTab))
            self.SettingsTab.removeTab(self.SettingsTab.indexOf(self.CreateJobTab))
            self.SettingsTab.removeTab(self.SettingsTab.indexOf(self.RotateTab))
            self.SettingsTab.removeTab(self.SettingsTab.indexOf(self.ObjBrowserTab))
            self.SettingsTab.removeTab(self.SettingsTab.indexOf(self.Skel2TIFFTab))
            
            self.JobTab.removeTab(self.JobTab.indexOf(self.ClassificationTab))
            
        self.BoundingBox=BoundingBox()   
        if self.QRWin.viewports.has_key("skeleton_viewport"):
            iviewport=self.QRWin.viewports["skeleton_viewport"]
            self.BoundingBox.AddBoundingBox(iviewport)

        if self._WorkingOffline==1:
            self.ActionWorkingOffline.setChecked(1)
        else:
            self.ActionWorkingOffline.setChecked(0)
        self.WorkingOffline()


    def closeEvent(self, event):
        if not self.Timer.changesSaved:            
            reply = QtGui.QMessageBox.question(self, 'Message',
                "There are unsaved changes. Do you want to save the changes before quitting?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No, QtGui.QMessageBox.Cancel)
            if reply == QtGui.QMessageBox.Yes:
                self.Save()
            elif reply == QtGui.QMessageBox.Cancel:
                event.ignore()
                return


        for filename,ddobj in self.DemDriFiles.iteritems():
            ddobj.delete()
        self.DemDriFiles.clear()
#        self.QRWin.Timer.stop()
#        self.QRWin.Timer.timeout.disconnect(self.QRWin.UpdateTimer)
#        self.QRWin.disconnect(self.QRWin.Timer, QtCore.SIGNAL('timeout()'), self.QRWin.TimerEvent)
#        if self.QRWin.Timer.timerId()>-1:
#            self.QRWin.Timer.killTimer(self.QRWin.Timer.timerId())
        CubeLoader.StopLoader()

        tempGeometry=self.geometry()
        self._StartPosition=\
            [tempGeometry.x(),tempGeometry.y(),tempGeometry.width(),tempGeometry.height()]
        
        if not myconfig.has_key("GUIstate"):
            myconfig["GUIstate"]={}
        for element in self.GUIElements:
            children=self.findChildren(element)
            for child in children:
                childName=unicode(child.objectName())
                if childName.startswith("_") or childName.startswith("qt_") or childName=="":
                    continue                
                if hasattr(child,'value'):
                    value=child.value()
                elif hasattr(child,'isChecked'):
                    value=child.isChecked()
                elif hasattr(child,'currentIndex'):
                    value=child.currentIndex()
                elif hasattr(child,'text'):
                    value=child.text()
                else:
                    continue
                myconfig["GUIstate"][childName]=value
                
        myconfig.SaveConfig(self,"ARIADNE")

        for key,iviewport in self.QRWin.viewports.iteritems():
            myconfig.SaveConfig(iviewport,key)
        myconfig.SaveConfig(self.QRWin,"QRWin")
        
        if hasattr(self,'synapse_browser'):
            myconfig.SaveConfig(self.synapse_browser,"SynapseBrowser")
        

        for key, iplane in self.planeROIs.iteritems():
            myconfig.SaveConfig(iplane,key)

        for idataset in range(self.menuDatasets.__len__()-1,-1,-1):
            dataset=self.menuDatasets[idataset]
            if dataset._File=='':
                break
            myconfig.SaveConfig(dataset,"Dataset{0}".format(self.menuDatasets.__len__()-1-idataset))

        for iplugin in range(self.menuPluginList.__len__()-1,-1,-1):
            mplugin=self.menuPluginList[iplugin]
            if mplugin._File=='':
                break
            myconfig.SaveConfig(mplugin,"Plugin{0}".format(self.menuPluginList.__len__()-1-iplugin))

        for ifile in range(self.Filelist.__len__()):
            recentFile=self.Filelist[ifile]
            if recentFile._File=='':
                break
            myconfig.SaveConfig(recentFile,"RecentFile{0}".format(ifile))

        for pluginName,plugin in self.Plugins.iteritems():
            myconfig.SaveConfig(plugin,pluginName)
            

        myconfig.write()

        if hasattr(self.console,'exit_interpreter'):
            self.console.exit_interpreter()
        event.accept()
        

    def SetupGUIState(self,GUIobj):
        if not myconfig.has_key("GUIstate"):
            return        
        if not hasattr(GUIobj,'findChildren'):
            return
        for element in self.GUIElements:
            children=GUIobj.findChildren(element)
            for child in children:
                childName=unicode(child.objectName())
                if childName.startswith("_"):
                    continue
                if not myconfig["GUIstate"].has_key(childName):
                    continue
                if hasattr(child,'value'):
                    template=child.value()
                    child.setValue(myconfig.Cast(myconfig["GUIstate"][childName],template))
                elif hasattr(child,'isChecked'):
                    template=child.isChecked()
                    child.setChecked(myconfig.Cast(myconfig["GUIstate"][childName],template))
                elif hasattr(child,'currentIndex'):
                    template=child.currentIndex()
                    child.setCurrentIndex(myconfig.Cast(myconfig["GUIstate"][childName],template))
                elif hasattr(child,'text'):
                    template=child.text()
                    child.setText(myconfig.Cast(myconfig["GUIstate"][childName],template))                    
                      

            
    def ConnectGUISlots(self):
        
        QtCore.QObject.connect(self.radioBtn_lines2Dpoints,QtCore.SIGNAL("toggled(bool)"),self.ChangeSkelVisMode)
        QtCore.QObject.connect(self.radioBtn_lines3Dpoints,QtCore.SIGNAL("toggled(bool)"),self.ChangeSkelVisMode)
        QtCore.QObject.connect(self.radioBtn_tubesflat,QtCore.SIGNAL("toggled(bool)"),self.ChangeSkelVisMode)
        QtCore.QObject.connect(self.radioBtn_full3D,QtCore.SIGNAL("toggled(bool)"),self.ChangeSkelVisMode)
        QtCore.QObject.connect(self.HideSkelNodes,QtCore.SIGNAL("stateChanged(int)"),self.ChangeSkelVisMode)        

        QtCore.QObject.connect(self.radioBtn_Synlines2Dpoints,QtCore.SIGNAL("toggled(bool)"),self.ChangeSynVisMode)
        QtCore.QObject.connect(self.radioBtn_Synlines3Dpoints,QtCore.SIGNAL("toggled(bool)"),self.ChangeSynVisMode)
        QtCore.QObject.connect(self.radioBtn_Syntubesflat,QtCore.SIGNAL("toggled(bool)"),self.ChangeSynVisMode)
        QtCore.QObject.connect(self.radioBtn_Synfull3D,QtCore.SIGNAL("toggled(bool)"),self.ChangeSynVisMode)
        QtCore.QObject.connect(self.HideSynNodes,QtCore.SIGNAL("stateChanged(int)"),self.ChangeSynVisMode)        
        QtCore.QObject.connect(self.SynStartNodeOnly,QtCore.SIGNAL("stateChanged(int)"),self.ChangeSynVisMode)        
        

        QtCore.QObject.connect(self.ckbx_HideSkelViewport,QtCore.SIGNAL("stateChanged(int)"),self.HideSkelViewport)  
        QtCore.QObject.connect(self.ckbx_ShowBoundingBox,QtCore.SIGNAL("stateChanged(int)"),self.ShowBoundingBox)  
        
        QtCore.QObject.connect(self.ckbx_HideYXplane,QtCore.SIGNAL("stateChanged(int)"),self.ChangePlaneVisMode)        
        QtCore.QObject.connect(self.ckbx_HideYZplane,QtCore.SIGNAL("stateChanged(int)"),self.ChangePlaneVisMode)        
        QtCore.QObject.connect(self.ckbx_HideZXplane,QtCore.SIGNAL("stateChanged(int)"),self.ChangePlaneVisMode)        
        QtCore.QObject.connect(self.ckbx_Hidearbitplane,QtCore.SIGNAL("stateChanged(int)"),self.ChangePlaneVisMode)        
        QtCore.QObject.connect(self.ckbx_HideBorder,QtCore.SIGNAL("stateChanged(int)"),self.ChangeBorderVisMode)        
        QtCore.QObject.connect(self.ckbx_SynZoom,QtCore.SIGNAL("stateChanged(int)"),self.ChangeSynZoom)  
        QtCore.QObject.connect(self.ckbx_ClipHulls,QtCore.SIGNAL("stateChanged(int)"),
                               lambda ph, vp='skeleton_viewport', state=None: self.ToggleClipHulls(vp,state))  

        QtCore.QObject.connect(self._ckbx_CaptureShowSkelVPOnly,QtCore.SIGNAL("stateChanged(int)"),self.ShowSkelVPOnly)  

        QtCore.QObject.connect(self.btn_loadReferenceFile,QtCore.SIGNAL("clicked()"),lambda: self.LoadReferenceFile('',True))
        
        QtCore.QObject.connect(self._ckbx_TaskDone,QtCore.SIGNAL("stateChanged(int)"),self.ChangeTaskState)        
        QtCore.QObject.connect(self.btn_nextTask,QtCore.SIGNAL("clicked()"),lambda direction='next': self.SearchTask(direction))
        QtCore.QObject.connect(self.btn_prevTask,QtCore.SIGNAL("clicked()"),lambda direction='previous': self.SearchTask(direction))

        QtCore.QObject.connect(self.btn_LoadSeedPath,QtCore.SIGNAL("clicked()"),lambda obj='SeedPath': self.SetPath(obj))
        QtCore.QObject.connect(self.btn_LoadJobPath,QtCore.SIGNAL("clicked()"),lambda obj='JobPath': self.SetPath(obj))
        QtCore.QObject.connect(self.btn_LoadDatasetPath,QtCore.SIGNAL("clicked()"),lambda obj='DatasetPath': self.SetPath(obj))
        QtCore.QObject.connect(self.btn_CreateJob,QtCore.SIGNAL("clicked()"),self.CreateJob)

        QtCore.QObject.connect(self.btn_CopyCoords,QtCore.SIGNAL("clicked()"),self.CopyCoords)

        QtCore.QObject.connect(self.btn_Convert2NML,QtCore.SIGNAL("clicked()"),self.Convert2NML)
        QtCore.QObject.connect(self.btn_Decrypt,QtCore.SIGNAL("clicked()"),self.Decrypt)
        QtCore.QObject.connect(self.btn_Encrypt,QtCore.SIGNAL("clicked()"),self.Encrypt)
        
        QtCore.QObject.connect(self.btn_Rotate,QtCore.SIGNAL("clicked()"),self.TransformOrthViewport)
        QtCore.QObject.connect(self.text_OrthcDir,QtCore.SIGNAL("textEdited(QString)"), 
           lambda  value, obj="cDir": self.TransformOrthViewport(obj,value))
        QtCore.QObject.connect(self.text_OrthvDir,QtCore.SIGNAL("textEdited(QString)"), 
           lambda  value, obj="vDir": self.TransformOrthViewport(obj,value))

        QtCore.QObject.connect(self.ckbx_ShowYXScaleBar,QtCore.SIGNAL("stateChanged(int)"),self.ChangeBorderVisMode)   
        QtCore.QObject.connect(self.ckbx_ShowYZScaleBar,QtCore.SIGNAL("stateChanged(int)"),self.ChangeBorderVisMode)   
        QtCore.QObject.connect(self.ckbx_ShowZXScaleBar,QtCore.SIGNAL("stateChanged(int)"),self.ChangeBorderVisMode)   
        QtCore.QObject.connect(self.ckbx_ShowArbitScaleBar,QtCore.SIGNAL("stateChanged(int)"),self.ChangeBorderVisMode)   
        QtCore.QObject.connect(self.SpinBoxScaleBar,QtCore.SIGNAL("editingFinished()"),
           lambda  value=None,parameter="length": self.ChangeScaleBar(parameter,value))
        QtCore.QObject.connect(self.SpinBox_ScaleBarWidth,QtCore.SIGNAL("editingFinished()"),
           lambda  value=None,parameter="width": self.ChangeScaleBar(parameter,value))
        QtCore.QObject.connect(self.btn_ScaleBarColor,QtCore.SIGNAL("clicked()"),self.ChangeScaleBarColor)

        QtCore.QObject.connect(self.btn_LoadGIFPath,QtCore.SIGNAL("clicked()"),lambda obj='GIFPath': self.SetPath(obj))
        QtCore.QObject.connect(self.btn_CreateGIF,QtCore.SIGNAL("clicked()"),self.CreateGIF)

        QtCore.QObject.connect(self.btn_LoadCapturePath,QtCore.SIGNAL("clicked()"),lambda obj='CapturePath': self.SetPath(obj))
        QtCore.QObject.connect(self.btn_Capture,QtCore.SIGNAL("clicked()"),self.Capture)

        QtCore.QObject.connect(self.radioBtn_Browsing,QtCore.SIGNAL("toggled(bool)"),self.ChangeWorkingMode)
        QtCore.QObject.connect(self.radioBtn_Tracing,QtCore.SIGNAL("toggled(bool)"),self.ChangeWorkingMode)
        QtCore.QObject.connect(self.radioBtn_Tagging,QtCore.SIGNAL("toggled(bool)"),self.ChangeWorkingMode)
        QtCore.QObject.connect(self.radioBtn_Synapses,QtCore.SIGNAL("toggled(bool)"),self.ChangeWorkingMode)

        QtCore.QObject.connect(self.SpinBoxVOISize,QtCore.SIGNAL("editingFinished()"),self.JumpToPoint)        
        QtCore.QObject.connect(self.ckbx_restrictVOI,QtCore.SIGNAL("stateChanged(int)"),self.RestrictVOI)        
        QtCore.QObject.connect(self.SpinBox_DemDriNeighbors,QtCore.SIGNAL("editingFinished()"),lambda: self.UpdateDemDriFiles(None,True))        

        QtCore.QObject.connect(self.radioBtn_singlenodes,QtCore.SIGNAL("toggled(bool)"),self.ChangeWorkingMode)
        QtCore.QObject.connect(self.radioBtn_connodes,QtCore.SIGNAL("toggled(bool)"),self.ChangeWorkingMode)

        QtCore.QObject.connect(self.radioBtn_showall,QtCore.SIGNAL("toggled(bool)"),self.ChangeNeuronVisMode)
        QtCore.QObject.connect(self.radioBtn_showactiveneuron,QtCore.SIGNAL("toggled(bool)"),self.ChangeNeuronVisMode)
        QtCore.QObject.connect(self.radioBtn_showconcomp,QtCore.SIGNAL("toggled(bool)"),self.ChangeNeuronVisMode)
        QtCore.QObject.connect(self.radioBtn_hideall,QtCore.SIGNAL("toggled(bool)"),self.ChangeNeuronVisMode)
        
        QtCore.QObject.connect(self.ckbx_hideReference,QtCore.SIGNAL("stateChanged(int)"),self.ChangeNeuronVisMode)
        QtCore.QObject.connect(self.ckbx_hideReference_2,QtCore.SIGNAL("stateChanged(int)"),self.ckbx_hideReference.setCheckState)
        QtCore.QObject.connect(self.ckbx_hideReference,QtCore.SIGNAL("stateChanged(int)"),self.ckbx_hideReference_2.setCheckState)
        
        QtCore.QObject.connect(self.ckbx_HideCrossHairs,QtCore.SIGNAL("stateChanged(int)"),self.QRWin.HideCrosshairs)
        QtCore.QObject.connect(self.ckbx_HideFocalpoint,QtCore.SIGNAL("stateChanged(int)"),self.QRWin.HideFocalpoint)

        QtCore.QObject.connect(self.RegionAlpha,QtCore.SIGNAL("valueChanged(int)"),self.SetRegionVisibility)                
        QtCore.QObject.connect(self.ckbx_HideRegionLabels,QtCore.SIGNAL("stateChanged(int)"),lambda: self.HideRegionLabels())        
        QtCore.QObject.connect(self.SomaAlpha,QtCore.SIGNAL("valueChanged(int)"),self.SetSomaVisibility)                
        QtCore.QObject.connect(self.ckbx_HideSomaLabels,QtCore.SIGNAL("stateChanged(int)"),lambda: self.HideSomaLabels())        
        QtCore.QObject.connect(self.SpinBox_LineWidth,QtCore.SIGNAL("editingFinished()"),self.SetSkelLineWidth)        

        QtCore.QObject.connect(self.SpinBox_CubesPerEdge,QtCore.SIGNAL("editingFinished()"),\
            lambda filename=None, recenter=0: self.ChangeCubeDataset(filename,recenter))        
        QtCore.QObject.connect(self.SpinBox_StreamingSlots,QtCore.SIGNAL("editingFinished()"),\
            lambda filename=None, recenter=0: self.ChangeCubeDataset(filename,recenter))      
        
        QtCore.QObject.connect(self.SpinBox_Radius,QtCore.SIGNAL("editingFinished()"),self.SetSkeletonRadius)        
        QtCore.QObject.connect(self.OverwriteRadius,QtCore.SIGNAL("stateChanged(int)"),lambda: self.SetSkeletonRadius())        

        QtCore.QObject.connect(self.SpinBox_SynRadius,QtCore.SIGNAL("editingFinished()"),self.SetSynapseRadius)        
        QtCore.QObject.connect(self.SpinBox_SynLineWidth,QtCore.SIGNAL("editingFinished()"),self.SetSynLineWidth)        

        QtCore.QObject.connect(self._comboBox_Shortcuts,QtCore.SIGNAL("editTextChanged(QString)"),self.ChangeShortcut)        


        QtCore.QObject.connect(self.comboBox_Coloring,QtCore.SIGNAL("currentIndexChanged(int)"),self.ChangeColorScheme)        
        
        self.span_contrast.lowerPositionChanged.connect(lambda  lower, upper=None: self.ChangeContrast(lower,upper))
        self.span_contrast.upperPositionChanged.connect(lambda  upper, lower=None: self.ChangeContrast(lower,upper))
        
        self.span_brightness.lowerPositionChanged.connect(lambda  lower, upper=None: self.ChangeBrightness(lower,upper))
        self.span_brightness.upperPositionChanged.connect(lambda  upper, lower=None: self.ChangeBrightness(lower,upper))

        QtCore.QObject.connect(self.SpinBoxX,QtCore.SIGNAL("editingFinished()"),self.JumpToPoint)        
        QtCore.QObject.connect(self.SpinBoxY,QtCore.SIGNAL("editingFinished()"),self.JumpToPoint)        
        QtCore.QObject.connect(self.SpinBoxZ,QtCore.SIGNAL("editingFinished()"),self.JumpToPoint)     
        
#        myValidator=QtGui.QValidator
#        myValidator.validate=self.CoordChanged
        self.SpinBoxX.findChild(QtGui.QLineEdit).setValidator(None)
        self.SpinBoxY.findChild(QtGui.QLineEdit).setValidator(None)
        self.SpinBoxZ.findChild(QtGui.QLineEdit).setValidator(None)

        self.SpinBoxX.findChild(QtGui.QLineEdit).textChanged.connect(self.CoordChanged)
        self.SpinBoxY.findChild(QtGui.QLineEdit).textChanged.connect(self.CoordChanged)
        self.SpinBoxZ.findChild(QtGui.QLineEdit).textChanged.connect(self.CoordChanged)

#        QtCore.QObject.connect(self.SpinBoxX,QtCore.SIGNAL("valueChanged(QString)"),self.CoordChanged)        
#        QtCore.QObject.connect(self.SpinBoxY,QtCore.SIGNAL("valueChanged(QString)"),self.CoordChanged)        
#        QtCore.QObject.connect(self.SpinBoxZ,QtCore.SIGNAL("valueChanged(QString)"),self.CoordChanged)     
#        self.SpinBoxX.findChild(QtGui.QLineEdit).textChanged.connect(self.CoordChanged)
#        self.SpinBoxY.findChild(QtGui.QLineEdit).textChanged.connect(self.CoordChanged)
#        self.SpinBoxZ.findChild(QtGui.QLineEdit).textChanged.connect(self.CoordChanged)

        QtCore.QObject.connect(self.ActionNewFile,QtCore.SIGNAL("triggered()"),self.New)        
        QtCore.QObject.connect(self.ActionSaveAs,QtCore.SIGNAL("triggered()"),self.SaveAs)        
        QtCore.QObject.connect(self.ActionSaveSeparately,QtCore.SIGNAL("triggered()"),self.SaveSeparately)
        QtCore.QObject.connect(self.ActionSaveActiveNeuron,QtCore.SIGNAL("triggered()"),self.SaveActiveNeuron)
        QtCore.QObject.connect(self.ActionExportSynapses,QtCore.SIGNAL("triggered()"),self.ExportSynapses)    
                
        QtCore.QObject.connect(self.ActionSave,QtCore.SIGNAL("triggered()"),self.Save)        
        QtCore.QObject.connect(self.ActionOpen,QtCore.SIGNAL("triggered()"),self.Open)     
        
        QtCore.QObject.connect(self.ActionLoadNewDataset,QtCore.SIGNAL("triggered()"),self.LoadCubeDataset)
        QtCore.QObject.connect(self.ActionSetDefaultDataPath,QtCore.SIGNAL("triggered()"),self.SetDefaultDataPath)
        QtCore.QObject.connect(self.ActionWorkingOffline,QtCore.SIGNAL("triggered()"),self.WorkingOffline)
        
        QtCore.QObject.connect(self.ActionLoadPlugin,QtCore.SIGNAL("triggered()"),self.RunPlugin)
        
        QtCore.QObject.connect(self._SpinBoxNeuronId,QtCore.SIGNAL("editingFinished()"),
           lambda source='spinbox1': self.GotoNode(source))        
        QtCore.QObject.connect(self._SpinBoxNeuronId_2,QtCore.SIGNAL("editingFinished()"),
           lambda  source='spinbox2': self.GotoNode(source))       
        QtCore.QObject.connect(self._SpinBoxNeuronId_3,QtCore.SIGNAL("editingFinished()"),
           lambda  source='spinbox3': self.GotoNode(source))       

        QtCore.QObject.connect(self._SpinBoxNodeId,QtCore.SIGNAL("editingFinished()"),
           lambda source='spinbox1': self.GotoNode(source))        
        QtCore.QObject.connect(self._SpinBoxNodeId_2,QtCore.SIGNAL("editingFinished()"),
           lambda  source='spinbox2': self.GotoNode(source))
        QtCore.QObject.connect(self._SpinBoxNodeId_3,QtCore.SIGNAL("editingFinished()"),
           lambda  source='spinbox3': self.GotoNode(source))
           
        QtCore.QObject.connect(self._text_Comment,QtCore.SIGNAL("textEdited(QString)"), 
           lambda  value, obj="SelObj", key="comment": self.ChangeComment(obj,key,unicode(value)))
        QtCore.QObject.connect(self._text_Comment_2,QtCore.SIGNAL("textEdited(QString)"), 
           lambda  value, obj="SelObj", key="comment": self.ChangeComment(obj,key,unicode(value)))
        QtCore.QObject.connect(self._text_Comment_3,QtCore.SIGNAL("textEdited(QString)"), 
           lambda  value, obj="SelObj", key="comment": self.ChangeComment(obj,key,unicode(value)))
        QtCore.QObject.connect(self._text_Comment_4,QtCore.SIGNAL("textEdited(QString)"), 
           lambda  value, obj="SelObj", key="comment": self.ChangeComment(obj,key,unicode(value)))

        QtCore.QObject.connect(self.ckbx_MaximizeJobTab,QtCore.SIGNAL("stateChanged(int)"),self.MaxMinJobTab)
        
        QtCore.QObject.connect(self._CertaintyLevel,QtCore.SIGNAL("valueChanged(int)"), 
           lambda : self.synapse_browser.attributes_changed())        
        

        QtCore.QObject.connect(self.btn_SplitConComp,QtCore.SIGNAL("clicked()"),self.SplitByConComp)
        QtCore.QObject.connect(self.btn_DelConComp,QtCore.SIGNAL("clicked()"),self.DelConComp)
        
        QtCore.QObject.connect(self.btn_MergeConComp,QtCore.SIGNAL("clicked()"),self.MergeConComp)
        QtCore.QObject.connect(self.btn_MergeNeuron,QtCore.SIGNAL("clicked()"),self.MergeNeuron)
        

        QtCore.QObject.connect(self.btn_NewNeuron,QtCore.SIGNAL("clicked()"),self.NewNeuron)
        QtCore.QObject.connect(self.btn_DelNeuron,QtCore.SIGNAL("clicked()"),self.DelNeuron)
        QtCore.QObject.connect(self.btn_DelNode,QtCore.SIGNAL("clicked()"),self.DelNode)

        QtCore.QObject.connect(self.btn_NewNeuron_2,QtCore.SIGNAL("clicked()"),self.NewNeuron)
        QtCore.QObject.connect(self.btn_NewNeuron_3,QtCore.SIGNAL("clicked()"),self.NewNeuron)
        QtCore.QObject.connect(self.btn_DelNeuron_2,QtCore.SIGNAL("clicked()"),self.DelNeuron)
        QtCore.QObject.connect(self.btn_DelNeuron_3,QtCore.SIGNAL("clicked()"),self.DelNeuron)
        QtCore.QObject.connect(self.btn_DelNode_2,QtCore.SIGNAL("clicked()"),self.DelNode)
        QtCore.QObject.connect(self.btn_DelNode_3,QtCore.SIGNAL("clicked()"),self.DelNode)


        QtCore.QObject.connect(self.btn_Syn_assign,QtCore.SIGNAL("clicked()"),self.synapse_browser.set_attributes)
#        QtCore.QObject.connect(self.btn_SynClassColor,QtCore.SIGNAL("clicked()"),self.synapse_browser.change_class_color)
        
        QtCore.QObject.connect(self.btn_findPrev,QtCore.SIGNAL("clicked()"),lambda direction="backward": self.SearchKeys(direction))
        QtCore.QObject.connect(self.btn_findNext,QtCore.SIGNAL("clicked()"),lambda direction="forward": self.SearchKeys(direction))
        QtCore.QObject.connect(self.btn_Reset,QtCore.SIGNAL("clicked()"),lambda: self.ResetSearch())


        QtCore.QObject.connect(self.btn_gotoPrev,QtCore.SIGNAL("clicked()"), lambda : self.synapse_browser.search_synapse("backward"))
        QtCore.QObject.connect(self.btn_gotoNext,QtCore.SIGNAL("clicked()"),  lambda : self.synapse_browser.search_synapse("forward"))

        QtCore.QObject.connect(self.btn_delete_syn,QtCore.SIGNAL("clicked()"),  lambda : self.synapse_browser.delete_synapse())

        QtCore.QObject.connect(self.btn_LoadDataset,QtCore.SIGNAL("clicked()"),self.AddDataset)
        QtCore.QObject.connect(self.btn_RemoveDataset,QtCore.SIGNAL("clicked()"),self.RemoveDataset)
        QtCore.QObject.connect(self.btn_SaveDatasetSettings,QtCore.SIGNAL("clicked()"),self.SaveDataset)
        
        self.ObjectBrowser.selectionModel().selectionChanged.connect(self.BrowserSelectionChanged)

        QtCore.QObject.connect(self.btn_LoadSkel2TIFFPath,QtCore.SIGNAL("clicked()"),lambda obj='Skel2TIFFPath': self.SetPath(obj))
        QtCore.QObject.connect(self.btn_CreateTIFFStack,QtCore.SIGNAL("clicked()"),self.Skel2TIFF)

        QtCore.QObject.connect(self.btn_SkelNodes2Soma,QtCore.SIGNAL("clicked()"),self.SkelNodes2Soma)

    def SearchTask(self,direction='next'):
        if not self.job:
            return
        checkstate=self.ckbx_skipDoneTasks.isChecked()
        if direction=='next':
            self.job.goto_next_task(checkstate)
            return
        if direction=='previous':
            self.job.goto_previous_task(checkstate)
            return

    def ChangeTaskState(self,done=None):
        if done==None:
            done=self._ckbx_TaskDone.isChecked()
        else:
            if done>0:
                done=True
                if not self._ckbx_TaskDone.isChecked():
                    self._ckbx_TaskDone.setChecked(1)
            else:
                done=False
                if self._ckbx_TaskDone.isChecked():
                    self._ckbx_TaskDone.setChecked(0)
            
        if (not self.job):
            return
        currTask=self.job.get_current_task()
        if currTask==None:
            return
        if currTask._done==done:
            return
        if done:
            self.job.DoneTasks+=1
        else:
            self.job.DoneTasks-=1            
        self.done_tasks.setText("Done tasks: %d/%d" % (self.job.DoneTasks,self.job.tasks.__len__()))

        currTask._done=done
        

    def ChangeShortcut(self,text=None):
        if text==None:
            return
        index=self._comboBox_Shortcuts.currentIndex()
        self._CommentShortcuts[index]=text
        self._comboBox_Shortcuts.setItemText(index,text)
        
    def ChangeColorScheme(self,index=None):
        if index==None:
            index=self.comboBox_Coloring.currentIndex()
        if index==0:
            1
        elif index==1:
            for ineuron,neuronID in enumerate(self.Neurons):
#                print neuronID
                color=self.get_autocolor(ineuron)                
                self.Neurons[neuronID].change_color(color)
        elif index==2:
            NNeurons=self.Neurons.__len__()
            if NNeurons<1:
                return
            LUT=vtk.vtkLookupTable()
            LUT.SetRampToLinear()
            LUT.SetNumberOfTableValues(NNeurons)
            LUT.SetTableRange(0,NNeurons-1)
            LUT.Build();

            zdir=np.array([-0.4410,-0.6302,0.6390],dtype=np.float);
            SomaCenter=list()
            NeuronIDs=list()
            for neuronId, neuron_obj in window1.Neurons.iteritems():
                if not "soma" in neuron_obj.children:
                    continue
                if neuron_obj.children["soma"]==[]:
                    continue
                Points= neuron_obj.children["soma"].data.GetPoints().GetData()
                NPoints=Points.GetNumberOfTuples()
                center=np.array([0,0,0],dtype=np.float)
                for ipoint in range(NPoints):
                    center+=Points.GetTuple(ipoint)
                SomaCenter.append(np.dot(zdir,center)/NPoints)
                NeuronIDs.append(neuronId)
                
            newColorIndex=sorted(range(SomaCenter.__len__()),key=lambda k: SomaCenter[k]);
            colorIdx=0
            for ineuron in newColorIndex:
                colorIdx+=1
                window1.Neurons[NeuronIDs[ineuron]].change_color(LUT.GetTableValue(colorIdx))
        if not (index==self.comboBox_Coloring.currentIndex()):
            self.comboBox_Coloring.setCurrentIndex(index)
        self.QRWin.Render()
        
    def DelConComp(self):
        nodeIds, seed=self.ExtractConComp("nodelist")
        if nodeIds==None:
            return
        self.Neurons[seed[1]].children[seed[0]].delete_node(nodeIds)     
        self.QRWin.Render()
    
    def ExtractConComp(self,format="data"):    
        SelObj=self.QRWin.SelObj
        if (not SelObj):
            return None,-1            
        if SelObj.__len__()<3:
            return None,-1
        NeuronID=SelObj[1]
        if not (NeuronID in self.Neurons):
            return None,-1
        if not (SelObj[0]=="skeleton"):
            return None,-1
        if not "skeleton" in self.Neurons[NeuronID].children:
            return None,-1
        child=self.Neurons[NeuronID].children["skeleton"]
        nodeId=SelObj[2]
        nodelist=child.extract_nodeId_seeded_region(nodeId)
        
        if nodelist.GetNumberOfTuples()==0:
            return None,-1

        if format=="nodelist":
            return nodelist, ["skeleton",NeuronID,nodeId]
            
        ConComp =vtk.vtkSelectionNode()
        ConComp.SetFieldType(vtk.vtkSelectionNode.POINT);
        ConComp.SetContentType(vtk.vtkSelectionNode.PEDIGREEIDS);        
        ConComp.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(),0);
        ConComp.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1);
        ConComp.SetSelectionList(nodelist)
        ConComp.GetSelectionList().DataChanged() #somehow this is not done internally by insertnextvalue
      
        Selection =vtk.vtkSelection()
        Selection.AddNode(ConComp);
        ExtractSelection =vtk.vtkExtractSelection()
        ExtractSelection.SetInputConnection(0,child.data.GetProducerPort());
        ExtractSelection.SetInputConnection(1,Selection.GetProducerPort());
        GeometryFilter=vtk.vtkGeometryFilter()
        GeometryFilter.SetInputConnection(ExtractSelection.GetOutputPort())
        GeometryFilter.Update()
        tempData=vtk.vtkPolyData()
        tempData.DeepCopy(GeometryFilter.GetOutput())
        tempData.GetPointData().SetActivePedigreeIds(None)
        tempData.Update()

        return tempData, ["skeleton",NeuronID,nodeId]
        
    def SplitByConComp(self,newNeuronID=None):
        tempData, seed=self.ExtractConComp("data")
        if tempData==None:
            return
        
        newNeuronID=self.NewNeuron(newNeuronID)
        if newNeuronID==None:
            return
        color=self.Neurons[newNeuronID].LUT.GetTableValue(\
            self.Neurons[newNeuronID].colorIdx)
        obj=skeleton(self.Neurons[newNeuronID].item,newNeuronID,color)
        self.Neurons[newNeuronID].children["skeleton"]=obj

        NodeID=tempData.GetPointData().GetArray("NodeID")
        obj.set_nodes(tempData.GetPoints(),NodeID)
        obj.add_branch(tempData.GetLines(),'Lines',1)
        
        obj.start_VisEngine(self)

        nodeIds=set(list(vtk_to_numpy(NodeID)))
        
        child=self.Neurons[seed[1]].children[seed[0]]
        for inode in nodeIds:
            obj.comments.set(inode,child.comments.get(inode))
            
        child.delete_node(nodeIds)     
        
        self.QRWin.SetActiveObj("skeleton",newNeuronID,seed[2])
        self.QRWin.GotoActiveObj()
        
    def MergeConComp(self):
        NeuronID=float(np.round(self._SpinBox_MergeNeuronId.value(),3))
        if not (NeuronID in self.Neurons):
            self.SplitByConComp(NeuronID)
            return
        
        tempData, seed=self.ExtractConComp("data")
        if tempData==None:
            return

        NPoints=tempData.GetNumberOfPoints()
        if NPoints==0:
            return
            
        if not "skeleton" in self.Neurons[NeuronID].children:
            color=self.Neurons[NeuronID].LUT.GetTableValue(\
                self.Neurons[NeuronID].colorIdx)
            obj=skeleton(self.Neurons[NeuronID].item,NeuronID,color)
            self.Neurons[newNeuronID].children["skeleton"]=obj
        else:           
            obj=self.Neurons[NeuronID].children["skeleton"]
        
        newNodeIds,newPointIdxs=obj.add_node(tempData.GetPoints())
        if newNodeIds==None:
            return
        if newPointIdxs.__class__.__name__=='list':
            startPointIdx=newPointIdxs[0]
            startNodeId=newNodeIds[0]
        else:
            startPointIdx=newPointIdxs
            startNodeId=newNodeIds
            
        tempLines=tempData.GetLines()
        tempLines.InitTraversal()
        NLines=tempData.GetNumberOfLines()
        if NLines>0:
            for icell in range(NLines):      
                tempCell=vtk.vtkIdList()     
                tempLines.GetNextCell(tempCell)
                for iid in range(tempCell.GetNumberOfIds()):
                    tempCell.SetId(iid,tempCell.GetId(iid)+startPointIdx)
                
                if icell==NLines-1:
                    obj.add_branch(tempCell,'vtkIdList',1)  #update of links
                else:
                    obj.add_branch(tempCell,'vtkIdList',0) #no updating of links

        NodeID=tempData.GetPointData().GetArray("NodeID")
        nodeIds=list(vtk_to_numpy(NodeID))
        
        child=self.Neurons[seed[1]].children[seed[0]]
        for inode,nodeId in enumerate(nodeIds):
            obj.comments.set(startNodeId+inode,child.comments.get(nodeId))    
            if nodeId==seed[2]:
                newSeedId=startNodeId+inode
        child.delete_node(nodeIds)     
        
        self.QRWin.SetActiveObj("skeleton",NeuronID,newSeedId)
        self.QRWin.GotoActiveObj()

    def MergeNeuron(self):        
        SelObj=self.QRWin.SelObj
        if (not SelObj):
            return           
        if SelObj.__len__()<3:
            return 
        SourceNeuronID=SelObj[1]
        if not (SourceNeuronID in self.Neurons):
            return
        if not (SelObj[0]=="skeleton"):
            return
        if not "skeleton" in self.Neurons[SourceNeuronID].children:
            return 
        source=self.Neurons[SourceNeuronID].children["skeleton"]
        tempData=source.data
        if tempData==None:
            return
        NPoints=tempData.GetNumberOfPoints()
        if NPoints==0:
            return

        NeuronID=float(np.round(self._SpinBox_MergeWholeNeuronId.value(),3))
        if not (NeuronID in self.Neurons):
            self.NewNeuron(NeuronID)

        if not "skeleton" in self.Neurons[NeuronID].children:
            color=self.Neurons[NeuronID].LUT.GetTableValue(\
                self.Neurons[NeuronID].colorIdx)
            target=skeleton(self.Neurons[NeuronID].item,NeuronID,color)
            self.Neurons[NeuronID].children["skeleton"]=target
        else:           
            target=self.Neurons[NeuronID].children["skeleton"]

        newNodeIds,newPointIdxs=target.add_node(tempData.GetPoints())
        if newPointIdxs.__class__.__name__=='list':
            startPointIdx=newPointIdxs[0]
            startNodeId=newNodeIds[0]
        else:
            startPointIdx=newPointIdxs
            startNodeId=newNodeIds
            
        tempLines=tempData.GetLines()
        tempLines.InitTraversal()
        NLines=tempData.GetNumberOfLines()
        if NLines>0:
            for icell in range(NLines):      
                tempCell=vtk.vtkIdList()     
                tempLines.GetNextCell(tempCell)
                for iid in range(tempCell.GetNumberOfIds()):
                    tempCell.SetId(iid,tempCell.GetId(iid)+startPointIdx)
                
                if icell==NLines-1:
                    target.add_branch(tempCell,'vtkIdList',1)  #update of links
                else:
                    target.add_branch(tempCell,'vtkIdList',0) #no updating of links

        NodeID=tempData.GetPointData().GetArray("NodeID")
        nodeIds=list(vtk_to_numpy(NodeID))
        
        for inode,nodeId in enumerate(nodeIds):
            target.comments.set(startNodeId+inode,source.comments.get(nodeId))    
            if nodeId==SelObj[2]:
                newSeedId=startNodeId+inode
        source.delete_node(nodeIds)     
        
        self.QRWin.SetActiveObj("skeleton",NeuronID,newSeedId)
        self.QRWin.GotoActiveObj()
        
    def NewNeuron(self,NeuronID=None,objtype='neuron'):
        if NeuronID==None:
            NeuronID=self._SpinBoxNeuronId.value()
            if NeuronID in self.Neurons:
                NeuronID=-1
                for neuronId, obj in self.Neurons.iteritems():
                    NeuronID=max(int(obj.NeuronID),NeuronID)
                NeuronID+=1
        
        if NeuronID in self.Neurons:
            print "There is already a neuron width id {0}.".format(NeuronID)
            return
        
        ineuron=self.Neurons.__len__()
        color=self.get_autocolor(ineuron)
        if objtype=='neuron':
            self.Neurons[NeuronID]=neuron(self.ObjectBrowser.model(),NeuronID,color)
        elif objtype=='area':
            self.Neurons[NeuronID]=area(self.ObjectBrowser.model(),NeuronID,color)
        else:
            print "Unknown objtype: ", objtype
            return
            
        self.Neurons[NeuronID].start_VisEngine(self)
        print "Created new neuron with ID {0}.".format(NeuronID)
        self.QRWin.SetActiveObj("neuron",NeuronID)
        self.Timer.action()
        self.GotoNode()
        return NeuronID

    
    def DelNeuron(self,NeuronID=None):
        if NeuronID==None:
            NeuronID=self.QRWin.SelObj[1]
        if not (NeuronID in self.Neurons):
            return
        self.Neurons[NeuronID].delete() 
        del self.Neurons[NeuronID]
        self.Timer.action()
        self.QRWin.Render()
#        self.GotoNode()
        
    
    def DelNode(self):
        if self.QRWin.SelObj[0]=="skeleton":
            self.QRWin.DeleteActiveObj()
        
    def GotoNode(self,source=None):
        if  source==None:
            NeuronId=self.QRWin.SelObj[1]        
        elif source=='spinbox1':
            NeuronId=float(np.round(self._SpinBoxNeuronId.value(),3))
        elif source=='spinbox2':
            NeuronId=float(np.round(self._SpinBoxNeuronId_2.value(),3))
        elif source=='spinbox3':
            NeuronId=float(np.round(self._SpinBoxNeuronId_3.value(),3))
#        print NeuronId
        oldObjType=self.QRWin.SelObj[0]
        oldNeuronId=self.QRWin.SelObj[1]
        if not self.Neurons:
            self._SpinBoxNeuronId.setValue(-1)
            self._SpinBoxNodeId.setValue(-1)
            self._SpinBoxNeuronId_2.setValue(-1)
            self._SpinBoxNodeId_2.setValue(-1)
            self._SpinBoxNeuronId_3.setValue(-1)
            self._SpinBoxNodeId_3.setValue(-1)
            return
        allNeurons=self.Neurons.keys()
        allNeurons.sort()
        if allNeurons.__len__()==0:
            NeuronId=-1
            NodeId=-1
        else:
            if not ( NeuronId in allNeurons):
                if not oldNeuronId in self.Neurons:
                    NeuronId=allNeurons[0]
                else:
                    if NeuronId>oldNeuronId:
                        newindex=allNeurons.index(oldNeuronId)+1
                    else:
                        newindex=allNeurons.index(oldNeuronId)-1
                    if newindex<0:
                        newindex=allNeurons.__len__()-1
                    elif newindex>(allNeurons.__len__()-1):
                        newindex=0
                    NeuronId=allNeurons[newindex]
    
            if oldObjType in self.Neurons[NeuronId].children:
                ObjType=oldObjType
            else:
                if "skeleton" in self.Neurons[NeuronId].children:
                    ObjType="skeleton"
                elif "soma" in self.Neurons[NeuronId].children:
                    ObjType="soma"
                else:
                    allChildren=self.Neurons[NeuronId].children.keys()
                    if allChildren.__len__()>0:
                        ObjType=allChildren[0]
                    else:
                        ObjType='None'
                    
            if  source==None:
                NodeId=self.QRWin.SelObj[2]        
            elif source=='spinbox1':
                NodeId=np.int(self._SpinBoxNodeId.value())
            elif source=='spinbox2':
                NodeId=np.int(self._SpinBoxNodeId_2.value())
            elif source=='spinbox3':
                NodeId=np.int(self._SpinBoxNodeId_3.value())

            oldNodeId=self.QRWin.SelObj[2]
            if (not ObjType) or (ObjType=='None'):
                self.QRWin.SetActiveObj("neuron",NeuronId,None)
            else:
                child=self.Neurons[NeuronId].children[ObjType]
                child.validData.Update()
                NodeID=child.validData.GetOutput().GetPointData().GetArray("NodeID")
                if NodeID==None:
                    self.QRWin.SetActiveObj("neuron",NeuronId,None)
                else:
                    NNodeIds=NodeID.GetNumberOfTuples()
                    if NNodeIds==0:
                        self.QRWin.SetActiveObj("neuron",NeuronId,None)
                    else:
                        if not child.isvalid_nodeId(NodeId):
                            if not child.isvalid_nodeId(oldNodeId):
                                NodeId=NodeID.GetValue(0)
                            else:
                                maxNodeId=NodeID.GetMaxNorm()
                                if NodeId>oldNodeId:
                                    NodeId=oldNodeId
                                    for inode in range(NNodeIds):
                                        NodeId+=1
                                        if NodeId>maxNodeId:
                                            NodeId=0
                                        if child.isvalid_nodeId(NodeId):
                                            break
                                else:
#                                    NodeId=oldNodeId
                                    for inode in range(NNodeIds):
                                        NodeId-=1
                                        if NodeId<0:
                                            NodeId=maxNodeId
                                        if child.isvalid_nodeId(NodeId):
                                            break
                        self._SpinBoxNodeId.setValue(np.int(NodeId))
                        self._SpinBoxNodeId_2.setValue(np.int(NodeId))
                        self._SpinBoxNodeId_3.setValue(np.int(NodeId))
                        self.QRWin.SetActiveObj(ObjType,NeuronId,NodeId)
                
        self._SpinBoxNeuronId.setValue(NeuronId)
        self._SpinBoxNeuronId_2.setValue(NeuronId)
        self._SpinBoxNeuronId_3.setValue(NeuronId)

        self.QRWin.GotoActiveObj()

    def ResetSearch(self):
        self.text_Comment_nav.setText("")

    def SearchKeys(self,direction):
        #process search mask
        keys=list()
        values=list()
        comment=unicode(self.text_Comment_nav.text())
        if not (not comment):
            keys.append("comment")
            values.append(comment)
        if not keys:
            return
        found=self.SearchComment(self.QRWin.SelObj,keys,values,direction)
        if (not found):
            print "No objects found for this search pattern"
        else:
#            print found
            self.QRWin.SetActiveObj(found[0],found[1],found[2])
            self.QRWin.GotoActiveObj()


    def SearchComment(self,StartObj,keys,values,direction="forward",WrapFlag=True):
        allNeurons=self.Neurons.keys()
        NNeurons=allNeurons.__len__()
        if NNeurons==0:
            return
        if (not StartObj):
            StartObj=[None,None,None]
        
        if StartObj[1]==None or (StartObj[1]<0) or not (StartObj[1] in allNeurons):
            if direction=="backward":
                ineuron=0
            else:
                ineuron=NNeurons-1
            NeuronID=allNeurons[ineuron]
            StartObj[1]=NeuronID
        else:
            ineuron=allNeurons.index(StartObj[1])
           
        if StartObj[0]==None or (StartObj[0]=='None'):
            StartObj[0]=self.Neurons[allNeurons[0]].__class__.__name__
#            StartObj[0]="neuron"
            
        if direction=="forward":
            neuronlist=range(ineuron,NNeurons)
            neuronlist.extend(range(0,ineuron))
        else:
            neuronlist=range(ineuron,-1,-1)
            neuronlist.extend(range(NNeurons-1,ineuron,-1))
        
        first=True
        first2=True
        for ineuron in neuronlist:
#            children=["neuron"]
            children=[self.Neurons[allNeurons[0]].__class__.__name__]
            children.extend(self.Neurons[allNeurons[ineuron]].children.keys())
            NChildren=children.__len__()
            if first:
                first=False                
                if StartObj[0] in children:
                    ichild=children.index(StartObj[0])
                else:
                    if direction=="forward":
                        ichild=0
                    else:
                        ichild=NChildren-1
            else:
                if direction=="forward":
                    ichild=0
                else:
                    ichild=NChildren-1

            if direction=="forward":
                childlist=range(ichild,NChildren)
    #                childlist.extend(range(0,ichild))
            else:
                childlist=range(ichild,-1,-1)
    #                childlist.extend(range(NChildren-1,ichild,-1))
            for ichild in childlist:
                if first2:
                    iid=StartObj[2]
                    first2=False
                else:
                    iid=None
                    
                if children[ichild]=="neuron" or children[ichild]=="area":
                    found=self.Neurons[allNeurons[ineuron]].search_comment(keys,values,iid,direction,False)
                else:
                    found=self.Neurons[allNeurons[ineuron]].children[children[ichild]].search_comment(keys,values,iid,direction,False)   
                if not (not found):
                    return found

            if direction=="forward":
                ichild=0
            else:
                ichild=NChildren-1

        if self.ckbx_Wraparound.isChecked() and WrapFlag:
            found=self.SearchComment([children[ichild],allNeurons[ineuron],None],\
                keys,values,direction,False)
            if not (not found):
                return found
        return None

    def ShowComments(self,NeuronID=None,objtype=None,Id=None):
        obj=None
        if NeuronID==None:
            objtype,NeuronID,Id=self.QRWin.SelObj
            
        if NeuronID==None:            
            comments=None
        elif not self.Neurons.has_key(NeuronID):
            comments=None
        elif not objtype:
            obj=self.Neurons[NeuronID]
        else:
            if self.Neurons[NeuronID].children.has_key(objtype):
                obj=self.Neurons[NeuronID].children[objtype]

        if objtype=="synapse" and not (obj==None):
            tagIdx=obj.nodeId2tagIdx(Id)
            if tagIdx==None:
                return
            nodeIds=obj.tagIdx2nodeId(tagIdx)
            if nodeIds==None:
                return
            Id=nodeIds[0]

        try:
            comments=obj.comments.get(Id)
        except:
            comments=OrderedDict()
        
        if comments==None:
            comments=OrderedDict()
        
        if comments.has_key("comment"):
            self._text_Comment.setText(comments["comment"])
            self._text_Comment_2.setText(comments["comment"])
            self._text_Comment_3.setText(comments["comment"])
            self._text_Comment_4.setText(comments["comment"])
        else:
            self._text_Comment.setText('')
            self._text_Comment_2.setText('')
            self._text_Comment_3.setText('')
            self._text_Comment_4.setText('')

        tagIdx=partner=certainty=className=item=None
        if objtype=="synapse" and not (obj==None):
            tagIdx=obj.nodeId2tagIdx(Id)
            partner=obj.comments.get(Id,"partner")
            certainty=str2num('int',obj.comments.get(Id,"certainty"))
            className=obj.comments.get(Id,"class")     
            for key,synclass in self.synapse_browser.classes.iteritems():
                if synclass._Name==className:
                    item=synclass.item[0]
                    break
        if tagIdx==None:
            self._Syn_label.setText("Synapse: None")
        else:
            self._Syn_label.setText("Synapse: %04d" % (Id/3.0))

        if (partner==None) or (partner=="None"):
            self._Syn_partner.setText("connects to: <b><font color='red'>None</font></b>")
        else:
            self._Syn_partner.setText("connects to: <b><font color='green'>%s</font></b>" % partner)

        if certainty==None:
            self._CertaintyLevel.setValue(0)
        else:
            self._CertaintyLevel.setValue(certainty)
        
        
        if item==None:
            self.synapse_browser.set_current_class(None,None)
        else:
            self.synapse_browser.imagelist.setCurrentItem(item);
            
        if (certainty==None) and (className==None):
            self.btn_Syn_assign.setStyleSheet("background-color: rgb(255, 0, 0)")
            self.btn_Syn_assign.setText("Assign")
        else:
            self.btn_Syn_assign.setStyleSheet("background-color: rgb(0, 255, 0)")
            self.btn_Syn_assign.setText("Assigned")

    def ChangeComment(self,objinfo=None,key=None,value=None):
        if not objinfo:
            return
        if not (self.QRWin.SynMode or (self.QRWin.TracingMode>0) or self.QRWin.TagMode):
            return

        if objinfo=="SelObj":
            objinfo=self.QRWin.SelObj
        NeuronID=objinfo[1]
        if not self.Neurons.has_key(NeuronID):
            return
        objtype=objinfo[0]
        if objtype=="neuron":
            obj=self.Neurons[NeuronID]
        elif objtype in self.Neurons[NeuronID].children:
            obj=self.Neurons[NeuronID].children[objtype]
        else:
            return

        Id=objinfo[2]
        if objtype=="synapse":
            tagIdx=obj.nodeId2tagIdx(Id)
            nodeIds=obj.tagIdx2nodeId(tagIdx)
            Id=nodeIds[0]
        obj.comments.set(Id,key,value)
        
        actiontime,dt=self.Timer.action()  
        obj.comments.set(Id,"time",actiontime)
#        print "changed comment"

    def RestrictVOI(self,state=False):
        objs2restrict=[skeleton]
        if self.ckbx_restrictVOI.isChecked():
            for iobj in objs2restrict:
                iobj.restrictVOI(True)
        else:
            for iobj in objs2restrict:
                iobj.restrictVOI(False)
        self.JumpToPoint()

    def ChangeNeuronVisMode(self,placeholder=-1,doRender=True):
        allobjs=[skeleton,synapse,tag,soma,region]
        if self.radioBtn_showall.isChecked():
            visibleNeurons=self.Neurons.keys()    
            hiddenNeurons=list()
        elif self.radioBtn_showactiveneuron.isChecked():
            SelObj=self.QRWin.SelObj
            if not (not SelObj):
                if SelObj.__len__()>1:
                    visibleNeurons=[self.QRWin.SelObj[1]]
                    hiddenNeurons=self.Neurons.keys()
                    if visibleNeurons[0] in hiddenNeurons:
                        hiddenNeurons.remove(visibleNeurons[0])
                    if not visibleNeurons[0] in self.Neurons.keys():
                        visibleNeurons=[]
        elif self.radioBtn_showconcomp.isChecked():
            SelObj=self.QRWin.SelObj
            if (not SelObj):
                return            
            if SelObj.__len__()<3:
                return
            if not (SelObj[0]=="skeleton"):
                return
            visibleNeurons=[SelObj[1]]
            hiddenNeurons=self.Neurons.keys()
            if visibleNeurons[0] in hiddenNeurons:
                hiddenNeurons.remove(visibleNeurons[0])
            if visibleNeurons[0] in self.Neurons.keys():
                for key, child in self.Neurons[visibleNeurons[0]].children.iteritems():
                    if key=="skeleton":
                        nodelist=child.extract_nodeId_seeded_region(SelObj[2]) 
                        if not nodelist:
                            return
                        VisibleNodes=child.data.GetPointData().GetArray("VisibleNodes")
                        NodeID=child.data.GetPointData().GetArray("NodeID")
                        NNodes=child.data.GetNumberOfPoints()
                        for inode in range(NNodes):
                            nodeId=NodeID.GetValue(inode)
                            if nodelist.LookupValue(nodeId)>-1:
                                VisibleNodes.SetValue(inode,1)
                            else:
                                VisibleNodes.SetValue(inode,0)
                        VisibleNodes.Modified()
                        child.visible=2 #partially visible
                    else: 
                        child.set_visibility(1)
            
            visibleNeurons=list()
            
        elif self.radioBtn_hideall.isChecked():
            for obj in allobjs:
                for vp in obj.viewports:
                    obj.hide_actors(vp)
            if doRender:
                self.QRWin.RenderWindow.Render()
            return
            
        for neuronId in visibleNeurons:
            self.Neurons[neuronId].set_visibility(1)

        for neuronId in hiddenNeurons: 
            self.Neurons[neuronId].set_visibility(0)

        for obj in allobjs:
            for vp in obj.viewports:
                obj.show_actors(vp)
            if doRender:
                self.QRWin.RenderWindow.Render()
    
    def ShowBoundingBox(self,state=None):
        if not hasattr(self,'BoundingBox'):
            return
        if "skeleton_viewport" in self.QRWin.viewports:
            viewport=self.QRWin.viewports["skeleton_viewport"]
        else:
            return

        if self.ckbx_ShowBoundingBox.isChecked():
            self.BoundingBox.AddBoundingBox(viewport)
        else:
            self.BoundingBox.RemoveBoundingBox(viewport)
        self.QRWin.Render()

    def ChangeBorderVisMode(self,state=None):
        
        hideborders=self.ckbx_HideBorder.isChecked()
        
        if not "skeleton_viewport" in self.QRWin.viewports:
            return
            
        iviewport=self.QRWin.viewports["skeleton_viewport"]
        
        for key,checkbox in [\
            ['Orth_planeROI',self.ckbx_ShowArbitScaleBar],
            ['YX_planeROI',self.ckbx_ShowYXScaleBar],
            ['YZ_planeROI',self.ckbx_ShowYZScaleBar],
            ['ZX_planeROI',self.ckbx_ShowZXScaleBar]]:
                         
            if key in iviewport._LinkedPlaneOutlines:
                if not (key in self.planeROIs):
                    continue
                plane=self.planeROIs[key]
                if checkbox.isChecked():
                    iviewport.AddViewProp(plane.ScaleBarActor)
                    plane.ScaleBarVisible=True
                else:
                    iviewport.RemoveViewProp(plane.ScaleBarActor)
                    plane.ScaleBarVisible=False
                        
        if hideborders>0:
            for key in iviewport._LinkedPlaneOutlines:
                if not (key in self.planeROIs):
                    continue
                plane=self.planeROIs[key]
                iviewport.RemoveViewProp(plane.OutlineActor)
                for intersection in iviewport.Intersections:
                    intersection.SetVisibility(0)        
        else:
            for key in iviewport._LinkedPlaneOutlines:
                if not (key in self.planeROIs):
                    continue
                plane=self.planeROIs[key]
                iviewport.AddViewProp(plane.OutlineActor)
                for intersection in iviewport.Intersections:
                    intersection.SetVisibility(1)
        self.QRWin.RenderWindow.Render()

    def ChangeSynZoom(self,synzoom=None):
        if synzoom==None:
            synzoom=self.ckbx_SynZoom.isChecked()
        if synzoom>0:
            self.QRWin.SynZoom=1
            for key,iviewport in self.QRWin.viewports.iteritems():
                if key=="skeleton_viewport":
                    continue;
                iviewport.ResetViewport()
        else:
            self.QRWin.SynZoom=0
    
    def SynchronizedZoom(self,dollyFactor):
        for key,iviewport in self.QRWin.viewports.iteritems():
            if key=="skeleton_viewport":
                continue;
#            old_distance=iviewport.Camera.GetDistance()
#            if old_distance==0.0:
#                old_distance=1.0
            iviewport.Zoom(dollyFactor)
    
    def ChangeScaleBarColor(self):
        color=None
        if "skeleton_viewport" in self.QRWin.viewports:
            iviewport=self.QRWin.viewports["skeleton_viewport"]       
            for key in iviewport._LinkedPlaneOutlines:
                if not (key in self.planeROIs):
                    continue
                plane=self.planeROIs[key]
                color=plane._ScaleBarColor
                break
        if not color:
            color=[1.0,0.0,0.0]
        color=QtGui.QColor().fromRgb(*color)
        color=QtGui.QColorDialog.getColor(color,self, "Scale bar color")

        if not color.isValid():
            return
        color=color.getRgb() 
        color=(color[0]/255.0,color[1]/255.0,color[2]/255.0)

        if not "skeleton_viewport" in self.QRWin.viewports:
            return
            
        iviewport=self.QRWin.viewports["skeleton_viewport"]
        
        for key in iviewport._LinkedPlaneOutlines:
            if not (key in self.planeROIs):
                continue
            plane=self.planeROIs[key]
            plane.UpdateScaleBars(None,None,color)
        self.QRWin.Render() 


    def ChangeScaleBar(self,parameter=None,value=None):
#        if self.ckbx_ShowScaleBar.isChecked():
#            visibility=True;
#        else:
#            visibility=False;
#        tempTxt=str(self.text_ScaleBarPos.text())
#        if not tempTxt:
#            pos=None
#        else:
#            pos=np.array(ast.literal_eval(tempTxt))
#            pos=np.multiply(pos,self.DataScale)
        length=self.SpinBoxScaleBar.value()*1000.0;
        width=self.SpinBox_ScaleBarWidth.value();
        if not "skeleton_viewport" in self.QRWin.viewports:
            return
            
        iviewport=self.QRWin.viewports["skeleton_viewport"]
        
        for key in iviewport._LinkedPlaneOutlines:
            if not (key in self.planeROIs):
                continue
            plane=self.planeROIs[key]
            plane.UpdateScaleBars(length,width)
        self.QRWin.Render() 
        
        
    def ToggleClipHulls(self,vp=None,state=None):
        ClippingClasses=[soma,region]
        if not vp:
            return
        if vp=='skeleton_viewport':
            if not state: 
                state=self.ckbx_ClipHulls.isChecked()
        if not (vp in self.QRWin.viewports):
            return
        viewport=self.QRWin.viewports[vp]
        viewport._ClipHulls=state

        planeKeys = [viewport._ViewportPlane]+ viewport._LinkedPlanes
        for planeKey in planeKeys:
            if not planeKey:
                continue
            if not self.planeROIs.has_key(planeKey):
                continue
            if self.planeROIs[planeKey].ClippingPlane.__len__()<1:
                continue
            FirstClippingPlane=self.planeROIs[planeKey].ClippingPlane[0]
            if not FirstClippingPlane:
                continue
            for obj in ClippingClasses:
                if not (FirstClippingPlane in obj.ClippingPlanes):
                    continue
                ind=obj.ClippingPlanes.index(FirstClippingPlane)
                PlaneClipperActor=obj.ClippingActors[ind]
                if not PlaneClipperActor:
                    continue
                if state:
                    if obj.allData.GetNumberOfInputConnections(0)==0:
                        continue
                    if not viewport.HasViewProp(PlaneClipperActor):
                        viewport.AddActor(PlaneClipperActor)
                else:
                    viewport.RemoveActor(PlaneClipperActor)

        for obj in ClippingClasses:
            if state:
                if obj.actor.GetClassName()=='vtkOpenGLActor':
                    viewport.RemoveActor(obj.actor)                     
                elif obj.actor.GetClassName()=='vtkActorCollection':
                    for iactor in range(obj.actor.GetNumberOfItems()):
                        actor=obj.actor.GetItemAsObject(iactor)
                        viewport.RemoveActor(actor)
            else:
                if obj.actor.GetClassName()=='vtkOpenGLActor':
                    if obj.allData.GetNumberOfInputConnections(0)==0:
                            continue
                    if not viewport.HasViewProp(obj.actor):
                        viewport.AddActor(obj.actor)                     
                elif obj.actor.GetClassName()=='vtkActorCollection':
                    if obj.allData.GetNumberOfInputConnections(0)==0:
                            continue
                    for iactor in range(obj.actor.GetNumberOfItems()):
                        actor=obj.actor.GetItemAsObject(iactor)
                        if not viewport.HasViewProp(actor):
                            viewport.AddActor(actor)
                        
        self.QRWin.RenderWindow.Render()
            

            
    def ChangePlaneVisMode(self,state=None):
        hide_YXplane=self.ckbx_HideYXplane.isChecked()
        hide_YZplane=self.ckbx_HideYZplane.isChecked()
        hide_ZXplane=self.ckbx_HideZXplane.isChecked()
        hide_arbitplane=self.ckbx_Hidearbitplane.isChecked()

        if not "skeleton_viewport" in self.QRWin.viewports:
            return
        iviewport=self.QRWin.viewports["skeleton_viewport"]
        
        plane=iviewport.LinkedPlaneROIs["YX_planeROI"]
        if hide_YXplane:
            iviewport.RemoveViewProp(plane.PlaneActor)
        else:
            iviewport.AddViewProp(plane.PlaneActor)

        plane=iviewport.LinkedPlaneROIs["YZ_planeROI"]
        if hide_YZplane:
            iviewport.RemoveViewProp(plane.PlaneActor)
        else:
            iviewport.AddViewProp(plane.PlaneActor)

        plane=iviewport.LinkedPlaneROIs["ZX_planeROI"]
        if hide_ZXplane:
            iviewport.RemoveViewProp(plane.PlaneActor)
        else:
            iviewport.AddViewProp(plane.PlaneActor)

        plane=iviewport.LinkedPlaneROIs["Orth_planeROI"]
        if hide_arbitplane:
            iviewport.RemoveViewProp(plane.PlaneActor)
        else:
            iviewport.AddViewProp(plane.PlaneActor)

        self.QRWin.RenderWindow.Render()
        
    def ChangeSkelVisMode(self,state=0,neuron_obj=None):
        showNodes=not self.HideSkelNodes.isChecked()
        if self.radioBtn_lines2Dpoints.isChecked():
            LineStyle="lines"
            PointStyle="points"
            vis3D="off"
        elif self.radioBtn_lines3Dpoints.isChecked():
            LineStyle="lines"
            PointStyle="spheres"
            vis3D="off"
        elif self.radioBtn_tubesflat.isChecked():
            LineStyle="tubes"
            PointStyle="spheres"
            vis3D="off"
        elif self.radioBtn_full3D.isChecked():
            LineStyle="tubes"
            PointStyle="spheres"
            vis3D="on"
            
        for obj in [skeleton,NodeSelection]:
            obj.DefaultLineStyle[0]=LineStyle
            obj.DefaultPointStyle[0]=PointStyle
            obj.Defaultvis3D[0]=vis3D
            obj.DefaultshowNodes[0]=showNodes
            obj.update_VisEngine(LineStyle, PointStyle, vis3D, showNodes)

        self.QRWin.RenderWindow.Render()

    def ChangeSynVisMode(self,state=0,neuron_obj=None):
        showNodes=not self.HideSynNodes.isChecked()
        startNodeOnly=self.SynStartNodeOnly.isChecked()
        if self.radioBtn_Synlines2Dpoints.isChecked():
            LineStyle="lines"
            PointStyle="points"
            vis3D="off"
        elif self.radioBtn_Synlines3Dpoints.isChecked():
            LineStyle="lines"
            PointStyle="spheres"
            vis3D="off"
        elif self.radioBtn_Syntubesflat.isChecked():
            LineStyle="tubes"
            PointStyle="spheres"
            vis3D="off"
        elif self.radioBtn_Synfull3D.isChecked():
            LineStyle="tubes"
            PointStyle="spheres"
            vis3D="on"

        for obj in [synapse]:     
            obj.DefaultStartNodeOnly[0]=startNodeOnly
            obj.DefaultLineStyle[0]=LineStyle
            obj.DefaultPointStyle[0]=PointStyle
            obj.Defaultvis3D[0]=vis3D
            obj.DefaultshowNodes[0]=showNodes
            obj.update_VisEngine(LineStyle, PointStyle, vis3D, showNodes,startNodeOnly)

        for obj in [tag]:
            obj.DefaultLineStyle[0]=LineStyle
            obj.DefaultPointStyle[0]=PointStyle
            obj.Defaultvis3D[0]=vis3D
            obj.DefaultshowNodes[0]=showNodes
            obj.update_VisEngine(LineStyle, PointStyle, vis3D, showNodes)

        self.QRWin.RenderWindow.Render()

    def ResetWorkingModes(self):
        self.radioBtn_Browsing.setEnabled(1)
        self.radioBtn_Tracing.setEnabled(1)
        self.radioBtn_Tagging.setEnabled(1)
        self.radioBtn_Synapses.setEnabled(1)

    def ChangeWorkingMode(self,state):
        if self.radioBtn_Browsing.isChecked():
            self.QRWin.TracingMode=0
            self.QRWin.SynMode=0
            self.QRWin.TagMode=0
            self._comboBox_Shortcuts.setEnabled(0)
            self._text_Comment.setEnabled(0)
            self._text_Comment_2.setEnabled(0)
            self._text_Comment_3.setEnabled(0)
            self._text_Comment_4.setEnabled(0)


        elif self.radioBtn_Tracing.isChecked():
            if self.radioBtn_singlenodes.isChecked():
                self.QRWin.TracingMode=2
            else:
                self.QRWin.TracingMode=1
            self.QRWin.SynMode=0
            self.QRWin.TagMode=0
            self._comboBox_Shortcuts.setEnabled(1)

            self._text_Comment.setEnabled(1)
            self._text_Comment_2.setEnabled(1)
            self._text_Comment_3.setEnabled(1)
            self._text_Comment_4.setEnabled(1)

        elif self.radioBtn_Synapses.isChecked():
            self.QRWin.TracingMode=0
            self.QRWin.SynMode=1
            self.QRWin.TagMode=0
            self.Job.setVisible(1)
            self.JobTab.setCurrentWidget(self.SynAnnotation)
            self._comboBox_Shortcuts.setEnabled(0)
            self._text_Comment.setEnabled(1)
            self._text_Comment_2.setEnabled(1)
            self._text_Comment_3.setEnabled(1)
            self._text_Comment_4.setEnabled(1)
        elif self.radioBtn_Tagging.isChecked():
            self.QRWin.TracingMode=0
            self.QRWin.SynMode=0
            self.QRWin.TagMode=1
            self._comboBox_Shortcuts.setEnabled(1)
            self._text_Comment.setEnabled(1)
            self._text_Comment_2.setEnabled(1)
            self._text_Comment_3.setEnabled(1)
            self._text_Comment_4.setEnabled(1)
               
        self.UpdateWindowTitle()

    def ChangeBrightness(self,lower=None,upper=None):
        for key, plane in self.planeROIs.iteritems():
            templower=lower
            tempupper=upper
#            oldrange=plane.Table.GetRange()
            if templower==None:
                templower=self.span_brightness.lowerPosition
#                templower=oldrange[0]
            if tempupper==None:
                tempupper=self.span_brightness.upperPosition
#                tempupper=oldrange[1]
            plane.Table.SetValueRange(templower/255.0,tempupper/255.0) # image intensity range
            plane.Table.Modified()
#        print lower, upper
        self.QRWin.Render()
        
    def ChangeContrast(self,lower=None,upper=None):
        for key, plane in self.planeROIs.iteritems():
            templower=lower
            tempupper=upper
#            oldrange=plane.Table.GetRange()
            if templower==None:
                templower=self.span_contrast.lowerPosition
#                templower=oldrange[0]
            if tempupper==None:
                tempupper=self.span_contrast.upperPosition
#                tempupper=oldrange[1]
            plane.Table.SetRange(templower,tempupper) # image intensity range
            plane.Table.Modified()
#        print lower, upper
        self.QRWin.Render()
        
    def CopyCoords(self):
        text=u"[{0},{1},{2}]".format(self.SpinBoxX.value(),\
            self.SpinBoxY.value(),self.SpinBoxZ.value())
        QtGui.QApplication.clipboard().setText(text)    
        
        
    def CoordChanged(self,text):
        if not (QtGui.QApplication.clipboard().text() == text):
            return            
        text=unicode(text)
        text=text.replace('[','')
        text=text.replace(']','')
        text=text.replace('(','')
        text=text.replace(')','')
        if ',' in text:
            text=text.split(',')
        elif ';' in text:
            text=text.split(';')
        else:
            text=text.split()                    
        if not text.__len__()==3:
            print "Invalid set of coordinates. Coordinates have to be seperated in either form: 1,2,3 or 1 2 3 or 1;2;3"
            return
        coords=list()
        for icoord in range(3):
            try:
                coords.append(float(text[icoord]))                  
            except ValueError:
                print "Invalid set of coordinates. Coordinates have to be seperated in either form: 1,2,3 or 1 2 3 or 1;2;3"
                return
        self.SpinBoxX.setValue(coords[0])
        self.SpinBoxY.setValue(coords[1])
        self.SpinBoxZ.setValue(coords[2])
        print "Pasted coordinates: {0},{1},{2}".format(coords[0],coords[1],coords[2])

    def JumpToPoint(self,NewFocalPoint=None,cDir=None,vDir=None,hDir=None):
        if invalidvector(NewFocalPoint):
            x=self.SpinBoxX.value()*self.DataScale[0]
            y=self.SpinBoxY.value()*self.DataScale[1]
            z=self.SpinBoxZ.value()*self.DataScale[2]
            self.QRWin.setFocus()
            NewFocalPoint=np.array([x,y,z],dtype=np.float);
        else:
            self.SpinBoxX.setValue(np.int(NewFocalPoint[0]/self.DataScale[0]))       
            self.SpinBoxY.setValue(np.int(NewFocalPoint[1]/self.DataScale[1]))
            self.SpinBoxZ.setValue(np.int(NewFocalPoint[2]/self.DataScale[2]))

        CubeLoader.UpdatePosition(NewFocalPoint)

        for key, iplane in self.planeROIs.iteritems():
            iplane.JumpToPoint(NewFocalPoint,cDir,vDir,hDir)
            
        for key, iviewport in self.QRWin.viewports.iteritems():
            if iviewport._FollowFocalPoint:
                iviewport.JumpToPoint(NewFocalPoint,cDir,vDir)

        self.UpdateDemDriFiles(NewFocalPoint)

        BoxSize=self.SpinBoxVOISize.value()*1000; #convert to nm
        CurrBounds=skeleton.VOIExtent
        if not ((BoxSize==CurrBounds[1]-CurrBounds[0]) and (NewFocalPoint[0]>CurrBounds[0]+BoxSize*0.2) and (NewFocalPoint[0]<CurrBounds[1]-BoxSize*0.2) and    
            (NewFocalPoint[1]>CurrBounds[2]+BoxSize*0.2) and (NewFocalPoint[1]<CurrBounds[3]-BoxSize*0.2) and    
            (NewFocalPoint[2]>CurrBounds[4]+BoxSize*0.2) and (NewFocalPoint[2]<CurrBounds[5]-BoxSize*0.2)):
            
            skeleton.VOIExtent=[NewFocalPoint[0]-BoxSize/2,NewFocalPoint[0]+BoxSize/2,\
                NewFocalPoint[1]-BoxSize/2,NewFocalPoint[1]+BoxSize/2,\
                NewFocalPoint[2]-BoxSize/2,NewFocalPoint[2]+BoxSize/2]
        if self.ckbx_restrictVOI.isChecked():
            skeleton.VOIFilter.SetExtent(skeleton.VOIExtent)
            skeleton.activeVOIFilter.SetExtent(skeleton.VOIExtent)
#                print "Updated VOI bounds: BoxSize=", BoxSize

        self.QRWin.Render_Intersect()

    def RemoveDataset(self):
        1
        
    def AddDataset(self):
        if os.path.isfile(self.CurrentDataset[1]):
            filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load dataset...",self.CurrentDataset[1],"*.conf");
        else:
            if os.path.isdir(application_path):
                filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load dataset...",application_path,"*.conf");
            else:
                filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load dataset...","","*.conf");
        if filelist.__len__()==0:
            return
        
        try:
            filename=unicode(filelist.last())           
        except:
            filename=unicode(filelist[filelist.__len__()-1])
            
        dataset=Dataset(self,filename)
#        if "skeleton_viewport" in self.QRWin.viewports:
#            dataset.add_toviewport(self.QRWin.viewports["skeleton_viewport"])
        for key,iviewport in self.QRWin.viewports.iteritems():
            dataset.add_toviewport(iviewport)

        dataset.LoadData()
        self.QRWin.RenderWindow.Render()
        self.Datasets=[dataset]

    def SaveDataset(self):
        if self.Datasets.__len__()==0:
            return
        self.Datasets[0].SaveDataSetInformation()
        
    def SetDefaultDataPath(self):
        if os.path.isdir(self._DefaultDataPath):
            startpath=self._DefaultDataPath
        else:
            startpath=application_path
            
        datapath = QtGui.QFileDialog.getExistingDirectory(self,"Choose the default data directory...",\
            startpath)
        if not (not datapath):
            self._DefaultDataPath=unicode(datapath)
        else:
            datapath=None
        return datapath
        
    def WorkingOffline(self):
        if self.ActionWorkingOffline.isChecked():
            CubeLoader.WorkingOffline[0]=1;
            self._WorkingOffline=1;
        else:
            CubeLoader.WorkingOffline[0]=0;
            self._WorkingOffline=0;
        self.UpdateWindowTitle()
        
        
    def LoadCubeDataset(self):
        if os.path.isfile(self.CurrentDataset[1]):
            filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load dataset...",self.CurrentDataset[1],"*.conf");
        else:
            if os.path.isdir(application_path):
                filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load dataset...",application_path,"*.conf");
            else:
                filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load dataset...","","*.conf");
        if filelist.__len__()==0:
            return
        self.ChangeCubeDataset(unicode(filelist[filelist.__len__()-1]))
    
    def LoadRecentCubeDataset(self):
        action = self.sender()
        if action:
            self.ChangeCubeDataset(action._File)
        
    def ChangeCubeDataset(self,filename=None,recenter=1):
        if not filename:
            filename=CubeLoader.filename;
        if not filename:
            return -1;
            
        if not CubeLoader.doload:
            print "Loader switched off"
            return 0

        if os.path.isfile(filename):
            DatasetName=CubeLoader.LoadDatasetInformation(filename)
            if not DatasetName:
                print "Error: Could not find/load dataset config file {0}".format(filename)
                return -1                
            try:         
                CubeLoader.LoadDataset()
            except:
                print "Error: Could not find/load dataset {0}".format(filename)
                return -1
        else:
            found=0
            for idataset in range(self.menuDatasets.__len__()-1,-1,-1):
                dataset=self.menuDatasets[idataset]
                if dataset._Name==filename:
                    DatasetName=CubeLoader.LoadDatasetInformation(dataset._File)
                    if not DatasetName:
                        found=0
                        continue;
                    else:
                        filename=dataset._File
                        found=1
                    try:         
                        CubeLoader.LoadDataset()
                        break;
                    except:
                        found=0
                        print "Error: Could not load dataset {0}".format(filename)
                        continue;
            if not found:
                print "Error: Could not find dataset {0}".format(filename)
                return -1
            
        for dim in range(3):
            self.DataScale[dim]=CubeLoader._DataScale[dim]

        print "Changed dataset {0}".format(filename)
        self.CurrentDataset=[DatasetName,filename]

        if any([dataset._File==filename for dataset in self.menuDatasets]):
            found=0
            for idataset in range(self.menuDatasets.__len__()-1):
                dataset=self.menuDatasets[idataset]
                if dataset._File==filename:
                    found=1
                if not found:
                    continue
                prev_dataset=self.menuDatasets[idataset+1]            
                dataset._Name=prev_dataset._Name
                dataset._File=prev_dataset._File
                text = "&%d %s" % (self.menuDatasets.__len__() - idataset, dataset._Name)
                dataset.setText(text)
                dataset.setVisible(not dataset._Name=='')
            dataset=self.menuDatasets[self.menuDatasets.__len__()-1]
            dataset._Name=DatasetName
            dataset._File=filename
            text = "&%d %s" % (1, DatasetName)
            dataset.setText(text)
            dataset.setVisible(not DatasetName=='')
        else:
            for idataset in range(self.menuDatasets.__len__()-1):
                dataset=self.menuDatasets[idataset]
                prev_dataset=self.menuDatasets[idataset+1]
                dataset._Name=prev_dataset._Name
                dataset._File=prev_dataset._File
                text = "&%d %s" % (self.menuDatasets.__len__() - idataset, dataset._Name)
                dataset.setText(text)
                dataset.setVisible(not dataset._Name=='')
            dataset=self.menuDatasets[self.menuDatasets.__len__()-1]
            dataset._Name=DatasetName
            dataset._File=filename
            text = "&%d %s" % (1, DatasetName)
            dataset.setText(text)
            dataset.setVisible(not DatasetName=='')
            
        self.initViewports()

#        ROISize=2.0*np.floor((CubeLoader._NCubesPerEdge[0]-1)*128.0/np.sqrt(2.0)/2.0)-1
#        InterPolFactor=CubeLoader.InterPolFactor;
#        if ROISize>361:
        ROISize=361
        InterPolFactor=(CubeLoader._NCubesPerEdge[0]-1)*float(CubeLoader._CubeSize[0])/np.sqrt(2.0)/361;
        CubeLoader.InterPolFactor=min(2.0,max(1.0,InterPolFactor));
        if InterPolFactor>2.0:
            ROISize=2.0*np.floor(361*InterPolFactor/4.0)+1.0

#        ROISize=540
        print "ROISize: ", ROISize

        for key, plane in self.planeROIs.iteritems():
           plane.SetImageSource([ROISize,ROISize])

        if hasattr(self,'synapse_browser'):
            self.synapse_browser.readin_classes(DatasetName)
                        
        bounds=(CubeLoader._DataScale[0]*CubeLoader._Origin[0],CubeLoader._DataScale[0]*(CubeLoader._Origin[0]+CubeLoader._Extent[0]),\
            CubeLoader._DataScale[1]*CubeLoader._Origin[1],CubeLoader._DataScale[1]*(CubeLoader._Origin[1]+CubeLoader._Extent[1]),\
            CubeLoader._DataScale[2]*CubeLoader._Origin[2],CubeLoader._DataScale[2]*(CubeLoader._Origin[2]+CubeLoader._Extent[2]));

        if "skeleton_viewport" in self.QRWin.viewports:
            viewport=self.QRWin.viewports["skeleton_viewport"]
            viewport.ResetCamera(bounds[0],bounds[1],bounds[2],bounds[3],bounds[4],bounds[5]);
            viewport.CenterCross.updatePosition()            

        if hasattr(self,'BoundingBox'):
            self.BoundingBox.UpdateBounds(bounds)
        
        if recenter:
            self.JumpToPoint(np.array([(bounds[1]+bounds[0])/2.0,(bounds[3]+bounds[2])/2.0,(bounds[5]+bounds[4])/2.0]))    

        self.UpdateWindowTitle()
        return 1

    def OpenRecentFile(self):
        action = self.sender()
        if action:
            self.Open(action._File)
            
    def UpdateCurrentFile(self,filename):
        if any([recentFile._File==filename for recentFile in self.Filelist]):
            found=0
            for ifile in range(self.Filelist.__len__()-1,0,-1):                
                recentFile=self.Filelist[ifile]
                if recentFile._File==filename:
                    found=1
                if not found:
                    continue
                prev_recentFile=self.Filelist[ifile-1]            
                recentFile._Name=prev_recentFile._Name
                recentFile._File=prev_recentFile._File
                text = "&%d %s" % (ifile+1, recentFile._Name)
                recentFile.setText(text)
                recentFile.setVisible(not recentFile._Name=='')
        else:
            for ifile in range(self.Filelist.__len__()-1,0,-1):
                recentFile=self.Filelist[ifile]
                prev_recentFile=self.Filelist[ifile-1]
                recentFile._Name=prev_recentFile._Name
                recentFile._File=prev_recentFile._File
                text = "&%d %s" % (ifile+1, recentFile._Name)
                recentFile.setText(text)
                recentFile.setVisible(not recentFile._Name=='')
        recentFile=self.Filelist[0]
        recentFile._Name=os.path.basename(filename)
        recentFile._File=filename
        text = "&%d %s" % (1, recentFile._Name)
        recentFile.setText(text)
        recentFile.setVisible(not recentFile._Name=='')
        
        self.CurrentFile=filename
    
    def LoadFileList(self,filename):
        Neurons=OrderedDict()
        SelObj=['neuron',None,None]
        txtfile = open(filename, 'r') 
        editPosition=None
        dataset=None
        for line in txtfile:
            linestr=line.strip().split(',')
            filename=linestr[0]
            if filename==None or filename=='':# or filename.endswith('.txt'):
                continue
            tempNeurons,SelObj,editPosition,dataset=self.Open(filename,"Append",1)
            if tempNeurons==None:
                continue
            color=None
            if linestr.__len__()>1:
                color=linestr[1]
                if color=='auto' or  color=='AUTO' or  color=='Auto':
                    1
                else:
                    toreplace=['[',']','(',')']
                    for ichar in toreplace: color=color.replace(ichar,'')
                    color=[float(x) for x in color.split()]   
#                if not (self.comboBox_Coloring.currentIndex()==0):
#                    self.comboBox_Coloring.setCurrentIndex(0)
            flags=None
            if linestr.__len__()>2:
                flags=linestr[2]
            ineuron=-1
            if not (not Neurons):
                ineuron+=Neurons.__len__()
            if not (not self.Neurons):
                ineuron+=self.Neurons.__len__()
            for neuronId, neuron_obj in tempNeurons.iteritems():
                ineuron+=1
                if neuronId in Neurons:
                    oldneuronId=neuronId
                    step=1
                    while Neurons.has_key(neuronId):
                        neuronId=oldneuronId+step*0.001
                        neuronId=float(np.round(neuronId,3))
                        step+=1

                    print "tree id: {0}".format(neuronId)
                    neuron_obj.set_new_neuronId(neuronId)
                if color=='auto' or  color=='AUTO' or  color=='Auto':
                    print "ineuron=",ineuron
                    color=self.get_autocolor(ineuron)
                if not (not color):
                    neuron_obj.change_color(color)
                if not (not flags):
                    neuron_obj.addflags(flags)
                    
                Neurons[neuronId]=neuron_obj
        return Neurons,SelObj,filename,editPosition,dataset
    
    def New(self):
        oldMode=self.ObjectBrowser.selectionMode();
        self.ObjectBrowser.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        for neuronId, neuron_obj in self.Neurons.iteritems():
            neuron_obj.delete()            
        self.Neurons.clear()

        for filename,ddobj in self.DemDriFiles.iteritems():
            ddobj.delete()
        self.DemDriFiles.clear()

        self.ObjectBrowser.setSelectionMode(oldMode)
        del self.job
        self.job=None
        self.QRWin.SelObj=[None,None,None]
        self.Timer.reset()
        self.CurrentFile='';
        self.ResetWorkingModes()
        self.UpdateWindowTitle()
        self.QRWin.RenderWindow.Render()

    def Open(self,filename=None,LoadingMode=None,SilentMode=0,UpdateCurrentFile=1):
        InitializeJob=UpdateCurrentFile;
        
        fileext="*.nmx *.amx *.txt *.csv *.mat *.nml *.aml *.ddx"
        if filename==None:
            if not self.CurrentFile and self.Filelist.__len__()>0:
                currentPath= os.path.split(unicode(self.Filelist[0]._File))
            else:
                currentPath= os.path.split(unicode(self.CurrentFile))
            if os.path.isdir(currentPath[0]):
                filelist = QtGui.QFileDialog.getOpenFileNames(self,"Open file...",currentPath[0],fileext);
            else:
                if os.path.isdir(application_path):
                    filelist= QtGui.QFileDialog.getOpenFileNames(self,"Open file...",application_path,fileext);
                else:
                    filelist= QtGui.QFileDialog.getOpenFileNames(self,"Open file...","",fileext);
        else:
            if filename.__class__.__name__=='list':
                filelist=filename;
            else:
                if not filename:
                    if SilentMode:    
                        return OrderedDict(), [None,None,None], None, None
                    return                    
                if not os.path.isfile(filename):
                    if SilentMode:    
                        return OrderedDict(), [None,None,None], None, None
                    return
                filelist=[QtCore.QString(filename)]                
        if filelist.__len__()==0:
            if SilentMode:    
                return OrderedDict(), [None,None,None], None, None
            return
        
        reply=0
        if self.Neurons.__len__()>0:
            if LoadingMode=="Overwrite":
                reply=0
            elif LoadingMode=="Append":
                reply=1
            else:
                reply = QtGui.QMessageBox.question(self, 'Message',
                    "There are already neurons loaded. What do you want to do?", QtCore.QString("Overwrite"),  QtCore.QString("Append"), QtCore.QString("Cancel"))
            if reply == 0: #overwrite
                oldMode=self.ObjectBrowser.selectionMode();
                self.ObjectBrowser.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
                for neuronId, neuron_obj in self.Neurons.iteritems():
                    neuron_obj.delete()            
                self.Neurons.clear()
                self.ObjectBrowser.setSelectionMode(oldMode)
                
                for filename,ddobj in self.DemDriFiles.iteritems():
                    ddobj.delete()
                self.DemDriFiles.clear()
                
                del self.job
                self.job=None
                self.QRWin.SelObj=[None,None,None]
                self.Timer.reset()
            elif reply == 1:
                1
                #do not do anything, append neurons
            else:
                if SilentMode:    
                    return OrderedDict(), [None,None,None], None, None
                return

        startTime=time.time()
        Neurons=OrderedDict()
        SelObj=[None,None,None]
        editPosition=[None,None,None]
        dataset=None
        
        if SilentMode:
            showprogress=0
        else:
            showprogress=1
        if showprogress:
            progress = QtGui.QProgressDialog("Loading file(s)...","Cancel", 0,filelist.__len__(), self)
            progress.setWindowTitle("Wait")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(1500)
            ifile=0;
        
        
        for filename in filelist:
            if showprogress:
                if progress.wasCanceled():
                    break
                progress.setValue(ifile)
                ifile+=1

            filename=unicode(filename)
#            try:
            if filename.endswith(".ddx"):
                filename=unicode(filename)
                if filename in self.DemDriFiles:
                    continue
                ddfobj=self.LoadDDXFile(filename);
                if not (not ddfobj):
                    dataset=ddfobj._Dataset
                    self.DemDriFiles[filename]=ddfobj

            elif filename.endswith(".nmx"):
                filename=unicode(filename)
                tempNeurons,SelObj,editPosition,dataset=self.LoadNMXFile(filename)
                for NeuronID, neuron_obj in tempNeurons.iteritems():
                    if NeuronID in Neurons:
                        oldNeuronID=NeuronID
                        step=1
                        while Neurons.has_key(NeuronID):
                            NeuronID=oldNeuronID+step*0.001
                            NeuronID=float(np.round(NeuronID,3))
                            step+=1
    
                        print "tree id: {0}".format(NeuronID)
                        neuron_obj.set_new_neuronId(NeuronID)
                    if not (not neuron_obj):
                        Neurons[NeuronID]=neuron_obj
            elif filename.endswith(".amx"):
                filename=unicode(filename)
                tempNeurons,SelObj,editPosition,dataset=self.LoadAMXFile(filename)
                for NeuronID, neuron_obj in tempNeurons.iteritems():
                    if NeuronID in Neurons:
                        oldNeuronID=NeuronID
                        step=1
                        while Neurons.has_key(NeuronID):
                            NeuronID=oldNeuronID+step*0.001
                            NeuronID=float(np.round(NeuronID,3))
                            step+=1
    
                        print "tree id: {0}".format(NeuronID)
                        neuron_obj.set_new_neuronId(NeuronID)
                    if not (not neuron_obj):
                        Neurons[NeuronID]=neuron_obj
            elif filename.endswith(".aml"):
                filename=unicode(filename)
                tempNeurons,SelObj=self.LoadXMLFile(filename)
                Neurons.update(tempNeurons)
            elif filename.endswith(".nml"):
                filename=unicode(filename)
                tempNeurons,SelObj,editPosition,dataset=self.LoadNMLFile(filename,parseflags=0)
                for NeuronID, neuron_obj in tempNeurons.iteritems():
                    if NeuronID in Neurons:
                        oldNeuronID=NeuronID
                        step=1
                        while Neurons.has_key(NeuronID):
                            NeuronID=oldNeuronID+step*0.001
                            NeuronID=float(np.round(NeuronID,3))
                            step+=1
    
                        print "tree id: {0}".format(NeuronID)
                        neuron_obj.set_new_neuronId(NeuronID)
                    if not (not neuron_obj):
                        Neurons[NeuronID]=neuron_obj
            elif filename.endswith(".mat") and not (usermode==1):
                filename=unicode(filename)
                tempNeurons,SelObj,editPosition,dataset=self.LoadSkeletonFile(filename)
                if not tempNeurons:
                    print "Could not load file: ", filename
                    continue
                for NeuronID, neuron_obj in tempNeurons.iteritems():
                    if NeuronID in Neurons:
                        oldNeuronID=NeuronID
                        step=1
                        while Neurons.has_key(NeuronID):
                            NeuronID=oldNeuronID+step*0.001
                            step+=1
    
                        print "tree id: {0}".format(NeuronID)
                        neuron_obj.set_new_neuronId(NeuronID)
                    if not (not neuron_obj):
                        Neurons[NeuronID]=neuron_obj
            elif filename.endswith(".txt"):
                filename=unicode(filename)
                tempNeurons,SelObj,tempfilename,editPosition,dataset=self.LoadFileList(filename)
                for NeuronID, neuron_obj in tempNeurons.iteritems():
                    if NeuronID in Neurons:
                        oldNeuronID=NeuronID
                        step=1
                        while Neurons.has_key(NeuronID):
                            NeuronID=oldNeuronID+step*0.001
                            NeuronID=float(np.round(NeuronID,3))
                            step+=1
    
                        print "tree id: {0}".format(NeuronID)
                        neuron_obj.set_new_neuronId(NeuronID)
                    if not (not neuron_obj):
                        Neurons[NeuronID]=neuron_obj
                if tempNeurons.__len__()==1:
                    if UpdateCurrentFile:
                        self.UpdateCurrentFile(filename)
                    filename=unicode(tempfilename)
            elif filename.endswith(".csv"):
                filename=unicode(filename)
                Neurons.update(self.LoadCSVFile(filename))
            else:
                continue            
            if UpdateCurrentFile:
                self.UpdateCurrentFile(filename)
#            except:
#                print "Error loading file: {0}".format(filename)
        if showprogress:
            progress.setValue(filelist.__len__())
            progress.deleteLater()
        if SilentMode:    
            return Neurons, SelObj, editPosition, dataset
        else:
            if not (not (dataset)):     
                if not self.CurrentDataset[0]==dataset:
                    #check if dataset exists in recent dataset history
                    status=self.ChangeCubeDataset(dataset);
                    
                    #dirty hack
                    if status<1 and dataset=='E085L01':
                        dataset='wanner16'
                        status=self.ChangeCubeDataset(dataset);
                        
                    if status<1:
                        datasetpath=os.path.join(self._DefaultDataPath,dataset)
                        configfile=os.path.join(datasetpath,"{0}.conf".format(dataset))    
                        if not os.path.isfile(configfile):
                            if os.path.isdir(self._DefaultDataPath):
                                startpath=self._DefaultDataPath
                            else:
                                startpath=application_path                    
                            datasetpath = QtGui.QFileDialog.getExistingDirectory(self,"Choose the PARENT directory of dataset: {0}".format(dataset) ,\
                                startpath)
                            datasetpath=unicode(datasetpath)
                            if not (not datasetpath):
                                if not datasetpath.endswith(dataset):
                                    datasetpath=os.path.join(datasetpath,dataset)                    
                                configfile=os.path.join(datasetpath,"{0}.conf".format(dataset))    
                        if os.path.isfile(configfile) and not (self.CurrentDataset[1]==configfile):
                            self.ChangeCubeDataset(configfile)
            
            for NeuronID, neuron_obj in Neurons.iteritems():
                if NeuronID in self.Neurons:
                    oldNeuronID=NeuronID
                    step=1
                    while self.Neurons.has_key(NeuronID):
                        NeuronID=oldNeuronID+step*0.001
                        NeuronID=float(np.round(NeuronID,3))
                        step+=1

                    print "tree id: {0}".format(NeuronID)
                    neuron_obj.set_new_neuronId(NeuronID)
                self.Neurons[NeuronID]=neuron_obj

            for key, iviewport in self.QRWin.viewports.iteritems():
                iviewport.MoveTag = None
                            
#            FocalPoint=np.array(self.QRWin.viewports["skeleton_viewport"].Camera.GetFocalPoint(),dtype=np.float);
#            self.QRWin.viewports["skeleton_viewport"].Camera.SetPosition(FocalPoint-np.array([0,-1,1],dtype=np.float))      
            
            self.ChangeSkelVisMode()
#            if not (not self.job):
#                self.ChangeColorScheme(0)
#            else:                    
#                self.ChangeColorScheme()
            self.ChangeNeuronVisMode()
            self.ChangePlaneVisMode()
            self.ChangeBorderVisMode()

            for obj in [neuron,skeleton,synapse,soma,region,tag]:
                obj.start_VisEngine(self)

            self.SetSomaVisibility()
            self.SetRegionVisibility()
            
            self.ResetWorkingModes()
            
            self.UpdateWindowTitle()
            if (not (not self.job)) and InitializeJob:
                self.job.goto_task(self.job._taskIdx)
            elif reply==0: #overwrite mode
                if not (not SelObj):
                    self.QRWin.SetActiveObj(SelObj[0],SelObj[1],SelObj[2])
                    self.QRWin.GotoActiveObj()
                if not (not editPosition):
                    if not None in editPosition:
                        self.JumpToPoint(np.multiply(np.array(editPosition),self.DataScale))
                
            if not (not self.DemDriFiles):
                self.UpdateDemDriFiles(None,True)
            else:
                self.QRWin.RenderWindow.Render()

            print "Loading time: ", time.time()-startTime
            return Neurons
                   
    def Save(self):
        if not self.CurrentFile:
            self.SaveAs()
            return

        filename, fileext = os.path.splitext(self.CurrentFile)
        if self.ckbx_incrementFile.isChecked():
            result = re.search(r'(.+)\.(\d+)$', filename)
            if result==None:
                filenumber=1
            else:
                filename=result.group(1)
                filenumber=np.int(result.group(2))+1
            
            newfile=r"{0}.{1:{fill}3}".format(filename,filenumber,fill=0)
        else:
            newfile=filename
            
        if not os.path.exists(os.path.dirname(newfile)):
            self.SaveAs()
            return
        whichNeurons=OrderedDict();
        for neuronId, neuron_obj in self.Neurons.iteritems():
            if not (\
                ('d' in neuron_obj.flags) or \
                ('x' in neuron_obj.flags)): #exclude any neurons belonging to a demand-driven pipeline from saving
                whichNeurons[neuronId]=neuron_obj
        
        if self.comboBox_AutoSave.currentIndex()==2 and not (usermode==1):
            newfile+=".nml"
            self.SaveNMLFile(newfile,whichNeurons,True,scaled=False)
        elif self.comboBox_AutoSave.currentIndex()==3 and not (usermode==1):
            newfile+=".nml"
            self.SaveNMLFile(newfile,whichNeurons,True,scaled=True)
        else: #default
            #save for (Py)KNOSSOS
            newfile+=".nmx"
            self.SaveNMXFile(newfile,whichNeurons)
        self.UpdateCurrentFile(newfile)
        self.Timer.changesSaved=1
        self.UpdateWindowTitle()                                

    def ExportSynapses(self):
        SynapsesFound=False
        if not (not self.job):
            currTask=self.job.get_current_task()
            if not (not currTask):
                if currTask._tasktype=="synapse_detection":    
                    SynapsesFound=True
        if not SynapsesFound:
            print "No synapse detection job/task found."
            return

        obj=self.Neurons[currTask._neuronId].children['synapse']
        NodeID=obj.data.GetPointData().GetArray('NodeID')
        DeletedNode=obj.data.GetPointData().GetArray('DeletedNodes')
        NSynapses=obj.data.GetNumberOfCells()
        tempSynapse=list()
        for isyn in range(NSynapses):
            skip=False
            PointIds=obj.data.GetCell(isyn).GetPointIds()
            Points=list()
            for ipoint in range(PointIds.GetNumberOfIds()):
                pointIdx=PointIds.GetId(ipoint)
                if DeletedNode.GetValue(pointIdx):
                    skip=True
                    break
                if ipoint==0:
                    tagId=int(NodeID.GetValue(pointIdx))
                Points.append(obj.data.GetPoints().GetPoint(pointIdx))
            if skip:
                continue
            tempSynapse.append(dict(obj.comments._Comments[tagId].items() + {'id' : tagId}.items()+{'points' : Points}.items()))        



        if os.path.exists(os.path.dirname(self.CurrentFile)):
            filename=self.CurrentFile
            basepath, basename = os.path.split(unicode(filename))
            basename, ext = os.path.splitext(basename)
            filename = os.path.join(basepath,"{0}.mat".format(basename))
            filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...",filename,"*.mat");
        elif os.path.isdir(application_path):
            filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...",application_path,"*.mat");
        else:
            filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...","","*.mat");
        if not filename:
            return 0
        else:            
            newfile=unicode(filename)
            if not newfile.endswith(".mat"):
                newfile+=".mat"
        scipy.io.savemat(newfile,{'synapse' : tempSynapse},oned_as='row')
        print "Exported synapses of neuron id ", currTask._neuronId

    def SaveActiveNeuron(self):
        SelObj=self.QRWin.SelObj
        whichNeuron=SelObj[1]
        if not whichNeuron in self.Neurons:
            return
        if ('d' in self.Neurons[whichNeuron].flags) or \
                ('x' in self.Neurons[whichNeuron].flags):
            return
        Neurons2Save=OrderedDict()        
        Neurons2Save[whichNeuron]=self.Neurons[whichNeuron]
        self.SaveAs(Neurons2Save)
        
    def SaveSeparately(self):
        if os.path.exists(os.path.dirname(self.CurrentFile)):
            CurrentPath=os.path.dirname(self.CurrentFile)
        elif os.path.isdir(application_path):
            CurrentPath=application_path
        else:
            CurrentPath=""
        
        selectedFilter=QtCore.QString();
        tempFileFormats=['PyKnossos (*.nmx)']
        if self.comboBox_AutoSave.currentIndex()==2:
            tempFileFormats=['KNOSSOS (*.nml)']
        elif self.comboBox_AutoSave.currentIndex()==3:
            tempFileFormats=['KNOSSOS (*.nml)']
        elif self.comboBox_AutoSave.currentIndex()==1:
            tempFileFormats=['PyKnossos (*.nmx)']
        
        for avFormat in availableFileFormats:
            if not avFormat in tempFileFormats:
                tempFileFormats.append(avFormat)
        tempFileFormats=';;'.join(tempFileFormats)
        for neuronId, neuron_obj in self.Neurons.iteritems():
            if ('d' in neuron_obj.flags) or \
                ('x' in neuron_obj.flags): #exclude any neurons from a demand-driven pipline from saving
                continue
            if not neuron_obj.filename:
                if neuronId==int(neuronId):    
                    newfile=os.path.join(CurrentPath,"Neuron_id{0:{fill}4}".format(int(neuronId),fill=0))
                else:
                    neuronIdparts=unicode(neuronId).split('.')
                    newfile=os.path.join(CurrentPath,"Neuron_id{0:{fill}4}_{1}".format(int(neuronIdparts[0]),neuronIdparts[1],fill=0))
            else:
                newfile=neuron_obj.filename

    
            filename, fileext = os.path.splitext(newfile)
            if self.ckbx_incrementFile.isChecked():
                result = re.search(r'(.+)\.(\d+)$', filename)
                if result==None:
                    filenumber=1
                else:
                    filename=result.group(1)
                    filenumber=np.int(result.group(2))+1
                
                newfile=r"{0}.{1:{fill}3}{2}".format(filename,filenumber,fileext,fill=0)
                                
            newfile = QtGui.QFileDialog.getSaveFileName(self,"Save neurons seperately as...",newfile,tempFileFormats,selectedFilter);
            if not newfile:
                return 0    
            newfile=unicode(newfile)
            selectedFilter=unicode(selectedFilter)
            Neuron=OrderedDict()
            Neuron[neuronId]=neuron_obj
            if newfile.endswith(".amx"):
                self.SaveAMXFile(newfile,Neuron)
            elif newfile.endswith(".nml"):
                if self.comboBox_AutoSave.currentIndex()==3 and not (usermode==1):
                    self.SaveNMLFile(newfile,Neuron,True,scaled=True)
                elif self.comboBox_AutoSave.currentIndex()==2 and not (usermode==1):
                    self.SaveNMLFile(newfile,Neuron,True)
                else: #default
                    self.SaveNMLFile(newfile,Neuron,True)
            elif newfile.endswith(".nmx"):
                self.SaveNMXFile(newfile,Neuron)
            else:
                newfile,fileext=os.path.splitext(newfile)
                if selectedFilter==u'Ariadne (*.amx)':
                    newfile+=".amx"
                    self.SaveAMXFile(newfile,Neuron)
                elif selectedFilter==u'KNOSSOS (*.nml)':
                    newfile+=".nml"
                    if self.comboBox_AutoSave.currentIndex()==3 and not (usermode==1):
                        self.SaveNMLFile(newfile,Neuron,True,scaled=True)
                    elif self.comboBox_AutoSave.currentIndex()==2 and not (usermode==1):
                        self.SaveNMLFile(newfile,Neuron,True)
                    else: #default
                        self.SaveNMLFile(newfile,Neuron)
                elif selectedFilter==u'PyKnossos (*.nmx)':
                    newfile+=".nmx"
                    self.SaveNMXFile(newfile,Neuron)

        self.Timer.changesSaved=1
        self.UpdateWindowTitle()                                
    
    def SaveAs(self,Neurons=None):        
        tempFileFormats=['PyKnossos (*.nmx)']
        if self.comboBox_AutoSave.currentIndex()==2:
            tempFileFormats=['KNOSSOS (*.nml)']
        elif self.comboBox_AutoSave.currentIndex()==3:
            tempFileFormats=['KNOSSOS (*.nml)']
        elif self.comboBox_AutoSave.currentIndex()==1:
            tempFileFormats=['PyKnossos (*.nmx)']
            
        for avFormat in availableFileFormats:
            if not avFormat in tempFileFormats:
                tempFileFormats.append(avFormat)
        tempFileFormats=';;'.join(tempFileFormats)
        
        if os.path.exists(os.path.dirname(self.CurrentFile)):
            filename=self.CurrentFile
            basepath, basename = os.path.split(unicode(filename))
            basename, ext = os.path.splitext(basename)
            filename = os.path.join(basepath,basename)
            filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...",filename,tempFileFormats);
        elif os.path.isdir(application_path):
            filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...",application_path,tempFileFormats);
        else:
            filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...","",tempFileFormats);
        if not filename:
            return 0

        newfile=unicode(filename)
        if not Neurons:
            Neurons=self.Neurons

        whichNeurons=OrderedDict();
        for neuronId, neuron_obj in Neurons.iteritems():
            if not (\
                ('d' in neuron_obj.flags) or \
                ('x' in neuron_obj.flags)): #exclude any neurons belonging to a demand-driven pipeline from saving
                whichNeurons[neuronId]=neuron_obj


        if newfile.endswith(".nml") and not (usermode==1):
            self.SaveNMLFile(newfile,whichNeurons,True)
        elif newfile.endswith(".amx"):
            self.SaveAMXFile(newfile,whichNeurons)
        else:
            if not newfile.endswith(".nmx"):
                newfile+=".nmx"
            self.SaveNMXFile(newfile,whichNeurons)
        self.UpdateCurrentFile(newfile)
        self.Timer.changesSaved=1
        self.UpdateWindowTitle()                                

    def SaveAMXFile(self,filename,Neurons):
        basepath, basename = os.path.split(unicode(filename))
        basename, ext = os.path.splitext(basename)
        if not self.job:
            jobfile=None
        else:
            jobfile=os.path.join(basepath,"{0}.job".format(basename))
            self.job.filename=jobfile
            self.job.save_job()
        amlfile=os.path.join(basepath,"{0}.aml".format(basename))
        self.SaveXMLFile(amlfile,Neurons)
        
        amxfile=os.path.join(basepath,"{0}.amx".format(basename))
        zipf=ZipFile(amxfile,'w')
        if not jobfile==None:
            zipf.write(jobfile,"{0}.job".format(basename))
        zipf.write(amlfile,"{0}.aml".format(basename))
        blockpath=os.path.join(basepath,basename)
        for root, dirs, files in os.walk(blockpath):
            for file in files:
                zipf.write(os.path.join(root, file),os.path.join(basename,file))
        zipf.close()
        if os.path.isdir(blockpath):
            shutil.rmtree(blockpath)
        if os.path.isfile(amlfile):
            os.remove(amlfile)
        if not jobfile==None:
            if os.path.isfile(jobfile):
                os.remove(jobfile)

    def LoadAMXFile(self,origfilename):
        basepath, basename = os.path.split(unicode(origfilename))
        basename, ext = os.path.splitext(basename)
        zipf=ZipFile(origfilename,'r')
        targetdir=os.path.join(application_path,'temp')
        targetdir=os.path.join(targetdir,basename)
        if not os.path.isdir(targetdir):
            os.makedirs(targetdir)
        zipf.extractall(targetdir)
        zipf.close()
    
        filelist = []
        for root, dirnames, filenames in os.walk(targetdir):
          for filename in fnmatch.filter(filenames, '*.aml'):
              filelist.append(os.path.join(root, filename))

        Neurons=None
        SelObj=[None,None,None]   
        if filelist.__len__()>0:
            amlfile=filelist[0]
            Neurons, SelObj, editPosition, dataset=self.LoadXMLFile(amlfile)
        filelist = []
        for root, dirnames, filenames in os.walk(targetdir):
          for filename in fnmatch.filter(filenames, '*.job'):
              filelist.append(os.path.join(root, filename))

        if not (not filelist):
            jobfile=filelist[0]
            self.job=job(self,jobfile)
            self.job.load_job()
        if os.path.isdir(targetdir):
            shutil.rmtree(targetdir)


        for neuronId, neuron_obj in Neurons.iteritems():
            neuron_obj.filename=origfilename
        return Neurons, SelObj, editPosition, dataset
        
    def HideSomaLabels(self,state=None,silent=0):
        if state==None:
            state=self.ckbx_HideSomaLabels.isChecked()
        for iitem in range(soma.labelactor.GetNumberOfItems()):
            labelactor=soma.labelactor.GetItemAsObject(iitem);  
            actorVisibility=labelactor.GetVisibility()
            if actorVisibility == (not state):
                continue
            if state:
                labelactor.VisibilityOff()
            else:
                labelactor.VisibilityOn()
        if not silent:
            self.QRWin.Render()
            
    def HideRegionLabels(self,state=None,silent=0):
        if state==None:
            state=self.ckbx_HideRegionLabels.isChecked()
        for iitem in range(region.labelactor.GetNumberOfItems()):
            labelactor=region.labelactor.GetItemAsObject(iitem);  
            actorVisibility=labelactor.GetVisibility()
            if actorVisibility == (not state):
                continue
            if state:
                labelactor.VisibilityOff()
            else:
                labelactor.VisibilityOn()
        if not silent:
            self.QRWin.Render()
        
    def SetSomaVisibility(self,alpha=None,neuron_obj=None):
        if alpha==None:
            alpha=self.SomaAlpha.value()
        soma.DefaultAlpha[0]=alpha
        soma.update_VisEngine(alpha)
        if self.ckbx_HideSomaLabels.isChecked():
            self.HideSomaLabels(1,1)
        else:
            if alpha==0:
                self.HideSomaLabels(1,1)
            else:
                self.HideSomaLabels(0,1)
        self.QRWin.RenderWindow.Render()

    def SetRegionVisibility(self,alpha=None,neuron_obj=None):
        if alpha==None:
            alpha=self.RegionAlpha.value()
        region.DefaultAlpha[0]=alpha
        region.update_VisEngine(alpha)
        if self.ckbx_HideRegionLabels.isChecked():
            self.HideRegionLabels(1,1)
        else:
            if alpha==0:
                self.HideRegionLabels(1,1)
            else:
                self.HideRegionLabels(0,1)
        self.QRWin.RenderWindow.Render()
            
    def SetSynLineWidth(self,width=None,defaultOnly=False):
        if width==None:
            width=self.SpinBox_SynLineWidth.value()
        
        if defaultOnly:
             self.QRWin.setFocus()
             return
        for obj in [synapse,tag]: 
            obj.DefaultLineWidth[0]=width
            obj.DefaultPointSize[0]=width+2+2
            obj.update_VisEngine()

        self.QRWin.RenderWindow.Render()
        self.QRWin.setFocus()

    def SetSkelLineWidth(self,width=None,defaultOnly=False):
        if width==None:
            width=self.SpinBox_LineWidth.value()
        
        if defaultOnly:
             self.QRWin.setFocus()
             return
        for obj in [skeleton]: 
            obj.DefaultLineWidth[0]=width
            obj.DefaultPointSize[0]=width+2+2
            obj.update_VisEngine()
            
        NodeSelection.DefaultLineWidth[0]=width+3+3
        NodeSelection.DefaultPointSize[0]=width+2+2+3+3
        NodeSelection.update_VisEngine()

         
        self.QRWin.RenderWindow.Render()
        self.QRWin.setFocus()


    def SetSkeletonRadius(self,radius=None,overwrite=None,defaultOnly=False):
        if radius==None:
            radius=self.SpinBox_Radius.value()
        
        if overwrite==None:
            overwrite=self.OverwriteRadius.isChecked()
        
        skeleton.DefaultRadius[0]=radius
        if defaultOnly:
             self.QRWin.setFocus()
             return
        for neuronId, neuron_obj in self.Neurons.iteritems():
            if "skeleton" in neuron_obj.children:
                child=neuron_obj.children["skeleton"]
                if not (not child):
                    child.SetRadius(radius,overwrite)
        self.QRWin.RenderWindow.Render()
        self.QRWin.setFocus()

    def SetSynapseRadius(self,radius=None,defaultOnly=False):
        if radius==None:
            radius=self.SpinBox_SynRadius.value()
        
        synapse.DefaultRadius[0]=radius
        tag.DefaultRadius[0]=radius
        if defaultOnly:
             self.QRWin.setFocus()
             return
        for neuronId, neuron_obj in self.Neurons.iteritems():
            if "synapse" in neuron_obj.children:
                child=neuron_obj.children["synapse"]
                if not (not child):
                    child.SetRadius(radius)
            if "tag" in neuron_obj.children:
                child=neuron_obj.children["tag"]
                if not (not child):
                    child.SetRadius(radius)
        self.QRWin.RenderWindow.Render()
        self.QRWin.setFocus()

    def UpdateWindowTitle(self):
        if self.QRWin.SynMode:
            WorkingMode="Synapse Annotation"
        elif self.QRWin.TagMode:
            WorkingMode="Tagging"
        elif self.QRWin.TracingMode:
            WorkingMode="Tracing"
        else:
            WorkingMode=""
        
        if not self.CurrentFile:
            currFile="None"
        else:
            basepath, currFile = os.path.split(unicode(self.CurrentFile))
        if not self.Timer.changesSaved:
            currFile +="*"
        
        if not CubeLoader._BaseName:
            Dataset="None"
        else:
            Dataset=CubeLoader._BaseName
        
        if (self._WorkingOffline==1) or (self.ActionWorkingOffline.isEnabled()==0):
            OnlineMode='offline';
        else:
            OnlineMode='online';
        Title="{0} - Work mode: {1} - Dataset: {2} ({3})- File: {4}".format(PyKNOSSOS_VERSION[0:12],WorkingMode,Dataset,OnlineMode,currFile)        
        self.setWindowTitle(Title)
        
    def TransformOrthViewport(self,obj=None,value=None):
        rotcDir=None
        rotvDir=None
        if not obj:
            if self.radioBtn_RotAxisYX.isChecked():
                rotAxis=[0.0,0.0,1.0]
            elif self.radioBtn_RotAxisYZ.isChecked():
                if self.radioButton_orthRef.isChecked():
                    rotAxis=self.QRWin.viewports['YZ_viewport'].Camera.GetDirectionOfProjection()
                else:
                    rotAxis=[1.0,0.0,0.0]
            elif self.radioBtn_RotAxisZX.isChecked():
                if self.radioButton_orthRef.isChecked():
                    rotAxis=self.QRWin.viewports['ZX_viewport'].Camera.GetDirectionOfProjection()
                else:
                    rotAxis=[0.0,1.0,0.0]
            elif self.radioBtn_RotAxisOrth.isChecked():
                rotAxis=self.QRWin.viewports['Orth_viewport'].Camera.GetDirectionOfProjection()
            rotAngle=self.SpinBox_RotAngle.value()
            cDir=self.QRWin.viewports['Orth_viewport'].Camera.GetDirectionOfProjection()
            vDir=self.QRWin.viewports['Orth_viewport'].Camera.GetViewUp()
            RotTransform=vtk.vtkTransform()
            RotTransform.RotateWXYZ(rotAngle,rotAxis)
            rotcDir=RotTransform.TransformDoubleVector(cDir)
            rotvDir=RotTransform.TransformDoubleVector(vDir)
            self.text_OrthcDir.setText(array2str(rotcDir))
            self.text_OrthvDir.setText(array2str(rotvDir))
            rotcDir=np.array(rotcDir)
            rotvDir=np.array(rotvDir)

                
        elif obj=='cDir' or obj=='vDir':
            tempTxt=str(self.text_OrthcDir.text())
            if not (not tempTxt):
                rotcDir=np.array(ast.literal_eval(tempTxt))
            tempTxt=str(self.text_OrthvDir.text())
            if not (not tempTxt):
                rotvDir=np.array(ast.literal_eval(tempTxt))
        else:
            return
        FPoint=self.QRWin.viewports['Orth_viewport'].Camera.GetFocalPoint()            
        self.JumpToPoint(np.array(FPoint),rotcDir,rotvDir)
        
    def ShowSkelVPOnly(self,state=0):
        1
        
    def HideSkelViewport(self,state=0):
        if not self.QRWin.viewports.has_key("skeleton_viewport"):
            return        
        if not self.QRWin.viewports["skeleton_viewport"]:
            return
        if self.ckbx_HideSkelViewport.isChecked():
            Layout=[[u'ZX_viewport', u'YX_viewport'], [u'Orth_viewport', u'YZ_viewport'],]
            self.QRWin.RenderWindow.RemoveRenderer(self.QRWin.viewports["skeleton_viewport"])
        else:
            Layout=[[u'ZX_viewport', u'YX_viewport'], [u'Orth_viewport', u'YZ_viewport'], [u'skeleton_viewport']]
            self.QRWin.RenderWindow.AddRenderer(self.QRWin.viewports["skeleton_viewport"])
        self.QRWin.DistributeViewports(Layout)
        
    def initViewports(self):
        for key,iviewport in self.QRWin.viewports.iteritems():
            iviewport.ResetCameraClippingRange()
            iviewport.ResetCamera()
            iviewport.Iren.SetInteractorStyle(None)
            iviewport.Zoom(0)
            iviewport.ResetViewport()
        self.QRWin.Render_Intersect()
            
    def ExtractMetaDataFromFieldData(self,fieldData,fields2extract):
        fieldValues=OrderedDict()
        for field in fields2extract:
            tempFieldData=fieldData.GetAbstractArray(field)
            if not tempFieldData:
                continue;
            NComponents=tempFieldData.GetNumberOfComponents()
            NTuples=tempFieldData.GetNumberOfTuples()
            
            tempData=OrderedDict();
            ival=0
            if NComponents==1:
                if NTuples==1:
                    tempData=tempFieldData.GetValue(ival)
                else:
                    tempData=[]
                    for ituple in range(NTuples):
                        tempData.append(tempFieldData.GetValue(ival))
                        ival+=1                
            else:
                compNames=[]
                for icomp in range(NComponents):
                    compName=tempFieldData.GetComponentName(icomp)
                    if compName==None:
                        continue
                    compNames.append(compName)
                if compNames.__len__()==0:
                    tempData=[]
                    for ituple in range(NTuples):                                
                        for icomp in range(NComponents):
                            tempData.append(tempFieldData.GetValue(ival))
                            ival+=1
                else:
                    compNames=[]
                    for icomp in range(NComponents):
                        compName=tempFieldData.GetComponentName(icomp)
                        if not compName:
                            compName="comp{0}".format(icomp)
                        compNames.append(compName)
                        tempData[compName]=list()
                    for ituple in range(NTuples):                                
                        for icomp in range(NComponents):
                            tempData[compNames[icomp]].append(tempFieldData.GetValue(ival))
                            ival+=1
            fieldValues[field]=tempData
        return fieldValues

    def SaveXMLFile(self,filename,Neurons):
        allData=vtk.vtkMultiBlockDataSet()
        
        idata=0 #the first block is the parameter block
        UniqueLookupTables=list()
        for neuronId, neuron_obj in Neurons.iteritems():
            if neuron_obj==[]:
                continue
            
            Block=vtk.vtkMultiBlockDataSet()
                        
            jdata=0
            metaData=neuron_obj.extract_metadata(UniqueLookupTables)

#            Block.SetFieldData(metaData) #does not work because field data of blocks is not written
            obj=vtk.vtkPolyData()     
            obj.SetFieldData(metaData)
            Block.SetBlock(jdata,obj);
            
            for key, child in neuron_obj.children.iteritems():
                if not child:
                    continue
                tempData=child.get_cleanedup_data()
                metaData=child.extract_metadata(UniqueLookupTables)
                tempData.SetFieldData(metaData)
                jdata+=1
                Block.SetBlock(jdata,tempData);
            idata+=1
            allData.SetBlock(idata,Block);
           
        Parameters=vtk.vtkPolyData()     
        for tableidx in range(UniqueLookupTables.__len__()):
            tempArray=UniqueLookupTables[tableidx].GetTable()
            tempArray.SetName("LUT{0}".format(tableidx))
            Parameters.GetFieldData().AddArray(tempArray)

        activeObj=vtk.vtkStringArray()
        activeObj.SetName("activeObj")
        activeObj.SetNumberOfComponents(3)
        activeObj.SetComponentName(0,"objtype")
        activeObj.SetComponentName(1,"neuronID")
        activeObj.SetComponentName(2,"nodeId")
        SelObj=self.QRWin.SelObj
        for iid in range(3):
            if SelObj[iid]==None:
                SelObj[iid]=='None'
        activeObj.InsertNextValue(unicode(SelObj[0]))
        activeObj.InsertNextValue(unicode(SelObj[1]))
        activeObj.InsertNextValue(unicode(SelObj[2]))        
        Parameters.GetFieldData().AddArray(activeObj)
        
        #the first block is the parameter block
        allData.SetBlock(0,Parameters)

        writer=vtk.vtkXMLMultiBlockDataWriter();
        if sys.maxsize==9223372036854775807: #64bit
            writer.SetBlockSize(4294967288)
            writer.SetIdTypeToInt64() #default
        else:
            writer.SetCompressorTypeToNone()
        writer.SetFileName(filename);
        writer.SetInput(allData);
        writer.Write();

    def LoadXMLFile(self,origfilename):
        reader=vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(origfilename)
        reader.Update()
        allData=reader.GetOutput()
                
        NBlocks=allData.GetNumberOfBlocks()
        if NBlocks<1:
            return
            
        Parameters=allData.GetBlock(0)
        
        #process lookuptables
        LookupTables=list()
        iLUT=0
        while 1:
            tempArray=Parameters.GetFieldData().GetArray("LUT{0}".format(iLUT))
            if not tempArray:
                break
            LookupTables.append(vtk.vtkLookupTable())            
            LookupTables[iLUT].SetTable(tempArray)
            LookupTables[iLUT].SetTableRange(0, LookupTables[iLUT].GetNumberOfAvailableColors()-1)
            LookupTables[iLUT].Build()
            iLUT+=1
        
        FieldValues=self.ExtractMetaDataFromFieldData(Parameters.GetFieldData(),["activeObj"])
        if "activeObj" in FieldValues:
            SelObj=list()
            if 'objtype' in FieldValues['activeObj']:
                value=FieldValues['activeObj']['objtype'][0]
                if value=='None':
                    value=None
            else:
                value=None
            SelObj.append(value)
            if 'neuronID' in FieldValues['activeObj']:
                value=FieldValues['activeObj']['neuronID'][0]
                if value=='None':
                    value=None
            else:
                value=None
            SelObj.append(value)
            if 'nodeId' in FieldValues['activeObj']:
                value=FieldValues['activeObj']['nodeId'][0]
                if value=='None':
                    value=None
                else:
                    value=np.int(value)
            else:
                value=None
            SelObj.append(value)
        else:
            SelObj=[None,None,None]
                
        
        timerOffset=self.Timer.timerOffset;
        totNNodes=0;
        Neurons=OrderedDict();
        for iblock in range(1,NBlocks):
            #load metadata
           
            ParentBlock=allData.GetBlock(iblock)
            
            #Block 0: parameters
            FieldValues=self.ExtractMetaDataFromFieldData(\
                ParentBlock.GetBlock(0).GetFieldData(),\
                ["type","text","id","coloridx","comments"])
            if not ("type" in FieldValues):
                print "type not found in field data of block {0}.".format(iblock)
                continue
            if not ("id" in FieldValues):
                print "id not found in field data of block {0}.".format(iblock)
                continue
            if not "coloridx" in FieldValues:
                print "coloridx not found in field of block {0}.".format(iblock)
                continue
            if FieldValues["type"]=="neuron":
                NeuronID=float(FieldValues["id"])
                oldNeuronID=NeuronID
                step=1
                while self.Neurons.has_key(NeuronID):
                    NeuronID=oldNeuronID+step*0.001
                    NeuronID=float(np.round(NeuronID,3))
                    step+=1

                LUTIdx=int(FieldValues["coloridx"][0])
                colorIdx=int(FieldValues["coloridx"][1])
                color=LookupTables[LUTIdx].GetTableValue(colorIdx)
                Neurons[NeuronID]=neuron(self.ObjectBrowser.model(),NeuronID,color)                    
                NSubBlocks=ParentBlock.GetNumberOfBlocks()
                for isubblock in range(1,NSubBlocks):
                    SubBlock=ParentBlock.GetBlock(isubblock)    
                    FieldValues=self.ExtractMetaDataFromFieldData(\
                        SubBlock.GetFieldData(),\
                        ["type","text","id","coloridx","comments"])

                    if not ("type" in FieldValues):
                        print "type not found in field data of subblock {0} of block {1}.".format(isubblock,iblock)
                        continue
                    if not ("id" in FieldValues):
                        print "id not found in field data of subblock {0} of block {1}.".format(isubblock,iblock)
                        continue
                    if not "coloridx" in FieldValues:
                        print "coloridx not found in field of subblock {0} of block {1}.".format(isubblock,iblock)
                        continue

                    LUTIdx=int(FieldValues["coloridx"][0])
                    colorIdx=int(FieldValues["coloridx"][1])
                    color=LookupTables[LUTIdx].GetTableValue(colorIdx)
                    if FieldValues["type"]=="skeleton":  
                        obj=skeleton(Neurons[NeuronID].item,NeuronID,color)
                        
                        NodeID=SubBlock.GetPointData().GetArray("NodeID")
                        totNNodes+=SubBlock.GetNumberOfPoints()
                        obj.set_nodes(SubBlock.GetPoints(),NodeID)
                        
#                        obj.add_branch(SubBlock.GetVerts(),'Verts',0) #don't need this because it's already done by set_nodes
                        obj.add_branch(SubBlock.GetLines(),'Lines',1)
                        
                    elif FieldValues["type"]=="soma":
                        if SubBlock.GetNumberOfPoints()>0:
                            obj=soma(Neurons[NeuronID].item,NeuronID,color)
                            NodeID=SubBlock.GetPointData().GetArray("NodeID")
                            if NodeID==None:
                                obj.set_nodes(SubBlock.GetPoints())
                            else:
                                obj.set_nodes(SubBlock.GetPoints(),NodeID)
                        
                    elif FieldValues["type"]=="tag":
                        obj=tag(Neurons[NeuronID].item,NeuronID,color)
                        NodeID=SubBlock.GetPointData().GetArray("NodeID")
                        obj.set_tags(SubBlock.GetPoints(),NodeID)
                    elif FieldValues["type"]=="synapse":#comments have to be added first before we can initialize the tags because of the class color assignment
                        obj=synapse(Neurons[NeuronID].item,NeuronID,color)
                        Points=SubBlock.GetPoints()  
                        Connections=SubBlock.GetLines()
                        NodeID=SubBlock.GetPointData().GetArray("NodeID")
                    else:
                        print "Unknown type: {0} of subblock {1} of block {2}.".format(FieldValues["type"],isubblock,iblock)
                        continue
                    if "comments" in FieldValues:
                        for idx,id in enumerate(FieldValues["comments"]["id"]):
                            value=FieldValues["comments"]["value"][idx]
                            if not value:
                                continue
                            dtype=value[0]
                            if dtype=='s':
                                value=np.unicode(value[1:])
                            elif dtype=='f':
                                value=np.float(value[1:])
                            elif dtype=='b':
                                value=np.bool(value[1:])
                            elif dtype=='i':
                                value=np.int(value[1:])
                            elif dtype=='u':
                                value=np.uint(value[1:])
                            elif dtype=='d':
                                value=np.double(value[1:])
                            elif dtype=='n': #ndarray
#                                print NeuronID, FieldValues["type"], FieldValues["comments"]["key"][idx], value
                                value=np.array(ast.literal_eval(value[1:]))
                            elif dtype=='l' or 't': #list or tuple
                                value=ast.literal_eval(value[1:])
                            else:
                                print "Unknown comment datatype {0} for id:{1},key:{2},value:{3} for subblock {4} of block {5}.".format(dtype,id,FieldValues["comments"]["key"][idx],value,isubblock,iblock)
                                value=np.unicode(value[1:])
                            
                            obj.comments.set(np.int(id),FieldValues["comments"]["key"][idx],value)
                            if FieldValues["comments"]["key"]=="time":
                                if value>timerOffset:
                                    timerOffset=value
                    if FieldValues["type"]=="synapse":#comments have to be added first before we can initialize the tags because of the class color assignment
                        nodeId,tagIdx=obj.set_tags(Points,Connections,NodeID)
                    Neurons[NeuronID].children[FieldValues["type"]]=obj
        self.Timer.timerOffset=timerOffset
        print "timerOffset after: ", timerOffset
        
        print "total number of nodes: {0}".format(totNNodes)
        for neuronId, neuron_obj in Neurons.iteritems():
            neuron_obj.filename=origfilename
        return Neurons, SelObj, None, None
        
    def get_autocolor(self,ineuron):
        NTableValues=self.tempLUT.GetNumberOfTableValues()
        ineuron=np.mod(ineuron,NTableValues)
        if ineuron>1:
            icolor=np.int(np.floor(((2*ineuron-1)-2**np.ceil(np.log(ineuron)/np.log(2)))/2**np.ceil(np.log(ineuron)/np.log(2))*NTableValues)-1)
        elif ineuron==1:
            icolor=NTableValues-1
        else:
            icolor=0
#        print "color idx: {0}".format(icolor)
        return self.tempLUT.GetTableValue(icolor)
    
    def LoadDDXFile(self,origfilename):
        if not origfilename:
            return None
        if not origfilename.endswith('.ddx'):
            return None
        if not os.path.isfile(origfilename):
            return None
        ddfobj=DemandDrivenFile(self,origfilename);
        return ddfobj
        

    def LoadNMXFile(self,origfilename,convertflag=False):
        if not origfilename:
            return
        if not origfilename.endswith('.nmx'):
            return
        basepath, basename = os.path.split(origfilename)
        basename, ext = os.path.splitext(basename)
        if not basename:
            return
        tempdir=os.path.join(application_path,'temp')
        tempdir=os.path.join(tempdir,basename)
        if os.path.isdir(tempdir):
            shutil.rmtree(tempdir) #remove any existing temporary host dir
        if not os.path.isdir(tempdir):
            os.makedirs(tempdir)
        
        if not zipfile.is_zipfile(origfilename):
            decrypted_nmxfile=unicode(os.path.join(tempdir,'{0}{1}'.format(basename,ext)))
            decrypt_file(encryptionkey, origfilename, decrypted_nmxfile, chunksize=64*1024)
            filename=decrypted_nmxfile
            zipf=ZipFile(filename,'r')
        else:
            zipf=ZipFile(origfilename,'r')
        zipf.extractall(tempdir)
        zipf.close()    

        filelist=glob.glob(os.path.join(tempdir,'*.nmx'))
        for filename in filelist:
            if not zipfile.is_zipfile(filename):
                basepath, basename = os.path.split(filename)
                decrypted_nmxfile=os.path.join(basepath,'temp_{0}'.format(basename))
                decrypt_file(key, filename, decrypted_nmxfile, chunksize=64*1024)
                filename=decrypted_nmxfile
            zipf=ZipFile(filename,'r')
            zipf.extractall(tempdir)
            zipf.close()    

        filelist = []
        txtfilelist=[]
        jobfiles= []
        for root, dirnames, filenames in os.walk(tempdir):
          for filename in fnmatch.filter(filenames, '*.nml'):
              filelist.append(os.path.join(root, filename))
          for filename in fnmatch.filter(filenames, '*.txt'):
              txtfilelist.append(os.path.join(root, filename))
          for filename in fnmatch.filter(filenames, '*.job'):
              jobfiles.append(os.path.join(root, filename))
        
        if convertflag:
            if not filelist.__len__()==1:
                print "Cannot convert tracing file: ", origfilename
                return 0
            temp=os.path.splitext(origfilename)
            newfilename=temp[0]+u'.nml'
            shutil.copy2(filelist[0],newfilename)
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir)
            return 1

        Neurons=OrderedDict()
        SelObj=[None,None,None]   
        editPosition=[None,None,None]
        dataset=None
        if filelist.__len__()>0:
            tempNeurons, SelObj, editPosition, dataset=self.LoadNMLFile(filelist,parseflags=1)
            Neurons.update(tempNeurons)
        for filename in filelist:
            if os.path.isfile(filename):
                os.remove(filename)
        for filename in txtfilelist:
            tempNeurons,SelObj,tempfilename,editPosition,dataset=self.LoadFileList(filename)
            Neurons.update(tempNeurons)
        for filename in txtfilelist:
            if os.path.isfile(filename):
                os.remove(filename)
                    
        for root, dirnames, filenames in os.walk(tempdir):
          for filename in fnmatch.filter(filenames, '*.nmx'):
              os.remove(os.path.join(root, filename))
        
        if jobfiles.__len__()>0:
            #should handle here the case of multiple job files.
            #for now we just select the first one.
            jobfile=jobfiles[0]

            self.job=job(self,jobfile)
            self.job.load_job()
            for filename in jobfiles:
                if os.path.isfile(filename):
                    os.remove(filename)

            if not (not (self.job._Dataset)):                
                dataset=self.job._Dataset
        for neuronId, neuron_obj in Neurons.iteritems():
            neuron_obj.filename=origfilename
            
        if os.path.isdir(tempdir):
            shutil.rmtree(tempdir)
        return Neurons, SelObj, editPosition, dataset

    def SaveNMXFile(self,filename,Neurons,Job=None,Dataset=None):
        basepath, basename = os.path.split(unicode(filename))
        basename, ext = os.path.splitext(basename)

        blockpath,ext=  os.path.splitext(unicode(filename))  
        if os.path.isdir(blockpath): #we do not want to overwrite existing block dirs
            blockpath=os.path.join(application_path,'temp')
            blockpath=os.path.join(blockpath,basename)
        if not os.path.isdir(blockpath):
            os.makedirs(blockpath)

        activeNode=None
        dataset=None
        editPosition=None
        if Job==None:
            Job=self.job            
        if Job==None:
            jobfile=None
        else:
            jobfile=os.path.join(blockpath,"{0}.job".format(basename))
            Job.filename=jobfile
            Job.save_job()
            if not (usermode==1): #???
                if hasattr(Job,'_Dataset'):
                    dataset=Job._Dataset
                else:
                    dataset=None
                currTask=Job.get_current_task()
                if not (not currTask):
                    activeNode=currTask._currNodeId
                    if (activeNode==None) or activeNode<0:
                        activeNode=None
                    editPosition=None
            else:
                editPosition=[int(np.round(CubeLoader.Position[idim]/self.DataScale[idim])) for idim in range(3)]
        if (not dataset) and not (not Dataset):
            filename,fileext = os.path.splitext(Dataset)
            temppath,dataset=os.path.split(filename)
        nmlfile=os.path.join(blockpath,"{0}.nml".format(basename))
        self.SaveNMLFile(nmlfile,Neurons,False,dataset,activeNode,editPosition)
        
        nmxfile=os.path.join(basepath,"{0}.nmx".format(basename))
        zipf=ZipFile(nmxfile,'w',8)
        for root, dirs, files in os.walk(blockpath):
            for file in files:
                zipf.write(os.path.join(root, file),os.path.join(basename,file))
        if not (Dataset==None) and self._ckbx_addDataset.isChecked():
            if os.path.isfile(Dataset):
                temppath,filename=os.path.split(Dataset)
                zipf.write(Dataset,filename)
        zipf.close()
        if os.path.isdir(blockpath):
            shutil.rmtree(blockpath) 
        if self.ckbx_encryptFile.isChecked() or (usermode==1):        
            encrypted_nmxfile=os.path.join(basepath,"{0}_temp.nmx".format(basename))
            encrypt_file(encryptionkey, nmxfile, encrypted_nmxfile, chunksize=64*1024)
            if os.path.isfile(nmxfile):
                os.remove(nmxfile)
            if os.path.isfile(encrypted_nmxfile):
                shutil.move(encrypted_nmxfile,nmxfile)
        
    def SaveNMLFile(self,filename,Neurons,KNOSSOSflag=False,dataset=None,activeNode=None,editPosition=None,scaled=False):
        UserInfo={'id':unicode(uuid.uuid1()),'path':os.path.expanduser('~'),'platform':sys.platform}

        if not dataset:
            dataset=CubeLoader._BaseName
        if not dataset:
            dataset='unknown'
            
        if scaled:
            scale=self.DataScale
        else:
            scale=[1.0,1.0,1.0]
                    
        if KNOSSOSflag:
            root =lxmlET.Element('things')
            parameters=lxmlET.SubElement(root,'parameters')
            #KNOSSOS 3.2 wants to have comments at the end of the file.
            #However, lxmlET saves the children unordered. 
            #Hence, we have to add it to the output file manually.                
            comments=lxmlET.Element('comments')
            maxTime=0.0 #reset time only once, as we may have multiple things in one file
        else:
            blockpath,ext=  os.path.splitext(unicode(filename))      
            if not os.path.isdir(blockpath):
                os.makedirs(blockpath)        
        NodeID=None    
        for neuronId, neuron_obj in Neurons.iteritems():
            if not neuron_obj:
                continue

            for objtype, child in neuron_obj.children.iteritems():               
                if not child:
                    continue
       
                if not KNOSSOSflag:
                    root =lxmlET.Element('things')
                    parameters=lxmlET.SubElement(root,'parameters')
                    #KNOSSOS 3.2 wants to have comments at the end of the file.
                    #However, lxmlET saves the children unordered. 
                    #Hence, we have to add it to the output file manually.                
                    comments=lxmlET.Element('comments')
                    
                    maxTime=0.0 #reset time as we save single files if KNOSSOSflag is off
                
                color=child.LUT.GetTableValue(child.colorIdx)
                comment=child.comments.get(child,'comment')
                if not comment:
                    comment=""
                childET=lxmlET.SubElement(root,'thing',{\
                    'id':unicode(neuronId),\
                    'color.r':unicode(color[0]),\
                    'color.g':unicode(color[1]),\
                    'color.b':unicode(color[2]),\
                    'color.a':unicode(color[3]),\
                    'comment':unicode(comment)})

                Data=child.get_cleanedup_data()
                NPoints=Data.GetNumberOfPoints()
                if NPoints==0:
                    continue
                Points=Data.GetPoints()
                NodeID=Data.GetPointData().GetArray("NodeID")
                nodesET=lxmlET.SubElement(childET,'nodes')
                for ipoint in range(NPoints):
                    point=Points.GetPoint(ipoint)
                    nodeId=NodeID.GetValue(ipoint)

                    node={'id':unicode(nodeId),'x':unicode(point[0]/scale[0]),'y':unicode(point[1]/scale[1]),'z':unicode(point[2]/scale[2])}
                    #parse attributes and comments
                    attributes=child.comments.get(nodeId)
                    if not attributes==None:
                        for key,value in attributes.iteritems():
                            if value==None:
                                continue
                            if key=='comment':
                                if value=="":
                                    continue
                                lxmlET.SubElement(comments,'comment',{'node':unicode(nodeId),'content':unicode(value)})
                                continue
                            elif key=='time':
                                maxTime=max(maxTime,value)
                                value=np.int(np.ceil(value*1000.0))
                            node[unicode(key)]=unicode(value)
    
                    lxmlET.SubElement(nodesET,'node',node)
    
                edgesET=lxmlET.SubElement(childET,'edges')
                NCells=Data.GetNumberOfCells()
                for icell in range(NCells):
                    CellIds=Data.GetCell(icell).GetPointIds()
                    NCellIds=CellIds.GetNumberOfIds()
                    for iid in range(1,NCellIds):
                        lxmlET.SubElement(edgesET,'edge',{\
                        'source':unicode(NodeID.GetValue(CellIds.GetId(iid-1))),\
                        'target':unicode(NodeID.GetValue(CellIds.GetId(iid)))})
                            
                if not KNOSSOSflag:
                    maxTime=np.int(np.ceil(maxTime*1000.0))
                    lxmlET.SubElement(parameters,'experiment',{'name':dataset})
                    lxmlET.SubElement(parameters,'scale',{\
                        'x':unicode(scale[0]),\
                        'y':unicode(scale[1]),\
                        'z':unicode(scale[2])})
                    lxmlET.SubElement(parameters,'time',{'ms':unicode(maxTime)})
                    if not invalidvector(editPosition):
                        lxmlET.SubElement(parameters,'editPosition',{\
                            'x':unicode(editPosition[0]),\
                            'y':unicode(editPosition[1]),\
                            'z':unicode(editPosition[2])})
                        
                    if (activeNode==None or activeNode<0):
                        if self.QRWin.SelObj[0]==objtype and self.QRWin.SelObj[1]==neuronId and not self.QRWin.SelObj[2]==None:
                            NodeID.ClearLookup()
                            nodeIdx=NodeID.LookupValue(self.QRWin.SelObj[2])
                            if nodeIdx>0:
                                activeNode=self.QRWin.SelObj[2]
                    if not (activeNode==None or activeNode<0):
                        lxmlET.SubElement(parameters,'activeNode',{'id':unicode(activeNode)})
    
                    lxmlET.SubElement(parameters,'lastsavedin',{'version':self.version})
                    lxmlET.SubElement(parameters,'userinfo',UserInfo)

                    #We have to add the comments mannually at the end.
                    prettystring=lxmlET.tostring(root,pretty_print=True, xml_declaration=True)#, 'utf-8')
                    prettystring2=lxmlET.tostring(comments,pretty_print=True)#, 'utf-8')
                    prettystring=prettystring.replace('</things>',prettystring2+'</things>')
                                        
                    flags='';
                    if hasattr(child,'flags'):
                        if not (not child.flags):
                            flags=child.flags

                    #default filename pattern: dataset_parent_id***_objtype_flags_date.nml
                    date=time.strftime('%Y%m%d%H%M',time.gmtime(int(time.time())))
                    objfilename=os.path.join(blockpath,\
                        "{0}_neuron_id{1}_{2}_{3}_{4}.nml".format(dataset,neuronId,objtype,flags,date))
                    text_file = open(objfilename, "w")
                    text_file.write(prettystring)
                    text_file.close()
        if KNOSSOSflag:
            maxTime=np.int(np.ceil(maxTime*1000))
            lxmlET.SubElement(parameters,'experiment',{'name':dataset})
            lxmlET.SubElement(parameters,'scale',{\
                'x':unicode(scale[0]),\
                'y':unicode(scale[1]),\
                'z':unicode(scale[2])})
            lxmlET.SubElement(parameters,'time',{'ms':unicode(maxTime)})
            if not invalidvector(editPosition):
                lxmlET.SubElement(parameters,'editPosition',{\
                    'x':unicode(editPosition[0]),\
                    'y':unicode(editPosition[1]),\
                    'z':unicode(editPosition[2])})
                
            if (activeNode==None or activeNode<0):
                if self.QRWin.SelObj[0]==objtype and self.QRWin.SelObj[1]==neuronId \
                    and not self.QRWin.SelObj[2]==None and not NodeID==None:
                        NodeID.ClearLookup()
                        nodeIdx=NodeID.LookupValue(self.QRWin.SelObj[2])
                        if nodeIdx>0:
                            activeNode=self.QRWin.SelObj[2]
            if not (activeNode==None or activeNode<0):
                lxmlET.SubElement(parameters,'activeNode',{'id':unicode(activeNode)})

            lxmlET.SubElement(parameters,'lastsavedin',{'version':self.version})
            lxmlET.SubElement(parameters,'userinfo',UserInfo)

            #We have to add the comments mannually at the end.
            prettystring=lxmlET.tostring(root,pretty_print=True, xml_declaration=True)#, 'utf-8')
            prettystring2=lxmlET.tostring(comments,pretty_print=True)#, 'utf-8')
            prettystring=prettystring.replace('</things>',prettystring2+'</things>')

            text_file = open(filename, "w")
            text_file.write(prettystring)
            text_file.close()
                
   
    def LoadCSVFile(self,filename):
        print "load csv file with node coordinates."
        print "assume coordinate order: Z (plane), Y, X"
        
        coords = np.loadtxt(filename, delimiter=',')
        if coords.shape==(3,):
             coords=coords.reshape([1,3])
        Points=vtk.vtkPoints()
        for inode in range(coords.shape[0]):
            Points.InsertNextPoint(\
                np.float(coords[inode,2]),\
                np.float(coords[inode,1]),\
                np.float(coords[inode,0]))    
#            Points.InsertNextPoint(\
#                np.float(node[2])*self.DataScale[0],\
#                np.float(node[1])*self.DataScale[1],\
#                np.float(node[0])*self.DataScale[2])    

        #new neuron
        if self.Neurons.__len__()>0:
            NeuronID=float(int(max(self.Neurons.keys()))+1)
        else:
            NeuronID=0.0
        color=self.get_autocolor(self.Neurons.keys().__len__())

        Neurons=OrderedDict()
        Neurons[NeuronID]=neuron(self.ObjectBrowser.model(),NeuronID,color)
        obj=tag(Neurons[NeuronID].item,NeuronID,color)
        obj.set_tags(Points)
        Neurons[NeuronID].children["tag"]=obj
        return Neurons

    def LoadReferenceFile(self,filename='',forcedlg=False,update_alltasks=True):
        if (not self.job):
            return None
        currTask=self.job.get_current_task()
        if currTask==None:
            return None
        if hasattr(currTask,'_ReferenceData'):
            if not filename:
                filename=currTask._ReferenceData;
            if not currTask._ReferenceData:
                return None
            oldfilename=filename
        else:
            return None
        fileext='.ddx'
        
        if not self.CurrentFile and self.Filelist.__len__()>0:
            recentbasepath, placeholder= os.path.split(unicode(self.Filelist[0]._File))
        else:
            recentbasepath, placeholder= os.path.split(unicode(self.CurrentFile))
        
        basepath, basename = os.path.split(unicode(filename))    
        
        if not os.path.isfile(filename):
            filename=os.path.join(recentbasepath,basename)

        if not os.path.isfile(filename):
            filename=os.path.join(application_path,basename)

        if not os.path.isfile(filename):
            forcedlg=True
        
        if forcedlg:
            if os.path.isdir(basepath):
                filename = QtGui.QFileDialog.getOpenFileName(self,"Choose neighborhood file...",os.path.join(basepath,basename),'*'+fileext);
            elif os.path.isdir(recentbasepath):
                filename = QtGui.QFileDialog.getOpenFileName(self,"Choose neighborhood file...",os.path.join(recentbasepath,basename),'*'+fileext);
            elif os.path.isdir(application_path):
                filename= QtGui.QFileDialog.getOpenFileName(self,"Choose neighborhood file...",os.path.join(application_path,basename),'*'+fileext);
            else:
                filename= QtGui.QFileDialog.getOpenFileName(self,"Choose neighborhood file...",basename,'*'+fileext);
            filename=unicode(filename)
        
        if os.path.isfile(filename):
            if update_alltasks:
                for taskobj in self.job.tasks: #update all referencedata filenames in all tasks
                    if hasattr(taskobj,'_ReferenceData'):
                        if unicode(oldfilename)==unicode(taskobj._ReferenceData):
                            taskobj._ReferenceData=filename;
            for key,ddobj in self.DemDriFiles.iteritems():
                ddobj.delete()
            self.DemDriFiles.clear()
            self.Open(filename,"Append",0,0)
            return filename
        else:
            return None
            
    def UpdateDemDriFiles(self,FocalPoint,forceflag=False):
        if not self.DemDriFiles:
            return
#        startTime=time.time()

        if invalidvector(FocalPoint):
            x=self.SpinBoxX.value()*self.DataScale[0]
            y=self.SpinBoxY.value()*self.DataScale[1]
            z=self.SpinBoxZ.value()*self.DataScale[2]
            FocalPoint=np.array([x,y,z],dtype=np.float);

        oldMode=self.ObjectBrowser.selectionMode();
        self.ObjectBrowser.setSelectionMode(QtGui.QAbstractItemView.NoSelection)

        NNeighborCubes=self.SpinBox_DemDriNeighbors.value();
        ph_neuronId=None
        ph_neuron_obj=None
        ph_neuron_obj2=None
        anyupdate=False
        for filename,ddobj in self.DemDriFiles.iteritems():
            files2load=set()
            if ddobj._LoadingMode=='FromFile':
                if not os.path.isdir(ddobj.BasePath):
                    continue;

            #determine cubes to load.
            edgeLength=min(ddobj._CubeSize)*(NNeighborCubes+0.5)
            if edgeLength<ddobj._minDist2Border:
                thres=min(ddobj._CubeSize)*0.5
            else:
                thres=edgeLength-ddobj._minDist2Border
            thres*=thres
            if not (not ddobj.CurrentCentralCube):
                CenterPoint=np.array(ddobj.CurrentCentralCube)+np.array([0.5,0.5,0.5])
                CenterPoint[0]*=ddobj._CubeSize[0]
                CenterPoint[1]*=ddobj._CubeSize[1]
                CenterPoint[2]*=ddobj._CubeSize[2]
                dist2=vtk.vtkMath.Distance2BetweenPoints(FocalPoint,CenterPoint)
#                print np.sqrt(dist2), np.sqrt(thres)    
                if dist2<thres and not forceflag:
                    continue;
            CentralCube=[\
                np.int(np.floor(FocalPoint[0]/ddobj._CubeSize[0])),\
                np.int(np.floor(FocalPoint[1]/ddobj._CubeSize[1])),\
                np.int(np.floor(FocalPoint[2]/ddobj._CubeSize[2]))]
            if ddobj.CurrentCentralCube==CentralCube and not forceflag:
                continue;

            for neuron_obj in ddobj.Neurons:
                neuronId=neuron_obj.NeuronID
                if ph_neuronId==None:
                    ph_neuronId=neuronId
                    ph_neuron_obj=neuron_obj;
                    continue
                neuron_obj.delete()
                del self.Neurons[neuronId]
                
            ddobj.Neurons=set()

            for xcube in range(CentralCube[0]-NNeighborCubes,CentralCube[0]+NNeighborCubes+1):
                for ycube in range(CentralCube[1]-NNeighborCubes,CentralCube[1]+NNeighborCubes+1):
                    for zcube in range(CentralCube[2]-NNeighborCubes,CentralCube[2]+NNeighborCubes+1):
                        for BaseName in ddobj._BaseFileNames:
                            filename=os.path.join(ddobj.BasePath,BaseName + (ddobj._CubePattern % (xcube,ycube,zcube)) + ddobj._FileExt)
                            if ddobj._LoadingMode=='FromFile':
                                if os.path.isfile(filename):
                                    files2load.add(filename)
                            elif ddobj._LoadingMode=='FromArchive':
                                if filename in ddobj.filelist:
                                    files2load.add(filename)
            ddobj.CurrentCentralCube=CentralCube
            
            files2delete=set()
            for filename in ddobj.LoadedCubes:
                if filename in files2load:
                    continue
                files2delete.add(filename)
            for filename in files2delete:
                del ddobj.LoadedCubes[filename]
#            print "#inputconn= ", skeleton.allData.GetNumberOfInputConnections(0), " files2delete= ", files2delete.__len__(), \
#            " files2load= ", files2load.__len__(), \
#            " loaded cubes= ", ddobj.LoadedCubes.keys().__len__()
            if not files2load:
                continue
            
#            if files2load.__len__()>9:
#                showprogress=True;
#            else:
#                showprogress=False;
            Neurons,SelObj,editPosition,dataset=self.LoadCubedNMLFile(ddobj,files2load,1,True)
            
#            print " loaded cubes= ", ddobj.LoadedCubes.keys().__len__()
            
            if not (not Neurons):
                anyupdate=True
                for NeuronID, neuron_obj in Neurons.iteritems():
                    if NeuronID in self.Neurons:
                        if ddobj._ComplementaryLoading==1 and not (NeuronID==ph_neuronId):
                            #Do not load skeletons of already exisiting 
                            #skeletons that are not part of the demand-driven file
                            neuron_obj.delete()
#                            print "Skiped ", NeuronID
                            continue
                        oldNeuronID=NeuronID
                        step=1
                        while self.Neurons.has_key(NeuronID):
                            if NeuronID==ph_neuronId and ph_neuron_obj2==None:
                                ph_neuron_obj2=neuron_obj
                                ddobj.Neurons.add(ph_neuron_obj2)
                                break;
                            NeuronID=oldNeuronID+step*0.001
                            NeuronID=float(np.round(NeuronID,3))
                            step+=1
    
    #                    print "tree id: {0}".format(NeuronID)
                        
                        if not (oldNeuronID==NeuronID):
                            neuron_obj.set_new_neuronId(NeuronID)
                    if not 'd' in neuron_obj.flags:
                        neuron_obj.flags+='d'
                        for obj_type,obj in neuron_obj.children.iteritems():
                            if not 'd' in obj.flags:
                                obj.flags+='d'
                    if not 'r' in neuron_obj.flags:
                        neuron_obj.flags+='r'
                        for obj_type,obj in neuron_obj.children.iteritems():
                            if not 'r' in obj.flags:
                                obj.flags+='r'
                    if not (neuron_obj==ph_neuron_obj2):
#                        print "Added ", NeuronID
                        self.Neurons[NeuronID]=neuron_obj
                        ddobj.Neurons.add(neuron_obj)

        if not (ph_neuronId==None):
            ph_neuron_obj.delete()
            del self.Neurons[ph_neuronId]
        
        if not (ph_neuron_obj2==None):
            self.Neurons[ph_neuron_obj2.NeuronID]=ph_neuron_obj2
        
        self.ObjectBrowser.setSelectionMode(oldMode)
        if anyupdate:
            self.ChangeNeuronVisMode(-1,False)
            for obj in [neuron,skeleton,synapse,soma,region,tag]:
                obj.start_VisEngine(self)
            for obj in [soma,region]:
                obj.update_VisEngine()
#            print "Demand-driven update time: ", time.time()-startTime
                        
    def LoadCubedNMLFile(self,ddobj,filelist,parseflags=0,showprogress=False):
        if not filelist.__class__.__name__=='set':
            filelist=set(filelist)
        SelObj=[None,None,None]
        dataset=None
        editPosition=[None,None,None]
#        timerOffset=self.Timer.timerOffset;
        Neurons=OrderedDict()
        showprogress=1
        if showprogress:
            progress = QtGui.QProgressDialog("Loading files...","Cancel", 0,filelist.__len__(), self)
            progress.setWindowTitle("Wait")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(1500)
            ifile=0;

        LoadingMode=ddobj._LoadingMode;
        if LoadingMode=='FromArchive':
            archive=ddobj.archive
            if not archive.__class__.__name__=='ZipFile':
                print "Error: Invalid archive for file: ", filename
                return
        root={}
        for filename in filelist:
            if showprogress:
                if progress.wasCanceled():
                    break
                progress.setValue(ifile)
                ifile+=1

            if filename in ddobj.LoadedCubes:
                continue
            if LoadingMode=='FromArchive':
                try:
                    bytestr=archive.read(filename)
                    if filename.endswith('.enml'):
                        bytestr=decrypt_string(encryptionkey,bytestr, chunksize=64*1024)
                    root[filename] = lxmlET.fromstring(bytestr)
                except:
                    print "Error: Could not load file from archive: ", filename
                continue;
            if LoadingMode=='FromFile':
                try:           
                    root[filename] = lxmlET.parse(filename).getroot()
                except:
                    print "Error: Could not load file ", filename
                continue;

#        start2=time.time()
#        results= ddobj.pool.map(parParseNML,root.items())
#        for newSkelCube in results:
#            ddobj.LoadedCubes.update(newSkelCube)
#        print "parallel loading of {0} cubes: {1}".format(results.__len__(),time.time()-start2)    
        for filename,rootdata in root.iteritems():                   
            CubeData=ParseNML(filename,rootdata,parseflags)
            ddobj.LoadedCubes[filename]=CubeData
#        print "serial loading of {0} cubes: {1}".format(root.keys().__len__(),time.time()-start2)    

        allData=dict()
        for filename,CubeData in ddobj.LoadedCubes.iteritems():
            for parenttype,CubeParent in CubeData.iteritems(): 
                if not parenttype in ('neuron','area'):
                    continue
                if not parenttype in allData:
                    allData[parenttype]=dict()
                for neuronId,CubeParentData in CubeParent.iteritems():
                    for objtype,CubeObjData in CubeParentData.iteritems():
                        if not objtype in ('skeleton','synapse','soma','tag'):
                            continue
                        if neuronId in allData[parenttype]:
                            parentData=allData[parenttype][neuronId]
                        else:
                            parentData=dict()
                            allData[parenttype][neuronId]=parentData
                        if objtype in parentData:                                
                            objData=parentData[objtype]
                        else:
                            objData=dict()
                            objData['Points']=list()
                            objData['edges']=set()
                            objData['idxEdges']=list()
                            objData['NodeID']=list()
                            objData['attributes']=dict()
                            objData['comments']=dict()
                            objData['flags']=''
                            parentData[objtype]=objData
                            
                        objData['obj_color']=CubeObjData['obj_color']
                             
                        objData['flags']="".join(set(objData['flags']+CubeData['flags'])) #creates a string with unique flags
                        objData['Points'].extend(CubeObjData['Points'])
                        objData['edges'].update(CubeObjData['edges'])
                        #has to be called before the assignement of new nodes to calculate the offset properly
                        offset=objData['NodeID'].__len__()                                    
                        objData['idxEdges'].extend([x+offset for x in CubeObjData['idxEdges']])
                        objData['NodeID'].extend(CubeObjData['NodeID'])
                        objData['attributes'].update(CubeObjData['attributes'])
                        objData['comments'].update(CubeObjData['comments'])
                                    
        Edges2PolyLines=vtk.vtkStripper()
        tempEdge=vtk.vtkIdList()
        tempEdge.SetNumberOfIds(2)
    
        ineuron=self.Neurons.__len__()
        for parenttype,parentData in allData.iteritems():
            for neuronId,neuronData in parentData.iteritems(): 
                for objtype,objData in neuronData.iteritems():
                    if not objtype in ('skeleton','synapse','soma','tag'):
                        continue
                    if neuronId in Neurons:
                        parent_obj=Neurons[neuronId]
                        obj_color=objData["obj_color"]
                        if not obj_color:
                            obj_color=parent_obj
                    else:
                        ineuron+=1
                        obj_color=objData["obj_color"]
                        if not obj_color:
                            obj_color=self.get_autocolor(ineuron)
                        parent_color=obj_color
                        parent_obj=globals()[parenttype](self.ObjectBrowser.model(),neuronId,parent_color)
                        Neurons[neuronId]=parent_obj
                    if objtype in parent_obj.children:
                        obj=parent_obj.children[objtype]
                    else:
                        obj=None
                    if not obj:
                        obj=globals()[objtype](parent_obj.item,neuronId,obj_color)
                        parent_obj.children[objtype]=obj
    
                    float_array=numpy_to_vtk(np.asarray(objData["Points"]), deep=1, array_type=vtk.VTK_FLOAT);
                    float_array.SetNumberOfComponents(3)
                    Points=vtk.vtkPoints()
                    Points.SetData(float_array)
                    
                    NodeID=numpy_to_vtk(np.asarray(objData["NodeID"]), deep=1, array_type=vtk.VTK_ID_TYPE);
    
                    if 'flags' in objData:
                        obj.flags="".join(set(obj.flags+objData['flags'])) #creates a string with unique flags
     
                    if "attributes" in objData:
                        for nodeId,attr in objData['attributes'].iteritems():
                            obj.comments.set(nodeId,attr)
                    if "comments" in objData:
                        for nodeId,value in objData['comments'].iteritems():
                            obj.comments.set(nodeId,'comment',value)
    
                    tempPolyData=vtk.vtkPolyData()
                    tempPolyData.Allocate()
                    tempPolyData.SetPoints(Points)
                    
                    if 'idxEdges' in objData:
                        NEdges=objData['idxEdges'].__len__()/2
                        vtkidxedges=numpy_to_vtk(np.hstack((2*np.ones(NEdges,dtype=np.int).reshape(NEdges,1),np.array(objData['idxEdges'],dtype=np.int).reshape(NEdges,2))).reshape(-1), deep=1, array_type=vtk.VTK_ID_TYPE)                        
                        idxEdges=vtk.vtkCellArray()
                        idxEdges.SetCells(NEdges,vtkidxedges)
                        tempPolyData.SetLines(idxEdges)                        
#                        for edge in objData['idxEdges']: 
#                            tempEdge.SetId(0,edge[0])
#                            tempEdge.SetId(1,edge[1])
#                            tempPolyData.InsertNextCell(vtk.VTK_LINE,tempEdge)                
                    if 'edges' in objData:
                        NodeID.ClearLookup()
                        for edge in objData['edges']: 
                            sourceIdx=NodeID.LookupValue(edge[0])
                            if sourceIdx<0:
                                continue
                            targetIdx=NodeID.LookupValue(edge[1])
                            if targetIdx<0:
                                continue
                            tempEdge.SetId(0,sourceIdx)
                            tempEdge.SetId(1,targetIdx)
                            tempPolyData.InsertNextCell(vtk.VTK_LINE,tempEdge)                
                    tempPolyData.BuildCells()
                    tempPolyData.Modified()
                    if tempPolyData.GetNumberOfCells()>0:
                        Edges2PolyLines.SetInputConnection(tempPolyData.GetProducerPort())
                        Edges2PolyLines.Update()
                        tempCells=Edges2PolyLines.GetOutput().GetLines()
                    else:
                        tempCells=None
    
                    if objtype=='skeleton':
                        obj.set_nodes(Points,NodeID)    
                        if not tempCells==None:
                            obj.add_branch(tempCells,'Lines',1)
                    elif objtype=='soma' or objtype=='region':
                        obj.set_nodes(Points,NodeID)    
                    elif objtype=='synapse':
                        if not tempCells==None:
                            obj.set_tags(Points,tempCells,NodeID)
        
                    elif objtype=='tag':
                        obj.set_tags(Points,NodeID)
        if showprogress:
            progress.setValue(filelist.__len__())
            progress.deleteLater()
    
        return Neurons, SelObj, editPosition, dataset

    def LoadNMLFile(self,filelist,parseflags=0,showprogress=False):
        if not filelist.__class__.__name__=='list':
            filelist=[filelist]
        SelObj=[None,None,None]
        dataset=None
        editPosition=[None,None,None]
        timerOffset=self.Timer.timerOffset;
        Neurons=OrderedDict()
        tempData=OrderedDict()
        totNNodes=0
        
        if showprogress:
            if filelist.__len__()<=2*2*2:
                showprogress=False;
            else:                
                progress = QtGui.QProgressDialog("Loading files...","Cancel", 0,filelist.__len__(), self)
                progress.setWindowTitle("Wait")
                progress.setWindowModality(QtCore.Qt.WindowModal)
                ifile=0;

        for fileobj in filelist:
            archive=None;
            origfilename=None;
            if fileobj.__class__.__name__=='str' or fileobj.__class__.__name__=='unicode':
                filename=fileobj;
                origfilename=filename;
                loadingmode='loadfromfile';
            elif fileobj.__class__.__name__=='list':
                try:                
                    archive=fileobj[0];
                    filename=fileobj[1];
                    loadingmode='loadfromarchive';
                except:
                    1
                if not archive.__class__.__name__=='ZipFile':
                    print "Error: Invalid archive for file: ", filename
                    continue
            else:
                print "Error: Unknown input type for file: ", fileobj
                continue
                
            if showprogress:
                if progress.wasCanceled():
                    break
                ifile+=1
                progress.setValue(ifile)
            
            #assuming filename like dataset_parent_id***_objtype_flags_date.nml
            temppath, tempname = os.path.split(unicode(filename))
            tempname, ext = os.path.splitext(tempname)
            parts=tempname.split('_')
            NParts=parts.__len__()
#            if NParts>0:
#                dataset=parts[0]
#            else:
#                dataset=''
            if NParts>1:
                parent=parts[1]
            else:
                parent='neuron'
            if NParts>2:
                id=parts[2]
            else:
                id=None
            if NParts>3:
                objtype=parts[3]
            else:
                objtype='none'
            if NParts>4 and parseflags==1:
                flags=parts[4]
            else:
                flags=''

#            neuronId=float(id.replace('id',''))
#            if not parent in globals():
#                continue
            if not (parent=='neuron' or parent=='area'):
                parent='neuron'
            if not objtype in ['skeleton','synapse','soma','tag','region']:
                for iobjtype in ['skeleton','synapse','soma','tag','region']:
                    if iobjtype in parts:
                        objtype=iobjtype
                        break
                for ipart in parts:
                    if 'id' in ipart:
                        id=ipart
                        break
                
            if not objtype in ['skeleton','synapse','soma','tag','region']:
                print "Unknown object type. Use default object type: skeleton"
                objtype='skeleton' #default objtype
                
            if loadingmode=='loadfromfile':            
                try:           
                    root = lxmlET.parse(filename).getroot()
                except:
                    print "Error: Skipped file ", filename
                    continue;
            elif loadingmode=='loadfromarchive':
                try:
                    bytestr=archive.read(filename)
                    if filename.endswith('.enml'):
                        bytestr=decrypt_string(encryptionkey,bytestr, chunksize=64*1024)
                    root = lxmlET.fromstring(bytestr)
                except:
                    print "Error: Skipped file in archive: ", filename
                    continue;
                

            parameters=root.find('parameters')

            try: 
                scale=parameters.find('scale')
                scale=[float(scale.get('x')),\
                    float(scale.get('y')),\
                    float(scale.get('z'))]
            except:
                scale=[1.0,1.0,1.0]

            try: 
                editPosition=parameters.find('editPosition')
                editPosition=[float(editPosition.get('x')),\
                    float(editPosition.get('y')),\
                    float(editPosition.get('z'))]
            except:
                1
            
            try: #dirty hacks for compability with previous datasets
                experiment=parameters.find('experiment')
                dataset=experiment.get('name')
                if (dataset=="E085L01_mag1" or dataset=="E085L01") and  (scale==[0.37008,0.37008,1.0] or scale==[0.37,0.37,1.0]):
                    scale=[9.252,9.252,25.0]
                elif (dataset=="E046L01_mag1" or dataset=="E046L01") and  scale==[0.40,0.40,1.0]:
                    scale=[11.29,11.29,30]
                elif (dataset=="cube") and scale==[0.36,0.36,1.0]:
                    scale=[9.0,9.0,25.0]
            except:
                1
                        
            try: 
                activeNode=parameters.find('activeNode')
                activeNode=int(activeNode.get('id'))
            except:
                activeNode=None

            #have to add node first in order to update soma/region labels properly
            comments=root.find('comments')

            for tree in root.iter('thing'):
                neuronId=None
                try:
                    neuronId=float(tree.get('id'))
                except:
                    try:
                        neuronId=float(id.replace('id',''))
                    except:
                        continue

                if neuronId==None:
                    continue

                try: 
                    obj_comment=unicode(tree.get('comment'))
                except:
                    obj_comment=unicode("")


                if neuronId in Neurons:
                    parent_obj=Neurons[neuronId]
                    parentData=tempData[neuronId]
                    try: 
                        obj_color=[float(tree.get('color.r')),\
                            float(tree.get('color.g')),\
                            float(tree.get('color.b')),\
                            float(tree.get('color.a'))]
                    except:
                        parent_color=parent_obj.LUT.GetTableValue(parent_obj.colorIdx)
                        obj_color=[parent_color[0],parent_color[1],parent_color[2],parent_color[3]]
                else:
                    try: 
                        obj_color=[float(tree.get('color.r')),\
                            float(tree.get('color.g')),\
                            float(tree.get('color.b')),\
                            float(tree.get('color.a'))]
                        parent_color=obj_color
                    except:
                        ineuron=self.Neurons.__len__()
                        parent_color=self.get_autocolor(ineuron)    
                        obj_color=[parent_color[0],parent_color[1],parent_color[2],parent_color[3]]
                    parent_obj=globals()[parent](self.ObjectBrowser.model(),neuronId,parent_color)
                    Neurons[neuronId]=parent_obj
                    parentData=OrderedDict()
                    tempData[neuronId]=parentData
                
                if not (not origfilename):
                    parent_obj.filename=origfilename
                
                if objtype in parent_obj.children:
                    obj=parent_obj.children[objtype]
                    NodeID=parentData[objtype]["NodeID"]
                    Points=parentData[objtype]["Points"]
                else:
                    obj=None
                if not obj:
                    obj=globals()[objtype](parent_obj.item,neuronId,obj_color)
                    parent_obj.children[objtype]=obj
                    NodeID=vtk.vtkIdTypeArray()
                    Points=vtk.vtkPoints()
                    parentData[objtype]=OrderedDict()
                    parentData[objtype]["NodeID"]=NodeID
                    parentData[objtype]["Points"]=Points
                
                obj.flags="".join(set(obj.flags+flags)) #creates a string with unique flags
                
                if not (not obj_comment):
                    if not (obj_comment=='None' or obj_comment=='NONE'):
                        obj.comments.set(obj,"comment",obj_comment)
                        obj.updateItem()

                NNodes=0
                try:
                    nodes=tree.find('nodes')
                    for node in nodes.iter('node'):
                        nodeId=int(node.attrib['id'])
                        point=[float(node.attrib['x'])*scale[0],\
                            float(node.attrib['y'])*scale[1],\
                            float(node.attrib['z'])*scale[2]]
                        if node.attrib.__len__()>4:
                            attributes=dict(node.attrib)
                            del attributes['id']
                            del attributes['x']
                            del attributes['y']
                            del attributes['z']
                            if attributes.has_key('time'):
                                if not attributes['time']:
                                    del attributes['time']
                                else:
                                    time=float(attributes['time'])/1000.0 #convert from ms to sec
                                    if time>0.0:
                                        attributes['time']=time
                                        timerOffset=max(timerOffset,time)
                                    else:                           
                                        del attributes['time']
                            if attributes.has_key('comment'):
                                value=unicode(attributes['comment'])     
                                if not value:
                                    del attributes['comment']
                            
                            if attributes.__len__()>0:
                                obj.comments.set(nodeId,attributes)
    
                        NodeID.InsertNextValue(nodeId)                                    
                        Points.InsertNextPoint(point)    
                        NNodes+=1
                except:
                    print "No nodes found."
                    continue

                totNNodes+=NNodes
                try:
                    if "Edges" in parentData[objtype]:
                        parentData[objtype]["Edges"].extend(tree.find('edges'))
                    else:
                        parentData[objtype]["Edges"]=tree.find('edges')
                except:
                    print "No edges found."

                #have to add node first in order to update soma/region labels properly
                if not comments==None:
                    NodeID=tempData[neuronId][objtype]["NodeID"]
                    NodeID.ClearLookup()
                    for comment in comments.iter('comment'):
                        if not (comment.attrib.has_key('node') and comment.attrib.has_key('content')):
                            continue
                        nodeId=np.int(comment.attrib['node'])
                        if NodeID.LookupValue(nodeId)==-1:
                            continue
                        value=unicode(comment.attrib['content'])     
                        if not value=="":
                            obj.comments.set(nodeId,'comment',value)

        Edges2PolyLines=vtk.vtkStripper()
        tempEdge=vtk.vtkIdList()
        tempEdge.SetNumberOfIds(2)
        for neuronId,neuron_obj in Neurons.iteritems():
            for objtype,obj in neuron_obj.children.iteritems():
                NodeID=tempData[neuronId][objtype]["NodeID"]
                NodeID.ClearLookup()
                Points=tempData[neuronId][objtype]["Points"]
                tempPolyData=vtk.vtkPolyData()
                tempPolyData.Allocate()
                tempPolyData.SetPoints(Points)
                try:
                    Edges=tempData[neuronId][objtype]["Edges"]
                    #TODO: remove dublicate edges
                    for edge in Edges.iter('edge'): 
                        sourceID=NodeID.LookupValue(int(edge.get('source')))
                        targetID=NodeID.LookupValue(int(edge.get('target')))
                        if sourceID==targetID:
#                                print "Invalid edge: source = {0} target = {1}".format(edge.get('source'),edge.get('target'))
                            continue
                        elif sourceID==-1 or targetID==-1:
#                                print "Invalid edge: source = {0} target = {1}".format(edge.get('source'),edge.get('target'))
                            continue
                        tempEdge.SetId(0,sourceID)
                        tempEdge.SetId(1,targetID)
                        tempPolyData.InsertNextCell(vtk.VTK_LINE,tempEdge)                
                    tempPolyData.BuildCells()
                    tempPolyData.Modified()
                    Edges2PolyLines.SetInputConnection(tempPolyData.GetProducerPort())
                    Edges2PolyLines.Update()
                    tempCells=Edges2PolyLines.GetOutput().GetLines()
                except:
                    tempCells=None;

                if objtype=='skeleton':
                    obj.set_nodes(Points,NodeID)    
                    if not tempCells==None:
                        obj.add_branch(tempCells,'Lines',1)
                elif objtype=='soma' or objtype=='region':
                    obj.set_nodes(Points,NodeID)    
                elif objtype=='synapse':
                    if not tempCells==None:
                        obj.set_tags(Points,tempCells,NodeID)
    
                elif objtype=='tag':
                    obj.set_tags(Points,NodeID)
                
                if not (activeNode==None):
                    SelObj=[objtype,neuronId,activeNode]
            #print "Loaded {0} of {1} id {2} with {3} nodes.".format(objtype,parent,neuronId,NNodes)
        if showprogress:
            progress.setValue(filelist.__len__())

        print "Total number of nodes: {0}".format(totNNodes)
        self.Timer.timerOffset=timerOffset;
        return Neurons, SelObj, editPosition, dataset

    def LoadSkeletonFile(self,origfilename):
        Data = scipy.io.loadmat(origfilename,struct_as_record=False, squeeze_me=True)
#        tempData=scipy.io.loadmat(origfilename,squeeze_me=True, chars_as_strings=False, mat_dtype=True, struct_as_record=False)
        if Data.has_key('Neuron'):
            Data=Data['Neuron']
        elif Data.has_key('ConsTree'):
            Data=Data['ConsTree']
        elif Data.has_key('Somas'):
            Data=Data['Somas']
        if Data.__class__.__name__=='mat_struct':
            Data=np.array([Data]);
        if not hasattr(Data,'size'):
            print origfilename, 'seems to be an invalid matlab skeleton.' 
            return None, None, None, None
        Neurons=OrderedDict()
        timerOffset=self.Timer.timerOffset
        for ineuron in range(Data.size):
            tempData=Data[ineuron]
            #NeuronID=np.round(tempData.Attributes.id.astype('float'),3)
            try:
                NeuronID=float(np.round(float(tempData.Attributes.id),3))
            except:
                NeuronID=float(np.round(tempData.Attributes.id.astype('float'),3))
            oldNeuronID=NeuronID
            step=1
            while self.Neurons.has_key(NeuronID):
                NeuronID=oldNeuronID+step*0.001
                NeuronID=float(np.round(NeuronID,3))
                step+=1
                
            if hasattr(tempData.Attributes, 'color'):
                #might want to validate input
                a=r=g=b=None
                color=tempData.Attributes.color
                if hasattr(color,'r'):
                    r=np.float(color.r)
                if hasattr(color,'g'):
                    g=np.float(color.g)
                if hasattr(color,'b'):
                    b=np.float(color.b)
                if hasattr(color,'a'):
                    a=np.float(color.a)
                if r==None or b==None or g==None:
                    color=self.get_autocolor(ineuron)
                else:
                    if a==None:
                        a=1.0
                    color=(r,g,b,a)
                    print "Loaded MATLAB color:", color
            else:
#                ineuron=self.Neurons.__len__()
                color=self.get_autocolor(ineuron)


            if hasattr(tempData.Attributes, 'comment'):
                #might want to validate input
                obj_comment=unicode(tempData.Attributes.comment)
            else:
#                ineuron=self.Neurons.__len__()
                obj_comment=unicode("")

            Neurons[NeuronID]=neuron(self.ObjectBrowser.model(),NeuronID,color)
                
            if hasattr(tempData, 'activity'):
                Neurons[NeuronID].activity=tempData.activity;        
            scale=self.DataScale
            dataset=None
            if hasattr(tempData,'Parameters'):
                if hasattr(tempData.Parameters,'scale'):
                    scale=tempData.Parameters.scale       
                if hasattr(tempData.Parameters,'experiment'):
                    if hasattr(tempData.Parameters.experiment,'name'):
                        dataset=str(tempData.Parameters.experiment.name)
            
            if tempData.__dict__.has_key('soma'):
                NNodes=tempData.soma.size
                if NNodes>0:
                    child=soma(Neurons[NeuronID].item,NeuronID,color)                    

                    Points=vtk.vtkPoints()
                    Points.SetNumberOfPoints(NNodes)
                    NodeID=vtk.vtkIdTypeArray()
                    for inode in range(NNodes):
                        if NNodes==1:
                            nodeobj=tempData.soma
                        else:
                            nodeobj=tempData.soma[inode]
                        if hasattr(nodeobj,'id'):
                            if nodeobj.id.__class__.__name__=='unicode':
                            	nodeId=np.int(nodeobj.id)
                            else:
                            	nodeId=np.int(nodeobj.id.astype(np.int))
                        else:
                             nodeId=inode
                        NodeID.InsertNextValue(nodeId)
                        Points.SetPoint(inode,\
                        np.float(nodeobj.x)*np.float(scale[0]),\
                        np.float(nodeobj.y)*np.float(scale[1]),\
                        np.float(nodeobj.z)*np.float(scale[2]))
                        if hasattr(nodeobj, 'comment'):
                            comment=nodeobj.comment
                            if comment.__class__.__name__=='unicode':
                                if comment.__len__()==0:
                                    continue
                            elif (comment.__class__.__name__=='float' or comment.__class__.__name__=='int' or comment.__class__.__name__=='double'):
                                1
                            else:
                                if comment.size==0:
                                    continue
                            comment=unicode(comment)
                            if not comment=='':
                                child.comments.set(nodeId,'comment',comment)                
                    child.set_nodes(Points,NodeID)
                    Neurons[NeuronID].children["soma"]=child
            
            if tempData.__dict__.has_key('branches'):
                child=skeleton(Neurons[NeuronID].item,NeuronID,color)
                
                if not (not obj_comment):
                    if not (obj_comment=='None' or obj_comment=='NONE'):
                        child.comments.set(child,"comment",obj_comment)
                        child.updateItem()
                
                NodeID=vtk.vtkIdTypeArray()
                Points=vtk.vtkPoints()
                if hasattr(tempData.nodes,'size'):
                    NNodes=tempData.nodes.size
                else:
                    NNodes=1
                Points.SetNumberOfPoints(NNodes)
                for inode in range(NNodes):
                    if NNodes==1:
                        nodeobj=tempData.nodes
                    else:
                        nodeobj=tempData.nodes[inode]
                    if nodeobj.id.__class__.__name__=='unicode':
                    	nodeId=np.int(nodeobj.id)
                    else:
                    	nodeId=np.int(nodeobj.id.astype(np.int))
                    NodeID.InsertNextValue(nodeId)
                    Points.SetPoint(inode,\
                    np.float(nodeobj.x)*np.float(scale[0]),\
                    np.float(nodeobj.y)*np.float(scale[1]),\
                    np.float(nodeobj.z)*np.float(scale[2]))    
                    if hasattr(nodeobj, 'comment'):
                        comment=nodeobj.comment
                        if comment.__class__.__name__=='unicode':
                            if comment.__len__()==0:
                                continue
                        elif (comment.__class__.__name__=='float' or comment.__class__.__name__=='int' or comment.__class__.__name__=='double'):
                            1
                        else:
                            if comment.size==0:
                                continue
                        comment=unicode(comment)
                        if not comment=='':
                            child.comments.set(nodeId,'comment',comment)
                    if hasattr(nodeobj, 'radius'):
                        radius=nodeobj.radius
                        if not (radius.__class__.__name__=='float' or radius.__class__.__name__=='int' or radius.__class__.__name__=='double'):
                            if radius.size==0:
                                continue
                        child.comments.set(\
                        nodeId,'radius',np.float(radius))
                    if hasattr(nodeobj, 'time'):
                        time=nodeobj.time
                        if time.__class__.__name__=='unicode':
                            if time.__len__()==0:
                                continue
                        else:
                            if time.__sizeof__()==0:
                                continue
                        time=np.float(time)
                        if time>0.0:
                            child.comments.set(nodeId,'time',time)
                            if time>timerOffset:
                                timerOffset=time
                child.set_nodes(Points,NodeID)
                NodeID.ClearLookup()
                if hasattr(tempData.branches,'size'):
                    NBranches=tempData.branches.size
                else:
                    NBranches=1;
                for ibranch in range(NBranches):
                    if NBranches==1:
                        branchobj=tempData.branches
                    else:
                        branchobj=tempData.branches[ibranch]
                    NBranchNodes=branchobj.branch.size
                    tempBranch=vtk.vtkIdList()
                    lastValidNode=0
                    if branchobj.branch[0].__class__.__name__=='unicode':
                        whichNode=NodeID.LookupValue(np.int(\
                            branchobj.branch[0]))
                    else:
                        whichNode=NodeID.LookupValue(np.int(\
                            branchobj.branch[0].astype(np.int)))
                    tempBranch.InsertNextId(whichNode)
                    for inode in range(1,NBranchNodes): 
                        if branchobj.branch[inode].__class__.__name__=='unicode':
                            whichNode=NodeID.LookupValue(np.int(\
                                branchobj.branch[inode]))
                        else:
                            whichNode=NodeID.LookupValue(np.int(\
                                branchobj.branch[inode].astype(np.int)))
                        if vtk.vtkMath.Distance2BetweenPoints(\
                            Points.GetPoint(tempBranch.GetId(lastValidNode)),\
                            Points.GetPoint(whichNode))<1.0e-5:
                            print "warning: coincident edge points."
                            continue       
                        lastValidNode+=1
                        tempBranch.InsertNextId(whichNode)
                    if tempBranch.GetNumberOfIds()<2:
                        continue
                    child.add_branch(tempBranch)
            Neurons[NeuronID].children["skeleton"]=child
        self.Timer.timerOffset=timerOffset

        for neuronId, neuron_obj in Neurons.iteritems():
            neuron_obj.filename=origfilename

        return Neurons, None, None, dataset

    def SetPath(self,obj=None):
        if not obj:
            return
        if obj=='SeedPath':
            fileext="*.nml"
            filename=unicode(self.text_SeedPath.text())
            dialogtype='file'
        elif obj=='JobPath':
            fileext="*.job" 
            filename=unicode(self.text_JobPath.text())
            dialogtype='file'
        elif obj=='DatasetPath':
            fileext="*.zip" 
            filename=unicode(self.text_DatasetPath.text())
            dialogtype='file'
        elif obj=='GIFPath':
            fileext="*.gif" 
            filename=unicode(self.text_GIFPath.text())
            dialogtype='dir'
        elif obj=='CapturePath':
            fileext="*.png"
            filename=unicode(self.text_CapturePath.text())
            dialogtype='savefile'            
        elif obj=='Skel2TIFFPath':
            fileext="*.tif"
            filename=unicode(self.text_Skel2TIFFPath.text())
            dialogtype='savefile'                        
        else:
            return
            
        if not (not filename):
            currentPath=filename
        elif (not self.CurrentFile) and self.Filelist.__len__()>0:
            currentPath= os.path.split(unicode(self.Filelist[0]._File))
        else:
            currentPath= os.path.split(unicode(self.CurrentFile))
        if os.path.isdir(currentPath[0]):
            if dialogtype=='file':
                filename = QtGui.QFileDialog.getOpenFileName(self,"Open file...",currentPath[0],fileext);
            elif dialogtype=='savefile':
                filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...",currentPath[0],fileext);
            else:
                filename = QtGui.QFileDialog.getExistingDirectory(self,"Choose save directory...",currentPath[0]);
                
        else:
            if os.path.isdir(application_path):
                if dialogtype=='file':
                    filename= QtGui.QFileDialog.getOpenFileName(self,"Open file...",application_path,fileext);
                elif dialogtype=='savefile':
                    filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...",application_path,fileext);
                else:
                    filename = QtGui.QFileDialog.getExistingDirectory(self,"Choose save directory...",application_path);
            else:
                if dialogtype=='file':
                    filename= QtGui.QFileDialog.getOpenFileName(self,"Open file...","",fileext);
                elif dialogtype=='savefile':
                    filename = QtGui.QFileDialog.getSaveFileName(self,"Save file as...",fileext);
                else:
                    filename = QtGui.QFileDialog.getExistingDirectory(self,"Choose save directory...","");

        filename=unicode(filename)

        if obj=='SeedPath':
            self.text_SeedPath.setText(filename)
        elif obj=='JobPath':
            self.text_JobPath.setText(filename)
        elif obj=='DatasetPath':
            self.text_DatasetPath.setText(filename)
        elif obj=='GIFPath':
            self.text_GIFPath.setText(filename)
        elif obj=='CapturePath':
            self.text_CapturePath.setText(filename)
        elif obj=='Skel2TIFFPath':
            self.text_Skel2TIFFPath.setText(filename)
            
    def Capture(self,magnification=1):        
        print "FPoint= {0}; CamPos= {1}; ViewUp= {2}".format(\
        self.QRWin.activeRenderer.Camera.GetFocalPoint(),\
        self.QRWin.activeRenderer.Camera.GetPosition(),\
        self.QRWin.activeRenderer.Camera.GetViewUp())
    
        filename=unicode(self.text_CapturePath.text())
        filename, fileext = os.path.splitext(filename)
        if self.ckbx_incrementCaptureFile.isChecked():
            result = re.search(r'(.+)_(\d+)$', filename)
            if result==None:
                filenumber=1
            else:
                filename=result.group(1)
                filenumber=np.int(result.group(2))
            
            tempfilename=r"{0}_{1:{fill}4}".format(filename,filenumber,fill=0)
            while os.path.isfile(tempfilename+fileext):
                filenumber+=1
                tempfilename=r"{0}_{1:{fill}4}".format(filename,filenumber,fill=0)
            filename=tempfilename    
        filename+=fileext
        print "capture screen: ", filename
        windowToImageFilter =vtk.vtkWindowToImageFilter();
        windowToImageFilter.ReadFrontBufferOff() #VERY IMPORTANT FOR LAPTOP AT LEAST 
        windowToImageFilter.SetInputBufferTypeToRGB()
        windowToImageFilter.SetMagnification(self.SpinBox_CaptureMag.value());
        windowToImageFilter.SetInput(self.QRWin.RenderWindow);
        windowToImageFilter.Update();
        vtkAVIWriter=vtk.vtkPNGWriter()#vtkFFMPEGWriter()
        vtkAVIWriter.SetInput(windowToImageFilter.GetOutput())
        vtkAVIWriter.SetFileName(filename)
        vtkAVIWriter.Write()
        self.QRWin.RenderWindow.Render()
        self.text_CapturePath.setText(filename)
        
    def SkelNodes2Soma(self):
        SelObj=self.QRWin.SelObj
        if (not SelObj):
            return None,-1            
        if SelObj.__len__()<3:
            return None,-1
        NeuronID=SelObj[1]
        if not (NeuronID in self.Neurons):
            return None,-1
        if not (SelObj[0]=="skeleton"):
            return None,-1
        if not "skeleton" in self.Neurons[NeuronID].children:
            return None,-1
        child=self.Neurons[NeuronID].children["skeleton"]
        
        Sphere = vtk.vtkSphereSource()
        Sphere.SetPhiResolution(self.SpinBoxSphereRes.value())
        Sphere.SetThetaResolution(self.SpinBoxSphereRes.value())
        Sphere.SetRadius(self.SpinBox_SomaRadius.value()*1000.0)
        Sphere.Update()
        
        child.validData.Update()
        SomaCenters=child.validData.GetOutput()

        NodeIDs=SomaCenters.GetPointData().GetArray("NodeID")
        nodeIds=list(vtk_to_numpy(NodeIDs))

        for ipoint in range(SomaCenters.GetNumberOfPoints()):
            Sphere.SetCenter(SomaCenters.GetPoint(ipoint))
            Sphere.Update()
            SomaData=Sphere.GetOutput()
            
            neuronId = self.NewNeuron()
            
            parent_obj=self.Neurons[neuronId]
            obj_color=parent_obj.LUT.GetTableValue(parent_obj.colorIdx)
            objtype="soma";
            obj=globals()[objtype](parent_obj.item,neuronId,obj_color)
            parent_obj.children[objtype]=obj
            obj.set_nodes(SomaData.GetPoints())
            obj.start_VisEngine(self)

            obj.comments.set(0,child.comments.get(nodeIds[ipoint]))
            
        child.delete_node(nodeIds)
        self.SetSomaVisibility()

        
    

    def CreateGIF(self):
        filename=unicode(self.text_GIFPath.text())
        filename, fileext = os.path.splitext(filename)
        if self.ckbx_AppendCoordsGIFFile.isChecked():
            filename += '_x{0}'.format(self.SpinBoxX.value()) + '_y{0}'.format(self.SpinBoxY.value()) + '_z{0}'.format(self.SpinBoxZ.value())            
        if self.ckbx_incrementGIFFile.isChecked():
            result = re.search(r'(.+)_ex(\d+)$', filename)
            if result==None:
                filenumber=1
            else:
                filename=result.group(1)
                filenumber=np.int(result.group(2))
            
            tempfilename=r"{0}_ex{1:{fill}2}".format(filename,filenumber,fill=0)
            while os.path.isfile(tempfilename+fileext):
                filenumber+=1
                tempfilename=r"{0}_ex{1:{fill}2}".format(filename,filenumber,fill=0)
            filename=tempfilename    

        if self.radioBtn_GIFviewport_YX.isChecked():
            whichviewport=self.QRWin.viewports['YX_viewport']
            direction='Z'
            whichplane=self.planeROIs['YX_planeROI']
            immodifier=0
        elif self.radioBtn_GIFviewport_YZ.isChecked():
            whichviewport=self.QRWin.viewports['YZ_viewport']
            if self.radioButton_orthRef.isChecked():
                direction='orth'
            else:
                direction='X'
            whichplane=self.planeROIs['YZ_planeROI']
            immodifier=0
        elif self.radioBtn_GIFviewport_ZX.isChecked():
            whichviewport=self.QRWin.viewports['ZX_viewport']
            if self.radioButton_orthRef.isChecked():
                direction='orth'
            else:
                direction='Y'
            whichplane=self.planeROIs['ZX_planeROI']
            immodifier=0
        elif self.radioBtn_GIFviewport_orth.isChecked():
            whichviewport=self.QRWin.viewports['Orth_viewport']
            direction='orth'
            whichplane=self.planeROIs['Orth_planeROI']
            immodifier=1
        else:
            return
            
                    

        if CubeLoader.ROIState.value==0:
            return 
            
#        maxROIEdge=np.max([[plane.cROISize[0]*plane.ROIRes[0]*plane._ROIScale[0],\
#            plane.cROISize[1]*plane.ROIRes[1]*plane._ROIScale[1]] \
#            for key,plane in window1.planeROIs.iteritems()])
        maxROIEdge=np.min([[plane.ROIRes[0]*plane._ROIScale[0],\
            plane.ROIRes[1]*plane._ROIScale[1]] \
            for key,plane in window1.planeROIs.iteritems()])
        Magnification=c_int(CubeLoader.CheckMagnification(maxROIEdge));
        
        cROISize=RawArray(c_int,[self.SpinBox_GIFSize.value(),self.SpinBox_GIFSize.value()]) #[vertical size,horizontal size]
        
        slicerange=self.SpinBox_GIFRange.value()
        
        frames=list()        
        for step in ((-1*np.ones([slicerange])).tolist()+np.ones([2*slicerange]).tolist()+\
            (-1*np.ones([slicerange])).tolist()):
            whichviewport.Move(\
                step*self.SpinBox_Steps.value()*CubeLoader.Magnification[0],direction) 
            ROI=RawArray(c_ubyte,255*np.ones([cROISize[0]*cROISize[1]],dtype=np.uint))
            complete=0
            starttime=time.time()
            while (not complete) and (time.time()-starttime)<2.0 :
                complete=extractROIlib.interp_ROI(whichplane.cCenter,\
                    whichplane.cvDir,whichplane.chDir,cROISize,ROI,Magnification,\
                    0,CubeLoader.Position,CubeLoader.Magnification,CubeLoader.LoaderState);

            frame = Image.new('L',cROISize)
            frame.putdata(ROI)
            if immodifier==0:
                frame=frame.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
            else:
                frame=frame.transpose(Image.ROTATE_90)
            frames.append(frame)
#            frames.append(frame.transpose(Image.ROTATE_270))
#            
#            im.save(filename+'.png')
        
        impath,tail=  os.path.split(unicode(filename))      
        if not os.path.isdir(impath):
            os.makedirs(impath)        
        images2gif.writeGif(filename+fileext,frames)
#        self.text_GIFPath.setText(filename+fileext)

    def Convert2NML(self):
        filelist=unicode(self.text_SeedPath.text())
        if '*' in filelist:
            filelist=glob.glob(filelist)
        else:
            filelist=[filelist]            
        for filename in filelist:
            success= self.LoadNMXFile(filename,convertflag=True)
            if not success:
                continue            
            if self.ckbx_deleteSeed.isChecked() and os.path.isfile(filename):
                os.remove(filename)

    def Encrypt(self):
        filelist=unicode(self.text_SeedPath.text())
        if '*' in filelist:
            filelist=glob.glob(filelist)
        else:
            filelist=[filelist]            
        for filename in filelist:
            if not filename:
                continue
            elif (filename.endswith('.nmx') and not zipfile.is_zipfile(filename)):
                continue;        
            elif not (filename.endswith('.nmx') or filename.endswith('.nml')):
                continue
            basepath, basename = os.path.split(filename)
            basename, ext = os.path.splitext(basename)  
            if not basename:
                continue
            if ext==u'.nml' or ext=='.nml':
                ext=u'.enml'
            tempdir=os.path.join(application_path,'temp')
            tempdir=os.path.join(tempdir,basename)
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir) #remove any existing temporary host dir
            if not os.path.isdir(tempdir):
                os.makedirs(tempdir)
            
            encrypted_filename=os.path.join(tempdir,'{0}{1}'.format(basename,ext))
            encrypt_file(encryptionkey,filename, encrypted_filename, chunksize=64*1024)
            if self.ckbx_deleteSeed.isChecked() and os.path.isfile(filename):
                os.remove(filename)
            filename=os.path.join(basepath,'{0}{1}'.format(basename,ext))
            shutil.move(encrypted_filename,filename)
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir) #remove any existing temporary host dir
            print "Created encrypted file: ", filename

    def Decrypt(self):
        if usermode==1:
            return
        filelist=unicode(self.text_SeedPath.text())
        if '*' in filelist:
            filelist=glob.glob(filelist)
        else:
            filelist=[filelist]            
        for filename in filelist:
            if not filename:
                continue
            if not (filename.endswith('.nmx') or filename.endswith('.enml')):
                continue
            if zipfile.is_zipfile(filename):
                continue;
            basepath, basename = os.path.split(filename)
            basename, ext = os.path.splitext(basename)
            if not basename:
                continue
            if ext==u'.enml' or ext=='.enml':
                ext=u'.nml'
            tempdir=os.path.join(application_path,'temp')
            tempdir=os.path.join(tempdir,basename)
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir) #remove any existing temporary host dir
            if not os.path.isdir(tempdir):
                os.makedirs(tempdir)
            
            decrypted_filename=os.path.join(tempdir,'{0}{1}'.format(basename,ext))
            decrypt_file(encryptionkey,filename, decrypted_filename, chunksize=64*1024)
            if self.ckbx_deleteSeed.isChecked() and os.path.isfile(filename):
                os.remove(filename)
            filename=os.path.join(basepath,'{0}{1}'.format(basename,ext))
            shutil.move(decrypted_filename,filename)
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir) #remove any existing temporary host dir
            print "Created decrypted file: ", filename

    def CreateJob(self):
        filelist=unicode(self.text_SeedPath.text())
        jobfilename=unicode(self.text_JobPath.text())
        
        if '*' in filelist:
            filelist=glob.glob(filelist)
        else:
            filelist=[filelist]            
        
        oldState=self.ckbx_encryptFile.isChecked()
        self.ckbx_encryptFile.setChecked(self.ckbx_encryptJobs.isChecked())
        
        if self.ckbx_allInOne.isChecked():
            if self.comboBox_TaskType.currentIndex()==0:
                self.CreateTracingJob(filelist,jobfilename)
            for filename in filelist:                
                if self.ckbx_deleteSeed.isChecked() and os.path.isfile(filename):
                    os.remove(filename)
        else:
            for filename in filelist:
                if self.comboBox_TaskType.currentIndex()==0:
                    self.CreateTracingJob(filename,jobfilename)
                
                if self.ckbx_deleteSeed.isChecked() and os.path.isfile(filename):
                    os.remove(filename)
                    
        self.ckbx_encryptFile.setChecked(oldState)
        
        
    def CreateTracingJob(self,filelist,jobfilename=None):
        #turns *.nml files into a proper *.nmx tracing job file.
        if not filelist.__class__.__name__=='list':
            filelist=[filelist]

        Neurons=OrderedDict()
        SelObj=[None,None,None]
        editPosition=[None,None,None]
        for filename in filelist:
            filename=unicode(filename)
            if not filename.endswith(".nml"):
                continue
            tempNeurons,SelObj, editPosition, dataset=self.LoadNMLFile(filename,parseflags=0)   
            if not (not tempNeurons):
                Neurons.update(tempNeurons)

        tempJob=job(self,jobfilename)
        tempJob.load_job()
        tempJob._Dataset=dataset
                        
        templateTask=tracing(self)
        if tempJob.tasks.__len__()>0:
            for attr in dir(tempJob.tasks[0]):
                if attr.startswith('__'):
                    continue
                if not attr.startswith('_'):
                    continue
                setattr(templateTask,attr,getattr(tempJob.tasks[0],attr))
        tempJob.tasks=[]
        if not (not Neurons):
            for neuronId,neuron_obj in Neurons.iteritems():
                templateTask._neuronId=neuronId
#                if neuronId==SelObj[1] and SelObj[0]=="skeleton":
#                    templateTask._currNodeId=SelObj[2]
                templateTask._currNodeId=-1 #TODO: synchronize currNodeId+ and activeNode?
                tempJob.tasks.append(templateTask)
        else:
            tempJob.tasks.append(templateTask)   
        if self._ckbx_addDataset.isChecked():
            Dataset=unicode(self.text_DatasetPath.text())
        else:
            Dataset=None
            
        if filelist.__len__()>0:
            filename=unicode(filelist[0])
        else:
            filename=None
        self.SaveNMXFile(filename,Neurons,tempJob,Dataset)
        if not (not Neurons):
            oldMode=self.ObjectBrowser.selectionMode();
            self.ObjectBrowser.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
            for neuronId, neuron_obj in Neurons.iteritems():
                neuron_obj.delete()            
            self.ObjectBrowser.setSelectionMode(oldMode)

            
    def setup_visualization(self):
        for neuronId, neuron_obj in self.Neurons.iteritems():
            neuron_obj.start_VisEngine(self)
    
        self.initViewports()
                            
        self.QRWin.RenderWindow.Render()

    def MaxMinJobTab(self,state):
        if self.ckbx_MaximizeJobTab.isChecked():
            self.ckbx_HideSkelViewport.setChecked(1)
            remainingWidth=self.QRWin.DistributeViewports()
            
            self.Job.setMinimumSize(350+remainingWidth,450)
            if not self.Job.isFloating():
                self.tabifyDockWidget(self.Settings,self.Job);
            else:
                self.Job.resize(350+remainingWidth,450)
        else:
            remainingWidth=350;
            self.ckbx_HideSkelViewport.setChecked(0)
            self.Job.setMinimumSize(remainingWidth,450)
            if not self.Job.isFloating():
                self.addDockWidget(QtCore.Qt.RightDockWidgetArea,self.Job)
            else:
                self.Job.resize(remainingWidth,450)

    def BrowserSelectionChanged(self):
        item=self.ObjectBrowser.model().itemFromIndex(self.ObjectBrowser.currentIndex())
        if not item:
            return
        if not hasattr(item,'objtype'):
            return
        if not hasattr(item,'neuronId'):
            return
        ObjType=item.objtype
        neuronID=item.neuronId
#        print ObjType, neuronID, self.QRWin.SelObj
        if hasattr(item,'nodeId'):
            nodeId=item.nodeId
        else:
            nodeId=None
        if (self.QRWin.SelObj[0]==ObjType) and (self.QRWin.SelObj[1]==neuronID) and (self.QRWin.SelObj[2]==nodeId):
            return
        if (self.QRWin.SelObj[0]==ObjType) and (self.QRWin.SelObj[1]==neuronID) and (nodeId==None):
            return
        self.QRWin.SetActiveObj(ObjType,neuronID,nodeId)
        self.QRWin.GotoActiveObj()
#        print "selected: ", item.text()


    def Skel2TIFF(self):                 
        if not CubeLoader._Origin.__len__()==0:
            dataorigin=[0,0,0]
        else:
            dataorigin=CubeLoader._Origin
        if CubeLoader._Extent.__len__()==0:
            print "Dataset extent not defined."
            return
        else:
            datadim=CubeLoader._Extent
            
        spacing=CubeLoader._DataScale[0:3];
        
        dataorigin=np.array(dataorigin,dtype=np.float)        
        spacing=np.array(spacing,dtype=np.float)        
        
        origin=np.array([0.0,0.0,0.0],dtype=np.float);
        origin[0]=spacing[0]*(dataorigin[0]+0.5*datadim[0])
        origin[1]=spacing[1]*(dataorigin[1]+0.5*datadim[1])
        
        vtkSplineFilter = vtk.vtkSplineFilter()
        vtkSplineFilter.SetSubdivideToLength()
        vtkSplineFilter.SetLength(min(spacing)*0.5)
        vtkSplineFilter.SetInputConnection(skeleton.Stripper.GetOutputPort())

        vtkSmoothPolyDataFilter=vtk.vtkSmoothPolyDataFilter()
        vtkSmoothPolyDataFilter.FeatureEdgeSmoothingOn()
        vtkSmoothPolyDataFilter.SetNumberOfIterations(5)
        vtkSmoothPolyDataFilter.SetInputConnection(vtkSplineFilter.GetOutputPort())
        
        BranchTubeFilter = vtk.vtkTubeFilter()
        BranchTubeFilter.SetNumberOfSides(11)
        BranchTubeFilter.CappingOn()
        BranchTubeFilter.SetRadius(skeleton.DefaultRadius[0])
        BranchTubeFilter.SetRadiusFactor(1.0)
        BranchTubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        BranchTubeFilter.SetInputConnection(vtkSmoothPolyDataFilter.GetOutputPort())

#        randDisplacement=vtk.vtkBrownianPoints()
#        randDisplacement.SetMinimumSpeed(max(spacing)*0.5)
#        randDisplacement.SetMaximumSpeed(max(spacing)*1.5)        
#        randDisplacement.SetInputConnection(BranchTubeFilter.GetOutputPort())

        stripper = vtk.vtkStripper();
        stripper.SetInputConnection(BranchTubeFilter.GetOutputPort());
        stripper.Update();
        data=stripper.GetOutput()     
        
#        geofilter=vtk.vtkGeometryFilter()
#        geofilter.SetInputConnection(BranchTubeFilter.GetOutputPort())
#        geofilter.Update()
#        data=geofilter.GetOutput()
    
        if CubeLoader._Origin.__len__()==0:
            dataorigin=[0,0,0]
        else:
            dataorigin=CubeLoader._Origin
        if CubeLoader._Extent.__len__()==0:
            print "Dataset extent not defined."
            return
        else:
            datadim=CubeLoader._Extent
        spacing=CubeLoader._DataScale[0:3];
        
        whiteImage = vtk.vtkImageData()
        whiteImage.SetExtent(0,datadim[0]-1,0,datadim[1]-1,0,datadim[2]-1)
        whiteImage.SetSpacing(spacing);
        whiteImage.SetOrigin(dataorigin);  
        whiteImage.SetScalarTypeToUnsignedChar();
        whiteImage.AllocateScalars();
    
        count = whiteImage.GetNumberOfPoints();
        fillarray=numpy_to_vtk(255*np.ones([count,1],dtype=np.uint8), deep=1, array_type=vtk.VTK_UNSIGNED_CHAR);
        whiteImage.GetPointData().SetScalars(fillarray)
        
#        stripper = vtk.vtkStripper();
#        stripper.SetInput(data);
#        stripper.Update();
     
        pol2stenc = vtk.vtkPolyDataToImageStencil();
        pol2stenc.SetTolerance(max(spacing)*0.5);
#        pol2stenc.SetTolerance(1.0e-9 );
#        pol2stenc.SetInput(stripper.GetOutput());
        pol2stenc.SetInput(data);
        pol2stenc.SetOutputOrigin(dataorigin);
        pol2stenc.SetOutputSpacing(spacing);
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent());
        pol2stenc.Update();
        
        imgstenc =  vtk.vtkImageStencil()
        imgstenc.SetInput(whiteImage);
        imgstenc.SetStencil(pol2stenc.GetOutput());
        imgstenc.ReverseStencilOff();
        imgstenc.SetBackgroundValue(0);
        imgstenc.Update();
        
        vtkImageFlip=vtk.vtkImageFlip()
        vtkImageFlip.SetFilteredAxis(1)
        vtkImageFlip.SetInput(imgstenc.GetOutput())
        vtkImageFlip.Update()
        
        writer=vtk.vtkTIFFWriter()    
        writer.SetInput(vtkImageFlip.GetOutput())
    #    writer.SetInput(imp.GetOutput())
    
        filename=unicode(self.text_Skel2TIFFPath.text())
        filename, fileext = os.path.splitext(filename)
        writer.SetFilePrefix(filename);
        writer.SetFilePattern("%s%d.tif");
        writer.SetFileDimensionality(2);
        writer.Update()
        writer.Write()
        return;

    def LoadPlugin(self,uri=None,absl=True): 
        if not uri:
            if os.path.isdir(application_path):
                filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load plugin...",application_path,"*.py;;*.pyc");
            else:
                filelist = QtGui.QFileDialog.getOpenFileNames(self,"Load plugin...","","*.py;;*.pyc");
            if filelist.__len__()==0:
                return None, None, None
            uri=unicode(filelist[filelist.__len__()-1])
        
        if not absl:
            uri = os.path.normpath(os.path.join(os.path.dirname(__file__), uri))

        path, fname = os.path.split(uri)
        mname, ext = os.path.splitext(fname)
        no_ext = os.path.join(path, mname)
        if ext=='.pyc' and os.path.exists(no_ext + '.pyc'):
            try:
                if mname in sys.modules:
                    del sys.modules[mname]
                filename= no_ext + '.pyc'
                module=imp.load_compiled(mname,filename)
                plugin=module.init(window1,CubeLoader)
                return mname,plugin,filename
            except OSError as err:
                print("OS error: {0}".format(err))
                return
        if ext=='.py' and os.path.exists(no_ext + '.py'):
            try:
                if mname in sys.modules:
                    del sys.modules[mname]
                filename = no_ext + '.py'
                module=imp.load_source(mname,filename)
                plugin=module.init(window1,CubeLoader)
                return mname,plugin,filename
            except OSError as err:
                print("OS error: {0}".format(err))
                return
#        if os.path.exists(no_ext + '.pyc'):
#            try:
#                if mname in sys.modules:
#                    del sys.modules[mname]
#                filename= no_ext + '.pyc'
#                module=imp.load_compiled(mname,filename)
#                return mname,module,filename
#            except:
#                pass
#        if os.path.exists(no_ext + '.py'):
#            try:
#                if mname in sys.modules:
#                    del sys.modules[mname]
#                filename = no_ext + '.py'
#                module=imp.load_source(mname,filename)
#                return mname,module,filename
#            except:
#                pass
        return None,None,None
        
    def RunPlugin(self,filename=None):
        if not filename:
            action = self.sender()
            if hasattr(action,'_File'):
                filename=action._File
        
        pluginName,plugin,filename=self.LoadPlugin(filename)
        
        if not plugin:
            return -1

        self.Plugins[pluginName]=plugin

        if any([mplugin._File==filename for mplugin in self.menuPluginList]):
            found=0
            for iplugin in range(self.menuPluginList.__len__()-1):
                mplugin=self.menuPluginList[iplugin]
                if mplugin._File==filename:
                    found=1
                if not found:
                    continue
                prev_plugin=self.menuPluginList[iplugin+1]            
                mplugin._Name=prev_plugin._Name
                mplugin._File=prev_plugin._File
                text = "&%d %s" % (self.menuPluginList.__len__() - iplugin, mplugin._Name)
                mplugin.setText(text)
                mplugin.setVisible(not mplugin._Name=='')
            mplugin=self.menuPluginList[self.menuPluginList.__len__()-1]
            mplugin._Name=pluginName
            mplugin._File=filename
            text = "&%d %s" % (1, pluginName)
            mplugin.setText(text)
            mplugin.setVisible(not pluginName=='')
        else:
            for iplugin in range(self.menuPluginList.__len__()-1):
                mplugin=self.menuPluginList[iplugin]
                prev_plugin=self.menuPluginList[iplugin+1]
                mplugin._Name=prev_plugin._Name
                mplugin._File=prev_plugin._File
                text = "&%d %s" % (self.menuPluginList.__len__() - iplugin, mplugin._Name)
                mplugin.setText(text)
                mplugin.setVisible(not mplugin._Name=='')
            mplugin=self.menuPluginList[self.menuPluginList.__len__()-1]
            mplugin._Name=pluginName
            mplugin._File=filename
            text = "&%d %s" % (1, pluginName)
            mplugin.setText(text)
            mplugin.setVisible(not pluginName=='')
        
        window1.SetupGUIState(self.Plugins[pluginName])
        myconfig.LoadConfig(self.Plugins[pluginName],pluginName)
        print "Loaded plugin: ", pluginName
        plugin.runPlugin()
        return 1

class ObjectBrowser(QtGui.QTreeView):
    def __init__(self, parent=None, *args):
#        super(ObjectBrowser, self).__init__(parent)
        QtGui.QTreeView.__init__(parent,*args)

        TreeModel=QtGui.QStandardItemModel(self)
        self.setModel(TreeModel)        
        self.setSelectionModel(QtGui.QItemSelectionModel(TreeModel))

        self.model().setHorizontalHeaderLabels(["Objects"])
        self.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(False)

        self.show()
        self.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
    
    def contextMenuEvent(self, event):
        self.menu = QtGui.QMenu(self)
        chcommentAction = QtGui.QAction('Change comment', self)
        chcommentAction.triggered.connect(self.changeComment)
        self.menu.addAction(chcommentAction)

        chcolorAction = QtGui.QAction('Change color', self)
        chcolorAction.triggered.connect(self.changeColor)
        self.menu.addAction(chcolorAction)
        # add other required actions
        self.menu.popup(QtGui.QCursor.pos())
        
    def changeComment(self):
        self.currentIndex()
        item=self.model().itemFromIndex(self.currentIndex())
        if not item:
            return
        if not hasattr(item,'obj'):
            return
        obj=item.obj
        comment=obj.comments.get(obj,"comment")
        if not comment:
            comment=""
        text, ok = QtGui.QInputDialog.getText(obj.ariadne, 'Change comment', 
            'Comment:',QtGui.QLineEdit.Normal,comment)
        
        if not ok:
            return
        obj.comments.set(obj,"comment",text)
        obj.updateItem()
    
    def changeColor(self):
        self.currentIndex()
        item=self.model().itemFromIndex(self.currentIndex())
        if not item:
            return
        if not hasattr(item,'obj'):
            return
        obj=item.obj

        color=obj.LUT.GetTableValue(obj.colorIdx)
        color=QtGui.QColor().fromRgb(*color)
        color=QtGui.QColorDialog.getColor(color,obj.ariadne, "Color")
        if not color.isValid():
            return
        color=np.array(color.getRgb())/255.00
        color[3]=1.0
        obj.change_color([color[0],color[1],color[2],color[3]])
        return color

class label_class:
    _Name=""
    _Description=""
    _color=(1.0,0.5,0.1,1.0)    
    _examples=list()
    label=None
    movie=None
    item=None

    def __init__(self,ariadne):
        self.ariadne=ariadne        
        self.item=QtGui.QTableWidgetItem()
        self.label=QtGui.QLabel()
        label=self.label
        movie=QtGui.QMovie(QtCore.QString(''),QtCore.QByteArray(), label)
        self.movie=movie
        movie.setCacheMode(QtGui.QMovie.CacheAll) 
        label.setMovie(movie)
        label.mouseDoubleClickEvent=self.animate_example
        label.wheelEvent=self.change_example

    def change_color(self,color=None):
        1


    def change_example(self,event=None):
        Nexamples=self._examples.__len__()
        if Nexamples<2:
            return
        if event==None:
            step=0
        else:
            step=event.delta()/120
        movie=self.movie
        if not movie.fileName():
            newExIdx=0
        else:
            newExIdx=self._examples.index(unicode(movie.fileName()))
        moviestate=movie.state()
        if not (not movie.fileName()):
            movie.stop()
        newExIdx+=step
        if newExIdx>(Nexamples-1):
            newExIdx=0
        elif newExIdx<0:
            newExIdx=(Nexamples-1)
        newFileName=unicode(self._examples[newExIdx])
        if not (not newFileName):
            movie.setFileName(QtCore.QString(newFileName))               
        if not (not movie.fileName()):
            if moviestate==2:
                movie.start()
            else:
                movie.jumpToNextFrame()
        
    def animate_example(self,event=None):
        movie=self.movie
        if not (not movie.fileName()):
            if movie.state()==movie.Running:
                movie.setPaused(True)
            elif movie.state()==movie.Paused:
                movie.setPaused(False)
            else:
                movie.start()

BorderSize=4;
                    
class synapse_browser:
    _IconSize=(180,180)
    currentClass=None
    browseList=list()
    
    def __init__(self,ariadne):
        self.classes=OrderedDict()

        self.ariadne=ariadne
        self.imagelist=ariadne._SynClassTable
        self.imagelist.horizontalHeader().setVisible(False);
        self.imagelist.verticalHeader().setVisible(False);
        self.imagelist.setSelectionBehavior(QtGui.QAbstractItemView.SelectColumns)

        self.imagelist.setStyleSheet("QTableWidget::item {padding: "+ "{0}".format(BorderSize)+ "px;}" );
        QtCore.QObject.connect(self.imagelist,QtCore.SIGNAL("currentItemChanged(QTableWidgetItem*, QTableWidgetItem*)"),self.set_current_class)        
        QtCore.QObject.connect(self.ariadne.btn_save_classes,QtCore.SIGNAL("clicked()"),self.SaveSynClasses)

        
    def delete_synapse(self):
        if self.ariadne.QRWin.SelObj[0]=="synapse":
            self.ariadne.QRWin.DeleteActiveObj()
            
    def search_synapse(self,direction="forward"):
        objtype,neuronID,nodeId=self.ariadne.QRWin.SelObj
        Neurons=self.ariadne.Neurons
        if self.ariadne.ckbx_randomizeSynBrowsing.isChecked():
            if self.browseList.__len__()==0:
                #populate list
                allNeuronIDs=Neurons.keys()
                objtype="synapse"
                for ineuronID in allNeuronIDs:
                    inodeId=None
                    while 1:
                        tagIdx=Neurons[ineuronID].search_child(objtype,inodeId,direction,False)
                        if tagIdx==None:
                            break
                        nodeIds=Neurons[ineuronID].children[objtype].tagIdx2nodeId(tagIdx)
                        inodeId=nodeIds[0]
                        self.browseList.append((ineuronID,inodeId))
#                        print (ineuronID,inodeId)
                random.shuffle(self.browseList)
#                print "Populated list: \n" + "{0}\n".format(self.browseList) + "-----"
            if (neuronID,nodeId) in self.browseList:
                oldidx=self.browseList.index((neuronID,nodeId))
            else:
                oldidx=-1;
            if direction=="backward":
                newidx=oldidx-1
            elif direction=="forward":
                newidx=oldidx+1
            else:
                return "synapse",None,-1

            if (newidx<0) or (newidx>(self.browseList.__len__()-1)):
                #populate list
                newlist=[]
                allNeuronIDs=Neurons.keys()
                objtype="synapse"
                for ineuronID in allNeuronIDs:
                    inodeId=None
                    while 1:
                        tagIdx=Neurons[ineuronID].search_child(objtype,inodeId,direction,False)
                        if tagIdx==None:
                            break
                        nodeIds=Neurons[ineuronID].children[objtype].tagIdx2nodeId(tagIdx)
                        inodeId=nodeIds[0]
                        newlist.append((ineuronID,inodeId))
                    
                self.browseList=[key for key in self.browseList if key in newlist]
                newlist=[key for key in newlist if key not in self.browseList]
                
                if newlist.__len__()>0:
#                    print "New list: " + "{0}\n".format(newlist) + "------"
                    random.shuffle(newlist)
                    self.browseList.extend(newlist)
#                    print "Re-populated list: \n" + "{0}\n".format(self.browseList) + "-----"
            objtype="synapse"
            while self.browseList.__len__()>0:
                if newidx<0:
                    newidx=self.browseList.__len__()-1
                if newidx>self.browseList.__len__()-1:
                    newidx=0
                if newidx>self.browseList.__len__()-1:
                    return "synapse",None,-1
                neuronID=self.browseList[newidx][0]
                nodeId=self.browseList[newidx][1]
                if not neuronID in Neurons:
#                    print "remove: " + "{0}".format(self.browseList[newidx])
                    self.browseList.remove(self.browseList[newidx])
                    continue
                tagIdx=Neurons[neuronID].search_child(objtype,nodeId,None)
                if tagIdx==None:
#                    print "remove: " + "{0}".format(self.browseList[newidx])
                    self.browseList.remove(self.browseList[newidx])
                    continue
#                print self.browseList[newidx]
                self.ariadne.QRWin.SetActiveObj(objtype,neuronID,nodeId)
                self.ariadne.QRWin.GotoActiveObj()
                return objtype,neuronID,nodeId
            return "synapse",None,-1


        if not (neuronID in Neurons):            
            allNeuronIDs=Neurons.keys()
            if direction=="backward":
                allNeuronIDs.reverse()

            objtype="synapse"
            for neuronID in allNeuronIDs:
                tagIdx=Neurons[neuronID].search_child(objtype,None,direction)
                if tagIdx==None:
                    continue
                nodeIds=Neurons[neuronID].children[objtype].tagIdx2nodeId(tagIdx)
                nodeId=nodeIds[0]
                self.ariadne.QRWin.SetActiveObj(objtype,neuronID,nodeId)
                self.ariadne.QRWin.GotoActiveObj()
                return objtype,neuronID,nodeId
            return "synapse",None,-1
        
        if objtype=="synapse":
            start_Id=nodeId
        else:
            objtype="synapse"
            start_Id=None
        tagIdx=Neurons[neuronID].search_child(objtype,start_Id,direction)
        if not (tagIdx==None):
            nodeIds=Neurons[neuronID].children[objtype].tagIdx2nodeId(tagIdx)
            nodeId=nodeIds[0]
            self.ariadne.QRWin.SetActiveObj(objtype,neuronID,nodeId)
            self.ariadne.QRWin.GotoActiveObj()
            return objtype,neuronID,nodeId

        allNeuronIDs=Neurons.keys()
        if direction=="backward":
            allNeuronIDs.reverse()

        ineuron=allNeuronIDs.index(neuronID)
        for jneuron in range(ineuron+1,allNeuronIDs.__len__())+range(0,ineuron+1):
            tagIdx=Neurons[allNeuronIDs[jneuron]].search_child(objtype,None,direction)
            if (not (tagIdx==None)) and (objtype in Neurons[neuronID].children):
                nodeIds=Neurons[neuronID].children[objtype].tagIdx2nodeId(tagIdx)
                nodeId=nodeIds[0]
                self.ariadne.QRWin.SetActiveObj(objtype,neuronID,nodeId)
                self.ariadne.QRWin.GotoActiveObj()
                return objtype,neuronID,nodeId
        return None
        
    def attributes_changed(self):
        self.ariadne.btn_Syn_assign.setStyleSheet("background-color: rgb(255, 0, 0)")
        self.ariadne.btn_Syn_assign.setText("Assign")

    def set_attributes(self):
        ObjType=self.ariadne.QRWin.SelObj[0]
        if not ObjType=="synapse":
            return
        NeuronID=self.ariadne.QRWin.SelObj[1]
        nodeId=self.ariadne.QRWin.SelObj[2]
        
        if ObjType in self.ariadne.Neurons[NeuronID].children:
            obj=self.ariadne.Neurons[NeuronID].children[ObjType]
        else:
            return
        tagIdx=obj.nodeId2tagIdx(nodeId)
        nodeIds=obj.tagIdx2nodeId(tagIdx)
        
        nodeId=nodeIds[0]
        obj.comments.set(nodeId,"certainty",self.ariadne._CertaintyLevel.value())
        if not self.currentClass==None:
            obj.comments.set(nodeId,"class",self.currentClass._Name)
            obj.assign_classcolor(tagIdx)
            self.ariadne.QRWin.Render()
        self.ariadne.btn_Syn_assign.setStyleSheet("background-color: rgb(0, 255, 0)")
        self.ariadne.btn_Syn_assign.setText("Assigned")

    def set_current_class(self,new_class=None,prev_class=None):
        key="None"
        if not prev_class==None:
            key=prev_class.synclassname
            if key in self.classes:
                movie=self.classes[key].movie
                if not (not movie.fileName()):
                    movie.stop()
                    movie.jumpToFrame(0)
        if not new_class==None:
            key=new_class.synclassname
            if key in self.classes:
                self.currentClass=self.classes[key]

#        if key=="None":
#            self.currentClass=None
#            self.namelabel.setText("Class: <b><font color='red'>None</font></b>")
#            self.descriptionlabel.setText("No synapse class assigned yet.")
#            color=(0.5,0.5,0.5,0.0)
#            self.ariadne.btn_SynClassColor.setStyleSheet("background-color: rgb({0}, {1}, {2}, {3})".format(color[0],color[1],color[2],color[3]))
#            return
#            
#        self.namelabel.setText("Class: {0}".format(self.currentClass._Name))
#        if not self.currentClass._Description:
#            self.descriptionlabel.setText("No description available.")            
#        else:
#            self.descriptionlabel.setText(self.currentClass._Description)
        self.attributes_changed()
        
    def get_classfile(self,DatasetName=None):
        if not DatasetName:
            return None, None
        class_folder = os.path.join(application_path, "SynapseClasses")
        class_folder = os.path.join(class_folder,DatasetName)
#        class_folder = os.path.join(application_path, "imageformats")
        if not os.path.isdir(class_folder):
            print "Synapse class directory {0} was not found. No classes loaded.".format(class_folder)
            return None, None
        filename=os.path.join(class_folder,'{0}_synapseclasses.conf'.format(DatasetName))
        return filename, class_folder

    def readin_classes(self,DatasetName=None):
        self.classes=OrderedDict()
        self.imagelist.clearContents() 
        while synapse.allclassnames.__len__()>0:
            synapse.allclassnames.remove(synapse.allclassnames[0])
        
        filename, class_folder =self.get_classfile(DatasetName)
        if filename==None:
            return
        ClassConfig=config(filename)

        self.imagelist.setRowCount(3)
        self.imagelist.setColumnCount(13)
        
        if not synapse.LUTinitialized[0]:
            synapse.LUT.Allocate(synapse.maxNofColors[0])
            #we allow maximally 12 classes, might want to change this later
            #the default color will be idx=14 (starting from 0) after calling setup_color
            synapse.LUT.SetNumberOfTableValues(2+12)
            synapse.LUT.SetTableRange(0,1)
            synapse.LUT.Build()
            synapse.LUTinitialized[0]=True
        
        
        NClasses=0;
        for iclass in range(1,13):
            self.imagelist.setColumnWidth(iclass-1,self._IconSize[1]+2*BorderSize)
            key="class{0}".format(iclass)
            if not ClassConfig.has_key(key):
                continue
            NClasses+=1
            if key in self.classes:
                synclass=self.classes[key]
            else:
                synclass=label_class(self.ariadne)
                self.classes[key]=synclass
            ClassConfig.LoadConfig(synclass,key)
            
            synclass.item=[None,None,None]            
            
            filename=''            
            synclass._examples=[os.path.join(class_folder,example) for example in synclass._examples]
            synapse.allclassnames.append(synclass._Name)
            
            movie =synclass.movie

            if synclass._examples.__len__()>0:
                filename=synclass._examples[0]
                movie.setFileName(QtCore.QString(filename))  
                
            movie.setScaledSize(QtCore.QSize(self._IconSize[0],self._IconSize[1]))
            if not (not movie.fileName()):
                movie.jumpToNextFrame()

            item=self.imagelist.item(1,iclass-1)
            if item==None:
                item=QtGui.QTableWidgetItem()
                self.imagelist.setItem(1,iclass-1,item)
            item.setSizeHint(QtCore.QSize(self._IconSize[0],self._IconSize[1]));
            item.synclassname=key
            item.setFlags( QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsEnabled )
            synclass.item[1]=item

            label=synclass.label

            self.imagelist.setCellWidget(1,iclass-1,label)

            descriptionitem=self.imagelist.item(2,iclass-1)
            if descriptionitem==None:
                descriptionitem=QtGui.QTableWidgetItem()
                self.imagelist.setItem(2,iclass-1,descriptionitem)
            descriptionitem.setText(synclass._Description);
            descriptionitem.synclassname=key
            descriptionitem.setFlags( QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsEnabled )
            synclass.item[2]=descriptionitem

            headeritem=self.imagelist.item(1,iclass-1)
            if headeritem==None:
                headeritem=QtGui.QTableWidgetItem();
                self.imagelist.setItem(1,iclass-1,headeritem)     
                
            headeritem.synclassname=key
            headeritem.setFlags( QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsEnabled )
             
            synclass.item[0]=headeritem
            headerbutton=QtGui.QPushButton(key)
            color=synclass._color
            headerbutton.setStyleSheet("background-color: rgb({0}, {1}, {2})".format(color[0],color[1],color[2]))
            if synclass._Name in synapse.allclassnames:
                classNo=synapse.allclassnames.index(synclass._Name)+1  
            else:
                classNo=14
            synapse.LUT.SetTableValue(classNo-1+2,[color[0]/255.0,color[1]/255.0,color[2]/255.0,color[3]/255.0])

            QtCore.QObject.connect(headerbutton,QtCore.SIGNAL("clicked()"),lambda obj=synclass: self.change_class_color(obj))
            synclass.header=headerbutton
            self.imagelist.setCellWidget(0,iclass-1,headerbutton)

        self.imagelist.setColumnCount(NClasses)
        self.imagelist.setWordWrap(1)
        self.imagelist.resizeRowsToContents()        
        self.imagelist.setRowHeight(0,22+2*BorderSize)
        self.imagelist.setRowHeight(1,self._IconSize[0]+2*BorderSize)
        synapse.LUT.Modified()

    def change_class_color(self,synclass=None,color=None):
        if not synclass:
            synclass=self.currentClass
        if not synclass:
            return
        if color==None:  
            color=synclass._color
            color=QtGui.QColor().fromRgb(*color)
            color=QtGui.QColorDialog.getColor(color,self.ariadne, "Color", QtGui.QColorDialog.ShowAlphaChannel)

            if not color.isValid():
                return
            color=color.getRgb() 
        synclass._color=color
        if synclass._Name in synapse.allclassnames:
            classNo=synapse.allclassnames.index(synclass._Name)+1 
        else:
            classNo=14
        synapse.LUT.SetTableValue(classNo-1+2,[color[0]/255.0,color[1]/255.0,color[2]/255.0,color[3]/255.0])
        synapse.LUT.Modified()
        synclass.header.setStyleSheet("background-color: rgb({0}, {1}, {2})".format(color[0],color[1],color[2]))
        self.ariadne.QRWin.Render()

    def SaveSynClasses(self):
        DatasetName=CubeLoader._BaseName
        
        if not DatasetName:
            return
            
        filename, class_folder =self.get_classfile(DatasetName)
        if filename==None:
            return

        ClassConfig=config(filename)
        for key, synclass in self.classes.iteritems():
            ClassConfig.SaveConfig(synclass,key)
        ClassConfig.write()

      
if __name__ == "__main__":
    # On Windows calling this function is necessary.
    if win:
        multiprocessing.freeze_support()
        print "activated freeze_support"
 #   resultQueue = multiprocessing.Queue()
  #  SendeventProcess(resultQueue)

    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(os.path.join('imageformats','PyKNOSSOS.ico')))
   
    #Loading default configuration
    if win:
        config_name = 'DefaultConfig_win.cfg'    
    else:
        config_name = 'DefaultConfig.cfg'    
    configfile = os.path.join(application_path, config_name)
    if not os.path.isfile(configfile):
        QtGui.QMessageBox.warning(None,"Error",
            "Configuration file missing:\n {0}".format(configfile),
            QtGui.QMessageBox.Cancel, QtGui.QMessageBox.NoButton,
            QtGui.QMessageBox.NoButton)        
        sys.exit()            
    
    try:
        myconfig = config(configfile)
    except:
        QtGui.QMessageBox.warning(None, "Error",
            "Configuration file corrupted:\n {0}".format(configfile),
            QtGui.QMessageBox.Cancel, QtGui.QMessageBox.NoButton,
            QtGui.QMessageBox.NoButton)
        sys.exit()
    
    
    CubeLoader=Loader(doload,mprocess)

    window1 = ARIADNE(os.path.join(application_path,"gui4.ui"))
        
    window1.Neurons=OrderedDict()
    window1.setup_visualization()
    window1.SetupGUIState(window1)

    window1.ToggleClipHulls('skeleton_viewport')    
    
    if "Orth_viewport" in window1.QRWin.viewports:
        cDir=np.array([0.0,0.0,1.0],dtype=np.float)
        vDir=np.array([0.0,-1.0,0.0],dtype=np.float)
        FPoint=np.array(window1.QRWin.viewports["Orth_viewport"].Camera.GetFocalPoint(),dtype=np.float)
        window1.QRWin.viewports["Orth_viewport"].JumpToPoint(FPoint,cDir,vDir)
    
    window1.Timer.reset()
#    try:
#        window1.Open(window1.Filelist[0]._File)
#    finally:
#        1

    #Reserve memory
    NCubesPerEdge=int(window1.SpinBox_CubesPerEdge.value())
    NPixels=CubeLoader._CubeSize[0]*CubeLoader._CubeSize[1]*CubeLoader._CubeSize[2];
    maxNCubesPerEdge=np.floor((float(maxContRAMBlock)/float(NPixels))**(1.0/3.0))
    window1.SpinBox_CubesPerEdge.setMaximum(maxNCubesPerEdge)
    window1.SpinBox_CubesPerEdge.setValue(maxNCubesPerEdge)
    CubeLoader.LoadDatasetInformation(None)
    CubeLoader.LoadDataset()
    window1.SpinBox_CubesPerEdge.setValue(NCubesPerEdge)
    
    window1.SetSkeletonRadius()
    window1.SetSynapseRadius()
    window1.SetSkelLineWidth()
    window1.SetSynLineWidth()

    window1.ChangeSkelVisMode()
#    if not (not window1.job):
#        window1.ChangeColorScheme(0)
#    else:                    
#        window1.ChangeColorScheme()
    window1.ChangeNeuronVisMode()
    window1.ChangePlaneVisMode()
    window1.ChangeBorderVisMode()
    window1.ShowBoundingBox()
    window1.ChangeScaleBar();
    window1.SetSomaVisibility()
    window1.SetRegionVisibility()
    window1.RestrictVOI()

    window1.btn_loadReferenceFile.setEnabled(False)

    if (usermode==1):
        window1.ckbx_encryptFile.setChecked(True)
        window1.ckbx_encryptFile.setEnabled(False)
        window1.ckbx_ClipHulls.setVisible(False)
        
        window1.btn_save_classes.setEnabled(False)
        window1.btn_save_classes.setVisible(False)

        window1.ckbx_randomizeSynBrowsing.setChecked(False)
        window1.ckbx_randomizeSynBrowsing.setEnabled(False)
        window1.ckbx_randomizeSynBrowsing.setVisible(False)

    if not experimental:
        window1.ckbx_SynZoom.setEnabled(False)
        window1.ckbx_SynZoom.setChecked(True)

        window1.ckbx_ClipHulls.setEnabled(False)
        window1.ckbx_ClipHulls.setChecked(False)
        window1.ckbx_ClipHulls.setVisible(False)
        
        window1.ckbx_HideLabelsSkelVP.setEnabled(False)
        window1.ckbx_HideLabelsSkelVP.setChecked(True)
        window1.ckbx_HideLabelsSkelVP.setVisible(False)

        window1.ckbx_HideLabelsDataVP.setEnabled(False)
        window1.ckbx_HideLabelsDataVP.setChecked(True)
        window1.ckbx_HideLabelsDataVP.setVisible(False)

        window1.ckbx_HideRegionLabels.setEnabled(False)
        window1.ckbx_HideRegionLabels.setChecked(True)
        window1.ckbx_HideRegionLabels.setVisible(False)
        window1.ckbx_HideSomaLabels.setEnabled(False)
        window1.ckbx_HideSomaLabels.setChecked(True)
        window1.ckbx_HideSomaLabels.setVisible(False)
        
        window1.ckbx_restrictVOI.setEnabled(False)
        window1.ckbx_restrictVOI.setChecked(False)
        window1.ckbx_restrictVOI.setVisible(False)
        window1.SpinBoxVOISize.setEnabled(False)
        window1.SpinBoxVOISize.setVisible(False)
        window1.VOIlabel1.setVisible(False)
        window1.VOIlabel2.setVisible(False)
        
        for checkbox in [window1.ckbx_ShowArbitScaleBar,window1.ckbx_ShowYXScaleBar,window1.ckbx_ShowYZScaleBar,window1.ckbx_ShowZXScaleBar]:
            checkbox.setChecked(False)                
        window1.group_ScaleBar.setVisible(False)
        
    window1.ChangeCubeDataset(window1.menuDatasets[4]._File,0)

    #for some reason the object browser would otherwise not extend
    #to fill the settings tab widget.
    currWidget=window1.SettingsTab.currentWidget()
    window1.SettingsTab.setCurrentWidget(window1.ObjBrowserTab)
    window1.SettingsTab.setCurrentWidget(currWidget)
    
    #jump to previous location as saved in config file
    window1.JumpToPoint()

    #if doload and win:
    #    CubeLoader.LoadDataset()    
    sys.exit(app.exec_())
