import vtk
import numpy as np


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

	skelvp=ariadne.QRWin.viewports["skeleton_viewport"]
	arbitvp=ariadne.QRWin.viewports["Orth_viewport"];

	arbitpl=arbitvp.ViewportPlane;
	tempLines = vtk.vtkPolyLine()
	tempLines.GetPointIds().SetNumberOfIds(2)
	tempLines.GetPointIds().SetId(0,0*3+0)
	tempLines.GetPointIds().SetId(1,0*3+1)

	for icell in range(arbitpl.ScaleBar.GetNumberOfCells()):
		arbitpl.ScaleBar.DeleteCell(icell);

	arbitpl.ScaleBar.RemoveDeletedCells();
	arbitpl.ScaleBar.InsertNextCell(tempLines.GetCellType(),tempLines.GetPointIds());

	ariadne.ChangeSynZoom(1);
	ariadne.SynchronizedZoom(0.034);
	ariadne.QRWin.Render();

	ariadne.RegionAlpha.setValue(15);
	ariadne.SomaAlpha.setValue(40);

	ariadne.ckbx_HideBorder.setChecked(0)
	ariadne.ckbx_ShowYXScaleBar.setChecked(0)
	ariadne.ckbx_ShowYZScaleBar.setChecked(0)
	ariadne.ckbx_ShowZXScaleBar.setChecked(0)
	ariadne.ckbx_ShowArbitScaleBar.setChecked(1)
	ariadne.ckbx_HideBorder.setChecked(1)

	FPoint=  (78233.9332778213, 61601.67070863842, 64849.07703562547) ;
	CamPos =  (237350.73945325083, 80656.08301123015, -39741.523057454404) ;
	ViewUp=  (0.12355078296512695, 0.9260262916142512, 0.3566658257639244) ;
	
	ariadne.SpinBox_ScaleBarWidth.setValue(2000)
	ariadne.radioBtn_tubesflat.setChecked(1)
	ariadne.SpinBox_Radius.setValue(150.0)
	ariadne.HideSkelNodes.setChecked(1)
	ariadne.ckbx_HideSomaLabels.setChecked(1)

	skelvp.Camera.SetViewUp(ViewUp)
	skelvp.Camera.SetFocalPoint(FPoint)
	skelvp.Camera.SetPosition(CamPos)
	skelvp.ResetCameraClippingRange();

	vDir=np.array(ViewUp)*1.0;
	vtk.vtkMath.Normalize(vDir)
	cDir=np.array(FPoint)-np.array(CamPos);
	vtk.vtkMath.Normalize(cDir)
	arbitvp.ViewportPlane.JumpToPoint(np.array(FPoint),np.array(cDir),np.array(vDir));

	skelvp.ResetCameraClippingRange();
	ariadne.QRWin.Render();

