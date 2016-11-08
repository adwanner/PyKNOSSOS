/*gcc extractROIlib.c -O3 -Wall -fPIC -shared -o clibraries/extractROIlib.so*/
/*i586-mingw32msvc-gcc extractROIlib.c -O3 -Wall -shared -o clibraries/extractROIlib.dll*/
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int initialized=0;
int *NMag;
int *NCubesPerEdge, NCubes;
int *NumberofCubes;
unsigned char *HyperCube[3];
unsigned char *invalidCube;

unsigned char completeLoading;
short int *AllCubes[1], **MagOffset;

float *DataScale;
int Mag=1;
int NCubesX, NCubesY, NCubesZ;
int ForceLoaderFlag=0;
float *LoaderCurrCoord;
int *LoaderMag;
int *LoaderState;
int *CubeSize, NPixelsPerCube, NPixelsInYX;


#define LERP(a,l,h)	(l+(h-l)*a)

unsigned char* find_cubeinplane(unsigned int x,unsigned int y,unsigned int z){
	unsigned int imag,ihypercube,prev_ihypercube;
	int cubepos=z;
	cubepos*=NCubesY;
	cubepos+=y;
	cubepos*=NCubesX;
	cubepos+=x;
	int pos=(int)(*(MagOffset[Mag]+cubepos));
	if (pos==-1){
		if (ForceLoaderFlag==0){
			completeLoading=0;
			return invalidCube;
		}		
		LoaderMag[0]=Mag+1;
		LoaderCurrCoord[0]=(((float) (x))*((float)CubeSize[0])+1.0)*DataScale[Mag*3];
		LoaderCurrCoord[1]=(((float) (y))*((float)CubeSize[1])+1.0)*DataScale[Mag*3+1];
		LoaderCurrCoord[2]=(((float) (z))*((float)CubeSize[2])+1.0)*DataScale[Mag*3+2];
		while ((pos==-1) && (LoaderState[0]>0)){
			if (LoaderState[0]!=2){
				LoaderState[0]=2;}
			pos=(int)(*(MagOffset[Mag]+cubepos));
		}
	}
	if (pos<=-2){
		//printf( "************Cube not found*************\n");
		return invalidCube;
	}
	cubepos=(unsigned int)pos;

	ihypercube=0;
	for (imag=0;imag<3;imag++){
		prev_ihypercube=ihypercube;
		ihypercube+=NCubesPerEdge[imag]*NCubesPerEdge[imag]*NCubesPerEdge[imag];
		if (cubepos<ihypercube){
			break;
		}
	}
	return (HyperCube[imag]+(cubepos-prev_ihypercube)*NPixelsPerCube);
}

unsigned char* find_bordercubeinplane(int x, int y, int z){
	//return invalidCube;
	if ((x>=NCubesX) || (y>=NCubesY) || (z>=NCubesZ)){
		return invalidCube;}	
	return find_cubeinplane((unsigned int)x,(unsigned int)y,(unsigned int)z);
}

int getBorderPixel(div_t divx0cube,div_t divy0cube, div_t divz0cube,\
unsigned char *d000,unsigned char *d100,unsigned char *d010,unsigned char *d001,unsigned char *d101,unsigned char *d110,unsigned char *d011,unsigned char *d111){
	/*bordercalls++;*/
	unsigned char *pos;
	int x1, y1, z1, dy,dz, dyz;
	dy=divy0cube.rem*CubeSize[0];
	dz=divz0cube.rem*NPixelsInYX;
	dyz=dy+dz;
	x1=divx0cube.quot+1;
	y1=divy0cube.quot+1;
	z1=divz0cube.quot+1;
	if (divx0cube.rem==CubeSize[0]-1){
		if (divy0cube.rem==CubeSize[1]-1){
			if (divz0cube.rem==CubeSize[2]-1){
				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,divz0cube.quot);
				pos+=divx0cube.rem+dyz;
				*d000=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,divy0cube.quot,divz0cube.quot);
				pos+=dyz;
				*d100=(unsigned char)*pos;

				pos=find_bordercubeinplane(divx0cube.quot,y1,divz0cube.quot);
				pos+=divx0cube.rem+dz;
				*d010=(unsigned char)*pos;

				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,z1);
				pos+=divx0cube.rem+dy;
				*d001=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,y1,divz0cube.quot);
				pos+=dz;
				*d110=(unsigned char)*pos;

				pos=find_bordercubeinplane(divx0cube.quot,y1,z1);
				pos+=divx0cube.rem;
				*d011=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,divy0cube.quot,z1);
				pos+=dy;
				*d101=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,y1,z1);
				*d111=(unsigned char)*pos;
				return 1;
			}
			else{
				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,divz0cube.quot);
				pos+=divx0cube.rem+dyz;
				*d000=(unsigned char)*pos;
				pos+=NPixelsInYX;
				*d001=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,divy0cube.quot,divz0cube.quot);
				pos+=dyz;
				*d100=(unsigned char)*pos;
				pos+=NPixelsInYX;
				*d101=(unsigned char)*pos;

				pos=find_bordercubeinplane(divx0cube.quot,y1,divz0cube.quot);
				pos+=divx0cube.rem+dz;
				*d010=(unsigned char)*pos;
				pos+=NPixelsInYX;
				*d011=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,y1,divz0cube.quot);
				pos+=dz;
				*d110=(unsigned char)*pos;
				pos+=NPixelsInYX;
				*d111=(unsigned char)*pos;
				return 1;
			}
		}
		else{
			if (divz0cube.rem==CubeSize[2]-1){
				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,divz0cube.quot);
				pos+=divx0cube.rem+dyz;
				*d000=(unsigned char)*pos;
				pos+=CubeSize[0];
				*d010=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,divy0cube.quot,divz0cube.quot);
				pos+=dyz;
				*d100=(unsigned char)*pos;
				pos+=CubeSize[0];
				*d110=(unsigned char)*pos;

				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,z1);
				pos+=divx0cube.rem+dy;
				*d001=(unsigned char)*pos;
				pos+=CubeSize[0];
				*d011=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,divy0cube.quot,z1);
				pos+=dy;
				*d101=(unsigned char)*pos;
				pos+=CubeSize[0];
				*d111=(unsigned char)*pos;
				return 1;
			}
			else{
				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,divz0cube.quot);
				pos+=divx0cube.rem+dyz;
				*d000=(unsigned char)*pos;
				pos+=CubeSize[0];
				*d010=(unsigned char)*pos;
				pos+=NPixelsInYX;
				*d011=(unsigned char)*pos;
				pos-=CubeSize[0];
				*d001=(unsigned char)*pos;

				pos=find_bordercubeinplane(x1,divy0cube.quot,divz0cube.quot);
				pos+=dyz;
				*d100=(unsigned char)*pos;						
				pos+=CubeSize[0];
				*d110=(unsigned char)*pos;
				pos+=NPixelsInYX;
				*d111=(unsigned char)*pos;
				pos-=CubeSize[0];
				*d101=(unsigned char)*pos;
				return 1;
			}
		}
	}
	else{
		if (divy0cube.rem==CubeSize[1]-1){
			if (divz0cube.rem==CubeSize[2]-1){
				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,divz0cube.quot);
				pos+=divx0cube.rem+dyz;
				*d000=(unsigned char)*pos;
				*d100=(unsigned char)*(++pos);

				pos=find_bordercubeinplane(divx0cube.quot,y1,divz0cube.quot);
				pos+=divx0cube.rem+dz;
				*d010=(unsigned char)*pos;
				*d110=(unsigned char)*(++pos);

				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,z1);
				pos+=divx0cube.rem+dy;
				*d001=(unsigned char)*pos;
				*d101=(unsigned char)*(++pos);

				pos=find_bordercubeinplane(divx0cube.quot,y1,z1);
				pos+=divx0cube.rem;
				*d011=(unsigned char)*pos;
				*d111=(unsigned char)*(++pos);
				return 1;
			}
			else{
				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,divz0cube.quot);
				pos+=divx0cube.rem+dyz;
				*d000=(unsigned char)*pos;
				*d100=(unsigned char)*(++pos);
				pos+=NPixelsInYX;
				*d101=(unsigned char)*pos;
				*d001=(unsigned char)*(--pos);

				pos=find_bordercubeinplane(divx0cube.quot,y1,divz0cube.quot);
				pos+=divx0cube.rem+dz;
				*d010=(unsigned char)*pos;
				*d110=(unsigned char)*(++pos);
				pos+=NPixelsInYX;
				*d111=(unsigned char)*pos;
				*d011=(unsigned char)*(--pos);
				return 1;
			}
		}
		else{
			if (divz0cube.rem==CubeSize[2]-1){
				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,divz0cube.quot);
				pos+=divx0cube.rem+dyz;
				*d000=(unsigned char)*pos;
				*d100=(unsigned char)*(++pos);
				pos+=CubeSize[0];
				*d110=(unsigned char)*pos;
				*d010=(unsigned char)*(--pos);

				pos=find_bordercubeinplane(divx0cube.quot,divy0cube.quot,z1);
				pos+=divx0cube.rem+dy;
				*d001=(unsigned char)*pos;
				*d101=(unsigned char)*(++pos);
				pos+=CubeSize[0];
				*d111=(unsigned char)*pos;
				*d011=(unsigned char)*(--pos);
				return 1;
			}
			else{
				return 1;
			}
		}
	}
	return -1;
}

int interp_ROI(float* currCoord,float* hDir,float* vDir,int* ROISize,unsigned char* ROI, int mag, int forceLoaderFlag, float* loaderCurrCoord, int* loaderMag,int* loaderState){
	if (initialized==0){
		return 0;
	}
	float \
	vCoord[3], HDir[3], VDir[3],\
	x,y,z,fx,fy,fz, scaling;

	unsigned char d000, d001, d010, d100, d110, d011, d111, d101;

	int iv,ih, i, x0, y0, z0; 

	unsigned char *pos;
	
	ForceLoaderFlag=forceLoaderFlag;
	LoaderMag=loaderMag;
	LoaderCurrCoord=loaderCurrCoord;
	LoaderState=loaderState;

	Mag=mag;	
	if (Mag<1){
		printf("ROIlib error: Magnifaction has to be an integer>0\n");
		return 0;
	}
	Mag-=1;
	NCubesX=NumberofCubes[Mag*3+0];
	NCubesY=NumberofCubes[Mag*3+1];
	NCubesZ=NumberofCubes[Mag*3+2];

	div_t divx0cube,divy0cube,divz0cube;

	completeLoading=1;
	for (i=0;i<3;i++){
		scaling=1.0/DataScale[Mag*3+i];
		VDir[i]=*(vDir+i) *scaling;
		HDir[i]=*(hDir+i) *scaling;
		vCoord[i]=*(currCoord+i) *scaling-1; //Have to subtract here 1, because KNOSSOS starts indexing with 1,1,1 - not 0,0,0.
		vCoord[i]-=VDir[i]*((float)ROISize[0])/2.0;
		vCoord[i]-=HDir[i]*((float)ROISize[1])/2.0;
	}
	//return 1;
	if ((VDir[0]==0.0) && (VDir[2]==0.0) && (HDir[0]==0.0) && (HDir[1]==0.0)){ //YZ plane
		x=vCoord[0];	
		x0=(int)(x);
		divx0cube=div(x0,CubeSize[0]);
		if ((x<0) || (divx0cube.quot>=NCubesX)){
			memset(ROI,(unsigned char)255,ROISize[0]*ROISize[1]);
			return completeLoading;
		}
		fx = x - x0;
		for (iv=0;iv<ROISize[0];iv++){
			vCoord[1]+=VDir[1];
			vCoord[2]+=VDir[2];
			y=vCoord[1];	
			z=vCoord[2];	
			for (ih=0;ih<ROISize[1];ih++){
				y+=HDir[1];	
				z+=HDir[2];	

				y0=(int)(y);
				divy0cube=div(y0,CubeSize[1]);
				if ((y<0) || (divy0cube.quot>=NCubesY)){
					*(ROI++)=(unsigned char)255;
					continue;
				}
				z0=(int)(z);
				divz0cube=div(z0,CubeSize[0]);
				if ((z<0) || (divz0cube.quot>=NCubesZ)){
					*(ROI++)=(unsigned char)255;
					continue;
				}
				
				fy = y - y0;
				fz = z - z0;

				if ((divx0cube.rem<CubeSize[0]-1) && (divy0cube.rem<CubeSize[1]-1) &&  (divz0cube.rem<CubeSize[2]-1)){
					pos=find_cubeinplane((unsigned int)divx0cube.quot,(unsigned int)divy0cube.quot,(unsigned int)divz0cube.quot);
					pos+=divx0cube.rem+divy0cube.rem*CubeSize[0]+divz0cube.rem*NPixelsInYX;
					d000=*pos;
					d100=*(++pos);

					pos+=CubeSize[0];
					d110=*pos;
					d010=*(--pos);

					pos+=NPixelsInYX;
					d011=*pos;
					d111=*(++pos);

					pos-=CubeSize[0];
					d101=*pos;
					d001=*(--pos);}
				else{
					getBorderPixel(divx0cube,divy0cube,divz0cube,&d000,&d100,&d010,&d001,&d101,&d110,&d011,&d111);}				

				*(ROI++)=(unsigned char)LERP(fz,LERP(fy,LERP(fx, d000, d100),LERP(fx, d010, d110)),LERP(fy,LERP(fx, d001, d101),LERP(fx, d011, d111)));
			}
		}
	}
	else if ((VDir[0]==0.0) && (VDir[1]==0.0) && (HDir[1]==0.0) && (HDir[2]==0.0)){ //ZX plane
		y=vCoord[1];	
		y0=(int)(y);
		divy0cube=div(y0,CubeSize[1]);
		if ((y<0) || (divy0cube.quot>=NCubesY)){
			memset(ROI,(unsigned char)255,ROISize[0]*ROISize[1]);
			return completeLoading;
		}

		fy = y - y0;
		for (iv=0;iv<ROISize[0];iv++){
			vCoord[0]+=VDir[0];
			vCoord[2]+=VDir[2];
			x=vCoord[0];	
			z=vCoord[2];	
			for (ih=0;ih<ROISize[1];ih++){
				x+=HDir[0];	
				z+=HDir[2];	

				x0=(int)(x);
				divx0cube=div(x0,CubeSize[0]);
				if ((x<0) || (divx0cube.quot>=NCubesX)){
					*(ROI++)=(unsigned char)255;
					continue;
				}

				z0=(int)(z);
				divz0cube=div(z0,CubeSize[2]);
				if ((z<0) || (divz0cube.quot>=NCubesZ)){
					*(ROI++)=(unsigned char)255;
					continue;
				}

				fx = x - x0;
				fz = z - z0;

				if ((divx0cube.rem<CubeSize[0]-1) && (divy0cube.rem<CubeSize[1]-1) &&  (divz0cube.rem<CubeSize[2]-1)){
					pos=find_cubeinplane((unsigned int)divx0cube.quot,(unsigned int)divy0cube.quot,(unsigned int)divz0cube.quot);
					pos+=divx0cube.rem+divy0cube.rem*CubeSize[0]+divz0cube.rem*NPixelsInYX;
					d000=*pos;
					d100=*(++pos);

					pos+=CubeSize[0];
					d110=*pos;
					d010=*(--pos);

					pos+=NPixelsInYX;
					d011=*pos;
					d111=*(++pos);

					pos-=CubeSize[0];
					d101=*pos;
					d001=*(--pos);}
				else{
					getBorderPixel(divx0cube,divy0cube,divz0cube,&d000,&d100,&d010,&d001,&d101,&d110,&d011,&d111);}				

				*(ROI++)=(unsigned char)LERP(fz,LERP(fy,LERP(fx, d000, d100),LERP(fx, d010, d110)),LERP(fy,LERP(fx, d001, d101),LERP(fx, d011, d111)));
			}
		}
	}
	else if ((VDir[0]==0.0) && (VDir[2]==0.0) && (HDir[1]==0.0) && (HDir[2]==0.0)){ //YX plane
		z=vCoord[2];
		z0=(int)(z);

		divz0cube=div(z0,CubeSize[2]);
		if ((z<0) || (divz0cube.quot>=NCubesZ)){
			memset(ROI,(unsigned char)255,ROISize[0]*ROISize[1]);
			return completeLoading;
		}
		fz = z - z0;

		for (iv=0;iv<ROISize[0];iv++){
			vCoord[0]+=VDir[0];
			vCoord[1]+=VDir[1];
			x=vCoord[0];	
			y=vCoord[1];	
			for (ih=0;ih<ROISize[1];ih++){
				x+=HDir[0];	
				y+=HDir[1];	

				x0=(int)(x);
				divx0cube=div(x0,CubeSize[0]);
				if ((x<0) || (divx0cube.quot>=NCubesX)){
					*(ROI++)=(unsigned char)255;
					continue;
				}

				y0=(int)(y);
				divy0cube=div(y0,CubeSize[1]);
				if ((y<0) || (divy0cube.quot>=NCubesY)){
					*(ROI++)=(unsigned char)255;
					continue;
				}

				fx = x - x0;
				fy = y - y0;

				if ((divx0cube.rem<CubeSize[0]-1) && (divy0cube.rem<CubeSize[1]-1) &&  (divz0cube.rem<CubeSize[2]-1)){
					pos=find_cubeinplane((unsigned int)divx0cube.quot,(unsigned int)divy0cube.quot,(unsigned int)divz0cube.quot);
					pos+=divx0cube.rem+divy0cube.rem*CubeSize[0]+divz0cube.rem*NPixelsInYX;
					d000=*pos;
					d100=*(++pos);

					pos+=CubeSize[0];
					d110=*pos;
					d010=*(--pos);

					pos+=NPixelsInYX;
					d011=*pos;
					d111=*(++pos);

					pos-=CubeSize[0];
					d101=*pos;
					d001=*(--pos);}
				else{
					getBorderPixel(divx0cube,divy0cube,divz0cube,&d000,&d100,&d010,&d001,&d101,&d110,&d011,&d111);}				

				*(ROI++)=(unsigned char)LERP(fz,LERP(fy,LERP(fx, d000, d100),LERP(fx, d010, d110)),LERP(fy,LERP(fx, d001, d101),LERP(fx, d011, d111)));
			}
		}
		}
	else {
		for (iv=0;iv<ROISize[0];iv++){
			vCoord[0]+=VDir[0];
			vCoord[1]+=VDir[1];
			vCoord[2]+=VDir[2];
			x=vCoord[0];	
			y=vCoord[1];	
			z=vCoord[2];	
			for (ih=0;ih<ROISize[1];ih++){
				x+=HDir[0];	
				y+=HDir[1];	
				z+=HDir[2];	

				x0=(int)(x);
				divx0cube=div(x0,CubeSize[0]);
				if ((x<0) || (divx0cube.quot>=NCubesX)){
					*(ROI++)=(unsigned char)255;
					continue;
				}

				y0=(int)(y);
				divy0cube=div(y0,CubeSize[1]);
				if ((y<0) || (divy0cube.quot>=NCubesY)){
					*(ROI++)=(unsigned char)255;
					continue;
				}	

				z0=(int)(z);
				divz0cube=div(z0,CubeSize[2]);
				if ((z<0) || (divz0cube.quot>=NCubesZ)){
					*(ROI++)=(unsigned char)255;
					continue;
				}

				fx = x - x0;
				fy = y - y0;
				fz = z - z0;

				if ((divx0cube.rem<CubeSize[0]-1) && (divy0cube.rem<CubeSize[1]-1) &&  (divz0cube.rem<CubeSize[2]-1)){
					pos=find_cubeinplane((unsigned int)divx0cube.quot,(unsigned int)divy0cube.quot,(unsigned int)divz0cube.quot);
					pos+=divx0cube.rem+divy0cube.rem*CubeSize[0]+divz0cube.rem*NPixelsInYX;
					d000=*pos;
					d100=*(++pos);

					pos+=CubeSize[0];
					d110=*pos;
					d010=*(--pos);

					pos+=NPixelsInYX;
					d011=*pos;
					d111=*(++pos);

					pos-=CubeSize[0];
					d101=*pos;
					d001=*(--pos);}
				else{
					getBorderPixel(divx0cube,divy0cube,divz0cube,&d000,&d100,&d010,&d001,&d101,&d110,&d011,&d111);}				

				*(ROI++)=(unsigned char)LERP(fz,LERP(fy,LERP(fx, d000, d100),LERP(fx, d010, d110)),LERP(fy,LERP(fx, d001, d101),LERP(fx, d011, d111)));
			}
		}
	}
	/*printf("bordercalls:%i",bordercalls);*/
	return completeLoading;
}

int release_ROI(void){
	if (initialized==0){
		return 1;
	}
	printf("Release ROI memory...");
	free(invalidCube);
	int imag;
	for (imag=0;imag<*NMag;imag++){
		MagOffset[imag]=NULL;
	}
	free(MagOffset);
	HyperCube[0]=NULL;
	HyperCube[1]=NULL;
	HyperCube[2]=NULL;
	AllCubes[0]=NULL;
	NMag=NULL;
	DataScale=NULL;
	CubeSize=NULL;
	NCubesPerEdge=NULL;
	NumberofCubes=NULL;

	LoaderMag=NULL;
	LoaderCurrCoord=NULL;
	LoaderState=NULL;

	printf("..done.\n");
	initialized=0;
	return 1;
}

int init_ROI(unsigned char* hyperCube0,unsigned char* hyperCube1,unsigned char* hyperCube2,short int* allCubes,int* nMag, float* dataScale,int* cubeSize, int* nCubesPerEdge,int* numberofCubes){
	if (initialized==1){
		release_ROI();
	}
	
	int imag;
	
	/* Assign input parameters*/
	CubeSize=cubeSize;
	NCubesPerEdge=nCubesPerEdge;
	NumberofCubes=numberofCubes;
	DataScale=dataScale;
	NMag=nMag;
	
	NPixelsInYX=CubeSize[0]*CubeSize[1];
	NPixelsPerCube=NPixelsInYX*CubeSize[2];
    
	NCubes=0;
	for (imag=0;imag<3;imag++){NCubes+=NCubesPerEdge[imag]*NCubesPerEdge[imag]*NCubesPerEdge[imag];}

	invalidCube=(unsigned char*)calloc(NPixelsPerCube,sizeof(unsigned char));
	if (invalidCube==NULL){printf("Could not allocate memory for invalidCube.\n");return -1;}	
	memset(invalidCube,(unsigned char)(255),NPixelsPerCube*sizeof(unsigned char));

	HyperCube[0]=hyperCube0;
	HyperCube[1]=hyperCube1;
	HyperCube[2]=hyperCube2;

	AllCubes[0]=allCubes;

	printf("Initialized ROI extraction with the following parameters:\nCubeSize: (%i,%i,%i), NCubesPerEdge: (%i,%i,%i)\n",\
		CubeSize[0],CubeSize[1],CubeSize[2],\
		NCubesPerEdge[0],NCubesPerEdge[1],NCubesPerEdge[2]);

	for (imag=0;imag<*NMag;imag++){
		printf("DataScale mag%i: (%f,%f,%f)\n",imag+1,DataScale[imag*3+0],DataScale[imag*3+1],DataScale[imag*3+2]);
	}
	printf("NMag: %i\n",*NMag);

	MagOffset=(short int**)calloc(*NMag,sizeof(short int*));
	if (MagOffset==NULL){printf("Could not allocate memory for MagOffset.\n");return -1;}

	printf("Number of cubes: ");
	MagOffset[0]=AllCubes[0];
	/*memset(MagOffset[0],(short int)(-1),(NumberofCubes[0]*NumberofCubes[1]*NumberofCubes[2])*sizeof(short int));*/	
	printf("(%i,%i,%i), ",NumberofCubes[0*3],NumberofCubes[0*3+1],NumberofCubes[0*3+2]);
	for (imag=1;imag<*NMag;imag++){
		MagOffset[imag]=MagOffset[imag-1]+NumberofCubes[(imag-1)*3]*NumberofCubes[(imag-1)*3+1]*NumberofCubes[(imag-1)*3+2];
		/*memset(MagOffset[imag],(short int)(-1),NumberofCubes[imag*3]*NumberofCubes[imag*3+1]*NumberofCubes[imag*3+2]*sizeof(short int));*/	
		printf("(%i,%i,%i), ",NumberofCubes[imag*3],NumberofCubes[imag*3+1],NumberofCubes[imag*3+2]);
	}
	printf("\n");
	initialized=1;
	return 1;
}
