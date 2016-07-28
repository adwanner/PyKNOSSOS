/*gcc -I/usr/include/libpng12 loaderlib.c readpng.c streamcube.h -lcurl -O3 -fPIC -shared -o clibraries/loaderlib.so*/

//FMI: /*i586-mingw32msvc-gcc -I/usr/local/i586-mingw32msvc/include -L/usr/local/i586-mingw32msvc/lib loaderlib.c readpng.c -O3 -shared -o clibraries/loaderlib.dll -lcurl -lpng -lz*/
//i586-mingw32msvc-gcc -O3 -shared -o clibraries/loaderlib.dll -DCURL_STATICLIB loaderlib.c readpng.c streamcube.h -lpthreadGC2 -lpng -lz -ljpeg -lcurl -lwldap32 -lz -lws2_32
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "readpng.h"
#include <pthread.h>
#include "streamcube.h" //has to be before jpeglib.h, otherwise conflict for INT32
#include <setjmp.h>
#include <jpeglib.h>

struct CubeIdentifier {
int flag;
int CubeID;
int mag;
int x;
int y;
int z;
};

int NThreads=5;
struct CubeIdentifier *Cube2Stream;
pthread_t *ThreadID;
int *validThreads;

int initialized=0;

float *DataScale, *CurrCoord;
int *NCubesPerEdge, NCubes, CumNCubes[4];
int prevCoord[3];
unsigned int PrevMag, CompletedLoading;

int *LoaderState;

int *Mag, *NMag, NMags2Load=3;
unsigned int *LoadingStrategy;

const char *BasePath, *BaseName, *BaseExt, *BaseURL, *UserName, *Password;
char *CubeFileName;
size_t BasePathLength, BaseNameLength, BaseExtLength;

unsigned int *Cubes2Load;

int *NCubes2Load;
int Cubes2LoadIdx;
int *OfflineMode;

int *NumberofCubes;
short int **CubeIDsInCache;
unsigned char *Cubes2Keep;
unsigned char *HyperCube[3];
short int *AllCubes[1], **MagOffset;

unsigned int NPriorities=3+3+1;
unsigned int Priority[3+3+1]; 
unsigned int cachepos;

unsigned int LoadingMode=0, ServerFormat;
int *CubeSize, NPixelsPerCube;

float *Dist2Center;

struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */

  jmp_buf setjmp_buffer;	/* for return to caller */
};
typedef struct my_error_mgr * my_error_ptr;

METHODDEF(void)
my_error_exit (j_common_ptr cinfo)
{
  printf("JPGERROR\n");
  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_ptr myerr = (my_error_ptr) cinfo->err;

  /* Always display the message. */
  /* We could postpone this until after returning, if we chose. */
  (*cinfo->err->output_message) (cinfo);

  /* Return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}

static void *pull_cube(void* userp){

	/*printf("Another thread was initialized");
	int i, res;
	for (i=0;i<100000;i++){
		res=res*res;
		res=res/i;
	}
	printf("Another thread gets closed.");	
	return NULL;*/
	struct CubeIdentifier *Cube2Stream = (struct CubeIdentifier *)userp;
	struct MemoryStruct CubeStream;
	int streamstate=0;
	unsigned int icube=0,found=0;

	CubeStream.memory = malloc(1);/* will be grown as needed by the realloc above */
	CubeStream.size = 0;/* no data at this point */
	InitStream((void *)&CubeStream);
	while (Cube2Stream->flag>=0){
		if (Cube2Stream->flag==0){
			usleep(50*1000);
		}
		else{
			CubeStream.memory = malloc(1);/* will be grown as needed by the realloc above */
			CubeStream.size = 0;/* no data at this point */
			streamstate=0;

			for (icube=1;icube<NCubes;icube=icube+5){
				if (*(Cubes2Load+icube)==Cube2Stream->CubeID){
					found=1;
					break;
				}
			};
			//printf("found: %d, cube2load: %d, cube2stream: %d\n",found,*(Cubes2Load+icube),Cube2Stream->CubeID);
			//if (*OfflineMode==1){printf("Working offline.\n");}
			if ((ServerFormat>0) && (BaseURL!=NULL) && (found==1) && (*OfflineMode==0)){
				streamstate=StreamCube((void *)&CubeStream,ServerFormat,BaseURL,BaseName,BaseExt,UserName,Password,Cube2Stream->mag,Cube2Stream->x,Cube2Stream->y,Cube2Stream->z,CubeSize);
				if (streamstate==1){
					WriteCube( (void *)&CubeStream,BasePath,BaseName,BaseExt,Cube2Stream->mag,Cube2Stream->x,Cube2Stream->y,Cube2Stream->z);
					*(MagOffset[Cube2Stream->mag-1]+Cube2Stream->CubeID)=(short int)-1;
				}
				else if (streamstate==2){ //time out
					*(MagOffset[Cube2Stream->mag-1]+Cube2Stream->CubeID)=(short int)-1;
				}
				else{
					*(MagOffset[Cube2Stream->mag-1]+Cube2Stream->CubeID)=(short int)-2;	
				};
				//printf("streamstate: %d, CubeStream.size: %lu, %p\n",streamstate,(long)CubeStream.size,&CubeStream.memory);
			}
			else{
				*(MagOffset[Cube2Stream->mag-1]+Cube2Stream->CubeID)=(short int)-1;
			};
			Cube2Stream->flag=0;
		};
	}
	if (CubeStream.curl_handle!=NULL){
		curl_easy_cleanup(CubeStream.curl_handle);
	}
	free(CubeStream.memory);
	return NULL;
}


#define CubeID(mag,x0,y0,z0) (x0+NumberofCubes[(mag-1)*3]*(y0+NumberofCubes[(mag-1)*3+1]*z0))
int GenerateLoadingStrategy1(){
	unsigned int icube=0, x, y, z,imag, halfEdge, iprior=0;
	Priority[iprior++]=0;
	for (imag=0;imag<3;imag++){
		halfEdge=(unsigned int)ceil(((float)NCubesPerEdge[imag])/2.0)-1;


		//ZX plane, omit cubes that have been loaded for YX and YZ planes
		y=halfEdge;
		for (x=0;x<NCubesPerEdge[imag];x++){
			for (z=0;z<NCubesPerEdge[imag];z++){
				*(LoadingStrategy+icube++)=imag;
				*(LoadingStrategy+icube++)=x;
				*(LoadingStrategy+icube++)=y;
				*(LoadingStrategy+icube++)=z;
			}
		}

		//YX plane
		z=halfEdge;
		for (x=0;x<NCubesPerEdge[imag];x++){
			for (y=0;y<halfEdge;y++){
				*(LoadingStrategy+icube++)=imag;
				*(LoadingStrategy+icube++)=x;
				*(LoadingStrategy+icube++)=y;
				*(LoadingStrategy+icube++)=z;
			}
			for (y=halfEdge+1;y<NCubesPerEdge[imag];y++){
				*(LoadingStrategy+icube++)=imag;
				*(LoadingStrategy+icube++)=x;
				*(LoadingStrategy+icube++)=y;
				*(LoadingStrategy+icube++)=z;
			}
		}

		//YZ plane, omit cubes that have been loaded for YX plane
		x=halfEdge;
		for (y=0;y<halfEdge;y++){
			for (z=0;z<halfEdge;z++){
				*(LoadingStrategy+icube++)=imag;
				*(LoadingStrategy+icube++)=x;
				*(LoadingStrategy+icube++)=y;
				*(LoadingStrategy+icube++)=z;
			}
			for (z=halfEdge+1;z<NCubesPerEdge[imag];z++){
				*(LoadingStrategy+icube++)=imag;
				*(LoadingStrategy+icube++)=x;
				*(LoadingStrategy+icube++)=y;
				*(LoadingStrategy+icube++)=z;
			}
		}
		for (y=halfEdge+1;y<NCubesPerEdge[imag];y++){
			for (z=0;z<halfEdge;z++){
				*(LoadingStrategy+icube++)=imag;
				*(LoadingStrategy+icube++)=x;
				*(LoadingStrategy+icube++)=y;
				*(LoadingStrategy+icube++)=z;
			}
			for (z=halfEdge+1;z<NCubesPerEdge[imag];z++){
				*(LoadingStrategy+icube++)=imag;
				*(LoadingStrategy+icube++)=x;
				*(LoadingStrategy+icube++)=y;
				*(LoadingStrategy+icube++)=z;
			}
		}
		Priority[iprior++]=icube/4;
	}
	for (imag=0;imag<3;imag++){		
		halfEdge=(unsigned int)ceil(((float)NCubesPerEdge[imag])/2.0)-1;
		//load the rest (8 holes at the corner of the hypercube)
		//left-front-top
		for (x=0;x<halfEdge;x++){
			for (y=0;y<halfEdge;y++){
				for (z=0;z<halfEdge;z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		//right-front-top
		for (x=halfEdge+1;x<NCubesPerEdge[imag];x++){
			for (y=0;y<halfEdge;y++){
				for (z=0;z<halfEdge;z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		//left-back-top
		for (x=0;x<halfEdge;x++){
			for (y=halfEdge+1;y<NCubesPerEdge[imag];y++){
				for (z=0;z<halfEdge;z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		//right-back-top
		for (x=halfEdge+1;x<NCubesPerEdge[imag];x++){
			for (y=halfEdge+1;y<NCubesPerEdge[imag];y++){
				for (z=0;z<halfEdge;z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		//left-front-bottom
		for (x=0;x<halfEdge;x++){
			for (y=0;y<halfEdge;y++){
				for (z=halfEdge+1;z<NCubesPerEdge[imag];z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		//right-front-bottom
		for (x=halfEdge+1;x<NCubesPerEdge[imag];x++){
			for (y=0;y<halfEdge;y++){
				for (z=halfEdge+1;z<NCubesPerEdge[imag];z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		//left-back-bottom
		for (x=0;x<halfEdge;x++){
			for (y=halfEdge+1;y<NCubesPerEdge[imag];y++){
				for (z=halfEdge+1;z<NCubesPerEdge[imag];z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		//right-back-bottom
		for (x=halfEdge+1;x<NCubesPerEdge[imag];x++){
			for (y=halfEdge+1;y<NCubesPerEdge[imag];y++){
				for (z=halfEdge+1;z<NCubesPerEdge[imag];z++){
					*(LoadingStrategy+icube++)=imag;
					*(LoadingStrategy+icube++)=x;
					*(LoadingStrategy+icube++)=y;
					*(LoadingStrategy+icube++)=z;
				}
			}
		}
		Priority[iprior++]=icube/4;
	}
	/*printf("Priorities: ");
	for (iprior=0;iprior<NPriorities;iprior++){
		printf("%i, ",Priority[iprior]);
	}
	printf("\n");*/
	return 1;
}

int compare (const void * idx1, const void * idx2){
  if ( Dist2Center[*(unsigned int*)idx1] <  Dist2Center[*(unsigned int*)idx2] ) return -1;
  if ( Dist2Center[*(unsigned int*)idx1] == Dist2Center[*(unsigned int*)idx2] ) return 0;
  if ( Dist2Center[*(unsigned int*)idx1] >  Dist2Center[*(unsigned int*)idx2] ) return 1;
}
//Load cubes according to the distance to the current position (center cube)
int GenerateLoadingStrategy2(){
	int icube, jcube, x, y, z,imag, halfEdge;
	unsigned int CubeIdx[NCubes];

	unsigned int tempLoadingStrategy[4*NCubes];

	Dist2Center=(float*)calloc(NCubes,sizeof(float));

	icube=0;
	jcube=0;
	for (imag=0;imag<3;imag++){
		halfEdge=(unsigned int)ceil(((float)NCubesPerEdge[imag])/2.0)-1;
		for (x=0;x<NCubesPerEdge[imag];x++){
			for (y=0;y<NCubesPerEdge[imag];y++){
				for (z=0;z<NCubesPerEdge[imag];z++){
					CubeIdx[jcube]=jcube;
					Dist2Center[jcube]=((float)(x-halfEdge)*(float)(x-halfEdge))+((float)(y-halfEdge)*(float)(y-halfEdge))+((float)(z-halfEdge)*(float)(z-halfEdge))+(float)(imag)/2.0; //(imag)/2.0 is added to priorize mag 0 over 1 over 2
					//printf("Dist2Center[%u]=%f\n",jcube,Dist2Center[jcube]);
					tempLoadingStrategy[icube++]=imag;
					tempLoadingStrategy[icube++]=x;
					tempLoadingStrategy[icube++]=y;
					tempLoadingStrategy[icube++]=z;
					jcube++;
				}
			}
		}
	}
	qsort(CubeIdx,NCubes, sizeof(unsigned int),compare);
	
	icube=0;
	for (jcube=0;jcube<NCubes;jcube++){
		//printf("Loading strategy2, idx %u (mag=%u,x=%u,y=%u,z=%u) has distance %f\n",CubeIdx[jcube],tempLoadingStrategy[CubeIdx[jcube]*4+0],tempLoadingStrategy[CubeIdx[jcube]*4+1],tempLoadingStrategy[CubeIdx[jcube]*4+2],tempLoadingStrategy[CubeIdx[jcube]*4+3],Dist2Center[CubeIdx[jcube]]);
		*(LoadingStrategy+icube++)=tempLoadingStrategy[CubeIdx[jcube]*4+0];
		*(LoadingStrategy+icube++)=tempLoadingStrategy[CubeIdx[jcube]*4+1];
		*(LoadingStrategy+icube++)=tempLoadingStrategy[CubeIdx[jcube]*4+2];
		*(LoadingStrategy+icube++)=tempLoadingStrategy[CubeIdx[jcube]*4+3];
	}
	free(Dist2Center);
}


int LoadCubesFromList(unsigned int icube, unsigned int icubeEnd){
	size_t bytes_read;
	FILE *fid;
	unsigned char *CachePosition;
	unsigned int cubeID, mag, imag;
	unsigned int x, y, z;
	

	unsigned long image_width, image_height;
	int image_channels;
	short int** CubeIDcachepos;

	int error;

	struct jpeg_decompress_struct cinfo;
	struct my_error_mgr jerr;
	/* More stuff */
	JSAMPARRAY buffer;		/* Output row buffer */
	int row_stride;		/* physical row width in output buffer */
	
	while(icube<icubeEnd){
		while (cachepos<NCubes){	
			if (*(Cubes2Keep+cachepos)){
				cachepos++;
				continue;
			}
			CubeIDcachepos=(CubeIDsInCache+cachepos);
			if (*CubeIDcachepos!=NULL){
				if (**CubeIDcachepos>-1){ 
					**CubeIDcachepos=(short int)(-1);
				}
			}
			break;	
		}
		if (cachepos>=NCubes){
			printf("Error: No free slot for new cube found. This should not happen...\n");
			return 0;
		}
		
		for (imag=0;imag<3;imag++){
			if (cachepos<CumNCubes[imag+1]){
				CachePosition=(HyperCube[imag]+(cachepos-CumNCubes[imag])*NPixelsPerCube);
				break;
			}
		}
		
		mag=*(Cubes2Load+icube++);		
		cubeID=*(Cubes2Load+icube++);	
		if (*(MagOffset[mag-1]+cubeID)==(short int)-2){
			//printf("Cube (mag=%u,x=%u,y=%u,z=%u) is non-existing\n",mag,x,y,z);
			icube+=3;
			(*NCubes2Load)--;
			continue;
		}
		else if (*(MagOffset[mag-1]+cubeID)==(short int)-3){
			//printf("Cube (mag=%u,x=%u,y=%u,z=%u) is non-existing\n",mag,x,y,z);
			icube+=3;
			continue;
		}
		else if (*(MagOffset[mag-1]+cubeID)>(short int)-1){
			//printf("Cube (mag=%u,x=%u,y=%u,z=%u) has already been loaded\n",mag,x,y,z);
			icube+=3;
			continue;
		}

		x=*(Cubes2Load+icube++);
		y=*(Cubes2Load+icube++);
		z=*(Cubes2Load+icube++);
		sprintf(CubeFileName,"%s/mag%u/x%04u/y%04u/z%04u/%s_mag%u_x%04u_y%04u_z%04u%s",\
			BasePath,mag,x,y,z,BaseName,mag,x,y,z,BaseExt);

 		//printf("Cube (mag=%u,x=%u,y=%u,z=%u) is in cache at: %u\n",mag,x,y,z,cachepos);
		fid = fopen (CubeFileName,"rb");
		error=0;
		if (fid!=NULL){
			/*printf("Loading cube: %s\n",CubeFileName);*/
			switch (LoadingMode){
				case 0:/*Raw image cubes*/
					bytes_read=fread(CachePosition,sizeof(unsigned char),NPixelsPerCube,fid);
					if (bytes_read != NPixelsPerCube) {
						error=1;
						printf("Read %zu bytes. Reading error for: %s\n",bytes_read,CubeFileName);
					}
					break;
				case 1:/*png image cubes, slices aranged from left to right*/
					if (readpng_init(fid, &image_width, &image_height) != 0) {
						error=1;
						printf("Error: Invalid PNG file: %s\n",CubeFileName);
					}
					else{
						if (image_width*image_height!=NPixelsPerCube){
							error=1;
							printf("Error: Invalid PNG image size: %s\n",CubeFileName);
						}
						else{
							readpng_get_image(CachePosition,&image_channels,1);
							readpng_cleanup();
						}
					}					    
					break;
				case 2:/*png image cubes, slices aranged in rows*/
					if (readpng_init(fid, &image_width, &image_height) != 0) {
						error=1;							
						printf("Error: Invalid PNG file: %s\n",CubeFileName);
					}
					else{
						if (image_width*image_height!=NPixelsPerCube){
							error=1;
							printf("Error: Invalid PNG image size: %s\n",CubeFileName);
						}
						else{
							readpng_get_image(CachePosition,&image_channels,2);
							readpng_cleanup();
						}
					}					    
					break;
				case 3:/*jpg image cubes, slices aranged in rows*/
					/* This struct contains the JPEG decompression parameters and pointers to
					* working space (which is allocated as needed by the JPEG library).
					*/
					/* Step 1: allocate and initialize JPEG decompression object */
					/* We set up the normal JPEG error routines, then override error_exit. */
					cinfo.err = jpeg_std_error(&jerr.pub);
					jerr.pub.error_exit = my_error_exit;

					if (setjmp(jerr.setjmp_buffer)) {
					  /* If we get here, the JPEG code has signaled an error.
					   * We need to clean up the JPEG object, close the input file, and return.
					   */
					  jpeg_destroy_decompress(&cinfo);
					  error=1;
					  printf("Error: Invalid JPG file: %s\n",CubeFileName);
					  break;
					}
					/* Now we can initialize the JPEG decompression object. */
			    		jpeg_create_decompress(&cinfo);

					/* Step 2: specify data source (eg, a file) */
					jpeg_stdio_src(&cinfo, fid);

					/* Step 3: read file parameters with jpeg_read_header() */
					(void) jpeg_read_header(&cinfo, TRUE);
					/* We can ignore the return value from jpeg_read_header since
					 *   (a) suspension is not possible with the stdio data source, and
					 *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
					 * See libjpeg.doc for more info.
					*/
					/* Step 5: Start decompressor */
					(void) jpeg_start_decompress(&cinfo);

					if (cinfo.output_width*cinfo.output_height!=NPixelsPerCube){
						error=1;
						printf("Error: Invalid JPG image size: %s, size: %i\n",CubeFileName,cinfo.output_width*cinfo.output_height);
						jpeg_destroy_decompress(&cinfo);
						break;
					};
					if ((cinfo.output_components>1) | (cinfo.jpeg_color_space != JCS_GRAYSCALE)){
						error=1;
						printf("Error: Not a grayscale JPG image: %s\n",CubeFileName);
						jpeg_destroy_decompress(&cinfo);
						break;
					};

					/* JSAMPLEs per row in output buffer */
					row_stride = cinfo.output_width * cinfo.output_components;
				  	/*printf("row_stride: %i\n",row_stride);
					printf("cinfo.output_height: %i\n",cinfo.output_height);*/

					/* Step 6: while (scan lines remain to be read) */
					/*           jpeg_read_scanlines(...); */

					/* Here we use the library's state variable cinfo.output_scanline as the
					 * loop counter, so that we don't have to keep track ourselves.
					*/
					buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
					while (cinfo.output_scanline < cinfo.output_height) {
					  /* jpeg_read_scanlines expects an array of pointers to scanlines.
					   * Here the array is only one element long, but you could ask for
					   * more than one scanline at a time if that's more convenient.
					   */
					  /*printf("irow: %i\n",cinfo.output_scanline);*/

					  
					  (void) jpeg_read_scanlines(&cinfo,buffer, 1);
					  //printf("%lu\n",(long)cinfo.output_scanline);

					  memcpy(CachePosition+(cinfo.output_scanline-1)*row_stride,buffer[0],row_stride);
					}

					/* Step 7: Finish decompression */
					if (cinfo.mem!=NULL){
						(void) jpeg_finish_decompress(&cinfo);
					}
					/* We can ignore the return value since suspension is not possible
					* with the stdio data source.
					*/

					/* Step 8: Release JPEG decompression object */

					/* This is an important step since it will release a good deal of memory. */
					
					jpeg_destroy_decompress(&cinfo);
										    
					break;
			}
		}
		else{
			error=1;
		};
		if (fid!=NULL){
	  		fclose(fid);
		}
		if (error==0){
	  		*CubeIDcachepos=(short int*)(MagOffset[mag-1]+cubeID);
			**CubeIDcachepos=(short int)cachepos;
			//printf("Load cube (mag=%u,x=%u,y=%u,z=%u) into cache at pos %u\n",mag,x,y,z,cachepos);
			cachepos++;
			(*NCubes2Load)--;
		}
		else{
			if ((ServerFormat>0) && (BaseURL!=NULL) && ((*(MagOffset[mag-1]+cubeID))!=-3)){
				int found=0, ithread;
				for (ithread=0;ithread<NThreads;ithread++){
					if ((Cube2Stream[ithread].flag==0) && (validThreads[ithread]==1)){
						//printf("Thread slot found: %d\n",ithread);
						found=1;
						break;
					}					
				}
				if (found==0){
					/*printf("No thread slot found. \n");*/
				}
				else{			
					*(MagOffset[mag-1]+cubeID)=-3;
					
					Cube2Stream[ithread].CubeID=cubeID;
					Cube2Stream[ithread].mag=mag;
					Cube2Stream[ithread].x=x;
					Cube2Stream[ithread].y=y;
					Cube2Stream[ithread].z=z;
					Cube2Stream[ithread].flag=1;

				}
			}
			else{
				printf("Cube not found: %s\n",CubeFileName);
				*(MagOffset[mag-1]+cubeID)=(short int)-2;
				(*NCubes2Load)--;
			};
			//printf("Cube %u (mag=%u,x=%u,y=%u,z=%u) is non-existing\n",mag,x,y,z);
		}
	}
}

int load_cubes(){
	if (initialized==0){
		return 0;
	}
	unsigned int idim, imag;
	unsigned int cubeID, whichMag[3];
	unsigned int icube, icubeEnd;
	int x, y, z, pos;
	unsigned int halfEdge;
	int start_cube[9];

	
	whichMag[0]=(unsigned int)(*Mag); //current MAGNIFICATION
	if (whichMag[0]<1){
		printf("loaderlib error: Magnification has to be an integer >0\n");
		return 0;
	}
	else if (whichMag[0]==1){
		whichMag[1]=whichMag[0]+2; //zoom-out 2x
		whichMag[2]=whichMag[0]+1; //zoom-out 1x
	}
	else if (whichMag[0]==*NMag){
		whichMag[1]=whichMag[0]-1; //zoom-in 1x
		whichMag[2]=whichMag[0]-2; //zoom-in 2x
	}
	else {
		whichMag[1]=whichMag[0]-1; //zoom-in MAGNIFICATION (except if we are already at max zoom)
		whichMag[2]=whichMag[0]+1; //zoom-out MAGNIFICATION (except if we are already at min zoom)
	}
	//printf("Magnifications: %u, %u, %u\n",whichMag[0],whichMag[1],whichMag[2]);
	
	for (imag=0;imag<3;imag++){
		if ((whichMag[imag]==0) || (whichMag[imag]>*NMag)){
			whichMag[imag]=0;
			continue;
		}
		halfEdge=(unsigned int)ceil(((float)(NCubesPerEdge[imag]))/2.0)-1;
		for (idim=0;idim<3;idim++){
			start_cube[imag*3+idim]=(int)(floor((CurrCoord[idim]/DataScale[(whichMag[imag]-1)*3+idim]-1)/((float)CubeSize[idim]))-halfEdge);
		}
	}
	
	if (*LoaderState<2 && CompletedLoading && (prevCoord[0]==start_cube[0]) && (prevCoord[1]==start_cube[1]) && (prevCoord[2]==start_cube[2]) && (PrevMag==whichMag[0])){
		return CompletedLoading;
	}

	PrevMag=whichMag[0];	
	for (idim=0;idim<3;idim++){
		prevCoord[idim]=start_cube[idim];
	}

	CompletedLoading=0;
	
	//First check which cubes have to be loaded and which are already in cache
	//reset memory
	memset(Cubes2Keep,(unsigned char)0,NCubes);
	Cubes2LoadIdx=0;
	icube=0;
	*NCubes2Load=0;
	while (icube<4*NCubes){
		imag=LoadingStrategy[icube++];
		if (whichMag[imag]==0){
			icube+=3;
			continue;
		}
		x=start_cube[imag*3]+LoadingStrategy[icube++];
		if ((x<0) || (x>=NumberofCubes[(whichMag[imag]-1)*3+0])){
			icube+=2;
			continue;
		}
		y=start_cube[imag*3+1]+LoadingStrategy[icube++];
		if ((y<0) || (y>=NumberofCubes[(whichMag[imag]-1)*3+1])){
			icube++;
			continue;
		}
		z=start_cube[imag*3+2]+LoadingStrategy[icube++];
		if ((z<0) || (z>=NumberofCubes[(whichMag[imag]-1)*3+2])){
			continue;
		}

		cubeID=(unsigned int)CubeID(whichMag[imag],x,y,z);
		pos=*(MagOffset[whichMag[imag]-1]+cubeID);
		if (pos>-1){
			*(Cubes2Keep+pos)=(unsigned char)1;
			continue;
		}
		(*NCubes2Load)++;
		*(Cubes2Load+Cubes2LoadIdx++)=whichMag[imag];
		*(Cubes2Load+Cubes2LoadIdx++)=cubeID;
		*(Cubes2Load+Cubes2LoadIdx++)=(unsigned int)x;
		*(Cubes2Load+Cubes2LoadIdx++)=(unsigned int)y;
		*(Cubes2Load+Cubes2LoadIdx++)=(unsigned int)z;
	}

	//printf("Number of cubes to load: %u\n",Cubes2LoadIdx);
	cachepos=0;
	icube=0; 
	halfEdge=(unsigned int)ceil(((float)(NCubesPerEdge[0]))/2.0)-1;
	while (icube<Cubes2LoadIdx){
		if (*LoaderState==0){
			return CompletedLoading;
		}	
		if (PrevMag!=(unsigned int)(*Mag)){
			//printf("Magnification changed...\n");
			load_cubes();
			return CompletedLoading;
		}
		for (idim=0;idim<3;idim++){
			if (start_cube[idim]!=(int)(floor((CurrCoord[idim]/DataScale[(whichMag[0]-1)*3+idim]-1)/((float)CubeSize[idim]))-halfEdge)){
				//printf("Coordinates have been updated...\n");
				load_cubes();
				return CompletedLoading;
			}
		}
		if ((ServerFormat>0) && (BaseURL!=NULL)){
			icubeEnd=icube+500; //5*100
			//icubeEnd=icube+NThreads;			
		}
		else{
			icubeEnd=icube+500; //5*100
		};
		if (icubeEnd>Cubes2LoadIdx){
			icubeEnd=Cubes2LoadIdx;
		}
		if ((ServerFormat>0) && (BaseURL!=NULL)){
			LoadCubesFromList(0,icubeEnd);
		}
		else{
			LoadCubesFromList(icube,icubeEnd);
		};
		icube=icubeEnd;
	}

	CompletedLoading=1;

	//printf("...done with loading cubes.\n");
	return CompletedLoading;
}

int release_loader(void){
	if (initialized==0){
		return 1;
	}
	printf("Release Loader memory...");
	if ((ServerFormat>0) && (BaseURL!=NULL)){
		int error, ithread;
		for(ithread=0; ithread< NThreads; ithread++) {
			Cube2Stream[ithread].flag=-1;
			pthread_detach(ThreadID[ithread]);
			validThreads[ithread]=0;
			/*if (ThreadID[ithread]!=0){
				error = pthread_join(ThreadID[ithread], NULL);
				fprintf(stderr, "Thread %d terminated\n",ithread);
			};*/
		};
		/* this function releases resources acquired by curl_global_init() */
		curl_global_cleanup();
	};
	free(CubeIDsInCache);
	int imag;
	for (imag=0;imag<*NMag;imag++){
		MagOffset[imag]=NULL;
	}


	free(MagOffset);
	free(CubeFileName);
	free(LoadingStrategy);
	free(Cubes2Load);
	free(Cubes2Keep);

	HyperCube[0]=NULL;
	HyperCube[1]=NULL;
	HyperCube[2]=NULL;
	AllCubes[0]=NULL;
	NCubesPerEdge=NULL;
	BasePath=NULL;
	BaseName=NULL;
	BaseExt=NULL;
	NMag=NULL;
	DataScale=NULL;
	CubeSize=NULL;
	NCubesPerEdge=NULL;
	NumberofCubes=NULL;
	Mag=NULL;

	BaseURL=NULL;
	UserName=NULL;
	Password=NULL;

	LoaderState=NULL;
	initialized=0;
	return 1;
}

int init_loader(float* currCoord,unsigned char* hyperCube0,unsigned char* hyperCube1,unsigned char* hyperCube2,short int* allCubes,int* nCubesPerEdge,const char* basePath,const char* baseName,char* baseExt, int* nMag, float* dataScale,int* cubeSize,int* numberofCubes,int* magnification,int* loadingMode,
const char* baseURL,const char* userName,const char* password, int* serverFormat,int*nThreads, int* offlineMode, int* nCubes2Load, int* loaderState){

  int idim,imag, pos, ithread, error;
	if (initialized==1){
		release_loader();
	}

	/* Assign input parameters*/
	CubeSize=cubeSize;
	NCubesPerEdge=nCubesPerEdge;
	BasePath=basePath;
	BaseName=baseName;
	BaseExt=baseExt;
	NMag=nMag;
	DataScale=dataScale;
	CurrCoord=currCoord;
	NumberofCubes=numberofCubes;
	Mag=(int *)magnification;

	BaseURL=baseURL;
	UserName=userName;
	Password=password;
	ServerFormat=(unsigned int) *serverFormat;
	NThreads=(int) (*nThreads);
	OfflineMode=offlineMode;

	NCubes2Load=nCubes2Load;
	NPixelsPerCube=CubeSize[0]*CubeSize[1]*CubeSize[2];


	if ((ServerFormat>0) && (BaseURL!=NULL)){
		curl_global_init(CURL_GLOBAL_ALL);
	};

	ThreadID=(pthread_t*)calloc(NThreads,sizeof(pthread_t));
	if (ThreadID==NULL){printf("Could not allocate memory for ThreadID.\n");return -1;}

	validThreads=(int*)calloc(NThreads,sizeof(int)); 
	if (validThreads==NULL){printf("Could not allocate memory for validThreads.\n");return -1;}
	
	Cube2Stream=(struct CubeIdentifier*)calloc(NThreads,sizeof(struct CubeIdentifier)); 
	if (Cube2Stream==NULL){printf("Could not allocate memory for Cube2Stream.\n");return -1;}

	LoaderState=loaderState;

	LoadingMode=(unsigned int) *loadingMode;
	printf("Started loader with the following parameters:\n");
	printf("BasePath: %s, BaseName: %s\n",BasePath,BaseName);
	printf("CubeSize: (%i,%i,%i), NCubesPerEdge: (%i,%i,%i), currCoord: (%f,%f,%f)\n",CubeSize[0],CubeSize[1],CubeSize[2],\
		NCubesPerEdge[0],NCubesPerEdge[1],NCubesPerEdge[2],currCoord[0],currCoord[1],currCoord[2]);

	
	for (imag=0;imag<*NMag;imag++){
		printf("DataScale mag%i: (%f,%f,%f)\n",imag+1,DataScale[imag*3+0],DataScale[imag*3+1],DataScale[imag*3+2]);
	}
	printf("NMag: %i, Mag: %i\n",*NMag,*Mag);

	NCubes=0;
	CumNCubes[0]=0;
	for (imag=0;imag<3;imag++){
		NCubes+=NCubesPerEdge[imag]*NCubesPerEdge[imag]*NCubesPerEdge[imag];
		CumNCubes[imag+1]=NCubes;
	}
	printf("NCubes to load:%i\n",NCubes);
  
	Cubes2Keep=(unsigned char*)calloc(NCubes,sizeof(unsigned char)); //loaded cubes that are still needed
	if (Cubes2Keep==NULL){printf("Could not allocate memory for Cubes2Keep.\n");return -1;}
	
	Cubes2Load=(unsigned int*)calloc(5*NCubes,sizeof(unsigned int)); //(x,y,z) cubes that have to be loaded
	if (Cubes2Load==NULL){printf("Could not allocate memory for Cubes2Load.\n");return -1;}

	CubeIDsInCache=(short int**)calloc(NCubes,sizeof(short int*));
	if (CubeIDsInCache==NULL){printf("Could not allocate memory for CubeIDsInCache.\n");return -1;}
	for (pos=0;pos<NCubes;pos++){
		*(CubeIDsInCache+pos)=(short int*)NULL;}

	LoadingStrategy=(unsigned int*)calloc(4*NCubes,sizeof(unsigned int));
	if (LoadingStrategy==NULL){printf("Could not allocate memory for LoadingStrategy.\n");return -1;}
	
	BasePathLength=strlen(BasePath);
	BaseNameLength=strlen(BaseName);
	BaseExtLength=strlen(BaseExt);

	CubeFileName=(char *)calloc((BasePathLength+5+2+4+2+4+2+4+1+BaseNameLength+5+2+4+2+4+2+4+BaseExtLength+1),sizeof(char));
	if (CubeFileName==NULL){printf("Could not allocate memory for CubeFileName.\n");return -1;}

	HyperCube[0]=hyperCube0;
	HyperCube[1]=hyperCube1;
	HyperCube[2]=hyperCube2;
	
	AllCubes[0]=allCubes;

	MagOffset=(short int**)calloc(*NMag,sizeof(short int*));
	if (MagOffset==NULL){printf("Could not allocate memory for MagOffset.\n");return -1;}

	printf("Number of cubes: ");
	MagOffset[0]=AllCubes[0];
	memset(MagOffset[0],(short int)(-1),(NumberofCubes[0]*NumberofCubes[1]*NumberofCubes[2])*sizeof(short int));	
	printf("(%i,%i,%i), ",NumberofCubes[0*3],NumberofCubes[0*3+1],NumberofCubes[0*3+2]);

	for (imag=1;imag<*NMag;imag++){
		MagOffset[imag]=MagOffset[imag-1]+NumberofCubes[(imag-1)*3]*NumberofCubes[(imag-1)*3+1]*NumberofCubes[(imag-1)*3+2];
		memset(MagOffset[imag],(short int)(-1),NumberofCubes[imag*3]*NumberofCubes[imag*3+1]*NumberofCubes[imag*3+2]*sizeof(short int));	
		printf("(%i,%i,%i), ",NumberofCubes[imag*3],NumberofCubes[imag*3+1],NumberofCubes[imag*3+2]);
	}
	printf("\n");
	CompletedLoading=0;
	prevCoord[0]=-1000;
	prevCoord[1]=-1000;
	prevCoord[2]=-1000;
	PrevMag=1000;
	if ((ServerFormat>0) && (BaseURL!=NULL)){
		printf("Start streamer with %d channels\n",NThreads);
		for (ithread=0;ithread<NThreads;ithread++){	
			Cube2Stream[ithread].flag=0;
			error = pthread_create(&ThreadID[ithread],NULL,pull_cube,(void *)&Cube2Stream[ithread]);
			if(0 != error){
				fprintf(stderr, "Couldn't run thread error no %d\n", error);
				validThreads[ithread]=0;
			}
			else{
				validThreads[ithread]=1;
			}
		};
	};
	GenerateLoadingStrategy2();

	initialized=1;
	return 1;
}
