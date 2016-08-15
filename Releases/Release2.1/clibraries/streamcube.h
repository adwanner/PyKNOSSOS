/* gcc URLcubeloadingTests20160629.c -lcurl -O3 -o URLcubeloadingTests20160629*/

//FMI: /*i586-mingw32msvc-gcc -I/usr/local/i586-mingw32msvc/include -L/usr/local/i586-mingw32msvc/lib loaderlib.c readpng.c -O3 -shared -o clibraries/loaderlib.dll -lpng -lz*/
//wannerad: /*i586-mingw32msvc-gcc -I/usr/local/i586-mingw32msvc/include -I/usr/include/libpng12 -I/usr/lib/syslinux/com32/include -L/usr/local/i586-mingw32msvc/lib loaderlib.c readpng.c -O3 -shared -o clibraries/loaderlib.dll -lpng -lz*/
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <curl/curl.h>
#include "mkpath.h"

struct MemoryStruct {
char *memory;
size_t size;
CURL *curl_handle;
};


int WriteCube(void *userp,const char *basepath, const char *basename, const char *baseExt, unsigned int mag, unsigned int x, unsigned int y, unsigned int z){
	FILE *fid;
	char *CubePath;
	char *CubeFileName;
	size_t BasePathLength, BaseNameLength, BaseExtLength;
	struct MemoryStruct *cube = (struct MemoryStruct *)userp;
	int state;

	BasePathLength=strlen(basepath);
	BaseNameLength=strlen(basename);
	BaseExtLength=strlen(baseExt);

	CubePath=(char *)calloc((BasePathLength+5+2+4+2+4+2+4+1),sizeof(char));
	if (CubePath==NULL){printf("Could not allocate memory for CubePath.\n");return -1;}

	CubeFileName=(char *)calloc((BasePathLength+5+2+4+2+4+2+4+1+BaseNameLength+5+2+4+2+4+2+4+BaseExtLength+1),sizeof(char));
	if (CubeFileName==NULL){printf("Could not allocate memory for CubeFileName.\n");return -1;}

	if (cube->size==0){
		state= 0;
		printf("ERROR: Have not written empty cube file %s\n",CubeFileName);
	}
	else{
		sprintf(CubePath,"%s/mag%u/x%04u/y%04u/z%04u",basepath,mag,x,y,z);
		mkpath(CubePath,0777);

		sprintf(CubeFileName,"%s/mag%u/x%04u/y%04u/z%04u/%s_mag%u_x%04u_y%04u_z%04u%s",\
					basepath,mag,x,y,z,basename,mag,x,y,z,baseExt);
		/*printf("Could write cube: %s\n",CubeFileName);
		return 1;*/
		fid = fopen(CubeFileName, "wb");
		if(fid!=NULL) {
			fwrite(cube->memory, 1,cube->size,fid);
			/* close the header file */
			fclose(fid);
			printf("Wrote cube: %s\n",CubeFileName);
			state= 1;
		}
		else{
			printf("ERROR: Could not write cube file %s\n",CubeFileName);
			state= 0;
		}
	}	
	free(CubePath);
	free(CubeFileName);
	return state;
}

static size_t HeaderCallback(void *contents, size_t size, size_t nmemb, void *userp){
	size_t realsize = size * nmemb;
	struct MemoryStruct *header = (struct MemoryStruct *)userp;

	header->memory = realloc(header->memory, header->size + realsize + 1);
	if(header->memory == NULL) {
		/* out of memory! */
		printf("not enough memory (realloc returned NULL)\n");
		return 0;
	}

	memcpy(&(header->memory[header->size]), contents, realsize);
	header->size += realsize;
	header->memory[header->size] = 0;

	return realsize;
}

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp){
	size_t realsize = size * nmemb;
	struct MemoryStruct *cube = (struct MemoryStruct *)userp;

	cube->memory = realloc(cube->memory, cube->size + realsize + 1);
	if(cube->memory == NULL) {
		/* out of memory! */
		printf("not enough memory (realloc returned NULL), cube->size: %lu,\n",(long)cube->size);
		return 0;
	}

	memcpy(&(cube->memory[cube->size]), contents, realsize);
	cube->size += realsize;
	cube->memory[cube->size] = 0;

	return realsize;
}
int InitStream(void *userp){
	long MaxLoadingTime=10;
	int returnsignal;


	struct MemoryStruct *InputCube = (struct MemoryStruct *)userp;
	InputCube->memory = malloc(1);/* will be grown as needed by the realloc above */
	InputCube->size = 0;/* no data at this point */

	/*struct MemoryStruct header;

	header.memory = malloc(1);// will be grown as needed by the realloc above
	header.size = 0;// no data at this point */

	/* init the curl session */
	InputCube->curl_handle = curl_easy_init();
	if(InputCube->curl_handle==NULL) {
		printf("ERROR: Could not curl_easy_init().\n");
		returnsignal= 2; /*signals time out */
	}
	else{
		curl_easy_setopt(InputCube->curl_handle,CURLOPT_NOSIGNAL, 1L);

		curl_easy_setopt(InputCube->curl_handle, CURLOPT_TIMEOUT,MaxLoadingTime); 

		/* Switch on full protocol/debug output while testing */
		//curl_easy_setopt(InputCube->curl_handle, CURLOPT_VERBOSE, 1L);

		/* disable progress meter, set to 0L to enable and disable debug output */
		curl_easy_setopt(InputCube->curl_handle, CURLOPT_NOPROGRESS, 1L);

		/* send all headerdata to this function*/
		/*curl_easy_setopt(InputCube->curl_handle, CURLOPT_HEADERFUNCTION,HeaderCallback);*/
	
		/* we pass our 'header' struct to the callback function */
		/*curl_easy_setopt(InputCube->curl_handle, CURLOPT_HEADERDATA, (void *)&header);*/

		/* send all data to this function*/
		curl_easy_setopt(InputCube->curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);

		curl_easy_setopt(InputCube->curl_handle, CURLOPT_FAILONERROR,1L);

		/* we pass our 'InputCube' struct to the callback function */
		curl_easy_setopt(InputCube->curl_handle, CURLOPT_WRITEDATA, (void *)InputCube);
	};
}

int StreamCube(void *userp,unsigned int ServerFormat, const char *baseurl,const char *baseName,const char *baseExt,const char *userName,const char *password,unsigned int mag, unsigned int x, unsigned int y, unsigned int z){
	char *CubeURL;
	unsigned int cubesize= 128;
	long MaxLoadingTime=10;
	CURLcode res;
	size_t BaseURLLength, baseNameLength, baseExtLength;
	int returnsignal;
	BaseURLLength=strlen(baseurl);
	baseNameLength=strlen(baseName);
	baseExtLength=strlen(baseExt);
	int doHTTPAUTH=0;

	struct MemoryStruct *InputCube = (struct MemoryStruct *)userp;
	InputCube->memory = malloc(1);/* will be grown as needed by the realloc above */
	InputCube->size = 0;/* no data at this point */

	switch (ServerFormat){
		case 1:
			CubeURL=(char *)calloc((BaseURLLength+1+2+1+ 10+1+10+1 + 10+1+10+1+ 10+1+10+1 +1),sizeof(char));
			if (CubeURL==NULL){printf("Could not allocate memory for CubeURL.\n");return -1;}

			sprintf(CubeURL,"%s/%02u/%010u,%010u/%010u,%010u/%010u,%010u/",\
				baseurl,mag-1,x*cubesize,(x+1)*cubesize,y*cubesize,(y+1)*cubesize,z*cubesize,(z+1)*cubesize);
			break;
		case 2:	
			doHTTPAUTH=1;
			CubeURL=(char *)calloc((BaseURLLength+5+2+4+2+4+2+4+1+baseNameLength+5+2+4+2+4+2+4+baseExtLength+1),sizeof(char));
			if (CubeURL==NULL){printf("Could not allocate memory for CubeURL.\n");return -1;}

			sprintf(CubeURL,"%s/mag%u/x%04u/y%04u/z%04u/%s_mag%u_x%04u_y%04u_z%04u%s",\
				baseurl,mag,x,y,z,baseName,mag,x,y,z,baseExt);
			break;

		printf("ERROR: Unknown server format: %u\n",ServerFormat);
		return 0;
	};
	//printf("Load cube from url: %s\n",CubeURL);

	if(InputCube->curl_handle==NULL) {
		printf("ERROR: Could not curl_easy_init().\n");
		returnsignal= 2; /*signals time out */
	}
	else{

		/* set URL to get here */
		curl_easy_setopt(InputCube->curl_handle, CURLOPT_URL, CubeURL);
		if (doHTTPAUTH){
			curl_easy_setopt(InputCube->curl_handle, CURLOPT_USERNAME,userName);
			curl_easy_setopt(InputCube->curl_handle, CURLOPT_PASSWORD,password);
		}
		/* get it! */
		res=curl_easy_perform(InputCube->curl_handle);

		/* check for errors */
		if(res != CURLE_OK) {
			if (res==CURLE_OPERATION_TIMEDOUT){
				returnsignal=2;
			}
			else{
				returnsignal= 0;
			}
		}
		else{	
			returnsignal= 1;
		}
		if (InputCube->size==0){ returnsignal=2;};
	};
	if (CubeURL!=NULL){
		free(CubeURL);
	}
	return returnsignal;
}
