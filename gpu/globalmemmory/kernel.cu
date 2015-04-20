/*****************************************************************************
 * Copyright (c) 2014-2015 The Parallel Search Team as listed in CREDITS.txt *
 * http://health-tourism.cpe.ku.ac.th/parallelsearch                         *
 *                                                                           *
 * This file is part of ParallelSearch                                       *
 * ParallelSearch is available under multiple licenses.                      *
 * The different licenses are subject to terms and condition as provided     *
 * in the files specifying the license. See "LICENSE.txt" for details        *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 * ParallelSearch is free software: you can redistribute it and/or modify    *
 * it under the terms of the GNU General Public License as published by      *
 * the Free Software Foundation, either version 3 of the License, or         *
 * (at your option) any later version. See "LICENSE-gpl.txt" for details.    *
 *                                                                           *
 * ParallelSearch is distributed in the hope that it will be useful,         *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the              *
 * GNU General Public License for more details.                              *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 * For non-commercial academic use see the license specified in the file     *
 * "LICENSE-academic.txt".                                                   *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 * If you are interested in other licensing models, including a commercial-  *
 * license, please contact the author at;                                    *
 * Chantana Chantrapornchai <fengcnc@ku.ac.th>                               *
 * Chidchanok Choksuchat <cchoksuchat@hotmail.com>                           *
 *                                                                           *
 *****************************************************************************/
  
///
/// \author Chantana Chantrapornchai <fengcnc@ku.ac.th>
/// \author Chidchanok Choksuchat <cchoksuchat@hotmail.com>
///

/// Version: parallel search on GPU

#define _LFS_LARGEFILE          1
#define _LFS64_LARGEFILE        1
#define _LFS64_STDIO			1
#define _LARGEFILE64_SOURCE    	1

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_SIZE 1014
#define MAX_THREAD_PER_BLOCK 1014
#define MAX 100

long long unsigned total_sub,total_data;
clock_t t_sub1,t_sub2,t_data1,t_data2;
const unsigned long long chunkSize = (1<<20);

unsigned long work_per_thread = 100;
char *pattern_arr[MAX];
int *count_found[MAX];
int total_pattern;
long unsigned total_found;
int TOTAL_THREADS_PER_BLOCK ;
int Rround=0;

__global__ void searchb(char* data, char* pattern, int len_data,int len_substring, bool*pos, unsigned long work_size)
{  

	//For all blocks
	unsigned long k;
	int j,i =blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * gridDim.x;

	for (; i < len_data; i+=numThreads  ) {
	

	if (data[i] == pattern[0]) {
	for ( j=1; i+j < len_data && j<len_substring; j++) {
	if (data[i+j] != pattern[j])     
		break;     
	 }

	if (j==len_substring) {
		 pos[i] =  true;
	}
	else  pos[i] =  false;	   
      }	  
   }
}//end of GPU Kernel

	 FILE* f_b, *pFile = NULL;
	 unsigned long long fileSize = 0;

	 size_t currByte=0;
 	  
	unsigned long long filesize(const char *filename)
	{
	FILE *f = fopen(filename,"rb");  /* open the file in read only */

		if (fseek(f,0,SEEK_END)==0) /* seek was successful for Linux*/
			fileSize = ftell(f);
		fclose(f);
		printf("fileSize = %llu", fileSize);
		return fileSize;
	}

	 int countR=0;

	long unsigned count_total_found(bool *arr, int n)
	{
		 int i;
		 long unsigned c=0;
		 for (i=0; i < n; i++)
			 if (arr[i]) c++;
		 return c;
	}


int main(int argc, char** argv)
{
    printf("start\n");
	int cuda_device = 0; // device ID
        long dposSize =0;
	int mb=0;           // pattern size bit S
	int nb = 0;           // number of ints in the bit data set
	int j,k;


	//start Timer
	cudaError_t error;   // capture returned error code
	cudaEvent_t start_event, stop_event; // data structures to capture events in GPU
	float time_main_b;
	double total_time_main_b=0.0;
	
	// Sanity checks
	{
	    // check the compute capability of the device A
		int num_devices=0;

		cudaGetDeviceCount(&num_devices) ;
		if(0==num_devices)
	    {
	        printf("your system does not have a CUDA capable A device\n");
	        return 1;
	    }
    	if( argc > 1 )
       		cuda_device = atoi( argv[1] );

	    // check if the command-line chosen device ID is within range, exit if not
	    if( cuda_device >= num_devices )
	    {
	        printf("choose device ID between 0 and %d\n", num_devices-1);
	        return 1;
	    }

    	cudaSetDevice( cuda_device );

		if ( argc < 4 ) {
      		printf("Usage: StringmatchingGPU <device_number> <data_file_b> <string_pattern1-..99>\n");
      		return -1;
    	}
	} // end of safe checks

	//Cuda Device 
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, cuda_device);
	if( (deviceProp.major == 2) && (deviceProp.minor < 2)){ 
		printf("\n%s does not have compute capability 2.2 or later\n",deviceProp.name);}
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cuda_device);
	printf(" num SMs %d\n",numSMs);

	//OpenFile
	if ((f_b = fopen(argv[2] , "r")) == NULL ) { 
	printf("Error : read file\n"); 
	return 0; 
	}

	unsigned long long currSize=fileSize;
	long double total_diff2 = 0.0;	
	long double total_time_data = 0.0, total_time_pat =0.0, total_time_pos=0.0;



	while (currSize>chunkSize){
		 currSize=(unsigned long) (currSize-chunkSize);
	 	countR++;
	}
   
	//Substring
	char* subString_b = (char*)malloc( (strlen(argv[3]) + 1) * sizeof(char) );
	strcpy(subString_b, argv[3]);
	 	
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	mb=0;
	for (j=3; j < argc; j++)
	 {
		  pattern_arr[total_pattern] = (char*)malloc( (strlen(argv[j]) + 1) * sizeof(char) ); 
		  count_found[total_pattern] = (int *) malloc( 2*sizeof(int));
		  count_found[total_pattern]=0;
		
		  strcpy(pattern_arr[total_pattern],argv[j]);
		  printf("pattern= %s \n",pattern_arr[total_pattern]);
		   mb= (mb > strlen(pattern_arr[total_pattern])? mb : strlen(pattern_arr[total_pattern]));
	
		  total_pattern++;
	 }

	char* mainString_b;
	char* d_data_b = 0,*data_b;
	bool* pos=false;
	bool* d_pos=false; 
	//Device's text


	// allocate Device's memory for substring
	char* d_substr_b = 0;

	// read in the filename and string pattern to be searched
	int alloc_size =(chunkSize+mb-1)*sizeof(char);
	int countc;
	unsigned int cur_size,my_size;
	char *cur_p,*next_p;
    
        data_b = (char *) malloc((chunkSize+mb-1)*sizeof(char));
	pos = (bool *) malloc((chunkSize+mb-1)*sizeof(bool));
	cudaMalloc((void**)&d_pos,(chunkSize+mb-1)*sizeof(bool));//
	if (d_pos == NULL)
		  printf("couldn't allocate d_pos\n");
	          dposSize = dposSize+(long) pos;
		  cudaMalloc((void**)&d_data_b, alloc_size) ;//
	
	if (d_data_b == NULL)
		  printf("couldn't allocate d_data_b\n");

	cudaMalloc((void**)&d_substr_b, (strlen(subString_b))*sizeof(char));

	size_t cur_free, cur_total;

	printf("\n");

	cudaMemGetInfo(&cur_free,&cur_total); 

	printf("%ld KB free of total %ld KB\n",cur_free/1024,cur_total/1024);


	while ( (countc=fread(data_b,sizeof(char),(chunkSize+mb-1),f_b))>0){
	mainString_b = data_b;
	nb = (int) countc/sizeof(char);
	nb= nb-(mb-1);
	printf("size read (byte) %d ", nb); 
		

		TOTAL_THREADS_PER_BLOCK = MAX_THREAD_PER_BLOCK ;

		
		dim3 threadsPerBlocks(TOTAL_THREADS_PER_BLOCK, 1);
		dim3 numBlocks((int)ceil((double)nb/TOTAL_THREADS_PER_BLOCK), 1);

		work_per_thread =(unsigned long) (ceil ((double) BLOCK_SIZE/TOTAL_THREADS_PER_BLOCK));

		printf("numblock %d  thread perblock %d work perThread %ld\n", numBlocks.x, threadsPerBlocks.x, work_per_thread);
		   
		if (work_per_thread <=0) work_per_thread = 1;


		t_data1= clock();
		cudaMemcpy(d_data_b, data_b, (nb+(mb-1)), cudaMemcpyHostToDevice );
		t_data2= clock();
	        long double diff2 = (((double)t_data2 - (double)t_data1) / CLOCKS_PER_SEC) *1000;
	
	        printf("timeCopyH2D-1 %Lf ms \n",diff2);
                Rround++;
                total_diff2 += diff2;
                total_time_data += diff2;

		// start timer!
		// using Kernel
		for (j=0; j < total_pattern; j++) {
			memset(pos,false,nb);
			cudaMemset(d_pos,false,nb);
			t_data1= clock();
			cudaMemcpy(d_substr_b, pattern_arr[j], sizeof(char)*(strlen(pattern_arr[j])), cudaMemcpyHostToDevice) ;
			
			t_data2= clock();
	                diff2 = (((long double)t_data2 - (double)t_data1) / CLOCKS_PER_SEC) *1000;
		        printf("timeCopyH2D-2 %Lf ms \n",diff2);

			Rround++;
	                total_diff2 += diff2;
		        total_time_pat += diff2;

			
			 cudaEventRecord(start_event, 0);
			
			 searchb <<<32*numSMs,1024>>>(d_data_b, d_substr_b,nb,strlen(pattern_arr[j]),d_pos, work_per_thread  );
			 
			cudaEventRecord(stop_event, 0);
			cudaEventSynchronize( stop_event );
			//Calculate time
			cudaEventElapsedTime( &time_main_b, start_event, stop_event );

			//Getting Error 
			error = cudaGetLastError();
			if ( error ) { 	printf("Error caught: %s\n", cudaGetErrorString( error ));}
		
			t_data1 =clock();
			cudaMemcpy(pos, d_pos, nb, cudaMemcpyDeviceToHost) ; // result position
			 t_data2= clock();
		     
			diff2 = (((long double)t_data2 - (double)t_data1) / CLOCKS_PER_SEC) *1000;
			 printf("timeCopyH2D-3 %Lf ms \n",diff2);
 			 Rround++;
                         total_diff2 += diff2;
                         total_time_pos +=diff2;

			//Print Time
			printf("timeOfMainSearch %f ms ", time_main_b);
			total_time_main_b += time_main_b;
			int t_f =count_total_found(pos,nb);
			printf(" cur_found %d  \n", t_f);
			total_found += t_f;
			// cleanup
			
		}
		// stop timer
	
		if (!feof(f_b)) fseeko(f_b,-((long long)mb-1),SEEK_CUR);
		else break;
             

		}//end while main stream
		
                //Free Substring
		cudaFree(d_substr_b);
		free(subString_b);
	
	
		//Free Input

		free(data_b);

		cudaFree(d_data_b); 
		cudaFree(d_pos); 

		cudaEventDestroy( start_event ); 
		cudaEventDestroy( stop_event ); 
               
		free(pos);

		for (j=0; j < total_pattern; j++) 
                { free(pattern_arr[j]); 
                  free(count_found[j]); }
		fclose(f_b);

		printf("\nEnd");
		return 0;
}
