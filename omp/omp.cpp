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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define MAX_THREADS 32

FILE* f_b;
long long unsigned total_found=0;
clock_t t1,t2;
int max_str;
unsigned long *pos;
unsigned long chunkSize =1<<30;
void  string_match(char* data, char* pattern, long unsigned len_data, int len_substring )
{  int j,i,k;
    i=0;

	#pragma omp parallel for private(j,k)
	for ( i=0;  i<len_data ; i++) 
	{
    int j;
		for ( j=0; i < len_data && j<len_substring; j++) {
			 if (data[i+j] != pattern[j])     break;
			}
	if(j== len_substring) {
		 k = omp_get_thread_num();
		 pos[k]++;
	  }
	 	
	}
	return ;
}

#define MAX 100
char *pattern_arr[MAX];
int total_pattern;
float total_diff=0;
int total_threads;

long unsigned  count_total_found (long unsigned *p, int numOfPos)
{
	   int i;
	   long unsigned totalPos=0;

	   for (i=0; i < numOfPos; i++)
		       totalPos += pos[i];
	   return totalPos;
}

int main(int argc, char** argv)
{    long unsigned nb,countc;
	 char *subString_b, *data_b;
	 int j,mb;
	 long unsigned count_c;

	if (argc < 3) {
		printf(" Error: require three arguments <filename> <pattern> <pattern2> ..\n"); 
		return 0; 
	}
	
	if ((f_b = fopen(argv[1] , "r")) == NULL ) { 
		printf("Error : read the dataset file\n"); 
		return 0; 
		}

	 max_str=0;

	 for (j=2; j < argc; j++)
	 {
		  pattern_arr[total_pattern] = (char*)malloc( (strlen(argv[j]) + 1) * sizeof(char) ); 
		  strcpy(pattern_arr[total_pattern],argv[j]);
		  printf("p%d %s ",j-1,pattern_arr[total_pattern]);
		  max_str = (max_str > strlen(pattern_arr[total_pattern])? max_str: strlen(pattern_arr[total_pattern]));
		  total_pattern++;
	 }

	 	subString_b = (char*)malloc( (max_str) * sizeof(char) );
		printf("Total Pattern= %d",total_pattern);
		int alloc_size =((max_str)+chunkSize)*sizeof(char);
		data_b = (char*)malloc( ((max_str)+chunkSize) * sizeof(char));

	if (data_b==NULL) {
		printf("Error allocate memory for data set\n"); 
		return -1; }

		pos = (long unsigned *)malloc( (MAX_THREADS) * sizeof(long unsigned)); 
		// Assume max of threads = 32
	
	if (pos==NULL) {
		printf("Error allocate memory for the results' positions\n"); 
		return -1; 
		}
	
	while (!feof(file_b))  
		{
		countc= (long unsigned) fread(data_b,sizeof(char),(max_str-1)+chunkSize,file_b);
		if (countc <= 0 ) break;
		if( ferror( file_b ) )      {
			perror( "Read error" );
			break;
		}

		nb = (long unsigned) countc/sizeof(char);
		printf("\n Read %ld %ld ", nb,countc); 

		if (!feof(f_b)) fseeko(f_b,-(long long)(max_str-1),SEEK_CUR);
		 
		int procs = omp_get_max_threads();
		omp_set_num_threads(procs);
		total_threads =omp_get_num_threads();
		
		printf("Processor %d %d\n",procs,total_threads);
		t1= clock();
		for (j=0; j < total_pattern; j++) {
			strcpy(subString_b, pattern_arr[j]);
			mb = strlen(subString_b);
			memset(pos,0,MAX_THREADS);
			string_match(data_b,subString_b, nb, mb);  
			printf("Keyword(s) %d %s ",j,subString_b);
			total_found += count_total_found(pos,MAX_THREADS); 
			printf(" Found %ld ", total_found);
		}
		t2=clock();

		float diff = (((float)t2 - (float)t1) / CLOCKS_PER_SEC) *1000;
		total_diff += diff;
		 
		if (feof(f_b) || countc <=0 )
		 break;
		
	}

	fclose(f_b);
	free(data_b);
	free(subString_b);
	free(pos);
	
	for (j=0; j < total_pattern; j++) free(pattern_arr[j]);
	printf("\n Total found %ld",total_found);
	printf("  Time %lf ", total_diff);
	return 0;
}