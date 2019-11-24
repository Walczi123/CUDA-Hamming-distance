
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime> 
#include <fstream>
#include <string>
#include<iostream>

#define SIZE 1000
#define SEQUENCES 100
#define BLOCK 32
#define THREATS 1024

//__global__ void sum(int* d_sum, int* d_data)
//{
//	extern __shared__ float temp[];
//	int tid = threadIdx.x;
//	temp[tid] = d_data[tid + blockIdx.x * blockDim.x];
//	for (int d = blockDim.x >> 1; d >= 1; d >>= 1) {
//		__syncthreads();
//		if (tid < d) temp[tid] += temp[tid + d];
//	}
//	if (tid == 0) d_sum[blockIdx.x] = temp[0];
//}

__global__ void HammingDistance(int *c, const int* a, const int* b ,long const int* size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; i < *size; i += stride) {
		//atomicAdd(c, a[i] != b[i]);
		//atomicAdd(c, a[i] ^ b[i]);
		a[i] != b[i] ? atomicAdd(c, 1) : 1;
	}

}

int* RandBinSeq(int n) {
	int* result = new int[n];

	for (int i = 0; i < n; i++) {
		result[i] = rand() % 2;
	}
	return result;
}

int** Many(int size, int num) {
	int** result = new int*[num];
	for (int i = 0; i < num; i++) {
		result[i] = RandBinSeq(size);
	}
	return result;
}

void writeToFile() {
	std::ofstream file;
	file.open("sequences.txt", std::ios::out);
	if (file.good() == true)
	{
		for (int i = 0; i < SEQUENCES; i++) {
			file << "[";
			for (int j = 0; j < SIZE; j++) {
				file << rand() % 2;
			}
			file << "]\n";
			file.flush();
		}
		file.close();
	}
	printf("Save in file\n");
}

int** readFromFile() {
	std::ifstream file;
	file.open("sequences.txt", std::ios::in);
	//char line [SIZE + 3] ;
	std::string line;
	int** result = new int* [SEQUENCES];
	int i = 0, l = 0;
	if (file.good() == true)
	{
		while (l<SEQUENCES) {
			int *r = new int[SIZE];
			getline(file, line);
			for (i = 1; line[i] != ']'; i++) {
				r[i-1] = (line[i] - '0');
			}
			result[l++] = r;
		}
		file.close();
	}
	printf("Load from file\n");
	return result;
}


int main()
{
	srand(time(NULL));
	//srand(1);
	//writeToFile();
	int** seq = readFromFile();
	long  int* size = (long int*)malloc(sizeof(long int));
	*size = SIZE;
	int* c = new int[SEQUENCES * (SEQUENCES - 1)];


	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	long int* sizeC = 0;
	cudaError_t cudaStatus;

	//cudaStatus = cudaMalloc((void**)&dev_c, sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_c, (SEQUENCES * (SEQUENCES - 1)) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, *size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, *size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&sizeC, sizeof(long int) * SEQUENCES);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(sizeC, size, sizeof(long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy b failed!");
		goto Error;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Launch a kernel on the GPU with one thread for each element.
	cudaEventRecord(start);
	int k = 0;
	for (int i = 0; i < SEQUENCES-1; i++) {
		for (int j = i+1; j < SEQUENCES; j++,k++) {
			cudaStatus = cudaMemcpy(dev_a, seq[i], *size * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy a failed!");
				goto Error;
			}

			cudaStatus = cudaMemcpy(dev_b, seq[j], *size * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy b failed!");
				goto Error;
			}

			//*c = 0;
			//cudaStatus = cudaMemcpy(dev_c, c, sizeof(int), cudaMemcpyHostToDevice);
			//if (cudaStatus != cudaSuccess) {
			//	fprintf(stderr, "cudaMemcpy b failed!");
			//	goto Error;
			//}
			 

			HammingDistance <<< BLOCK, THREATS >>> (dev_c+k, dev_a, dev_b, sizeC);
			//HammingDistance <<< BLOCK, THREATS >>> (dev_c, dev_a, dev_b, sizeC);

			//cudaStatus = cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
			//if (cudaStatus != cudaSuccess) {
			//	fprintf(stderr, "cudaMemcpy c failed!");
			//	goto Error;
			//}
			//printf("The Hamming distance between %d and %d seqence is %d. ", i, j, *c);
			//if (*c == 1)printf("Pair with distance equal 1");
			//printf("\n");
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching HammingDistance!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, sizeof(int) * SEQUENCES * (SEQUENCES - 1), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy c failed!");
		goto Error;
	}
	k = 0;
	int s = 0;
	for (int i = 0; i < SEQUENCES - 1; i++) {
		for (int j = i+1; j < SEQUENCES; j++,k++) {
			if (c[k] == 1) {
				s++;
				printf("Pair with distance equal 1. ");
				printf("The Hamming distance between %d and %d seqence is %d.\n", i, j, *(c + k));
			}
		}
	}
	printf("There is %d pairs with the Hamming distance equal 1\n", s);


	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time ms %f \n\n\n",milliseconds);

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	delete(c);
	delete(size);
	for (int i = 0; i < SEQUENCES; i++) {
		delete(seq[i]);
	}
	delete(seq);

    return 0;
}