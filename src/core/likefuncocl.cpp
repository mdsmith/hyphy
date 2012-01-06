// *********************************************************************
// OpenCL likelihood function Notes:
//
// Runs computations with OpenCL on the GPU device and then checks results
// against basic host CPU/C++ computation.
//
//
// *********************************************************************

#ifdef MDSOCL

#include <string>
#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "calcnode.h"

#include <opencl_kernels.h>

//#define FLOAT
//#define OCLVERBOSE

const unsigned int MAX_GPU_COUNT = 8;

#if defined(__APPLE__) || defined(APPLE)
#include <OpenCL/OpenCL.h>
typedef float fpoint;
typedef cl_float clfp;
#define FLOATPREC "typedef float fpoint; \n"
//#define PRAGMADEF "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n"
#define PRAGMADEF " \n"
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(__NVIDIAOCL__)
//#define __GPUResults__
#define __OCLPOSIX__
//#include <oclUtils.h>
#include <CL/opencl.h>
typedef double fpoint;
typedef cl_double clfp;
#define FLOATPREC "typedef double fpoint; \n"
#define PRAGMADEF "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n"
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(__AMDOCL__) 
//#define __GPUResults__
#define __OCLPOSIX__
#include <CL/opencl.h>
typedef double fpoint;
typedef cl_double clfp;
#define FLOATPREC "typedef double fpoint; \n"
#define PRAGMADEF "#pragma OPENCL EXTENSION cl_amd_fp64: enable \n"
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#elif defined(FLOAT)
#include <CL/opencl.h>
typedef float fpoint;
typedef cl_float clfp;
#define FLOATPREC "typedef float fpoint; \n"
#define PRAGMADEF " \n"
#endif

//#define __VERBOSE__
#define OCLGPU
#ifdef OCLGPU
#define OCLTARGET " #define BLOCK_SIZE 16 \n"
#else
#define OCLTARGET " #define BLOCK_SIZE 1 \n"
#endif

#ifdef __GPUResults__
#define OCLGPUResults " #define __GPUResults__ \n"
#else
#define OCLGPUResults " \n"
#endif


// #define MIN(a,b) ((a)>(b)?(b):(a))

// time stuff:
#define BILLION 1E9
struct timespec mainStart, mainEnd, bufferStart, bufferEnd, queueStart, queueEnd, setupStart, setupEnd;
double mainSecs;
double buffSecs;
double queueSecs;
double setupSecs;

bool clean;

cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_command_queue commandQueues[MAX_GPU_COUNT];
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id* cdDevices;          // OpenCL device
cl_device_id cdDevice;          // OpenCL device
cl_uint ciDeviceCount;
int deviceNr[MAX_GPU_COUNT];
cl_program cpMLProgram;
//cl_program cpLeafProgram;
//cl_program cpInternalProgram;
//cl_program cpAmbigProgram;
//cl_program cpResultProgram;
cl_kernel ckLeafKernel[MAX_GPU_COUNT];
cl_kernel ckInternalKernel[MAX_GPU_COUNT];
cl_kernel ckAmbigKernel[MAX_GPU_COUNT];
cl_kernel ckResultKernel[MAX_GPU_COUNT];
cl_kernel ckReductionKernel[MAX_GPU_COUNT];
size_t szGlobalWorkSize[2];        // 1D var for Total # of work items
size_t szLocalWorkSize[2];         // 1D var for # of work items in the work group
size_t localMemorySize;         // size of local memory buffer for kernel scratch
size_t szParmDataBytes;         // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErr1, ciErr2;          // Error code var
cl_int count;

cl_mem cmNode_cache;
cl_mem cmModel_cache;
cl_mem cmNodRes_cache;
cl_mem cmNodFlag_cache;
cl_mem cmroot_cache;
cl_mem cmroot_scalings;
cl_mem cmScalings_cache;
cl_mem cmFreq_cache;
cl_mem cmProb_cache;
cl_mem cmResult_cache;
long realSiteCount, realAlphabetDimension;
long* lNodeFlags;
_SimpleList     updateNodes,
                flatParents,
                flatNodes,
                flatCLeaves,
                flatLeaves,
                flatTree,
                theFrequencies;
_Parameter      *iNodeCache,
                *theProbs;
_SimpleList taggedInternals;
_GrowingVector* lNodeResolutions;
float scalar;
int sitesPerGPU;

void *node_cache, *nodRes_cache, *nodFlag_cache, *scalings_cache, *prob_cache, *freq_cache, *root_cache, *result_cache, *root_scalings, *model;

void _OCLEvaluator::init(   long esiteCount,
                                    long ealphabetDimension,
                                    _Parameter* eiNodeCache)
{
    clean = false;
    contextSet = false;
    realSiteCount = esiteCount;
    realAlphabetDimension = ealphabetDimension;
    iNodeCache = eiNodeCache;
    mainSecs = 0.0;
    buffSecs = 0.0;
    queueSecs = 0.0;
    setupSecs = 0.0;
    scalar = 10.0;
}

// So the two interfacing functions will be the constructor, called in SetupLFCaches, and launchmdsocl, called in ComputeBlock.
// Therefore all of these functions need to be finished, the context needs to be setup separately from the execution, the data needs
// to be passed piecewise, and a pointer needs to be passed around in likefunc2.cpp. After that things should be going a bit faster,
// though honestly this solution is geared towards analyses with a larger number of sites.

// *********************************************************************
void _OCLEvaluator::setupDevices(void)
{

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

//    printf("clGetPlatformID...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }


    //Get the devices
#ifdef OCLGPU
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Could not get number of devices");
    }
    //ciDeviceCount = 1;
    cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id));
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Could not get devices");
    }

#else
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
#endif
 //   printf("clGetDeviceIDs...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

/* Broken with multiGPU setup

    size_t maxWorkGroupSize;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                             sizeof(size_t), &maxWorkGroupSize, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Getting max work group size failed!\n");
    }
    //printf("Max work group size: %lu\n", (unsigned long)maxWorkGroupSize);

    size_t maxLocalSize;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_LOCAL_MEM_SIZE,
                             sizeof(size_t), &maxLocalSize, NULL);
    size_t maxConstSize;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                             sizeof(size_t), &maxConstSize, NULL);
    printf("LocalSize: %ld, Const size: %ld\n", (long unsigned) maxLocalSize, (long unsigned) maxConstSize);

    printf("sites: %ld\n", siteCount);

*/
    //Create the context
    cxGPUContext = clCreateContext(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErr1);
//    printf("clCreateContext...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Create a command-queue
    //cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    for (int i = 0; i < ciDeviceCount; i++)
    {
        commandQueues[i] = clCreateCommandQueue(cxGPUContext, cdDevices[i], 0, &ciErr1);
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clCreateCommandQueue #%i, Line %u in file %s !!!\n\n", i,__LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
    
    }
    printf("Number of GPUs in use: %u\n", ciDeviceCount);
//    printf("clCreateCommandQueue...\n");
}


int _OCLEvaluator::setupContext(int siteCount, int alphabetDimension)
{
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &setupStart);
#endif

    long nodeFlagCount = flatLeaves.lLength*siteCount;
    long nodeResCount = lNodeResolutions->GetUsed();
    int roundCharacters = roundUpToNextPowerOfTwo(alphabetDimension);
    //printf("Got the sizes of nodeRes and nodeFlag: %i, %i\n", nodeResCount, nodeFlagCount);

    bool ambiguousNodes = true;
    if (nodeResCount == 0)
    {
        nodeResCount++;
        ambiguousNodes = false;
    }

    //node_cache      = (void*)malloc(sizeof(cl_float)*roundCharacters*siteCount*(flatNodes.lLength));
    nodRes_cache    = (void*)malloc(sizeof(cl_float)*roundUpToNextPowerOfTwo(nodeResCount));
    nodFlag_cache   = (void*)malloc(sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount));
    //scalings_cache  = (void*)malloc(sizeof(cl_int)*roundCharacters*siteCount*(flatNodes.lLength));
    prob_cache      = (void*)malloc(sizeof(cl_float)*roundCharacters);
    //freq_cache      = (void*)malloc(sizeof(cl_int)*siteCount);
    freq_cache      = (void*)malloc(sizeof(cl_int)*siteCount);
    //root_cache      = (void*)malloc(sizeof(cl_float)*siteCount*roundCharacters);
    //root_scalings   = (void*)malloc(sizeof(cl_int)*siteCount*roundCharacters);
#ifndef __pinned_reads__
    result_cache    = (void*)malloc(sizeof(clfp)*roundUpToNextPowerOfTwo(siteCount));
#endif
    model           = (void*)malloc(sizeof(cl_float)*roundCharacters*roundCharacters*(flatParents.lLength-1));

    //printf("Allocated all of the arrays!\n");
    //printf("setup the model, fixed tagged internals!\n");
    //printf("flatleaves: %ld\n", flatLeaves.lLength);
    //printf("flatParents: %ld\n", flatParents.lLength);
    //printf("flatCleaves: %i\n", flatCLeaves.lLength);
    //printf("flatNodes: %ld\n", flatNodes.lLength);
    //printf("updateNodes: %ld\n", updateNodes.lLength);
    //printf("flatTree: %ld\n", flatTree.lLength);
    //printf("nodeFlagCount: %i\n", nodeFlagCount);
    //printf("nodeResCount: %i\n", nodeResCount);

    //for (int i = 0; i < nodeCount*siteCount*alphabetDimension; i++)
    //printf("siteCount: %ld, alphabetDimension: %ld \n", siteCount, alphabetDimension);
    if (ambiguousNodes)
        for (int i = 0; i < nodeResCount; i++)
            ((float*)nodRes_cache)[i] = (float)(lNodeResolutions->theData[i]);
    for (int i = 0; i < nodeFlagCount; i++)
        ((long*)nodFlag_cache)[i] = lNodeFlags[i];
    for (int i = 0; i < siteCount; i++)
        ((int*)freq_cache)[i] = theFrequencies[i];
    for (int i = 0; i < alphabetDimension; i++)
        ((float*)prob_cache)[i] = theProbs[i];

    //printf("Created all of the arrays!\n");

    // alright, by now taggedInternals have been taken care of, and model has
    // been filled with all of the transition matrices.


    //**************************************************


    // set and log Global and Local work size dimensions

#ifdef OCLGPU
    szLocalWorkSize[0] = 16; // All of these will have to be generalized.
    szLocalWorkSize[1] = 16;
#else
    szLocalWorkSize[0] = 1; // All of these will have to be generalized.
    szLocalWorkSize[1] = 1;
#endif
    szGlobalWorkSize[0] = 64;
    szGlobalWorkSize[1] = ((siteCount + 16)/16)*16;
    //szGlobalWorkSize[1] = roundUpToNextPowerOfTwo(siteCount);
    printf("Global Work Size \t\t= %ld, %ld\nLocal Work Size \t\t= %ld, %ld\n# of Work Groups \t\t= %ld\n\n",
           (long unsigned) szGlobalWorkSize[0],
           (long unsigned) szGlobalWorkSize[1],
           (long unsigned) szLocalWorkSize[0],
           (long unsigned) szLocalWorkSize[1],
           (long unsigned) ((szGlobalWorkSize[0]*szGlobalWorkSize[1])/(szLocalWorkSize[0]*szLocalWorkSize[1])));

/*
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_VENDOR, sizeof(vendor_name),
                             vendor_name, &returned_size);
    ciErr1 |= clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(device_name),
                              device_name, &returned_size);
    assert(ciErr1 == CL_SUCCESS);
//    printf("Connecting to %s %s...\n", vendor_name, device_name);
*/



    //printf("Setup all of the OpenCL stuff!\n");

    // Allocate the OpenCL buffer memory objects for the input and output on the
    // device GMEM
    cmNode_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                    sizeof(cl_float)*roundCharacters*siteCount*(flatNodes.lLength), NULL,
                    &ciErr1);
    cmModel_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(cl_float)*roundCharacters*roundCharacters*(flatParents.lLength-1),
                    NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmScalings_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                    sizeof(cl_int)*roundCharacters*siteCount*flatNodes.lLength, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmNodRes_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(cl_float)*roundUpToNextPowerOfTwo(nodeResCount), NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmNodFlag_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount), NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmroot_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                    sizeof(cl_float)*siteCount*roundCharacters, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmroot_scalings = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                    sizeof(cl_int)*siteCount*roundCharacters, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmProb_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(cl_float)*roundCharacters, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmFreq_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(cl_float)*siteCount, NULL, &ciErr2);
    ciErr1 |= ciErr2;

#ifdef __pinned_reads__
    cmResult_cache = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                    sizeof(clfp)*roundUpToNextPowerOfTwo(siteCount), NULL, &ciErr2);
    result_cache = (fpoint*)clEnqueueMapBuffer(cqCommandQueue, cmResult_cache, CL_TRUE, 
                    CL_MAP_READ, 0, sizeof(clfp)*roundUpToNextPowerOfTwo(siteCount), 0, 
                    NULL, NULL, NULL);
#else
    cmResult_cache = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,
                    sizeof(clfp)*roundUpToNextPowerOfTwo(siteCount), NULL, &ciErr2);
#endif
    ciErr1 |= ciErr2;
//    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        switch(ciErr1)
        {
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_BUFFER_SIZE: printf("CL_INVALID_BUFFER_SIZE\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n");
        }
        Cleanup(EXIT_FAILURE);
    }

    //printf("Made all of the buffers on the device!\n");

//    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Create the program
    const char * program_source = "" OCLTARGET PRAGMADEF FLOATPREC OCLGPUResults KERNEL_STRING;

    cpMLProgram = clCreateProgramWithSource(cxGPUContext, 1, &program_source,
                                          NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    ciErr1 = clBuildProgram(cpMLProgram, 0, NULL, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    //ciErr1 = clBuildProgram(cpMLProgram, 1, &cdDevice, NULL, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break;
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }


/*
    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(cpMLProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];
    // Second call to get the log
    clGetProgramBuildInfo(cpMLProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    printf("%s", build_log);
    delete[] build_log;

    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break;
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
*/


}

void _OCLEvaluator::setupQueueContext(int siteCount, int alphabetDimension, int nodeResCount, int nodeFlagCount, int i)
{

    // Create the kernel
    //ckKernel = clCreateKernel(cpProgram, "FirstLoop", &ciErr1);
    //for (int i = 0; i < ciDeviceCount; i++)
    {

        ckLeafKernel[i] = clCreateKernel(cpMLProgram, "LeafKernel", &ciErr1);
        //printf("clCreateKernel (LeafKernel)...\n");
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
        ckAmbigKernel[i] = clCreateKernel(cpMLProgram, "AmbigKernel", &ciErr1);
        //printf("clCreateKernel (AmbigKernel)...\n");
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
        ckInternalKernel[i] = clCreateKernel(cpMLProgram, "InternalKernel", &ciErr1);
        //printf("clCreateKernel (InternalKernel)...\n");
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
        ckResultKernel[i] = clCreateKernel(cpMLProgram, "ResultKernel", &ciErr1);
        //printf("clCreateKernel (ResultKernel)...\n");
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
        ckReductionKernel[i] = clCreateKernel(cpMLProgram, "ReductionKernel", &ciErr1);
        //printf("clCreateKernel (ReductionKernel)...\n");
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }

        long tempLeafState = 1;
        long tempSiteCount = siteCount;
        long tempCharCount = alphabetDimension;
        long tempChildNodeIndex = 0;
        long tempParentNodeIndex = 0;
        long tempRoundCharCount = roundUpToNextPowerOfTwo(alphabetDimension);
        int  tempTagIntState = 0;
        int   tempNodeID = 0;
        float tempScalar = scalar;
        // this is currently ignored, 1 is hardcoded into the kernel code. 
        float tempuFlowThresh = 0.000000001f;

        ciErr1  = clSetKernelArg(ckLeafKernel[i], 0, sizeof(cl_mem), (void*)&cmNode_cache);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 1, sizeof(cl_mem), (void*)&cmModel_cache);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 3, sizeof(cl_mem), (void*)&cmNodFlag_cache);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 4, sizeof(cl_long), (void*)&tempSiteCount);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 5, sizeof(cl_long), (void*)&tempCharCount);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 6, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 7, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 8, sizeof(cl_long), (void*)&tempRoundCharCount);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 9, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 10, sizeof(cl_int), (void*)&tempNodeID); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 11, sizeof(cl_mem), (void*)&cmScalings_cache);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 12, sizeof(cl_float), (void*)&tempScalar);
        ciErr1 |= clSetKernelArg(ckLeafKernel[i], 13, sizeof(cl_float), (void*)&tempuFlowThresh);

        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 0, sizeof(cl_mem), (void*)&cmNode_cache);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 1, sizeof(cl_mem), (void*)&cmModel_cache);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 3, sizeof(cl_mem), (void*)&cmNodFlag_cache);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 4, sizeof(cl_long), (void*)&tempSiteCount);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 5, sizeof(cl_long), (void*)&tempCharCount);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 6, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 7, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 8, sizeof(cl_long), (void*)&tempRoundCharCount);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 9, sizeof(cl_int), (void*)&tempTagIntState);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 10, sizeof(cl_int), (void*)&tempNodeID);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 11, sizeof(cl_mem), (void*)&cmScalings_cache);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 12, sizeof(cl_float), (void*)&tempScalar);
        ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 13, sizeof(cl_float), (void*)&tempuFlowThresh);

        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 0, sizeof(cl_mem), (void*)&cmNode_cache);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 1, sizeof(cl_mem), (void*)&cmModel_cache);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 3, sizeof(cl_long), (void*)&tempSiteCount);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 4, sizeof(cl_long), (void*)&tempCharCount);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 5, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 6, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 7, sizeof(cl_long), (void*)&tempRoundCharCount);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 8, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 9, sizeof(cl_int), (void*)&tempNodeID); // reset this in the loop
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 10, sizeof(cl_mem), (void*)&cmroot_cache);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 11, sizeof(cl_mem), (void*)&cmScalings_cache);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 12, sizeof(cl_float), (void*)&tempScalar);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 13, sizeof(cl_float), (void*)&tempuFlowThresh);
        ciErr1 |= clSetKernelArg(ckInternalKernel[i], 14, sizeof(cl_mem), (void*)&cmroot_scalings);

        ciErr1 |= clSetKernelArg(ckResultKernel[i], 0, sizeof(cl_mem), (void*)&cmFreq_cache);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 1, sizeof(cl_mem), (void*)&cmProb_cache);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 2, sizeof(cl_mem), (void*)&cmResult_cache);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 3, sizeof(cl_mem), (void*)&cmroot_cache);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 4, sizeof(cl_mem), (void*)&cmroot_scalings);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 5, sizeof(cl_long), (void*)&tempSiteCount);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 6, sizeof(cl_long), (void*)&tempRoundCharCount);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 7, sizeof(cl_float), (void*)&tempScalar);
        ciErr1 |= clSetKernelArg(ckResultKernel[i], 8, sizeof(cl_long), (void*)&tempCharCount);

        ciErr1 |= clSetKernelArg(ckReductionKernel[i], 0, sizeof(cl_mem), (void*)&cmResult_cache);

        //printf("clSetKernelArg 0 - 12...\n\n");
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }

/* 
        ciErr1 |= clEnqueueWriteBuffer(commandQueues[i], cmNode_cache, CL_FALSE, 0,
                    sizeof(cl_float)*roundCharacters*siteCount*(flatNodes.lLength), node_cache, 0, NULL, NULL);
        ciErr1 |= clEnqueueWriteBuffer(commandQueues[i], cmScalings_cache, CL_FALSE, 0,
                    sizeof(cl_int)*roundCharacters*siteCount*(flatNodes.lLength), scalings_cache, 0, NULL, NULL);
*/
        ciErr1 |= clEnqueueWriteBuffer(commandQueues[i], cmNodRes_cache, CL_FALSE, 0,
                    sizeof(cl_float)*roundUpToNextPowerOfTwo(nodeResCount), nodRes_cache, 0, NULL, NULL);
        ciErr1 |= clEnqueueWriteBuffer(commandQueues[i], cmNodFlag_cache, CL_FALSE, 0,
                    sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount), nodFlag_cache, 0, NULL, NULL);
        ciErr1 |= clEnqueueWriteBuffer(commandQueues[i], cmProb_cache, CL_FALSE, 0,
                    sizeof(cl_float)*roundCharacters, prob_cache, 0, NULL, NULL);
        ciErr1 |= clEnqueueWriteBuffer(commandQueues[i], cmFreq_cache, CL_FALSE, 0,
                    sizeof(cl_int)*siteCount, freq_cache, 0, NULL, NULL);
        
        //printf("clEnqueueWriteBuffer (root_cache, etc.)...");
        if (ciErr1 != CL_SUCCESS)
        {
            printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
    }

#ifdef __VERBOSE__
/*
    size_t maxKernelSize;
    ciErr1 = clGetKernelWorkGroupInfo(ckLeafKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Leaf Kernel Work Group Size: %ld \n", (long unsigned) maxKernelSize);
    ciErr1 = clGetKernelWorkGroupInfo(ckAmbigKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Ambig Kernel Work Group Size: %ld \n", (long unsigned) maxKernelSize);
    ciErr1 = clGetKernelWorkGroupInfo(ckInternalKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Internal Kernel Work Group Size: %ld \n", (long unsigned) maxKernelSize);
    ciErr1 = clGetKernelWorkGroupInfo(ckResultKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Result Kernel Work Group Size: %ld \n", (long unsigned) maxKernelSize);
    ciErr1 = clGetKernelWorkGroupInfo(ckReductionKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Reduction Kernel Work Group Size: %ld \n", (long unsigned) maxKernelSize);
*/
#endif


    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back
    // Asynchronous write of data to GPU device
    //printf(" Done!\n");
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &setupEnd);
    setupSecs += (setupEnd.tv_sec - setupStart.tv_sec)+(setupEnd.tv_nsec - setupStart.tv_nsec)/BILLION;
#endif
}

void _OCLEvaluator::doLF(int siteCount, int alphabetDimension, int i)
{
    //printf("newLF!\n");
    //printf("LF");
    // so far this wholebuffer rebuild takes almost no time at all. Perhaps not true re:queue
    // Fix the model cache
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &bufferStart);
#endif
    int roundCharacters = roundUpToNextPowerOfTwo(alphabetDimension);
/*
    printf("Update Nodes:");
    for (int i = 0; i < updateNodes.lLength; i++)
    {
        printf(" %i ", updateNodes.lData[i]);
    }
    printf("\n");

    printf("Tagged Internals:");
    for (int i = 0; i < taggedInternals.lLength; i++)
    {
        printf(" %i", taggedInternals.lData[i]);
    }
    printf("\n");
*/
    long nodeCode, parentCode;
    bool isLeaf;
    _Parameter* tMatrix;
    int a1, a2;
    //printf("updateNodes.lLength: %i", updateNodes.lLength);
    //#pragma omp parallel for default(none) shared(updateNodes, flatParents, flatLeaves, flatCLeaves, flatTree, alphabetDimension, model, roundCharacters) private(nodeCode, parentCode, isLeaf, tMatrix, a1, a2)

    // rebuild the model cache and move it over to the GPU
    for (int nodeID = 0; nodeID < updateNodes.lLength; nodeID++)
    {
        nodeCode = updateNodes.lData[nodeID];
        parentCode = flatParents.lData[nodeCode];

        isLeaf = nodeCode < flatLeaves.lLength;

        if (!isLeaf) nodeCode -= flatLeaves.lLength;

        tMatrix = (isLeaf? ((_CalcNode*) flatCLeaves (nodeCode)):
                   ((_CalcNode*) flatTree    (nodeCode)))->GetCompExp(0)->theData;

        for (a1 = 0; a1 < alphabetDimension; a1++)
        {
            for (a2 = 0; a2 < alphabetDimension; a2++)
            {
                ((float*)model)[nodeID*roundCharacters*roundCharacters+a1*roundCharacters+a2] =
                   (float)(tMatrix[a1*alphabetDimension+a2]);
            }
        }
    }

    // enqueueing the read and write buffers takes 1/2 the time, the kernel takes the other 1/2.
    // with no queueing, however, we still only see ~700lf/s, which isn't much better than the threaded CPU code.

    //for (int i = 0; i < ciDeviceCount; i++)
    {
        ciErr1 |= clEnqueueWriteBuffer(commandQueues[i], cmModel_cache, CL_FALSE, 0,
                    sizeof(cl_float)*roundCharacters*roundCharacters*(flatParents.lLength-1),
                    model, 0, NULL, NULL);
    }

#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &bufferEnd);
    buffSecs += (bufferEnd.tv_sec - bufferStart.tv_sec)+(bufferEnd.tv_nsec - bufferStart.tv_nsec)/BILLION;

    clock_gettime(CLOCK_MONOTONIC, &queueStart);
#endif
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_MEM_OBJECT: printf("CL_INVALID_MEM_OBJECT\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
                //          case   CL_MISALIGNED_SUB_BUFFER_OFFSET: printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
                //          case   CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    /*
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    */
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &queueStart);
#endif
    //printf("Finished writing the model stuff\n");
    // Launch kernel
    for (int nodeIndex = 0; nodeIndex < updateNodes.lLength; nodeIndex++)
    {
        //printf("NewNode\n");
        long    nodeCode = updateNodes.lData[nodeIndex],
                parentCode = flatParents.lData[nodeCode];

        //printf("NewNode: %i, NodeCode: %i\n", nodeIndex, nodeCode);
        bool isLeaf = nodeCode < flatLeaves.lLength;

        if (isLeaf)
        {
            long nodeCodeTemp = nodeCode;
            int tempIntTagState = taggedInternals.lData[parentCode];
            int ambig = 0;
            for (int aI = 0; aI < siteCount; aI++)
                if (lNodeFlags[nodeCode*siteCount + aI] < 0)
                    {
                        ambig = 1;
                        break;
                    }
            //for (int i = 0; i < ciDeviceCount; i++)
            {
                if (!ambig)
                {
                    ciErr1 |= clSetKernelArg(ckLeafKernel[i], 6, sizeof(cl_long), (void*)&nodeCodeTemp);
                    ciErr1 |= clSetKernelArg(ckLeafKernel[i], 7, sizeof(cl_long), (void*)&parentCode);
                    ciErr1 |= clSetKernelArg(ckLeafKernel[i], 9, sizeof(cl_int), (void*)&tempIntTagState);
                    ciErr1 |= clSetKernelArg(ckLeafKernel[i], 10, sizeof(cl_int), (void*)&nodeIndex);

                    //printf("Leaf!\n");
    #ifdef __VERBOSE__
                    printf("Leaf/Ambig Started (ParentCode: %i)...", parentCode);
    #endif
                    ciErr1 |= clEnqueueNDRangeKernel(commandQueues[i], ckLeafKernel[i], 2, NULL,
                                                    szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
                }
                else
                {
                    ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 6, sizeof(cl_long), (void*)&nodeCodeTemp);
                    ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 7, sizeof(cl_long), (void*)&parentCode);
                    ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 9, sizeof(cl_int), (void*)&tempIntTagState);
                    ciErr1 |= clSetKernelArg(ckAmbigKernel[i], 10, sizeof(cl_int), (void*)&nodeIndex);

                    //printf("ambig!\n");
    #ifdef __VERBOSE__
                    printf("Leaf/Ambig Started ...");
    #endif
                    ciErr1 |= clEnqueueNDRangeKernel(commandQueues[i], ckAmbigKernel[i], 2, NULL,
                                                    szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
                }
                ciErr1 |= clFlush(commandQueues[i]);
            }
            taggedInternals.lData[parentCode] = 1;
#ifdef __VERBOSE__
            printf("Finished\n");
#endif
        }
        else
        {
            long tempLeafState = 0;
            nodeCode -= flatLeaves.lLength;
            long nodeCodeTemp = nodeCode;
            int tempIntTagState = taggedInternals.lData[parentCode];
            //for (int i = 0; i < ciDeviceCount; i++)
            {
                ciErr1 |= clSetKernelArg(ckInternalKernel[i], 5, sizeof(cl_long), (void*)&nodeCodeTemp);
                ciErr1 |= clSetKernelArg(ckInternalKernel[i], 6, sizeof(cl_long), (void*)&parentCode);
                ciErr1 |= clSetKernelArg(ckInternalKernel[i], 8, sizeof(cl_int), (void*)&tempIntTagState);
                ciErr1 |= clSetKernelArg(ckInternalKernel[i], 9, sizeof(cl_int), (void*)&nodeIndex);
    #ifdef __VERBOSE__
                printf("Internal Started (ParentCode: %i)...", parentCode);
    #endif
                ciErr1 |= clEnqueueNDRangeKernel(commandQueues[i], ckInternalKernel[i], 2, NULL,
                                                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);

                //printf("internal!\n");
                ciErr1 |= clFlush(commandQueues[i]);
            }
            taggedInternals.lData[parentCode] = 1;
#ifdef __VERBOSE__
            printf("Finished\n");
#endif
        }
        if (ciErr1 != CL_SUCCESS)
        {
            printf("%i\n", ciErr1); //prints "1"
            switch(ciErr1)
            {
                case   CL_INVALID_PROGRAM_EXECUTABLE: printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
                case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
                case   CL_INVALID_KERNEL: printf("CL_INVALID_KERNEL\n"); break;
                case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
                case   CL_INVALID_KERNEL_ARGS: printf("CL_INVALID_KERNEL_ARGS\n"); break;
                case   CL_INVALID_WORK_DIMENSION: printf("CL_INVALID_WORK_DIMENSION\n"); break;
                case   CL_INVALID_GLOBAL_WORK_SIZE: printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;
                case   CL_INVALID_GLOBAL_OFFSET: printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
                case   CL_INVALID_WORK_GROUP_SIZE: printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
                case   CL_INVALID_WORK_ITEM_SIZE: printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
                case   CL_INVALID_IMAGE_SIZE: printf("CL_INVALID_IMAGE_SIZE\n"); break;
                case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
                case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
                case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
                case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
                default: printf("Strange error\n"); //This is printed
            }
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
    }

    //for (int i = 0; i < ciDeviceCount; i++)
    {
        #ifdef __GPUResults__
            size_t szGlobalWorkSize2 = 256;
            size_t szLocalWorkSize2 = 256;
            ciErr1 |= clEnqueueNDRangeKernel(commandQueues[i], ckResultKernel, 1, NULL,
                &szGlobalWorkSize2, &szLocalWorkSize2, 0, NULL, NULL);
            ciErr1 |= clEnqueueNDRangeKernel(commandQueues[i], ckReductionKernel, 1, NULL,
                &szGlobalWorkSize2, &szLocalWorkSize2, 0, NULL, NULL);
        #else
            ciErr1 |= clEnqueueNDRangeKernel(commandQueues[i], ckResultKernel[i], 2, NULL,
                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        #endif
    }
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM_EXECUTABLE: printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
            case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
            case   CL_INVALID_KERNEL: printf("CL_INVALID_KERNEL\n"); break;
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_KERNEL_ARGS: printf("CL_INVALID_KERNEL_ARGS\n"); break;
            case   CL_INVALID_WORK_DIMENSION: printf("CL_INVALID_WORK_DIMENSION\n"); break;
            case   CL_INVALID_GLOBAL_WORK_SIZE: printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;
            case   CL_INVALID_GLOBAL_OFFSET: printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
            case   CL_INVALID_WORK_GROUP_SIZE: printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
            case   CL_INVALID_WORK_ITEM_SIZE: printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
                //          case   CL_MISALIGNED_SUB_BUFFER_OFFSET: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            case   CL_INVALID_IMAGE_SIZE: printf("CL_INVALID_IMAGE_SIZE\n"); break;
            case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
}

double _OCLEvaluator::oclmain(int siteCount, int alphabetDimension)
{  

    for (int i = 0; i < ciDeviceCount; i++)
        doLF(siteCount, alphabetDimension, i);

    for (int i = 0; i < ciDeviceCount; i++)
        clFinish(commandQueues[i]);
    // Synchronous/blocking read of results, and check accumulated errors
#ifdef __GPUResults_
    //ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
     //       sizeof(cl_double)*roundUpToNextPowerOfTwo(siteCount), result_cache, 0,
      //      NULL, NULL);
    ciErr1 = clEnqueueReadBuffer(commandQueues[0], cmResult_cache, CL_FALSE, 0,
            sizeof(clfp), result_cache, 0,
            NULL, NULL);
    //ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
     //       sizeof(cl_double)*1, result_cache, 0,
      //      NULL, NULL);
#else
    //ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
     //       sizeof(cl_float)*roundUpToNextPowerOfTwo(siteCount), result_cache, 0,
      //      NULL, NULL);
    //ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
     //       sizeof(cl_float)*roundUpToNextPowerOfTwo(siteCount), result_cache, 0,
      //      NULL, NULL);
    //ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
    //        sizeof(clfp)*roundUpToNextPowerOfTwo(siteCount), result_cache, 0,
     //       NULL, NULL);
    ciErr1 = clEnqueueReadBuffer(commandQueues[0], cmResult_cache, CL_TRUE, 0,
            sizeof(clfp)*siteCount, result_cache, 0,
            NULL, NULL);
#endif
/*
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
            sizeof(clfp)*roundUpToNextPowerOfTwo(siteCount), result_cache, 0,
            NULL, NULL);
*/
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_MEM_OBJECT: printf("CL_INVALID_MEM_OBJECT\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
                //          case   CL_MISALIGNED_SUB_BUFFER_OFFSET: printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
                //          case   CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    //--------------------------------------------------------


    for (int i = 0; i < ciDeviceCount; i++)
        clFinish(commandQueues[i]);
    double oResult = 0.0;

#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &queueEnd);
    queueSecs += (queueEnd.tv_sec - queueStart.tv_sec)+(queueEnd.tv_nsec - queueStart.tv_nsec)/BILLION;
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
#endif

#ifdef __GPUResults__
	/*
    for (int i = 0; i < siteCount; i++)
    {
        oResult += ((fpoint*)result_cache)[i];
    }
#ifdef __VERBOSE__
    printf("Result_Cache: \n");
    for (int i = 0; i < siteCount; i++)
        printf("%4.10g ", ((fpoint*)result_cache)[i]);
    printf("\n\n");
#endif
    for (int i = 0; i < siteCount; i++)
    {
        oResult += ((float*)result_cache)[i];
    }
    //oResult = ((double*)result_cache)[0];
    printf("Result_Cache: \n");
    for (int i = 0; i < siteCount; i++)
        printf("%4.10g ", ((fpoint*)result_cache)[i]);
    printf("\n\n");
	*/
    oResult = ((fpoint*)result_cache)[0];
#else
    //#pragma omp parallel for reduction (+:oResult) schedule(static)
    for (int i = 0; i < siteCount; i++)
    {
        oResult += ((fpoint*)result_cache)[i];
    }
#ifdef __VERBOSE__
    printf("Result_Cache: \n");
    for (int i = 0; i < siteCount; i++)
        printf("%4.10g ", ((float*)result_cache)[i]);
    printf("\n\n");
#endif
#endif
/*
*/
/*
    //#pragma omp parallel for reduction (+:oResult) schedule(static)
    for (int i = 0; i < siteCount; i++)
    {
        oResult += ((float*)result_cache)[i];
    }
    printf("Result_Cache: \n");
    for (int i = 0; i < siteCount; i++)
        printf("%4.10g ", ((float*)result_cache)[i]);
    printf("\n\n");
*/
/*
    //printf("! ");
    //return result;
    printf("oResult: %4.10g, gpuResult: %4.10g\n", oResult, ((double*)result_cache)[4]);
    if (oResult != ((double*)result_cache)[4])
    {
        printf("Result_Cache: \n");
        //for (int i = 0; i < roundUpToNextPowerOfTwo(siteCount); i++)
        for (int i = 0; i < 5; i++)
            printf("%4.10g ", ((double*)result_cache)[i]);
        printf("\n\n");
    }
*/
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
    mainSecs += (mainEnd.tv_sec - mainStart.tv_sec)+(mainEnd.tv_nsec - mainStart.tv_nsec)/BILLION;
#endif
    return oResult;
}


double _OCLEvaluator::launchmdsocl( _SimpleList& eupdateNodes,
                                    _SimpleList& eflatParents,
                                    _SimpleList& eflatNodes,
                                    _SimpleList& eflatCLeaves,
                                    _SimpleList& eflatLeaves,
                                    _SimpleList& eflatTree,
                                    _Parameter* etheProbs,
                                    _SimpleList& etheFrequencies,
                                    long* elNodeFlags,
                                    _SimpleList& etaggedInternals,
                                    _GrowingVector* elNodeResolutions)
{
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
#endif


    updateNodes = eupdateNodes;
    taggedInternals = etaggedInternals;
    theFrequencies = etheFrequencies;


    if (!contextSet)
    {
        setupDevices();
        theProbs = etheProbs;
        flatNodes = eflatNodes;
        flatCLeaves = eflatCLeaves;
        flatLeaves = eflatLeaves;
        flatTree = eflatTree;
        flatParents = eflatParents;
        lNodeFlags = elNodeFlags;
        lNodeResolutions = elNodeResolutions;
        setupContext(realSiteCount, realAlphabetDimension);
        for (int i = 0; i < ciDeviceCount; i++)
            setupQueueContext(realSiteCount/ciDeviceCount, realAlphabetDimension, i);
        contextSet = true;
    }

#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
    mainSecs += (mainEnd.tv_sec - mainStart.tv_sec)+(mainEnd.tv_nsec - mainStart.tv_nsec)/BILLION;
#endif

    return oclmain();
}


void _OCLEvaluator::Cleanup (int iExitCode)
{
    if (!clean)
    {
        printf("Time in main: %.4lf seconds\n", mainSecs);
        printf("Time in updating transition buffer: %.4lf seconds\n", buffSecs);
        printf("Time in queue: %.4lf seconds\n", queueSecs);
        printf("Time in Setup: %.4lf seconds\n", setupSecs);
        // Cleanup allocated objects
        printf("Starting Cleanup...\n\n");
        //if(cpLeafProgram)clReleaseProgram(cpLeafProgram);
        //if(cpInternalProgram)clReleaseProgram(cpInternalProgram);
        //if(cpAmbigProgram)clReleaseProgram(cpAmbigProgram);
        //if(cpResultProgram)clReleaseProgram(cpResultProgram);
        if(cpMLProgram)clReleaseProgram(cpMLProgram);
        if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
        for (int i = 0; i < ciDeviceCount; i++)
        {
            if(ckLeafKernel[i])clReleaseKernel(ckLeafKernel[i]);
            if(ckInternalKernel[i])clReleaseKernel(ckInternalKernel[i]);
            if(ckAmbigKernel[i])clReleaseKernel(ckAmbigKernel[i]);
            if(ckResultKernel[i])clReleaseKernel(ckResultKernel[i]);
            if(ckReductionKernel[i])clReleaseKernel(ckReductionKernel[i]);
            if(commandQueues[i])clReleaseCommandQueue(commandQueues[i]);
        }
        printf("Halfway...\n\n");
        if(cxGPUContext)clReleaseContext(cxGPUContext);

        if(cmNode_cache)clReleaseMemObject(cmNode_cache);
        if(cmModel_cache)clReleaseMemObject(cmModel_cache);
        if(cmNodRes_cache)clReleaseMemObject(cmNodRes_cache);
        if(cmNodFlag_cache)clReleaseMemObject(cmNodFlag_cache);
        if(cmroot_cache)clReleaseMemObject(cmroot_cache);
        if(cmroot_scalings)clReleaseMemObject(cmroot_scalings);
        if(cmScalings_cache)clReleaseMemObject(cmScalings_cache);
        if(cmFreq_cache)clReleaseMemObject(cmFreq_cache);
        if(cmProb_cache)clReleaseMemObject(cmProb_cache);
        if(cmResult_cache)clReleaseMemObject(cmResult_cache);

        /*
        clGetMemObjectInfo(cmNode_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount nodeCache = %d\n", count);
        clGetMemObjectInfo(cmModel_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount ModelCache = %d\n", count);
        clGetMemObjectInfo(cmNodRes_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount nodResCache = %d\n", count);
        clGetMemObjectInfo(cmNodFlag_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount nodFlagCache = %d\n", count);
        clGetMemObjectInfo(cmroot_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount rootCache = %d\n", count);
        clGetMemObjectInfo(cmroot_scalings, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount cmroot_scalings = %d\n", count);
        clGetMemObjectInfo(cmScalings_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount cmScalings_cache = %d\n", count);
        clGetMemObjectInfo(cmFreq_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount cmFreq_cache = %d\n", count);
        clGetMemObjectInfo(cmProb_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount cmProb_cache = %d\n", count);
        clGetMemObjectInfo(cmResult_cache, CL_MEM_REFERENCE_COUNT, sizeof(cl_int), &count, NULL);
        printf("refCount cmResult_cache = %d\n", count);
        */
        printf("Done with ocl stuff...\n\n");
        // Free host memory
        free(node_cache);
        free(model);
        free(nodRes_cache);
        free(nodFlag_cache);
        free(scalings_cache);
        free(prob_cache);
        free(freq_cache);
        free(root_cache);
        free(result_cache);
        free(root_scalings);
        printf("Done!\n\n");
        clean = true;
        exit(0);

        if (iExitCode = EXIT_FAILURE)
            exit (iExitCode);
    }
}

unsigned int _OCLEvaluator::roundUpToNextPowerOfTwo(unsigned int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

double _OCLEvaluator::roundDoubleUpToNextPowerOfTwo(double x)
{
    return pow(2, ceil(log2(x)));
}
#endif

