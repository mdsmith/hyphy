// Author: Martin Smith
// Project: Flexible MULtiple MULtiplicaiton LIBrary in C Plus Plus.
// Created: Early August 2012
// Last Edited: 8/23/2012

#include "gpuMuller.h"
#include <iostream>
#include <stdio.h>
#include "helperFunctions.h"
#include "naiveFunctions.h"
#include <sys/stat.h>
#include <sys/time.h>
#if defined(__APPLE__) || defined(APPLE)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#define STRINGIFY(src) #src

#define kernel_header "#define BLOCK_SIZE 16\n#define SCALAR 10\n#define SCAL_THRESH 1e-10\n#define MAX(a, b) (((a) > (b)) ? (a) : (b))\n#define MIN(a, b) (((a) < (b)) ? (a) : (b))\n"

inline const char* Kernels()
{
    static const char* kernels =
        kernel_header
        #include "oclKernels.cl"
        ;
    return kernels;
}

using namespace std;

#define BLOCK_SIZE 16
#define PAD_SIZE 64
#define SCALAR 10
#define SCAL_THRESH 1e-10
#define OCL_VERBOSE

bool evaluated;
bool cleanBuff;
bool a_dirt;
bool b_dirt;
bool c_dirt;
bool ctx_set;
cl_context ctx;
cl_kernel kernel;
cl_command_queue queue;
size_t global_work_size[2];
size_t local_work_size[2];
cl_int err_num;
double totalMulTime;
double totalReadTime;
double totalWriteTime;

cl_mem d_A;
cl_mem d_As;
cl_mem d_B;
cl_mem d_Bs;
cl_mem d_C;
cl_mem d_Cs;

GPUMuller::GPUMuller()
{
    totalMulTime = 0;
    totalReadTime = 0;
    totalWriteTime = 0;
    //cout << "naiveMuller constructed" << endl;
    cleanBuff = false;
    a_dirt = false;
    b_dirt = false;
    c_dirt = false;
    evaluated = false;
    overwrite = true;
    ctx_set = false;
    //setup_context();
}


const char* GPUMuller::get_name()
{
    return "GPU";
}


void GPUMuller::transpose_B()
{
    B.transpose();
}

void GPUMuller::set_A(float* A, int num_rows, int num_cols)
{
    if (ctx != NULL)
        cleanBuff = false;
    Muller::set_A(A, num_rows, num_cols);
}

void GPUMuller::set_A(double* A, int num_rows, int num_cols)
{
    //cout << "printing treecache: " << endl;
    //print_double_mat(A, 0,0, num_rows, num_cols, num_rows, num_cols);
    if (ctx != NULL)
        cleanBuff = false;
    Muller::set_A(A, num_rows, num_cols);
}

void GPUMuller::set_B(float* B, int num_rows, int num_cols)
{
    if (ctx != NULL)
        cleanBuff = false;
    Muller::set_B(B, num_rows, num_cols);
}
void GPUMuller::set_B(double* B, int num_rows, int num_cols)
{
    if (ctx != NULL)
        cleanBuff = false;
    Muller::set_B(B, num_rows, num_cols);
}

void GPUMuller::set_C(float* C, int num_rows, int num_cols)
{
    if (ctx != NULL)
        cleanBuff = false;
    Muller::set_C(C, num_rows, num_cols);
}
void GPUMuller::set_C(double* C, int num_rows, int num_cols)
{
    if (ctx != NULL)
        cleanBuff = false;
    Muller::set_C(C, num_rows, num_cols);
}

void GPUMuller::update_A(float* A, int row_offset, int col_offset, int ah, int ud, int num_rows, int num_cols)
{
    Muller::update_A(A, row_offset, col_offset, ah, ud, num_rows, num_cols);
    if (ctx != NULL)
        a_dirt = true;
}
void GPUMuller::update_A(double* A, int row_offset, int col_offset, int ah, int ud, int num_rows, int num_cols)
{
    Muller::update_A(A, row_offset, col_offset, ah, ud, num_rows, num_cols);
    if (ctx != NULL)
        a_dirt = true;
}

void GPUMuller::update_B(float* B, int row_offset, int col_offset, int ud, int bw, int num_rows, int num_cols)
{
    Muller::update_B(B, row_offset, col_offset, ud, bw, num_rows, num_cols);
    if (ctx != NULL)
        b_dirt = true;
}
void GPUMuller::update_B(double* B, int row_offset, int col_offset, int ud, int bw, int num_rows, int num_cols)
{
    Muller::update_B(B, row_offset, col_offset, ud, bw, num_rows, num_cols);
    if (ctx != NULL)
        b_dirt = true;
}

void GPUMuller::bound_A(int row_offset, int col_offset, int ah, int ud)
{
    Muller::bound_A(row_offset, col_offset, ah, ud);
}

void GPUMuller::bound_B(int row_offset, int col_offset, int ud, int bw)
{
    Muller::bound_B(row_offset, col_offset, ud, bw);
}


void GPUMuller::setup_context()
{

    cl_platform_id plat = NULL;
    cl_device_id *devices = NULL;
    cl_device_id device = NULL;
    cl_uint dev_count = 0;
    err_num = CL_SUCCESS;

    // Plat setup
    err_num = clGetPlatformIDs(1, &plat, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "Plat fail" << endl;
        exit(err_num);
    }

    // XXX move to multiple devices
    // Dev setup
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
    devices = (cl_device_id *)malloc(dev_count * sizeof(cl_device_id));
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, dev_count, devices, NULL);
    device = devices[0];
    if (err_num != CL_SUCCESS)
    {
        cout << "Dev fail" << endl;
        exit(err_num);
    }

    // Context setup
    // 1 == my device count (arbitrary)
    ctx = clCreateContext(0, 1, &device, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "Ctx fail" << endl;
        exit(err_num);
    }

    // get device info
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    err_num = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name),
                             vendor_name, &returned_size);
    err_num |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                              device_name, &returned_size);
    if (err_num != CL_SUCCESS)
    {
        cout << "Name fetch fail" << endl;
        exit(err_num);
    }
    printf("Connecting to %s %s...\n", vendor_name, device_name);

    // queue setup
    queue = clCreateCommandQueue(   ctx,
                                    device,
                                    0,
                                    &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "queue fail" << endl;
        exit(err_num);
    }

    // prog setup
    const char* source = Kernels();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &source, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "compile fail" << endl;
        exit(err_num);
    }

    // build program
    err_num = clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "build fail" << endl;
        exit(err_num);
    }

    // kernel setup
    kernel = clCreateKernel(prog, "matMul", &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make kernel fail" << endl;
        exit(err_num);
    }
    ctx_set = true;
}

// XXX Makes this more efficient
void GPUMuller::update_buffers()
{
    if (!C.is_set())
        set_C(  new float[A.get_bound_rows()*B.get_bound_cols()],
                A.get_bound_rows(),
                B.get_bound_cols()
                );
    A.get_data()->check_scalings();
    B.get_data()->check_scalings();
    C.get_data()->check_scalings();
    // array rounding
    //if (    A.get_data()->get_scaled_float() == B.get_data()->get_scaled_float()
        //&&  A.get_data()->get_scaled_float() == C.get_data()->get_scaled_float())
    if (    A.get_data()->get_id() == B.get_data()->get_id()
        &&  A.get_data()->get_id() == C.get_data()->get_id())
    {
        A.get_data()->pad_to(PAD_SIZE);
        // The other matrices use the same MatrixData, so you only have to
        // pad one matrice's MatrixData
        /*
        B.get_data()->set_data_and_scalings(A.get_data()->get_scaled_float(),
                                A.get_data()->get_scalings(),
                                A.get_data()->get_total_rows(),
                                A.get_data()->get_total_cols());
        C.get_data()->set_data_and_scalings(A.get_data()->get_scaled_float(),
                                A.get_data()->get_scalings(),
                                A.get_data()->get_total_rows(),
                                A.get_data()->get_total_cols());
        */
    }
    else if (B.get_data()->get_scaled_float() == A.get_data()->get_scaled_float())
    {
        A.get_data()->pad_to(PAD_SIZE);
        /*
        B.get_data()->set_data_and_scalings(A.get_data()->get_scaled_float(),
                                A.get_data()->get_scalings(),
                                A.get_data()->get_total_rows(),
                                A.get_data()->get_total_cols());
        */
        C.get_data()->pad_to(PAD_SIZE);
    }
    else if (C.get_data()->get_scaled_float() == B.get_data()->get_scaled_float())
    {
        A.get_data()->pad_to(PAD_SIZE);
        B.get_data()->pad_to(PAD_SIZE);
        /*
        C.get_data()->set_data_and_scalings(B.get_data()->get_scaled_float(),
                                B.get_data()->get_scalings(),
                                B.get_data()->get_total_rows(),
                                B.get_data()->get_total_cols());
        */
    }
    else if (C.get_data()->get_scaled_float() == A.get_data()->get_scaled_float())
    {
        //cout << "A pre: " << A.get_data()->get_total_rows() << endl;
        //cout << "C pre: " << C.get_data()->get_total_rows() << endl;
        A.get_data()->pad_to(PAD_SIZE);
        //cout << "A post: " << A.get_data()->get_total_rows() << endl;
        //cout << "C post: " << C.get_data()->get_total_rows() << endl;
        B.get_data()->pad_to(PAD_SIZE);
        /*
        C.get_data()->set_data_and_scalings(A.get_data()->get_scaled_float(),
                                A.get_data()->get_scalings(),
                                A.get_data()->get_total_rows(),
                                A.get_data()->get_total_cols());
        */
    }
    else
    {
        A.get_data()->pad_to(PAD_SIZE);
        B.get_data()->pad_to(PAD_SIZE);
        C.get_data()->pad_to(PAD_SIZE);
    }

    //cout << "Post padding: " << endl;
    //cout << "A total: " << endl;
    //A.print_total();
    //cout << "A bound: " << endl;
    //A.print_bound();
    //print_A();
    //cout << "B total: " << endl;
    //B.print_total();
    //cout << "B bound: " << endl;
    //B.print_bound();
    //print_B();
    //cout << "C total: " << endl;
    //C.print_total();
    //cout << "C bound: " << endl;
    //C.print_bound();
    //print_C();
/*
*/

    // work dim setup
    global_work_size[0] = round_up(B.get_bound_cols(),PAD_SIZE);
    global_work_size[1] = round_up(A.get_bound_rows(),PAD_SIZE);
    local_work_size[0] = BLOCK_SIZE;
    local_work_size[1] = BLOCK_SIZE;

    if (d_A != NULL)
    {
        err_num = clReleaseMemObject(d_A);
        err_num |= clReleaseMemObject(d_As);
    }
    if (err_num != CL_SUCCESS)
    {
        cout << "release buffer fail" << endl;
        exit(err_num);
    }
    d_A = clCreateBuffer(   ctx,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            A.get_data()->get_total_rows() * A.get_data()->get_total_cols() * sizeof(float),
                            A.get_data()->get_scaled_float(),
                            &err_num);
    d_As = clCreateBuffer(   ctx,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            A.get_data()->get_total_rows() * A.get_data()->get_total_cols() * sizeof(int),
                            A.get_data()->get_scalings(),
                            &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }

    if (d_B != NULL)
    {
        err_num = clReleaseMemObject(d_B);
        err_num |= clReleaseMemObject(d_Bs);
        if (err_num != CL_SUCCESS)
        {
            cout << "release buffer fail" << endl;
            exit(err_num);
        }
    }
    if (B.get_data()->get_scaled_float() == A.get_data()->get_scaled_float())
    {
        d_B = d_A;
        d_Bs = d_As;
    }
    else
    {
        d_B = clCreateBuffer(   ctx,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                B.get_data()->get_total_rows() *
                                B.get_data()->get_total_cols() *
                                sizeof(float),
                                B.get_data()->get_scaled_float(),
                                &err_num);
        d_Bs = clCreateBuffer(  ctx,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                B.get_data()->get_total_rows() *
                                B.get_data()->get_total_cols() *
                                sizeof(int),
                                B.get_data()->get_scalings(),
                                &err_num);
    }
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }

    if (d_C != NULL)
    {
        //err_num = clReleaseMemObject(d_C);
        //err_num |= clReleaseMemObject(d_Cs);
        if (err_num != CL_SUCCESS)
        {
            cout << "release buffer fail" << endl;
            exit(err_num);
        }
    }
    if (C.get_data()->get_scaled_float() == A.get_data()->get_scaled_float())
    {
        d_C = d_A;
        d_Cs = d_As;
    }
    else if (C.get_data()->get_scaled_float() == B.get_data()->get_scaled_float())
    {
        d_C = d_B;
        d_Cs = d_Bs;
    }
    else
    {
        d_C = clCreateBuffer(   ctx,
                                CL_MEM_READ_WRITE,
                                A.get_data()->get_total_rows() *
                                B.get_data()->get_total_cols() *
                                sizeof(float),
                                NULL,
                                &err_num);
        d_Cs = clCreateBuffer(   ctx,
                                CL_MEM_READ_WRITE,
                                A.get_data()->get_total_rows() *
                                B.get_data()->get_total_cols() *
                                sizeof(int),
                                NULL,
                                &err_num);
    }
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    cleanBuff = true;
}


// INFO updating buffers only works if you're replacing the whole buffer
// (nothing on the GPU will be written back before being overwritten)
// XXX make this more efficient
void GPUMuller::check_buffers()
{
    if (!cleanBuff)
        update_buffers();

    #ifdef OCL_VERBOSE
    timeval t1, t2;
    clFinish(queue);
    gettimeofday(&t1, NULL);
    #endif
    if (a_dirt)
    {
        err_num = clEnqueueWriteBuffer( queue,
                                        d_A,
                                        CL_FALSE,
                                        0,
                                        sizeof(cl_float)* A.get_data()->get_total_rows()
                                                        * A.get_data()->get_total_cols(),
                                        A.get_data()->get_scaled_float(),
                                        0,
                                        NULL,
                                        NULL
                                        );
        err_num = clEnqueueWriteBuffer( queue,
                                        d_As,
                                        CL_FALSE,
                                        0,
                                        sizeof(cl_int)  * A.get_data()->get_total_rows()
                                                        * A.get_data()->get_total_cols(),
                                        A.get_data()->get_scalings(),
                                        0,
                                        NULL,
                                        NULL
                                        );
    }
    if (b_dirt)
    {
        err_num |= clEnqueueWriteBuffer( queue,
                                        d_B,
                                        CL_FALSE,
                                        0,
                                        sizeof(cl_float)* B.get_data()->get_total_rows()
                                                        * B.get_data()->get_total_cols(),
                                        B.get_data()->get_scaled_float(),
                                        0,
                                        NULL,
                                        NULL
                                        );
        err_num |= clEnqueueWriteBuffer( queue,
                                        d_Bs,
                                        CL_FALSE,
                                        0,
                                        sizeof(cl_int)  * B.get_data()->get_total_rows()
                                                        * B.get_data()->get_total_cols(),
                                        B.get_data()->get_scalings(),
                                        0,
                                        NULL,
                                        NULL
                                        );
    }
    if (c_dirt)
    {
        err_num |= clEnqueueWriteBuffer( queue,
                                        d_C,
                                        CL_FALSE,
                                        0,
                                        sizeof(cl_float)*C.get_data()->get_total_rows()*C.get_data()->get_total_cols(),
                                        C.get_data()->get_scaled_float(),
                                        0,
                                        NULL,
                                        NULL
                                        );
        err_num |= clEnqueueWriteBuffer( queue,
                                        d_Cs,
                                        CL_FALSE,
                                        0,
                                        sizeof(int)*C.get_data()->get_total_rows()*C.get_data()->get_total_cols(),
                                        C.get_data()->get_scalings(),
                                        0,
                                        NULL,
                                        NULL
                                        );
    }
    if (err_num != CL_SUCCESS)
    {
        cout << "Error rewritting buffers" << endl;
        exit(err_num);
    }
    #ifdef OCL_VERBOSE
    clFinish(queue);
    gettimeofday(&t2, NULL);
    totalWriteTime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    totalWriteTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    #endif
}


void GPUMuller::multiply()
{
    cl_int temp_ah = A.get_bound_rows();
    cl_int temp_bw = B.get_bound_cols();
    cl_int temp_bw_round = B.get_data()->get_total_cols();
    cl_int temp_ud = A.get_bound_cols();
    cl_int temp_ud_round = A.get_data()->get_total_cols();
    cl_int temp_a_row_offset = A.get_row_offset();
    cl_int temp_a_col_offset = A.get_col_offset();
    cl_int temp_b_row_offset = B.get_row_offset();
    cl_int temp_b_col_offset = B.get_col_offset();
    cl_int temp_c_row_offset = C.get_row_offset();
    cl_int temp_c_col_offset = C.get_col_offset();
    cl_bool temp_overwrite = overwrite;

/*
    cout    << "AH, AW, ARO, ACO: "
            << temp_ah << ", "
            << temp_ud << ", "
            << temp_a_row_offset << ", "
            << temp_a_col_offset << ", "
            << endl
            << "BH, BW, BRO, BCO: "
            << temp_ud << ", "
            << temp_bw << ", "
            << temp_b_row_offset << ", "
            << temp_b_col_offset << ", "
            << endl
            << "CH, CW, CRO, CCO: "
            << temp_ah << ", "
            << temp_bw << ", "
            << temp_c_row_offset << ", "
            << temp_c_col_offset << ", "
            << endl
            << "row bound: "
            << temp_bw << ", "
            << " col bound: "
            << temp_ah
            << endl
            << "overwrite: "
            << overwrite
            << endl;
*/


    // set kernel args
    err_num  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &d_A);
    err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_As);
    err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_B);
    err_num |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_Bs);
    err_num |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &d_C);
    err_num |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &d_Cs);
    err_num |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *) &temp_ud);
    err_num |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void *) &temp_ud_round);
    err_num |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void *) &temp_bw_round);
    err_num |= clSetKernelArg(kernel, 9, sizeof(cl_int), (void *) &temp_bw);
    err_num |= clSetKernelArg(kernel, 10, sizeof(cl_int), (void *) &temp_ah);
    err_num |= clSetKernelArg(  kernel,
                                11,
                                sizeof(cl_int),
                                (void *) &temp_a_row_offset);
    err_num |= clSetKernelArg(  kernel,
                                12,
                                sizeof(cl_int),
                                (void *) &temp_a_col_offset);
    err_num |= clSetKernelArg(  kernel,
                                13,
                                sizeof(cl_int),
                                (void *) &temp_b_row_offset);
    err_num |= clSetKernelArg(  kernel,
                                14,
                                sizeof(cl_int),
                                (void *) &temp_b_col_offset);
    err_num |= clSetKernelArg(  kernel,
                                15,
                                sizeof(cl_int),
                                (void *) &temp_c_row_offset);
    err_num |= clSetKernelArg(  kernel,
                                16,
                                sizeof(cl_int),
                                (void *) &temp_c_col_offset);
    err_num |= clSetKernelArg(  kernel,
                                17,
                                sizeof(cl_bool),
                                (void *) &temp_overwrite);

    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }

    // launch kernel
    timeval t1, t2;
    double elapsedTime;
    #ifdef OCL_VERBOSE
    clFinish(queue);
    #endif
    gettimeofday(&t1, NULL);

    //cout << "Global work size: " << global_work_size[0];
    //cout << ", " << global_work_size[1] << endl;
    //cout << "Local work size: " << local_work_size[0];
    //cout << ", " << local_work_size[1] << endl;
    err_num = clEnqueueNDRangeKernel(   queue,
                                        kernel,
                                        2,
                                        0,
                                        global_work_size,
                                        local_work_size,
                                        0,
                                        NULL,
                                        NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel launch fail" << endl;
        exit(err_num);
    }
    clFinish(queue);
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec -t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    totalMulTime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    totalMulTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    //cout << "Multiplication: " << elapsedTime << " ms.\n";
}

// Read from the significant and exponent buffers on the GPU
// XXX this singular offset should probably be changed...?
void GPUMuller::read_C( int offset, // This is the offset for both the GPU
                                    // buffer and the data_ptr
                        int size,   // This is the size of the chunk to write
                                    // from this location in the GPU buffer
                                    // to this location in data_ptr
                        float* data_ptr,    // The host significand buffer ptr
                        int* scalings_ptr)  // The host exponent buffer ptr
{
    // get results
    err_num = clEnqueueReadBuffer(  queue,
                                    d_C,
                                    CL_TRUE,
                                    offset * sizeof(float),
                                    //C.get_total_rows() * C.get_total_cols() *
                                    //sizeof(float),
                                    size * sizeof(float),
                                    //C.get_scaled(),
                                    data_ptr + offset,
                                    //temp_sigs,
                                    0,
                                    NULL,
                                    NULL);

/*
    cout    << "Offset: " << offset
            << ", size: " << size;
    cout << "C fresh off the gpu: " << endl;
    cout << "size " << size << endl;
    for (int i = 0; i < size; i++)
    {
        if (i % C.get_data()->get_total_cols() == 0)
            cout << endl;
        cout << data_ptr[offset + i] << " ";
    }
    cout << endl;
*/
/*
    cout << "temp_sigs" << endl;
    print_float_mat(temp_sigs, 0, 0, C.get_total_rows(), C.get_total_cols(),
                            C.get_total_rows(), C.get_total_cols());
*/
    if (err_num != CL_SUCCESS)
    {
        cout << "significand read fail" << endl;
        exit(err_num);
    }
    #ifdef OCL_VERBOSE
    timeval t1, t2;
    clFinish(queue);
    gettimeofday(&t1, NULL);
    #endif
    err_num = clEnqueueReadBuffer(  queue,
                                    d_Cs,
                                    CL_TRUE,
                                    offset * sizeof(int),
                                    //C.get_total_rows() * C.get_total_cols()*
                                    //sizeof(int),
                                    size * sizeof(int),
                                    //C.get_scalings(),
                                    scalings_ptr + offset,
                                    0,
                                    NULL,
                                    NULL);
    #ifdef OCL_VERBOSE
    clFinish(queue);
    gettimeofday(&t2, NULL);
    totalReadTime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    totalReadTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    #endif
    if (err_num != CL_SUCCESS)
    {
        cout << "exponent read fail" << endl;
        exit(err_num);
    }
    clFinish(queue);
    C.set_set(true);
}


void GPUMuller::eval_C(int row_offset, int col_offset, int height, int width)
{
    C.bound_data(row_offset, col_offset, height, width);

    //cout << "Before buffer creation: " << endl;
    //print_A();
    //print_B();


    if (!ctx_set)
    {
        setup_context();
        check_buffers();
    }
    else
    {
        check_buffers();
    }

    //print_A(); // XXX temp
    /*
    double* temp_A = A.get_data()->get_slice_double(A.get_row_offset(),
                                                    A.get_col_offset(),
                                                    A.get_bound_rows(),
                                                    A.get_bound_cols());
    double* temp_B = B.get_data()->get_slice_double(B.get_row_offset(),
                                                    B.get_col_offset(),
                                                    B.get_bound_rows(),
                                                    B.get_bound_cols());
    */
    /*
    print_double_mat(   temp_A,
                        0,
                        0,
                        A.get_bound_rows(),
                        A.get_bound_cols(),
                        A.get_bound_rows(),
                        A.get_bound_cols());
    */
    //print_B(); // XXX temp
    //print_C();

    multiply();
    /*
    read_C( 0,
            C.get_data()->get_total_rows() * C.get_data()->get_total_cols(),
            C.get_data()->get_scaled_float(),
            C.get_data()->get_scalings()); // XXX temp
    */
    /*
    double* temp_C = C.get_data()->get_slice_double(C.get_row_offset(),
                                                    C.get_col_offset(),
                                                    C.get_bound_rows(),
                                                    C.get_bound_cols());
    double* naive_C = naive_matrix_multiply_double( temp_A,
                                                    temp_B,
                                                    0, // ARO
                                                    0, // ACO
                                                    0, // BRO
                                                    0, // BCO
                                                    A.get_bound_rows(), // AH
                                                    A.get_bound_rows(), //ANR
                                                    A.get_bound_cols(), // UD
                                                    A.get_bound_cols(), //UDt
                                                    B.get_bound_cols(), // BW
                                                    B.get_bound_cols()); //BNC

    */
    /*
    print_double_mat(   temp_C,
                        0,
                        0,
                        C.get_bound_rows(),
                        C.get_bound_cols(),
                        C.get_bound_rows(),
                        C.get_bound_cols());
    print_double_mat(   naive_C,
                        0,
                        0,
                        C.get_bound_rows(),
                        C.get_bound_cols(),
                        C.get_bound_rows(),
                        C.get_bound_cols());
    */
    /*
    double thresh = 1e-6;
    for (int i = 0; i < C.get_bound_rows() * C.get_bound_cols(); i++)
    {
        double error = (temp_C[i] - naive_C[i])/non_zero_max(temp_C[i], naive_C[i]);
        if (error > thresh)
        {
            int row = i / C.get_bound_cols();
            int col = i % C.get_bound_cols();
            cout    << "Answer is Wrong. Thresh: "
                    << thresh
                    << ", error: "
                    << error
                    << " at: "
                    << row
                    << ", "
                    << col
                    << endl
                    << "Values: "
                    << temp_C[i]
                    << ", "
                    << naive_C[i]
                    << endl;

            break;
        }
    }
    */
    evaluated = true;
    //print_C(); // XXX temp
    //cout << "C after multiply: " << endl;
    //print_C(0, 0, C.get_total_rows(), C.get_total_cols());
}


float* GPUMuller::get_C(int row_offset, int col_offset, int height, int width)
{
    //cout << "A id is: " << A.get_data()->get_id() << endl;
    //cout << "B id is: " << B.get_data()->get_id() << endl;
    //cout << "C id is: " << C.get_data()->get_id() << endl;
    if (!evaluated)
    {
        eval_C(row_offset, col_offset, height, width);
    }
    // XXX um, how about we use the actual offsets?
    /*
    read_C( 0,
            C.get_data()->get_total_rows() * C.get_data()->get_total_cols(),
            C.get_data()->get_scaled_float(),
            C.get_data()->get_scalings());
    */

    read_C( row_offset*C.get_data()->get_total_cols(),
            height * C.get_data()->get_total_cols(),
            C.get_data()->get_scaled_float(),
            C.get_data()->get_scalings());

    // XXX something is broken here if they're printing different things.
    /*
    C.print_total();
    cout << endl << endl << endl;
    cout << "root conditionals: " << endl;
    float* test = C.get_slice(row_offset, col_offset, width, height);
    for (int i = 0; i < width*height; i++)
    {
        cout << test[i] << " ";
        if (i % width == 0)
            cout << endl;
    }
    */

    //cout << "RO: " << row_offset << " CO: " << col_offset << " h: " << height
    //<< " w: " << width << endl;

    return C.get_data()->get_slice(row_offset, col_offset, height, width);
}

double* GPUMuller::get_C_double(int row_offset, int col_offset, int height, int width)
{
    //cout << "A id is: " << A.get_data()->get_id() << endl;
    //cout << "B id is: " << B.get_data()->get_id() << endl;
    //cout << "C id is: " << C.get_data()->get_id() << endl;
    // XXX um, how about we use the actual offsets?
    /*
    if (!evaluated)
    {
        eval_C(row_offset, col_offset, height, width);
    }
    */

    /*
    read_C( 0,
            C.get_data()->get_total_rows() * C.get_data()->get_total_cols(),
            C.get_data()->get_scaled_float(),
            C.get_data()->get_scalings());
    */
    read_C( row_offset*C.get_data()->get_total_cols(),
            height * C.get_data()->get_total_cols(),
            C.get_data()->get_scaled_float(),
            C.get_data()->get_scalings());
    /*
    */


    //C.get_data()->print_mat(row_offset, col_offset, height, width);
    /*
    cout << endl << endl << endl;
    cout << "Root conditionals earlier: " << endl;
    double* test = C.get_data()->get_slice_double(row_offset, col_offset,
    height, width);
    cout    << "RO, CO W, H: "
            << row_offset
            << ", "
            << col_offset
            << ", "
            << width
            << ", "
            << height
            << endl;
    for (int i = 0; i < width*height; i++)
    {
        cout << test[i] << " ";
        if (i % width == width-1)
            cout << endl;
    }
    */
    /*
    cout << "openCL model: " << endl;
    B.get_data()->print_mat();

    cout << "openCL data: " << endl;
    C.get_data()->print_mat();
    */

    //cout << "RO: " << row_offset << " CO: " << col_offset << " h: " << height
    //<< " w: " << width << endl;

    return C.get_data()->get_slice_double(row_offset, col_offset, height,
    width);
}


void GPUMuller::test()
{
    cout << "GPUMuller checking in" << endl;
}


GPUMuller::~GPUMuller()
{
    #ifdef OCL_VERBOSE
    printf("Total multiplication time: %g\n", totalMulTime);
    printf("Total read time: %g\n", totalReadTime);
    printf("Total write time: %g\n", totalWriteTime);
    #endif
    // cleanup
    /*
    err_num |= clReleaseMemObject(d_A);
    err_num |= clReleaseMemObject(d_B);
    err_num |= clReleaseMemObject(d_C);
    err_num |= clReleaseKernel(kernel);
    err_num |= clReleaseCommandQueue(queue);
    err_num |= clReleaseContext(ctx);
    */
    if (err_num != CL_SUCCESS)
    {
        cout << "free fail" << endl;
        exit(err_num);
    }

    //cout << "naiveMuller destructed" << endl;
}
