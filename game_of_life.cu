/*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Author:     Sami Byaruhanga 
Purpose:    CUDA-accelerated, global memory version of game of life
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>*/


//INCLUDES + GLOBAL VARS ********************************************
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for resizing
#include <vector>
#include <sstream>
#include <string>

#include <limits.h>
#include <fstream>
#include <cassert>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#define MAX_SIZE 1024 
// #define MAX_SIZE 10000


using namespace std;
//************************************************* INCLUDES END HERE


//FUNCTIONS  ********************************************************
//-------------------------------------------------------
// cuda error handling 
#define cudaErrorCheck(result) { cudaAssert((result), __FILE__, __FUNCTION__, __LINE__); }

inline void cudaAssert(cudaError_t err, const char *file,  const char *function, int line, bool quit=true)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr,"cudaAssert failed with error \'%s\', in File: %s, Function: %s, at Line: %d\n", cudaGetErrorString(err), file, function, line);
        if (quit) exit(err);
    }
}
//-------------------------------------------------------


//-------------------------------------------------------
// printCUDADevice
// PURPOSE: prints cude 
// INPUT PARAMETERS: cuda device properties
void printCUDADevice(cudaDeviceProp properties)
{
    std::cout << "CUDA Device: " << std::endl;
    std::cout << "\tDevice name              : " << properties.name << std::endl;
    std::cout << "\tMajor revision           : " << properties.major << std::endl;
    std::cout << "\tMinor revision           : " << properties.minor << std::endl;
    std::cout << "\tGlobal memory            : " << properties.totalGlobalMem/1024.0/1024.0/1024.0 << " Gb" << std::endl;
    std::cout << "\tShared memory per block  : " << properties.sharedMemPerBlock/1024.0 << " Kb" << std::endl;
    std::cout << "\tRegisters per block      : " << properties.regsPerBlock << std::endl;
    std::cout << "\tWarp size                : " << properties.warpSize << std::endl;
    std::cout << "\tMax threads per block    : " << properties.maxThreadsPerBlock << std::endl;
    std::cout << "\tMaximum x dim of block   : " << properties.maxThreadsDim[0] << std::endl;
    std::cout << "\tMaximum y dim of block   : " << properties.maxThreadsDim[1] << std::endl;
    std::cout << "\tMaximum z dim of block   : " << properties.maxThreadsDim[2] << std::endl;
    std::cout << "\tMaximum x dim of grid    : " << properties.maxGridSize[0] << std::endl;
    std::cout << "\tMaximum y dim of grid    : " << properties.maxGridSize[1] << std::endl;
    std::cout << "\tMaximum z dim of grid    : " << properties.maxGridSize[2] << std::endl;
    std::cout << "\tClock frequency          : " << properties.clockRate/1000.0 << " MHz" << std::endl;
    std::cout << "\tConstant memory          : " << properties.totalConstMem << std::endl;
    std::cout << "\tNumber of multiprocs     : " << properties.multiProcessorCount << std::endl;
}
//-------------------------------------------------------


//-------------------------------------------------------
// cudagameoflife
// PURPOSE: 
// INPUT PARAMETERS: 
// RETURNS: 
// __global__ void cudaGameOfLife(int ny, int nx, int maxiter, int visualizationIter, cv::Mat population, cv::Mat new_population, cv::Mat image_for_viewing){
__global__ void cudaGameOfLife(uchar* population, uchar* new_population, int ny, int nx){
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if(iy < ny && ix < nx){
        int occupied_neighbours = 0;
        for (int jy = iy - 1; jy <= iy + 1; jy++)
        {
            for (int jx = ix - 1; jx <= ix + 1; jx++)
            {
                if (jx == ix && jy == iy) continue;
                
                int row = jy;
                if (row == ny) row = 0;
                if (row == -1) row = ny-1;
                
                int col = jx;
                if (col == nx) col = 0;
                if (col == -1) col = nx - 1;
                
                // if (population.at<uchar>(row,col) == 0) occupied_neighbours++;
                //prof data[m*row + col] ==> (row,col)
                if (population[nx*row + col] == 0) occupied_neighbours++;
                
            }
        }
    
        // if (population.at<uchar>(iy,ix) == 0)   //alive
        //prof data[m*row + col] ==> (row,col)
        if (population[nx*iy + ix] == 0)   //alive
        {
            if (occupied_neighbours <= 1 || occupied_neighbours >= 4) new_population[nx*iy + ix] = 255; //dies
            if (occupied_neighbours == 2 || occupied_neighbours == 3) new_population[nx*iy + ix] = 0; //same as population
        }
        else if (population[nx*iy + ix] == 255) //dead
        {
            if (occupied_neighbours == 3)
            {
                new_population[nx*iy + ix] = 0; //reproduction
            }
        }
    }
}
//------------------------------------------

//-------------------------------------------------------

//-------------------------------------------------------
// serialGameOfLife
// PURPOSE: Implements a serial game of life
// INPUT PARAMETERS: rows, cols, maxiteration, visualization iterations, matrix for population, new population, image reviewed 
void serialGameOfLife(int ny, int nx, int maxiter, int visualizationIter, cv::Mat population, cv::Mat new_population, cv::Mat image_for_viewing){
    for (int iter = 0; iter < maxiter; iter++)
    {
        // std::cout << "iteration count: " << iter << std::endl;
        if( iter % visualizationIter ==0 && visualizationIter != -1){ //ie, remainder == 0 therefore its divisible by iteration and if -1 dont show results
            // std::cout<<" \n\t---------" << iter <<std::endl;
            //something new here - we will resize our image up to MAX_SIZE x MAX_SIZE so its not really tiny on the screen
            cv::resize(population,image_for_viewing,image_for_viewing.size(),cv::INTER_LINEAR); //resize image to MAX_SIZE x MAX_SIZE
            cv::imshow("SERIAL", image_for_viewing);
            std::string name = "./imgs/SERIAL iteration " + std::to_string(iter) + ".jpg";
            cv::imwrite(name, image_for_viewing);
            // std::cout<<" show image at iteration " << iter <<std::endl;
            cv::waitKey(10);	//wait 10 seconds before closing image (or a keypress to close)
        }

        
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int occupied_neighbours = 0;
 
                for (int jy = iy - 1; jy <= iy + 1; jy++)
                {
                    for (int jx = ix - 1; jx <= ix + 1; jx++)
                    {
                        if (jx == ix && jy == iy) continue;
                        
                        int row = jy;
                        if (row == ny) row = 0;
                        if (row == -1) row = ny-1;
                        
                        int col = jx;
                        if (col == nx) col = 0;
                        if (col == -1) col = nx - 1;
                        
                        if (population.at<uchar>(row,col) == 0) occupied_neighbours++;
                    }
                }
            
                if (population.at<uchar>(iy,ix) == 0)   //alive
                {
                    if (occupied_neighbours <= 1 || occupied_neighbours >= 4) new_population.at<uchar>(iy,ix) = 255; //dies
                    if (occupied_neighbours == 2 || occupied_neighbours == 3) new_population.at<uchar>(iy,ix) = 0; //same as population
                }
                else if (population.at<uchar>(iy,ix) == 255) //dead
                {
                    if (occupied_neighbours == 3)
                    {
                        new_population.at<uchar>(iy,ix) = 0; //reproduction
                    }
                }
            }
        }
        population = new_population.clone();
    }
}
//-------------------------------------------------------

//************************************************ FUNCTIONS END HERE


//MAIN **************************************************************
int main(int argc, char**argv){
    //Get and print dev info ---------------------------
	cudaError_t err;
    cudaDeviceProp device_properties;
	cudaGetDeviceProperties	(&device_properties, 0);
	// printCUDADevice(device_properties);    
    //--------------------------------------------------


    //CMD args (nx, ny, maxiter, visualizationIter) -----
	// assert(argc == 7); 
    int ny = atoi(argv[1]);
	int nx = atoi(argv[2]);
    int maxiter = atoi(argv[3]);
    assert(ny <= MAX_SIZE);
    assert(nx <= MAX_SIZE);
    int visualizationIter = atoi(argv[4]);
    int block_size_x = atoi(argv[5]);
    int block_size_y = atoi(argv[6]);
    //--------------------------------------------------


    //Create our image rand (show image) ---------------
    srand(clock());
    cv::Mat population(ny, nx, CV_8UC1);
    for (unsigned int iy = 0; iy < ny; iy++)
    {
        for (unsigned int ix = 0; ix < nx; ix++)
        {
            //seed a 1/2 density of alive (just arbitrary really)
            int state = rand()%2;
            if (state == 0) population.at<uchar>(iy,ix) = 255; //dead
            else population.at<uchar>(iy,ix) = 0;   //alive
        }
    }
    cv::Mat new_population = population.clone();     //for cuda
    cv::Mat new_population1 = population.clone();    //for serial
    // cv::namedWindow("Population", cv::WINDOW_AUTOSIZE );
    cv::Mat image_for_viewing(MAX_SIZE,MAX_SIZE,CV_8UC1);
    //--------------------------------------------------

    
    //Set up timing using cuda events  -----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);    
    //--------------------------------------------------


    //Create GPU(DEV)BUFFERS ---------------------------
    uchar* cuda_population;
    uchar* cuda_new_population;

    //allocate and copy
    err = cudaMalloc((void**)&cuda_population, ny * nx * sizeof(uchar)); cudaErrorCheck(err);
    err = cudaMalloc((void**)&cuda_new_population, ny * nx * sizeof(uchar)); cudaErrorCheck(err);
    err = cudaMemcpy(cuda_population, population.data, ny * nx * sizeof(uchar), cudaMemcpyHostToDevice); cudaErrorCheck(err);
    err = cudaMemcpy(cuda_new_population, new_population.data, ny * nx * sizeof(uchar), cudaMemcpyHostToDevice); cudaErrorCheck(err);
    //--------------------------------------------------


    //Set execution config + check dimensions ----------
    dim3 block_size(block_size_x,block_size_y);
    int gridy = (int)ceil((double)ny/(double)block_size_y); 
    int gridx = (int)ceil((double)nx/(double)block_size_x); 
    dim3 grid_size(gridx, gridy);    

    if (block_size.x*block_size.y > device_properties.maxThreadsPerBlock)
	{
		std::cerr << "Block Size of " << block_size.x << " x " << block_size.y << " is too big. " << std::endl;
		std::cerr << "Maximum threads per block = " << device_properties.maxThreadsPerBlock << std::endl;
		return -1;
	}
	else if (block_size.x > device_properties.maxThreadsDim[0] || block_size.y > device_properties.maxThreadsDim[1])
	{
		std::cerr << "Block Size of " << block_size.x << " x " << block_size.y << " is too big. " << std::endl;
		std::cerr << "Maximum threads for dimension 0 = " << device_properties.maxThreadsDim[0] << std::endl;
		std::cerr << "Maximum threads for dimension 1 = " << device_properties.maxThreadsDim[1] << std::endl;
		return -1;
	}
	else if (grid_size.x > device_properties.maxGridSize[0] || grid_size.y > device_properties.maxGridSize[1])
	{
		std::cerr << "Grid Size of " << grid_size.x << " x " << grid_size.y << " is too big. " << std::endl;
		std::cerr << "Maximum grid dimension 0 = " << device_properties.maxGridSize[0] << std::endl;
		std::cerr << "Maximum grid dimension 1 = " << device_properties.maxGridSize[1] << std::endl;		
		return -1;
	}
    //--------------------------------------------------

/*V1: 
    //call kernel for each generation ------------------
    for (int iter = 0; iter < maxiter; iter++) {
        if( iter % visualizationIter ==0 && visualizationIter != -1){ //ie, remainder == 0 therefore its divisible by iteration and if -1 dont show results
            // std::cout<<" \n\t---------" << iter <<std::endl;
            err = cudaMemcpy(new_population.data, cuda_population, ny * nx * sizeof(uchar), cudaMemcpyDeviceToHost); cudaErrorCheck(err);    
            cv::resize(new_population, image_for_viewing, image_for_viewing.size(), cv::INTER_LINEAR);
            // cv::imshow("CUDA", image_for_viewing);
            // std::cout << " show image at iteration " << iter << std::endl;
            std::string name = "./imgs/CUDA iteration " + std::to_string(iter) + ".jpg";
            cv::imwrite(name, image_for_viewing);
            cv::waitKey(10);
        }

        // std::cout  << "Calling global kernel... " << std::endl;
        cudaGameOfLife<<<grid_size, block_size>>>(cuda_population, cuda_new_population, ny, nx);
        // std::cout  << "Calling done... " << std::endl;
        
        // Update population for the next iteration
        // std::swap(cuda_population, cuda_new_population);
        cuda_population = cuda_new_population;
    }
    //--------------------------------------------------
*/

    //call kernel for each generation ------------------
    for (int iter = 0; iter < maxiter; iter++) {
        if( iter % visualizationIter ==0 && visualizationIter != -1){ //ie, remainder == 0 therefore its divisible by iteration and if -1 dont show results
            cv::resize(new_population, image_for_viewing, image_for_viewing.size(), cv::INTER_LINEAR);
            cv::imshow("CUDA", image_for_viewing);
            std::cout << " show image at iteration " << iter << std::endl;
            std::string name = "./imgs/CUDA iteration " + std::to_string(iter) + ".jpg";
            cv::imwrite(name, image_for_viewing);
            cv::waitKey(10);
        }
        

        cudaGameOfLife<<<grid_size, block_size>>>(cuda_population, cuda_new_population, ny, nx);
        err = cudaMemcpy(new_population.data, cuda_new_population, ny * nx * sizeof(uchar), cudaMemcpyDeviceToHost); cudaErrorCheck(err);    

        // uchar *temp = cuda_population;
        cuda_population = cuda_new_population;
        // cuda_new_population = temp;
    }
    //--------------------------------------------------


    //Stop timing --------------------------------------
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);   //time in milliseconds
    gpu_time /= 1000.0;
    std::cout << "Done GPU Computations in " << gpu_time << " seconds" << std::endl; std::cout.flush();
    //--------------------------------------------------


    //clean up (free and destroy events) ---------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(cuda_population);
    cudaFree(cuda_new_population);
    //--------------------------------------------------


    //Call serial code ---------------------------------
    double cpu_t_start = (double)clock()/(double)CLOCKS_PER_SEC;
    serialGameOfLife(ny, nx, maxiter, visualizationIter, population, new_population1, image_for_viewing);
    double cpu_time = (double)clock()/(double)CLOCKS_PER_SEC - cpu_t_start;
    //--------------------------------------------------


    //Print timing results -----------------------------
    std::cout << "\tBlock Size = " << block_size.x << " x " << block_size.y << "\n";
    std::cout << "\tGrid Size  = " << grid_size.x << " x " << grid_size.y << "\n";
    std::cout << "GPU Time = " << gpu_time << std::endl;
    std::cout << "CPU Time = " << cpu_time << std::endl;
    std::cout << "Speedup = " << cpu_time/gpu_time << std::endl;
    //--------------------------------------------------
}
//**************************************************** MAIN ENDS HERE


/** NOTES ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * QUESTIONS: 
 * 1. is blocks similar to ny and nx?? 
        ans: I dont think so coz the ny and nx specify size of our population cv thingy
 * 2. grid size: might need to find better option coz rn am just using profs
 
 * IMRPOVEMENTS:
 * 1. 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/


/** STEPS TO SOLVE PROGRAM *******************************************
 * MAIN (Borrowed from 23b and 26a)
    * 1. Get and print dev info
    * 2. Get cmdline args for game of life (nx, ny, maxiteration, visualizationIteration)
    * 3. Create our image rand (show image)
    * 4. Set up timing using cuda events 
    * 5. Create GPU(DEV)BUFFERS: 
    *       - ALLOCATE MEMORY (next, current states)
    *       - MEM COPY
    * 6. Set execution configuration & check dimensions based on properties
    * 8. call kernel for each generation 
    * 9. clean up (free and destroy events)
* KERNEL (based off 23b)
    * get the row, cols
    * pass uchar, instead of row,col use data[m*row+col]
    * use visualization paramter to show image after certain display??
*********************************************************************/
