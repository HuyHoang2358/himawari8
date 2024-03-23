#include "jbutil.h"
#include <vector>
#include <limits>
#include <istream>
#include <cmath>
#include <string>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define PI 3.14159265

// whether to measure block similarity using all image channel
int MULTI_CHANNELS = 1;

//struct to hold the motion vectors and MAD for each macroblock
struct block_data
{
	int motion_vector_x;
	int motion_vector_y;
	float MAD;
};

//struct to hold the MAD - Searchblock pairs
struct MAD_per_Macroblock
{
	// IMPORTANT: need to change the value according to search_dist and compile before run.
	float blocks[961];
};


//Function used to load frames
//Inputs: path for the images, the 2 frame images that will hold the frames
//Output: True if all load correctly, False if not
bool Load_Frames(std::string path1, std::string path2, jbutil::image<uint8_t> &frame1,jbutil::image<uint8_t> &frame2)
{
	//Load the 2 frames
	std::ifstream file1(path1.c_str(), std::ios::binary);
	if(file1)
	{
		frame1.load(file1);
			#ifndef NDEBUG
		std::cerr << "First Frame Loaded \n" << std::flush;
			#endif
	}
	else
	{
			#ifndef NDEBUG
		std::cerr << "Error Loading First Frame \n" << std::flush;
			#endif
		return false;
	}
	file1.close();

	std::ifstream file2(path2.c_str(), std::ios::binary);
	if(file2)
	{
		frame2.load(file2);
			#ifndef NDEBUG
		std::cerr << "Second Frame Loaded \n" << std::flush;
			#endif
	}
	else
	{
			#ifndef NDEBUG
		std::cerr << "Error Loading Second Frame" << std::flush;
			#endif
		return false;
	}
	file2.close();

	return true;
}

//Function used to check the input parameters
//Inputs: image used to check parameters
//Output: True if all parameters are correct, False if not
bool Parameter_Check(jbutil::image<uint8_t> &frame_1, jbutil::image<uint8_t> &frame_2, int search_dist, int grid_spacing)
{
	// check to ensure two frame has equal size
	if (frame_1.get_rows() != frame_2.get_rows() || frame_1.get_cols() != frame_2.get_cols())
	{
		#ifndef NDEBUG
		std::cerr << "Error: Two frames must be in same size! \n" << std::flush;
		#endif
		return false;
	}

	if(search_dist == 0 || grid_spacing == 0)
	{
		#ifndef NDEBUG
		std::cerr<<"Integer parameters must be non-zero \n"<<std::flush;
		#endif
		return false;
	}

	return true;
}

//Function to perform the linearization of the image
//Inputs: Image to be linearized, output array
//Output: None
void Linearize_Image(jbutil::image<uint8_t> &image, uint8_t* array, int n_rows, int n_cols, int n_channels)
{
	//linearize the image in such a way that memory is coalesced
	int index = 0;
	for (int row = 0; row < n_rows; row++)
	{
		for(int col = 0; col < n_cols; col++)
		{
			for(int channel = 0; channel < n_channels; channel++)
			{
				array[index] = image(channel, row, col);
				index++;
			}
		}
	}
}

__global__ void mean(int n, int n_searchs, MAD_per_Macroblock *x, MAD_per_Macroblock *y, MAD_per_Macroblock *sum)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		for (int search = 0; search < n_searchs; search ++)
			sum[i].blocks[search] = (x[i].blocks[search] + y[i].blocks[search]) / 2.0;
	}
}

__global__ void max(int n, int n_searchs, MAD_per_Macroblock *x, MAD_per_Macroblock *y, MAD_per_Macroblock *max)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		for (int search = 0; search < n_searchs; search ++)
			max[i].blocks[search] = x[i].blocks[search] > y[i].blocks[search] ? x[i].blocks[search] : y[i].blocks[search];
	}
}

__global__ void Block_Match_Kernel(MAD_per_Macroblock* device_MAD_all_searches, 
								uint8_t* device_frame_1,  uint8_t* device_frame_2, 
								int n_rows, int n_cols, int n_channels,
								int n_blocks_y, int n_blocks_x,
								int grid_spacing, int block_size, int search_dist)
{
	int anchor_ind_x = blockIdx.x % n_blocks_x;
	int anchor_ind_y = blockIdx.x / n_blocks_x;

	int anchor_start_x = anchor_ind_x * grid_spacing - block_size >= 0 ? anchor_ind_x * grid_spacing - block_size : 0;
	int anchor_stop_x = (anchor_ind_x + 1) * grid_spacing + block_size <= n_cols ? (anchor_ind_x + 1) * grid_spacing + block_size : n_cols;

	int anchor_start_y = anchor_ind_y * grid_spacing - block_size >= 0 ? anchor_ind_y * grid_spacing - block_size : 0;
	int anchor_stop_y = (anchor_ind_y + 1) * grid_spacing + block_size <= n_rows ? (anchor_ind_y + 1) * grid_spacing + block_size : n_rows;

	int search_block = threadIdx.x;
	int ref_delta_x =  search_block % (2 * search_dist + 1) - search_dist;
	int ref_delta_y =  search_block / (2 * search_dist + 1) - search_dist;

	if (anchor_stop_x + ref_delta_x < n_cols && 
		anchor_start_x + ref_delta_x >= 0 && 
		anchor_stop_y + ref_delta_y < n_rows && 
		anchor_start_y + ref_delta_y >= 0)
	{
		float SAD = 0.0;
		for (int y = anchor_start_y; y < anchor_stop_y; y ++)
		{
			for (int x = anchor_start_x; x < anchor_stop_x; x ++)
			{
				for (int c = 0; c < n_channels; c ++)
				{
					SAD += abs(device_frame_1[(n_cols * y + x) * n_channels + c] - 
							device_frame_2[(n_cols * (y + ref_delta_y) + (x + ref_delta_x)) * n_channels + c]);
				}
			}
		}

		device_MAD_all_searches[blockIdx.x].blocks[search_block] = SAD / ((anchor_stop_x - anchor_start_x) * (anchor_stop_y - anchor_start_y));
	}
}

void Motion_Image_Construct(jbutil::image<uint8_t> &frame_motion,
							MAD_per_Macroblock* MAD_all_searches,
							int n_blocks_y, int n_blocks_x, 
							int search_dist, int n_searchs)
{
	//The following holds the lowest MAD and motion vector for each macroblock in a linear manner
	block_data* macroblocks = (block_data*)malloc(n_blocks_x * n_blocks_y * sizeof(block_data));
	//Initialisation of the parmeters for each block
	for (int x = 0; x < n_blocks_x; x++)
	{
		for (int y = 0; y < n_blocks_y; y++)
		{
			int index = x + y * n_blocks_x;

			macroblocks[index].motion_vector_x = 0;
			macroblocks[index].motion_vector_y = 0;
			macroblocks[index].MAD = std::numeric_limits<float>::max();
		}
	}

	// find the block with lowest MAD for each anchor block
	for (int y = 0; y < n_blocks_y; y++)
	{
		for (int x = 0; x < n_blocks_x; x++)
		{
			int index = x + y * n_blocks_x;
			for (int search = 0; search < n_searchs; search ++)
			{
				if (MAD_all_searches[index].blocks[search] < macroblocks[index].MAD)
				{
					macroblocks[index].MAD = MAD_all_searches[index].blocks[search];
					macroblocks[index].motion_vector_x =  search % (2 * search_dist + 1) - search_dist;
					macroblocks[index].motion_vector_y =  search / (2 * search_dist + 1) - search_dist;
				}
			}
		}
	}

	// Once the process is done, construct the motion image
	for (int y = 0; y < n_blocks_y; y++)
	{
		for (int x = 0; x < n_blocks_x; x++)
		{
			int index = x + y * n_blocks_x;
			
			// calculate angle in radian
			double angle = atan2(macroblocks[index].motion_vector_y, macroblocks[index].motion_vector_x);
			// convert angle in range [-PI, PI] to range [0, 2 * PI]
			angle = angle + 2.0 * PI;
			// convert angle to degree
			angle = angle * 180.0 / PI;
			
			// calculate maginute
			double mag = sqrt(macroblocks[index].motion_vector_x * macroblocks[index].motion_vector_x
								+ macroblocks[index].motion_vector_y * macroblocks[index].motion_vector_y);
			// normalize magnitude to range [0, 1] by dividing to highest possible magnitude
			mag = mag / sqrt(2 * search_dist * search_dist);

			int rgb[3];
			jbutil::HSVtoRGB(angle, 1.0, mag, rgb);
			
			frame_motion(0, y, x) = rgb[0];
			frame_motion(1, y, x) = rgb[1];
			frame_motion(2, y, x) = rgb[2];
		}
	}

	std::free(macroblocks);
}

void Block_Match(jbutil::image<uint8_t> &frame_1, 
				 jbutil::image<uint8_t> &frame_2, 
				 MAD_per_Macroblock* &MAD_all_searches,
				 int n_rows, int n_cols, int n_channels,
				 int n_blocks_y, int n_blocks_x,
				 int grid_spacing, int block_size, 
				 int search_dist, int n_searchs)
{	
	//Initialisation of the parmeters for each block
	for (int x = 0; x < n_blocks_x; x++)
	{
		for (int y = 0; y < n_blocks_y; y++)
		{
			int index = x + y * n_blocks_x;
			for (int search = 0; search < n_searchs; search++)
			{
				MAD_all_searches[index].blocks[search] = std::numeric_limits<float>::max();
			}
		}
	}

	//Linearize the images, allocate space for them on the device and pass them to the device
	int array_size = n_cols * n_rows * n_channels;

	uint8_t* array_frame_1 = (uint8_t*)std::malloc(sizeof(uint8_t)*array_size);
	Linearize_Image(frame_1, array_frame_1, n_rows, n_cols, n_channels);
	uint8_t* device_frame_1;
	cudaMalloc((void**)&device_frame_1, sizeof(uint8_t)*array_size);
	cudaMemcpy(device_frame_1, array_frame_1, sizeof(uint8_t)*array_size, cudaMemcpyHostToDevice);

	uint8_t* array_frame_2 = (uint8_t*)std::malloc(sizeof(uint8_t)*array_size);
	Linearize_Image(frame_2, array_frame_2, n_rows, n_cols, n_channels);
	uint8_t* device_frame_2;
	cudaMalloc((void**)&device_frame_2, sizeof(uint8_t)*array_size);
	cudaMemcpy(device_frame_2, array_frame_2, sizeof(uint8_t)*array_size, cudaMemcpyHostToDevice);

	//create the the macroblock - searc block pair MAD data structure for the device and copy to device
	MAD_per_Macroblock* device_MAD_all_searches;
	cudaMalloc((void**)&device_MAD_all_searches, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock));
	cudaMemcpy(device_MAD_all_searches, MAD_all_searches, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock), cudaMemcpyHostToDevice);

	//call the kernel
	int kernal_block_size = n_searchs;
	int kernel_num_blocks = n_blocks_x * n_blocks_y;

	Block_Match_Kernel<<<kernel_num_blocks,kernal_block_size>>>
		(device_MAD_all_searches, device_frame_1, device_frame_2, n_rows, n_cols, n_channels, n_blocks_y, n_blocks_x, grid_spacing, block_size, search_dist);
	cudaDeviceSynchronize();

	cudaMemcpy(MAD_all_searches, device_MAD_all_searches, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock), cudaMemcpyDeviceToHost);
	

	//Free all the memory allocations
	cudaFree(device_frame_1);
	cudaFree(device_frame_2);
	cudaFree(device_MAD_all_searches);

	std::free(array_frame_1);
	std::free(array_frame_2);
}

//Main Function
int main(int argc, char* argv[])
{
	if(argc != 11)
	{
		#ifndef NDEBUG
		std::cerr << "Not enough input arguments\n" << std::flush;
		#endif
		return 0;
	}

	int search_dist 	= atoi(argv[1]);
	int grid_spacing 	= atoi(argv[2]); // --> grid_spacing
	int block_size_1 	= atoi(argv[3]); // --> 2 * block_size + 1
	int block_size_2 	= atoi(argv[4]);
	std::string in_dir(argv[5]);
	std::string filename1(argv[6]);
	std::string filename2(argv[7]);
	std::string out_dir(argv[8]);
	std::string filename_motion(argv[9]);
	int debug			= atoi(argv[10]);

	std::ofstream file_motion;
	
	// number of referenced blocks for each anchor block
	int n_searchs = (2 * search_dist + 1) * (2 * search_dist + 1);
	//Objects to hold the 2 frames
	jbutil::image<uint8_t> frame_1;
	jbutil::image<uint8_t> frame_2;
	
	//load frames
	if(!Load_Frames(in_dir + filename1, in_dir + filename2, frame_1, frame_2))
		return 0;
	//Image details
	int n_rows = frame_1.get_rows();
	int n_cols = frame_1.get_cols();
	int n_channels = 1;
	if (MULTI_CHANNELS)
		n_channels = frame_1.channels();

	//check the frames and parameters
	if(!Parameter_Check(frame_1, frame_2, search_dist, grid_spacing))
		return 0;

	//Object to hold the motion vector
	jbutil::image<uint8_t> frame_motion(frame_2.get_rows() / grid_spacing, 
										frame_2.get_cols() / grid_spacing, 3);
	//the number of macroblocks along the x and y directions
	int n_blocks_x = frame_motion.get_cols();
	int n_blocks_y = frame_motion.get_rows();
	
	//The following holds the MAD for all the search blocks for every macroblock in a linear manner
	MAD_per_Macroblock* mad_all_searches_1 = (MAD_per_Macroblock*)malloc(n_blocks_x * 
																		 n_blocks_y *
																		 sizeof(MAD_per_Macroblock));
	//Run the Block Matching and Reconstruction
	Block_Match(frame_1, frame_2, mad_all_searches_1, 
				n_rows, n_cols, n_channels,
				n_blocks_y, n_blocks_x, 
				grid_spacing, block_size_1, 
				search_dist, n_searchs);
	
	if (debug > 0){
		// saving motion vectors as image
		Motion_Image_Construct(frame_motion, mad_all_searches_1,
								n_blocks_y, n_blocks_x, 
								search_dist, n_searchs);
		file_motion.open((out_dir + filename_motion + "_1.ppm").c_str(), std::ios::out | std::ios::binary);
		frame_motion.save(file_motion);
		file_motion.close();
	}
		
	//The following holds the MAD for all the search blocks for every macroblock in a linear manner
	MAD_per_Macroblock* mad_all_searches_2 = (MAD_per_Macroblock*)malloc(n_blocks_x * 
																		 n_blocks_y * 
																		 sizeof(MAD_per_Macroblock));
	//Run the Block Matching and Reconstruction
	Block_Match(frame_1, frame_2, mad_all_searches_2, 
				n_rows, n_cols, n_channels,
				n_blocks_y, n_blocks_x,
				grid_spacing, block_size_2, 
				search_dist, n_searchs);

	if (debug > 0){
		Motion_Image_Construct(frame_motion, mad_all_searches_2,
								n_blocks_y, n_blocks_x, 
								search_dist, n_searchs);
		// saving motion vectors as image
		file_motion.open((out_dir + filename_motion + "_2.ppm").c_str(), std::ios::out | std::ios::binary);
		frame_motion.save(file_motion);
		file_motion.close();
	}

	// #ifndef NDEBUG
	// std::cerr << "Entering motion image construction with block_size_1 and block_size_2\n" << std::flush;
	// #endif
	MAD_per_Macroblock* mad_all_searches = (MAD_per_Macroblock*)malloc(n_blocks_x * 
																	n_blocks_y * 
																	sizeof(MAD_per_Macroblock));
	MAD_per_Macroblock* device_mad_all_searches;
	cudaMalloc((void**)&device_mad_all_searches, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock));
	cudaMemcpy(device_mad_all_searches, mad_all_searches, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock), cudaMemcpyHostToDevice);

	MAD_per_Macroblock* device_mad_all_searches_1;
	cudaMalloc((void**)&device_mad_all_searches_1, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock));
	cudaMemcpy(device_mad_all_searches_1, mad_all_searches_1, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock), cudaMemcpyHostToDevice);

	MAD_per_Macroblock* device_mad_all_searches_2;
	cudaMalloc((void**)&device_mad_all_searches_2, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock));
	cudaMemcpy(device_mad_all_searches_2, mad_all_searches_2, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock), cudaMemcpyHostToDevice);

	mean<<<(n_blocks_y * n_blocks_x + 256 - 1) / 256, 256>>>(n_blocks_y * n_blocks_x, n_searchs, device_mad_all_searches_1, device_mad_all_searches_2, device_mad_all_searches);
	cudaDeviceSynchronize();
	cudaMemcpy(mad_all_searches, device_mad_all_searches, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock), cudaMemcpyDeviceToHost);

	Motion_Image_Construct(frame_motion, mad_all_searches,
							n_blocks_y, n_blocks_x, 
							search_dist, n_searchs);
	// saving motion vectors as image
	file_motion.open((out_dir + filename_motion + ".ppm").c_str(), std::ios::out | std::ios::binary);
	frame_motion.save(file_motion);
	file_motion.close();

	// max<<<(n_blocks_y * n_blocks_x + 256 - 1) / 256, 256>>>(n_blocks_y * n_blocks_x, n_searchs, device_mad_all_searches_1, device_mad_all_searches_2, device_mad_all_searches);
	// cudaDeviceSynchronize();
	// cudaMemcpy(mad_all_searches, device_mad_all_searches, n_blocks_x*n_blocks_y*sizeof(MAD_per_Macroblock), cudaMemcpyDeviceToHost);
	// Motion_Image_Construct(frame_motion, mad_all_searches,
	// 						n_blocks_y, n_blocks_x, 
	// 						search_dist, n_searchs);
	// file_motion.open((out_dir + filename_motion + "_max.ppm").c_str(), std::ios::out | std::ios::binary);
	// frame_motion.save(file_motion);
	// file_motion.close();
	
	cudaFree(device_mad_all_searches);
	cudaFree(device_mad_all_searches_1);
	cudaFree(device_mad_all_searches_2);

	// free up data
	std::free(mad_all_searches);
	std::free(mad_all_searches_1);
	std::free(mad_all_searches_2);

	return 0;
}