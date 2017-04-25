#include "interlacing.h"

/*
    flip_image_kernel
    A CUDA kernel for inverting the image.
    Inputs:
        input_image -> a linearized array of the image
        num_rows -> number of rows in image
        num_cols -> number of columns in image
        num_elements -> number of elements in image

*/

__global__ void flip_image_kernel(uint8_t* input_image, size_t num_rows, size_t num_cols, size_t num_elements)
{
    uint8_t temp;
    int ty = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = blockIdx.y * blockDim.y + threadIdx.y;

    size_t pixel_ind = ty*num_rows + tx;

    // Similar to the pixel swapping algorithm below,
    // but this time we assign each thread to swap two pixels.
    // the pixels are determined by the thread indices, both in x
    // and in y. Using those, we can map the threads to a block
    // and gain more parallelism that way.
    if(pixel_ind < num_elements /  2)
    {
        temp = input_image[pixel_ind];
        input_image[pixel_ind] = input_image[num_elements - (pixel_ind + 1)];
        input_image[num_elements - (pixel_ind + 1)] = temp;
    }
}

Interlacer::Interlacer()
{
    // initialize everything to just a basic constructor, no arguments, everything is null
    reset_videos();
    height = 0;
    width = 0;
    video_1 = NULL;
    video_2 = NULL;
    output_video = NULL;
}

Interlacer::~Interlacer()
{
    // call reset_videos to do the dirty work
    reset_videos();
}

Interlacer::Interlacer(const Interlacer & interlace_obj)
{
    reset_videos();
    video_1 = interlace_obj.video_1;
    video_2 = interlace_obj.video_2;
    height = interlace_obj.height;
    width = interlace_obj.width;
}

Interlacer::Interlacer(const string & video_1_name, const string & video_2_name, const string & video_output_name)
{
    reset_videos();
    initialize_interlacer(video_1_name, video_2_name, video_output_name);
}

void Interlacer::initialize_interlacer(const string & video_1_name, const string & video_2_name, const string & video_output_name)
{
    reset_videos();

    // read the videos
    read_first_video(video_1_name);
    read_second_video(video_2_name);

    // set appropriate height and width
    height = std::max(video_1->get(CV_CAP_PROP_FRAME_HEIGHT), video_2->get(CV_CAP_PROP_FRAME_HEIGHT));
    width = std::max(video_1->get(CV_CAP_PROP_FRAME_WIDTH), video_2->get(CV_CAP_PROP_FRAME_WIDTH));

    initialize_output_video(video_output_name);

}

void Interlacer::reset_videos()
{
    // if we want to call the destructor,
    // release all videos first,
    // and then delete the pointer
    if(video_1 != NULL)
    {
        video_1 -> release();
        delete video_1;
        video_1 = NULL;
    }
    if(video_2 != NULL)
    {
        video_2 -> release();
        delete video_2;
        video_2 = NULL;
    }
    if(output_video != NULL)
    {
        output_video->release();
        delete output_video;
        output_video = NULL;
    }
}

void Interlacer::interlace()
{
    if(video_1 == NULL || video_2 == NULL || output_video == NULL)
    {
        std::cout << "Videos not initialized!" << std::endl;
        return;
    }

    Mat video_frame_1, video_frame_2;
    Mat gray_frame_1, gray_frame_2;
    Mat resized_frame_1, resized_frame_2;

    // create a stream
    cudaStreamCreate(&stream);

    // create a device pointer
    uint8_t * image_1;
    gpuErrchk(cudaMalloc((void**)&image_1, sizeof(uint8_t)*height*width));

    // grid and block dimensions for the video
    dim3 dimGrid(ceil(height * 1.0/BLOCK_SIZE), ceil(width * 1.0/BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    int i = 0;
    // loop through each frame until we have
    while(video_1->read(video_frame_1) && video_2->read(video_frame_2))
    {
        // convert to grayscale in case it already isn't
        // for now, we assume everything is grayscale because of ambiguity
        cvtColor(video_frame_1, gray_frame_1, CV_BGR2GRAY);
        cvtColor(video_frame_2, gray_frame_2, CV_BGR2GRAY);

        // resize to max resolution
        resize(gray_frame_1, resized_frame_1, cv::Size(width, height));
        resize(gray_frame_2, resized_frame_2, cv::Size(width, height));

        // copy first image to GPU
        cudaMemcpyAsync(image_1, (uint8_t*)resized_frame_1.data, sizeof(uint8_t)*height*width, cudaMemcpyHostToDevice, stream);
        // execute kernel, doesn't block so we can continue doing work on CPU
        flip_image_kernel<<<dimGrid, dimBlock, 0, stream>>>(image_1, width, height, height*width);

        // flip the second image
        flip_image(resized_frame_2);

        // copy the kernel code which should be done by now - in case it isn't, we block until previous kernel calls finish
        cudaMemcpy((uint8_t*)resized_frame_1.data, image_1, sizeof(uint8_t)*height*width, cudaMemcpyDeviceToHost);
        output_video->write(resized_frame_1);
        output_video->write(resized_frame_2);

        // synchronize devices
        cudaDeviceSynchronize();
        i++;
        std::cout << "Processed frames " << i << std::endl;
    }

    cudaFree(image_1);
    cudaStreamDestroy(stream);

    std::cout << "Completed interlacing!" << std::endl;

}

void Interlacer::flip_image(Mat & image)
{
    // get some parameters relating to the data,
    // including the number of channels and total
    // elements in the data
    int num_channels = image.channels();
    uint8_t* imdata = (uint8_t*)image.data;
    size_t num_elements = image.total();

    // We're going to do this by linearizing the entire data
    // and flipping the first and last elements, and then iterating by
    // num_channels. This allows us to still remain in O(n) time,
    // but also handle color or grayscale images regardless of
    // the image type
    // i points to a block of elements in data, and j points to each
    // element in that block. We then swap the i+jth element with
    // it's corresponding element at the end of the array, addressed by
    // (num_elements - (num_channels)*(i+1) + j
    for(size_t i = 0; i < (num_elements / num_channels) / 2; i += num_channels)
    {
        for(int j = 0; j < num_channels; j++)
        {
            std::swap(imdata[i+j], imdata[num_elements - num_channels*(i+1) + j]);
        }
    }

}

void Interlacer::read_first_video(const string & video_1_name)
{
    video_1 = new VideoCapture();
    if(!video_1->open(video_1_name))
    {
        reset_videos();
        throw std::invalid_argument("Video 1 file not found!");
    }

}

void Interlacer::read_second_video(const string & video_2_name)
{
    video_2 = new VideoCapture();
    if(!video_2->open(video_2_name))
    {
        reset_videos();
        throw std::invalid_argument("Video 2 file not found!");
    }
}

void Interlacer::initialize_output_video(const string & video_output_name)
{
    output_video = new VideoWriter();
    if(!output_video->open(video_output_name, CV_FOURCC('M','J','P','G'), 10, cv::Size(width, height), false))
    {
        reset_videos();
        throw std::invalid_argument("Output video invalid!");
    }
}
