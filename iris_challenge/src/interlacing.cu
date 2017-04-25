#include "interlacing.h"

/*
    TODO:
    * write test cases for differing heights/widths
    * write test cases for null pointers
    * write test case for kernels

*/

__global__ void flip_image_kernel(double* input_image, size_t image_size)
{
    double temp;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tx < image_size / 2)
    {
        temp = input_image[i];
        input_image[i] = input_image[image_size - i];
        imput_image[image_size - i] = temp;
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

    read_first_video(video_1_name);
    read_second_video(video_2_name);

    // set appropriate height and width
    height = std::max(video_1->get(CV_CAP_PROP_FRAME_HEIGHT), video_2->get(CV_CAP_PROP_FRAME_HEIGHT));
    width = std::max(video_1->get(CV_CAP_PROP_FRAME_WIDTH), video_2->get(CV_CAP_PROP_FRAME_WIDTH));

    initialize_output_video(video_output_name);

}

void Interlacer::reset_videos()
{
    if(video_1 != NULL)
    {
        delete video_1;
        video_1 = NULL;
    }
    if(video_2 != NULL)
    {
        delete video_2;
        video_2 = NULL;
    }
    if(output_video != NULL)
    {
        delete output_video;
        output_video = NULL;
    }
}

void Interlacer::interlace()
{
    Mat video_frame_1, video_frame_2;

    // create a stream
    cudaStreamCreate(&stream);

    double * image_1;
    cudaMalloc((void**)&image_1, sizeof(double)*height*width);

    Dim3 dimBlock()

    while(video_1->read(video_frame_1) && video_2->read(video_frame_2))
    {
        cudaMemcpyAsync(&stream, (double)video_frame_1, sizeof(double)*height*width);
        flip_image_kernel<<<
    }

}

void Interlacer::flip_image(Mat & image)
{
    int image_type = cv::Mat::type(image);
    size_t num_rows = image.rows;
    size_t num_cols = image.cols;

    for(MatIterator im_it_start = image.begin(), MatIterator im_it_end = image.begin(); im_it_start < im_it_end; ++im_it_start, --im_it_end)
    {
        std::swap(*im_it_start, *im_it_end);
    }
}

void Interlacer::read_first_video(const string & video_1_name)
{
    video_1 = new VideoCapture();
    if(!video_1->open(video_1_name))
    {
        throw std::invalid_argument("Bad file for video 1!");s
    }

}

void Interlacer::read_second_video(const string & video_2_name)
{
    video_2 = new VideoCapture();
    if(!video_2->open(video_2_name))
    {
        throw std::invalid_argument("Bad file for video 2!");
    }
}

void Interlacer::initialize_output_video(const string & video_output_name)
{
    output_video = new VideoWriter();
    if(!output_video->open(video_output_name, CV_FOURCC('j','p','e','g'), 30.0, cv::Size(height, width)))
    {
        throw std::bad_alloc("Bad file for initializing output video!");
    }
}
