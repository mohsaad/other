/*
    Mohammad Saad
    4/23/2017
    interlacing.h

    Given two video files, extract each frame from each video
    and interlace them such that a stream aaaaa and a stream bbbbb outputs
    ababababab.





*/



#ifndef __INTERLACING_H__
#define __INTERLACING_H__

#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <exception>
#include <string>
#include <algorithm>

#define BLOCK_SIZE 256


using namespace cv;
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
    This class defines the functions which we will be using to interlace,
    both on the host device and in CUDA.
*/
class Interlacer
{
    public:
        /*
            Public constructor. Creates an interlacer object.
            Doesn't really do anything for now, as we don't have any member variables,
            Sets most everything to null.
        */
        Interlacer();

        /* Destructor. Releases any VideoCapture objects and VideoWriter objects. */
        ~Interlacer();

        /* Copy constructor. Copies fields. */
        Interlacer(const Interlacer & interlace);

        /* A different constructor with filename inputs. */
        Interlacer(const string & video_1_name, const string & video_2_name, const string & video_output_name);

        /* An initialization function for both videos */
        void initialize_interlacer(const string & video_1_name, const string & video_2_name, const string & video_output_name);

        /*  The interlacing function. */
        void interlace();





    private:
        /* Define two videocapture objects. This will be object that loads
        both our streams in. We will use these to read and write files in and
        out of our program. */
        VideoCapture* video_1 = NULL; // in this example, a
        VideoCapture* video_2 = NULL; // in this example, b

        VideoWriter* output_video = NULL; // our output file writer.

        /* Properties of the video files themselves. We need these to do some
        calculations regarding when to put in a frame. */
        size_t height;
        size_t width;

        /* Method for reading the first video file */
        void read_first_video(const string & video_1_name);

        /* Method for reading the first video file */
        void read_second_video(const string & video_2_name);

        /* Method for initializing the output file */
        void initialize_output_video(const string & video_output_name);

        /* A reset/clear function */
        void reset_videos();

        // A CPU-based image inverter.
        void flip_image(Mat & im);

        // A cuda stream for parallelization
        cudaStream_t stream;

};


#endif
