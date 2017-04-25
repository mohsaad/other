#include "interlacing.h"
#include <iostream>
#include <string>


int main(int argc, char** argv)
{
    if(argc != 4)
    {
        std::cout << "Usage: ./interlace <video file 1> <video file 2> <output file>" << std::endl;
        return 0;
    }

    string video_file_1(argv[1]);
    string video_file_2(argv[2]);
    string output_file(argv[3]);

    Interlacer stitch(video_file_1, video_file_2, output_file);
    stitch.interlace();

    return 0;
}
