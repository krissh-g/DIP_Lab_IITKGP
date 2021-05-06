#include <iostream>
#include <string>
#include <fstream>
#include <cstring>

#define BM 0x4D42
#define RGB_CHANNEL 3
#define GRAY_CHANNEL 1
#define GRAY_SCALE_OFFSET 1078
#define GRAY_BITS_PER_PIXEL 8
#define SIZE_COLOR_TABLE 256

#pragma pack(2)
/* BMP Header structure */
typedef struct BMPHeader
{
    /* BMP File Header */
    uint16_t signature;      /* Signature of Image file */
    uint32_t file_size;      /* File size in bytes */
    uint32_t reserved;       /* Unused (=0) Application Specific */
    uint32_t data_offset;    /* Offset from beginning of file to the beginning of the bitmap data */
    /* BMP Info Header */
    uint32_t size;           /* Size of InfoHeader */
    int32_t  width;          /* Horizontal width of bitmap in pixels */
    int32_t  height;         /* Vertical height of bitmap in pixels */
    uint16_t planes;         /* Number of Planes (=1) */
    uint16_t bits_per_pixel; /* Bits per Pixel used to store palette information */
    uint32_t compression;    /* Type of Compression */
    uint32_t image_size;     /* (compressed) Size of Image */
    int32_t  xpixels_per_M;  /* horizontal resolution */
    int32_t  ypixels_per_M;  /* vertical resolution */
    uint32_t colors_used;    /* Number of actually used colors */
    uint32_t imp_colors;     /* Number of important colors */
}BMPHeader;

/* Color Table entry structure */
typedef struct rgb
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char resv;
}rgb;

#pragma pack()

/* Bitmap structure */
typedef struct Bitmap
{
    BMPHeader header;
    unsigned char*** ImgPixarray;
}BitMap;

/* Utility function to calculate no of channels */
uint32_t CalculateChannel(uint16_t& bits_per_pixel)
{
    uint32_t channel = 0;
    if (8 == bits_per_pixel)
    {
        channel = 1;
    }
    else if (24 == bits_per_pixel)
    {
        channel = 3;
    }
    else
    {
        std::cerr << "Currently not supported" << std::endl;
        exit(1);
    }

    return channel;
}

/* Utility function to clean up heap memory */
void destroy(unsigned char*** temp, int32_t height, uint16_t bpp)
{
    std::cout << "destroy called" << std::endl;
    uint32_t no_of_channel = CalculateChannel(bpp);
    for (int32_t i = 0; i < no_of_channel; i++)
    {
        for (int32_t j = 0; j < height; j++)
        {
            delete[] temp[i][j];
        }
        delete[] temp[i];
    }
    delete[] temp;
}

/* Utility function to calculate stride */
int32_t CalculateStride(int32_t width, uint16_t bits_per_pixel)
{
    int32_t stride = (int32_t)((((width * (int32_t)bits_per_pixel) + 31) & ~31) >> 3);
    return stride;
}

/* Utility function to create Grayscale Image */
unsigned char*** ConvertGrayscale(Bitmap* bitmap)
{
    std::cout << "convert grayscale" << std::endl;
    unsigned char*** GrayImgPixarray = new unsigned char** [1];
    GrayImgPixarray[0] = new unsigned char* [bitmap->header.height];
    for (int32_t i = 0; i < bitmap->header.height; i++)
    {
        GrayImgPixarray[0][i] = new unsigned char[bitmap->header.width];
    }

    // Convert to grayscale if channel is RGB
    if (RGB_CHANNEL == CalculateChannel(bitmap->header.bits_per_pixel))
    {
        for (int32_t i = 0; i < bitmap->header.height; i++)
        {
            for (int32_t j = 0; j < bitmap->header.width; j++)
            {
                // Calculate grayscale value from RGB arrays : use grayval = R*0.30 + G*0.59 + B*0.11
                unsigned char temp = (unsigned char)(bitmap->ImgPixarray[0][i][j] * 0.30) + (bitmap->ImgPixarray[1][i][j] * 0.59)
                    + (bitmap->ImgPixarray[2][i][j] * 0.11);
                GrayImgPixarray[0][i][j] = temp;
            }
        }
    }
    else if (GRAY_CHANNEL == CalculateChannel(bitmap->header.bits_per_pixel))
    {
        GrayImgPixarray = bitmap->ImgPixarray;
    }
    return GrayImgPixarray;
}

/* Utility function to print important header information */
void PrintHeaderInfo(const BMPHeader& header)
{
    std::cout << "Image Height:" << header.height << "\n";
    std::cout << "Image width:" << header.width << "\n";
    std::cout << "Image bitwidth:" << header.bits_per_pixel << "\n";
    std::cout << "File size in bytes:" << header.file_size << "\n";
    std::cout << "Offset size:" << header.data_offset << "\n";
}

/* ReadBMP reads the bitmap input file, stores the header and the Image pixel array */
/* Input: input_filename ->input file string.,.bitmap-> Bitmap structure pointer... */
/* Output: BMPHeader sturucture.....................................................*/

BMPHeader ReadBMP(const std::string& input_filename, BitMap* bitmap)
{
    BMPHeader header = { 0 };
    FILE* fp;
    fopen_s(&fp, input_filename.c_str(), "rb");

    if (NULL == fp)
    {
        std::cerr << "File ptr is NULL , return" << std::endl;
        return header;
    }

    /* Read image header info into header structure */
    fread(&header, sizeof(BMPHeader), 1, fp);
    std::cout << "header sig:" << header.signature << " header size:" << sizeof(BMPHeader) << "\n";

    /* Return error if header signature is not BM (Bitmap signature) */
    if (BM != header.signature)
    {
        std::cerr << "Not a BMP file, return" << std::endl;
        return header;
    }

    /* Print important header information */
    PrintHeaderInfo(header);

    fseek(fp, 0, SEEK_SET);

    uint32_t no_of_channel = CalculateChannel(header.bits_per_pixel);
    int32_t stride = CalculateStride(header.width, header.bits_per_pixel);
    int32_t padding = stride - (header.width * no_of_channel);

    std::cout << "\n" << "nc: " << no_of_channel << "stride: " << stride << "pad: " << padding << std::endl;

    /* Initialize Bitmap Imagepixel array */
    bitmap->ImgPixarray = new unsigned char** [no_of_channel];
    for (int32_t i = 0; i < no_of_channel; i++)
    {
        std::cout << "channel i:" << i << std::endl;
        bitmap->ImgPixarray[i] = new unsigned char* [header.height];
        for (int32_t j = 0; j < header.height; j++)
        {
            bitmap->ImgPixarray[i][j] = new unsigned char[header.width];
            memset(bitmap->ImgPixarray[i][j], '\0', sizeof(unsigned char) * header.width);
        }
    }

    /* Read Image pixel data in Imagepixel array from Bitmap */
    for (int32_t row = header.height - 1; row >= 0; row--)
    {
        int32_t i = header.height - 1 - row;
        fseek(fp, header.data_offset + stride * row, SEEK_SET);
        for (int32_t j = 0; j < header.width; j++) 
        {
            for (int32_t k = no_of_channel - 1; k >= 0; k--) 
            {
                fread(&(bitmap->ImgPixarray[k][i][j]), sizeof(unsigned char), 1, fp);
            }
        }
    }

    bitmap->header = header;
    fclose(fp);
    return bitmap->header;
}

/* ConvertFlipGrayscale converts the image in Grayscale and flips it along diagonal */
/* Input: bitmap-> Bitmap structure pointer........................................ */
/* Output: Pointer to flipped image pixel array.....................................*/

unsigned char*** ConvertFlipGrayscale(BitMap* bitmap)
{
    /* Convert Image to Grayscale first */
    unsigned char*** GrayImgPixarray = ConvertGrayscale(bitmap);

    /* Initialize flipped image pixel array */
    unsigned char*** FlipImgPixarray = new unsigned char** [GRAY_CHANNEL];
    FlipImgPixarray[0] = new unsigned char* [bitmap->header.width];
    for (int32_t i = 0; i < bitmap->header.width; i++)
    {
        FlipImgPixarray[0][i] = new unsigned char[bitmap->header.height];
    }

    /* Flip Image array along diagonal */
    for (int32_t i = 0; i < bitmap->header.height; i++)
    {
        for (int32_t j = 0; j < bitmap->header.width; j++)
        {
            /* Flip image along main diagonal */
            //FlipImgPixarray[0][j][i] = GrayImgPixarray[0][i][j];
            /* FLip Image along secondary diagonal */
            FlipImgPixarray[0][j][i] = GrayImgPixarray[0][bitmap->header.height - 1 - i][bitmap->header.width - 1 - j];
        }
    }

    destroy(GrayImgPixarray, bitmap->header.height, 8);
    //  Swapping value of the width and height
    int32_t temp = bitmap->header.width;
    bitmap->header.width = bitmap->header.height;
    bitmap->header.height = temp;
    bitmap->header.data_offset = GRAY_SCALE_OFFSET;
    bitmap->header.bits_per_pixel = GRAY_BITS_PER_PIXEL;
    bitmap->header.image_size = bitmap->header.height * bitmap->header.width;
    return FlipImgPixarray;
}


/* RotateGrayscale rotates the grayscale image by 90 degrees anticlockwise......... */
/* Input: header-> Bitmap header pointer , ImgPixarray->Image pixel array pointer.. */
/* Output: Pointer to rotated image pixel array.................................... */

unsigned char*** RotateGrayscale(BMPHeader* header, unsigned char*** ImgPixarray)
{
    unsigned char*** rotatePixarray = new unsigned char** [1];
    rotatePixarray[0] = new unsigned char* [header->width];
    for (int32_t j = 0; j < header->width; j++)
    {
        rotatePixarray[0][j] = new unsigned char[header->height];
    }

    /* Rotate image anticlockwise by 90 degrees */
    int32_t m = 0;
    for (int32_t j = header->width - 1; j >= 0; j--)
    {
        int32_t n = 0;
        for (int32_t i = 0; i < header->height; i++)
        {
            rotatePixarray[0][m][n] = ImgPixarray[0][i][j];
            n++;
        }
        m++;
    }

    /* Swap header width and height */
    int32_t temp = header->width;
    header->width = header->height;
    header->height = temp;
    return rotatePixarray;
}

/* WriteBMP writes to bitmap output file, stores the header and the Image pixel array*/
/* Input: output_filename ->output file string.,.header-> Bitmap header, ............*/
/* .......ImgPixarray -> Image pixel array pointer...................................*/
/* Output: None......................................................................*/

void WriteBMP(std::string output_filename, BMPHeader header, unsigned char*** ImgPixarray)
{
    std::cout << "\n" << "enter writebmp" << std::endl;

    //  opening the Bitmap file
    output_filename = output_filename + ".bmp";
    FILE* fp;
    fopen_s(&fp, output_filename.c_str(), "wb");

    //  If error occured in opening file
    if (NULL == fp)
    {
        std::cout << "File ptr is NULL , return" << std::endl;
        return;
    }

    std::cout << "file opened:" << std::endl;

    /* Write headerinfo to the file */
    fwrite(&header, sizeof(BMPHeader), 1, fp);
    std::cout << "header written" << std::endl;

    /* Write color Table */
    if (GRAY_BITS_PER_PIXEL == header.bits_per_pixel)
    {
        std::cout << "write colortable" << std::endl;
        rgb* colortable = new rgb[SIZE_COLOR_TABLE];
        std::cout << "size rgb:" << sizeof(rgb) << std::endl;
        for (int32_t i = 0; i < SIZE_COLOR_TABLE; i++)
        {
            colortable[i].r    = (unsigned char)i;
            colortable[i].g    = (unsigned char)i;
            colortable[i].b    = (unsigned char)i;
            colortable[i].resv = (unsigned char)i;
        }
        fwrite(colortable, sizeof(rgb), SIZE_COLOR_TABLE, fp);
        delete[] colortable;
    }

    int32_t stride = CalculateStride(header.width, header.bits_per_pixel);
    uint32_t no_of_channel = CalculateChannel(header.bits_per_pixel);
    int32_t padding = stride - (header.width * no_of_channel);

    /* Write Image pixel data to file */
    for (int32_t row = header.height - 1; row >= 0; row--)
    {
        unsigned char c = '\0';
        int32_t i = header.height - 1 - row;
        fseek(fp, header.data_offset + stride * row, SEEK_SET);
        for (int32_t j = 0; j < header.width; j++) {
            for (int32_t k = no_of_channel - 1; k >= 0; k--)
            {
                fwrite(&(ImgPixarray[k][i][j]), sizeof(unsigned char), 1, fp);
            }
        }
        if (padding > 0)
        {
            for (int32_t p = 0; p < padding; p++)
            {
                fwrite(&c, sizeof(unsigned char), 1, fp);
            }
        }
    }

    fclose(fp);
}

int main()
{
    std::string input_filename;
    BitMap bitmap = { 0 };
    BMPHeader bmpheader = { 0 };
    std::cout << "Enter input file name with extension:" << std::endl;
    std::cin >> input_filename;
    bmpheader = ReadBMP(input_filename, &bitmap);

    unsigned char*** FlipImgPixarray = ConvertFlipGrayscale(&bitmap);
    WriteBMP("output_flip", bitmap.header, FlipImgPixarray);

    unsigned char*** rotatePixarray = RotateGrayscale(&(bitmap.header), FlipImgPixarray);
    WriteBMP("output_rotate", bitmap.header, rotatePixarray);

    destroy(FlipImgPixarray, bitmap.header.height, 8);
    destroy(rotatePixarray, bitmap.header.height, 8);

    std::cout << "Successfully done" << std::endl;
    return 0;
}