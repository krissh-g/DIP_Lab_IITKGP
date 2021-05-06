/*------- SPATIAL FILTERING-------------*/
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>

#define NO_OF_IMAGES 9
#define NO_OF_FILTERS 6
#define NO_OF_KERNELS 3
#define NO_OF_ORIENTS 3

using namespace std;
using namespace cv;

/* List of input images */
string input_file[] = {
"Cameraman_Salt&Pepper_0.08.jpg",
"jetplane.jpg",
"lena_gray_512.jpg",
"livingroom.jpg",
"mandril_gray.jpg",
"walkbridge.jpg",
"Pepper_Salt&Pepper_0.08.jpg",
"Camerman_Gaussian_0.05.jpg",
"Pepper_Salt&Pepper_0.08.jpg",
"Pepper_Gaussian_0.005.jpg"
};

string filter_name[] = {
"Mean filter",
"Median filter",
"Prewitt filter",
"Laplacian filter",
"Sobel filter",
"Gaussian filter",
"LOG filter"
};

string kernel_size[] = {
"Invalid kernel",
"3*3",
"5*5",
"7*7"
};

string orient_dir[] = {
"Default/Horizontal",
"Vertical",
"Diagonal"
};

enum filter
{
	MEAN,
	MEDIAN,
	PREWITT,
	LAPLACIAN,
	SOBEL,
	GAUSSIAN,
	LOG
};

/* Variables to hold the trackbar value */
/* select orient = 0 :DEFAULT,HORIZONTAL */
/* select orient = 1 :VERTICAL */
/* select orient = 0 :DIAGONAL */

int select_image;
int select_filter;
int select_size;
int select_orient;

/* Coefficients for different filters */
int mean_3_3[][3] = { { 1, 1, 1 },
				      { 1, 1, 1 },
					  { 1, 1, 1 } };

int mean_5_5[][5] = { { 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1 } };

int mean_7_7[][7] = { { 1, 1, 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1, 1, 1 },
					  { 1, 1, 1, 1, 1, 1, 1 } };

int prewitt_H_3_3[][3] = { {  1,  1,  1 },
					       {  0,  0,  0 },
					       { -1, -1, -1 } };

int prewitt_H_5_5[][5] =  { {  1,  1,  1,  1,  1 },
				            {  2,  2,  2,  2,  2 },
				            {  0,  0,  0,  0,  0 },
				            { -2, -2, -2, -2, -2 },
				            { -1, -1, -1, -1, -1 } };

int prewitt_H_7_7[][7] = { {  1,  1,  1,  1,  1,  1,  1 },
					       {  2,  2,  2,  2,  2,  2,  2 },
					       {  3,  3,  3,  3,  3,  3,  3 },
					       {  0,  0,  0,  0,  0,  0,  0 },
					       { -3, -3, -3, -3, -3, -3, -3 },
					       { -2, -2, -2, -2, -2, -2, -2 },
					       { -1, -1, -1, -1, -1, -1, -1 } };

int prewitt_V_3_3[][3] = { { 1, 0, -1 },
				           { 1, 0, -1 },
				           { 1, 0, -1 } };

int prewitt_V_5_5[][5] = { { 1, 2, 0, -2, -1},
						   { 1, 2, 0, -2, -1},
                           { 1, 2, 0, -2, -1},
						   { 1, 2, 0, -2, -1},
						   { 1, 2, 0, -2, -1} };

int prewitt_V_7_7[7][7] = { { 1, 2, 3, 0, -3, -2, -1},
						    { 1, 2, 3, 0, -3, -2, -1},
						    { 1, 2, 3, 0, -3, -2, -1},
						    { 1, 2, 3, 0, -3, -2, -1},
						    { 1, 2, 3, 0, -3, -2, -1},
						    { 1, 2, 3, 0, -3, -2, -1},
						    { 1, 2, 3, 0, -3, -2, -1} };

int laplacian_3_3[][3] = { {-1, -1, -1},
						   {-1,  8, -1},
						   {-1, -1, -1} };

int laplacian_5_5[][5] = { {-1, -1, -1, -1, -1},
					       {-1, -1, -1, -1, -1},
					       {-1, -1, 24, -1, -1},
					       {-1, -1, -1, -1, -1},
					       {-1, -1, -1, -1, -1} };

int laplacian_7_7[][7] =  { {-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, 48, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1},
							{-1, -1, -1, -1, -1, -1, -1} };

int sobel_H_3_3[][3] = { {  1,  2,  1},
						 {  0,  0,  0},
						 { -1, -2, -1} };

int sobel_H_5_5[][5] = { {  1,  4,   6,  4,  1},
						 {  2,  8,  12,  8,  2},
						 {  0,  0,   0,  0,  0},
						 { -2, -8, -12, -8, -2},
						 { -1, -4,  -6, -4, -1} };

int sobel_H_7_7[][7] = { {  1,   6,  15,   20,  15,   6,  1},
						 {  4,  24,  60,   80,  60,  24,  4},
						 {  5,  30,  75,  100,  75,  30,  5},
						 {  0,   0,   0,    0,   0,   0,  0},
						 { -5, -30, -75, -100, -75, -30, -5},
						 { -4, -24, -60,  -80, -60, -24, -4},
						 { -1,  -6, -15,  -20, -15,  -6, -1} };

int sobel_V_3_3[][3] = { { 1, 0, -1 },
						 { 2, 0, -2 },
						 { 1, 0, -1 } };

int sobel_V_5_5[][5] = { { 1,  2, 0,  -2, -1 },
						 { 4,  8, 0,  -8, -4 },
						 { 6, 12, 0, -12, -6 },
						 { 4,  8, 0,  -8, -4 },
						 { 1,  2, 0,  -2, -1 } };

int sobel_V_7_7[][7] = { {  1,  4,   5, 0,   -5,  -4,  -1},
				         {  6, 24,  30, 0,  -30, -24,  -6},
				         { 15, 60,  75, 0,  -75, -60,  -15},
				         { 20, 80, 100, 0, -100, -80,  -20},
				         { 15, 60,  75, 0,  -75, -60,  -15},
				         {  6, 24,  30, 0,  -30, -24,  -6},
				         {  1,  4,   5, 0,   -5,  -4,  -1} };

int sobel_D_3_3[][3] = { { 0, -1, -2},
						  { 1,  0, -1},
						  { 2,  1,  0} };

int sobel_D_5_5[][5] = { { 0, -2, -1, -4,  -6 },
						 { 2,  0, -8, -12, -4 },
						 { 1,  8,  0, -8,  -1 },
						 { 4, 12,  8,  0,  -2 },
						 { 6,  4,  1,  2,   0 } };

int gaussian_5_5[][5] = { { 1, 4,   7,  4, 1 },
						  { 4, 16, 26, 16, 4 },
						  { 7, 26, 41, 26, 7 },
						  { 4, 16, 26, 16, 4 },
						  { 1,  4,  7,  4, 1 } };

int LOG_5_5[][5] = { { 0, 0,   1, 0, 0 },
				     { 0, 1,   2, 1, 0 },
				     { 1, 2, -16, 2, 1 },
					 { 0, 1,   2, 1, 0 },
					 { 0, 0,   1, 0, 0 } };

/* Utility function for heap memory cleanup */

void destroy(int **mask,int size)
{
	for (int i = 0; i < size; i++)
	{
		delete[] mask[i];
	}

	delete[] mask;
	return;
}

/* Utility function for generating filter masks from existing */
/* filter coefficients */

int** getMask(int* m,int size)
{
	int** mask = new int*[size];
	for (int i = 0; i < size; i++)
	{
		mask[i] = new int[size];
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			mask[i][j] = *((m + (int64_t)i * size) + j);
		}
	}

	return mask;
}

/* Utility function for convolution. Each value is normalized */
/* to the sum of weight of the mask coefficients */

void handleConv(Mat img, int **mask, int size)
{
	int left = size/2;
	int top = size/2;
	Mat padded_img(img.rows + (int)size/2, img.cols + (int)size/2, img.type(), Scalar(0));
	Mat final_img(img.rows, img.cols, img.type(), Scalar(0));
	img.copyTo(padded_img(Rect(left, top, img.rows, img.cols)));
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int sum = 0;
			int weight = 0;
			int m = 0, n = 0;
			for (int k = - (int)size/2,m = 0; k <= (int)size/2; k++,m++)
			{
				if (((i + k) >= 0) && ((i + k) < img.cols))
				{
					for (int l = -(int)size / 2, n = 0; l <= (int)size / 2; l++, n++)
					{
						if (((j + l) >= 0) && ((j + l) < img.rows))
						{
							/*multiply and accumulate pixel values with mask coeffs */
							sum = sum + mask[m][n] * (int)padded_img.at<uchar>(i + k, j + l);
							weight = weight + mask[m][n];
						}
					}
				}
			}
			final_img.at<uchar>(i, j) = saturate_cast<uchar>(sum/weight);
		}
	}

	cv::imshow("Output Image", final_img);
	destroy(mask,size);
	return;
}

/* Utility function for convolution. Each pixel value is recalculated */
/* using the neighborhood pixels and the mask coeffs */

void handleSpatialFilter(Mat img,int **mask,int size)
{
	int left = size / 2;
	int top = size / 2;
	Mat padded_img(img.rows + (int)size / 2, img.cols + (int)size / 2, img.type(), Scalar(0));
	Mat final_img(img.rows, img.cols, img.type(), Scalar(0));
	img.copyTo(padded_img(Rect(left, top, img.rows, img.cols)));
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int sum = 0;
			int count = 0;
			int m = 0, n = 0;
			for (int k = -(int)size / 2, m = 0; k <= (int)size / 2; k++, m++)
			{
				if (((i + k) >= 0) && ((i + k) < img.cols))
				{
					for (int l = -(int)size / 2, n = 0; l <= (int)size / 2; l++, n++)
					{
						if (((j + l) >= 0) && ((j + l) < img.rows))
						{
							/*multiply and accumulate pixel values with mask coeffs */
							sum = sum + mask[m][n] * (int)padded_img.at<uchar>(i + k, j + l);
							count++;
						}
					}
				}
			}
			final_img.at<uchar>(i, j) = saturate_cast<uchar>(sum);
		}
	}

	cv::imshow("Output Image", final_img);
	destroy(mask, size);
	return;

}

/* Utility function to handle mean filtering */
/* Get the mask coeffs corresponding to the kernel size and run 2D convolution*/

void handleMeanFilter()
{
	int** mask;
	Mat img = imread(input_file[select_image], IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}

	cv::imshow("Input Image",img);
	if (1 == select_size)
	{
		mask = getMask(&mean_3_3[0][0], 3);
		handleConv(img, mask, 3);
	}
	else if (2 == select_size)
	{
		mask = getMask(&mean_5_5[0][0], 5);
		handleConv(img, mask, 5);
	}
	else if (3 == select_size)
	{
		mask = getMask(&mean_7_7[0][0], 7);
		handleConv(img, mask, 7);
	}

	return;
}

/* Utility function to handle median filtering */

void handleMedianFilter()
{
	Mat img = imread(input_file[select_image], IMREAD_GRAYSCALE);
	cv::imshow("Input Image", img);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}

	if (select_size <= 0)
	{
		return;
	}

	int arr_size = (2 * select_size + 1) * (2 * select_size + 1); //neighborhood size
	int k_size = 2 * select_size + 1; // define kernel size side
	vector<int> to_sort(arr_size, 0); // initialize vector to hold neighborhood elements
	Mat final_img = img.clone();
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int count = 0;
			to_sort.clear();
			for (int k = - (k_size / 2); k <= (k_size / 2); k++)
			{
				if (((i + k) >= 0) && ((i + k) < img.cols))
				{
					for (int l = - (k_size / 2); l <= (k_size / 2); l++)
					{
						if (((j + l) >= 0) && ((j + l) < img.rows))
						{
							to_sort.push_back(img.at<uchar>(i + k, j + l));
							count++;
						}
					}
				}
			}
			/* sort the neighborhood pixels and assign the median pixel as output */
			sort(to_sort.begin(), to_sort.end());
			final_img.at<uchar>(i, j) = to_sort[count/2];
		}
	}
	
	cv::imshow("Output Image", final_img);
	return;
}

/* Utility function to handle prewitt filtering */

void handlePrewtittFilter()
{
	int** mask;
	Mat img = imread(input_file[select_image], IMREAD_GRAYSCALE);
	cv::imshow("Input Image", img);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}

	if (0 == select_orient)
	{
		if (1 == select_size)
		{
			mask = getMask(&prewitt_H_3_3[0][0], 3);
			handleSpatialFilter(img, mask, 3);
		}
		else if (2 == select_size)
		{
			mask = getMask(&prewitt_H_5_5[0][0], 5);
			handleSpatialFilter(img, mask, 5);
		}
		else if (3 == select_size)
		{
			mask = getMask(&prewitt_H_7_7[0][0], 7);
			handleSpatialFilter(img, mask, 7);
		}
	}
	if (1 == select_orient)
	{
		if (1 == select_size)
		{
			mask = getMask(&prewitt_V_3_3[0][0], 3);
			handleSpatialFilter(img, mask, 3);
		}
		else if (2 == select_size)
		{
			mask = getMask(&prewitt_V_5_5[0][0], 5);
			handleSpatialFilter(img, mask, 5);
		}
		else if (3 == select_size)
		{
			mask = getMask(&prewitt_V_7_7[0][0], 7);
			handleSpatialFilter(img, mask, 7);
		}
	}
}

/* Utility function to handle laplacian filtering */

void handleLaplacianFilter()
{
	int** mask;
	Mat img = imread(input_file[select_image], IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}
	cv::imshow("Input Image", img);
	if (1 == select_size)
	{
		mask = getMask(&laplacian_3_3[0][0], 3);
		handleSpatialFilter(img, mask, 3);
	}
	else if (2 == select_size)
	{
		mask = getMask(&prewitt_H_5_5[0][0], 5);
		handleSpatialFilter(img, mask, 5);
	}
	else if (3 == select_size)
	{
		mask = getMask(&prewitt_H_7_7[0][0], 7);
		handleSpatialFilter(img, mask, 7);
	}

	return;
}

/* Utility function to handle sobel filtering */

void handleSobelFilter()
{
	int** mask;
	Mat img = imread(input_file[select_image], IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}
	cv::imshow("Input Image", img);
	if (0 == select_orient)
	{
		if (1 == select_size)
		{
			mask = getMask(&sobel_H_3_3[0][0], 3);
			handleSpatialFilter(img, mask, 3);
		}
		else if (2 == select_size)
		{
			mask = getMask(&sobel_H_5_5[0][0], 5);
			handleSpatialFilter(img, mask, 5);
		}
		else if (3 == select_size)
		{
			mask = getMask(&sobel_H_7_7[0][0], 7);
			handleSpatialFilter(img, mask, 7);
		}
	}
	if (1 == select_orient)
	{
		if (1 == select_size)
		{
			mask = getMask(&sobel_V_3_3[0][0], 3);
			handleSpatialFilter(img, mask, 3);
		}
		else if (2 == select_size)
		{
			mask = getMask(&sobel_V_5_5[0][0], 5);
			handleSpatialFilter(img, mask, 5);
		}
		else if (3 == select_size)
		{
			mask = getMask(&sobel_V_7_7[0][0], 7);
			handleSpatialFilter(img, mask, 7);
		}
	}
	if (2 == select_orient)
	{
		if (1 == select_size)
		{
			mask = getMask(&sobel_D_3_3[0][0], 3);
			handleSpatialFilter(img, mask, 3);
		}
		else if (2 == select_size)
		{
			mask = getMask(&sobel_D_5_5[0][0], 5);
			handleSpatialFilter(img, mask, 5);
		}
	}
}
void handleGaussianFilter()
{
	if (2 != select_size)
	{
		return;
	}
	int** mask;
	Mat img = imread(input_file[select_image], IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}
	cv::imshow("Input Image", img);

	mask = getMask(&gaussian_5_5[0][0], 5);
	handleConv(img, mask, 5);
}

void handleLOGFilter()
{
	if (2 != select_size)
	{
		return;
	}
	int** mask;
	Mat img = imread(input_file[select_image], IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}
	cv::imshow("Input Image", img);

	mask = getMask(&LOG_5_5[0][0], 5);
	handleSpatialFilter(img, mask, 5);
}

/* callback function to handle any change in trackbar parameters */

void OnSelection(int,void*)
{
	cout << "image selected:" << input_file[select_image] << endl;
	cout << "filter selected:" << filter_name[select_filter] << endl;
	cout << "kernel size:" << kernel_size[select_size] << endl;
	cout << "Orientation:" << orient_dir[select_orient] << endl;
	switch (select_filter)
	{
	case MEAN:
		handleMeanFilter();
		break;
	case MEDIAN:
		handleMedianFilter();
		break;
	case PREWITT:
		handlePrewtittFilter();
		break;
	case LAPLACIAN:
		handleLaplacianFilter();
		break;
	case SOBEL:
		handleSobelFilter();
		break;
	case GAUSSIAN:
		handleGaussianFilter();
		break;
	case LOG:
		handleLOGFilter();
		break;
	default:
		break;
	}
	return;
}

int main()
{
	namedWindow("spatial filtering");
	resizeWindow("spatial filtering", 320, 240);
	createTrackbar("ImageID", "spatial filtering", &select_image, NO_OF_IMAGES,OnSelection);
	createTrackbar("FilterID", "spatial filtering", &select_filter, NO_OF_FILTERS,OnSelection);
	createTrackbar("KernelID", "spatial filtering", &select_size, NO_OF_KERNELS,OnSelection);
	createTrackbar("OrientID", "spatial filtering", &select_orient, NO_OF_ORIENTS, OnSelection);
    
	waitKey(0);
	return 0;
}

