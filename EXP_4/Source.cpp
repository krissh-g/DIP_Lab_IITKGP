#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <complex>

#define NO_OF_IMAGES 7
#define NO_OF_FILTERS 5
#define NO_OF_CUTOFF_FREQ 3
#define PI 3.141592

using namespace std;
using namespace cv;

string input_file[] = {
"lena_gray_512.jpg",
"lake.jpg",
"livingroom.jpg",
"jetplane.jpg",
"pirate.jpg",
"walkbridge.jpg",
"mandril_gray.jpg",
"cameraman.jpg"
};

string filter_name[] = {
"Ideal_LPF",
"Ideal_HPF",
"Gaussian_LPF",
"Gaussian_HPF",
"Butterworth_LPF",
"Butterworth_HPF"
};

enum filter
{
	Ideal_LPF,
	Ideal_HPF,
	Gaussian_LPF,
	Gaussian_HPF,
	Butterworth_LPF,
	Butterworth_HPF
};

int select_image = 0;
int select_filter = 0;
int select_cutoff = 1;

void destroy(complex<double>** fft,int N)
{
	for (int i = 0; i < N; i++)
	{
		delete[] fft[i];
	}

	delete[] fft;
}

complex<double>** calc_transpose(complex<double>** fft, int N)   //to perform transpose of given 2-D array
{
	complex<double> cmplx;
	for (int i = 0; i < N; i++) {

		for (int j = i + 1; j < N; j++) {
			cmplx = fft[i][j];
			fft[i][j] = fft[j][i];
			fft[j][i] = cmplx;
		}
	}

	return fft;
}
void shift_FFT(Mat& img)
{
	int N = img.rows;
	for (int i = 0; i < N / 2; i++)
	{
		for (int j = 0; j < N / 2; j++)
		{
			float ch = img.at<float>(i, j);
			img.at<float>(i, j) = img.at<float>(i + N / 2, j + N / 2);
			img.at<float>(i + N / 2, j + N / 2) = ch;
		}
	}

	for (int i = N / 2; i < N; i++) {
		for (int j = 0; j < N / 2; j++) {
			float ch = img.at<float>(i, j);
			img.at<float>(i, j) = img.at<float>(i - N / 2, j + N / 2);
			img.at<float>(i - N / 2, j + N / 2) = ch;
		}
	}
}

Mat normalize(complex<double>** fft, int N,int max_range)
{
	cout << "Enter normalize" << endl;
	Mat output(N, N, CV_32F, Scalar(0));
	double mag = 0.0f;
	double max = abs(fft[0][0]) / N;
	double min = abs(fft[0][0]) / N;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			fft[i][j] = fft[i][j] / (double)N;
			mag = abs(fft[i][j]);
			if (mag < min)
			{
				min = mag;
			}
			else if (mag > max)
			{
				max = mag;
			}
		}
	}
	cout << "values:" << endl;
	cout << "max: " << max << "min: " << min << "N:" << N << endl;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			double temp = ((abs(fft[i][j]) - min) / (max - min)) * max_range;
			output.at<float>(i, j) = (float)(temp);
		}
	}

	return output;
}
void calc_1DFFT(complex<double>* fft, int N)
{
	if (N <= 1)
	{
		return;
	}

	/* Declare even and odd arrays */
	complex<double>* odd = new complex<double>[N / 2];
	complex<double>* even = new complex<double>[N / 2];
	for (int i = 0; i < N / 2; i++)
	{
		even[i] = fft[i * 2];
		odd[i] = fft[i * 2 + 1];
	}

	/* Recursively call even and odd array to compute FFT */
	calc_1DFFT(even, N / 2);
	calc_1DFFT(odd, N / 2);


	/* Join the recursively calculated array */
	for (int k = 0; k < N / 2; k++)
	{
		complex<double> w = exp(complex<double>(0, -2 * PI * k / N));
		complex<double> t = exp(complex<double>(0, -2 * PI * k / N)) * odd[k];
		fft[k] = even[k] + t;
		fft[N / 2 + k] = even[k] - t;
	}

	delete[] even;
	delete[] odd;
}

void calc_1DIFFT(complex<double>* fft, int N)
{
	if (N <= 1)
	{
		return;
	}

	/* Declare even and odd arrays */
	complex<double>* odd = new complex<double>[N / 2];
	complex<double>* even = new complex<double>[N / 2];
	for (int i = 0; i < N / 2; i++)
	{
		even[i] = fft[i * 2];
		//cout << "loop even: real" << real(even[i]) << "imag:" << imag(even[i]) << endl;
		odd[i] = fft[i * 2 + 1];
		//cout << "loop odd: real" << real(odd[i]) << "imag:" << imag(odd[i]) << endl;
	}

	/* Recursively call even and odd array to compute IFFT */
	calc_1DIFFT(even, N / 2);
	calc_1DIFFT(odd, N / 2);


	/* Join the recursively calculated array */
	for (int k = 0; k < N / 2; k++)
	{
		complex<double> w = exp(complex<double>(0, 2 * PI * k / N));
		complex<double> t = exp(complex<double>(0, 2 * PI * k / N)) * odd[k];
		fft[k] = even[k] + t;
		fft[N / 2 + k] = even[k] - t;
	}

	delete[] even;
	delete[] odd;
}

complex<double>** calc_IFFT(complex<double>** fft, int N)
{
	complex<double>** ifft = new complex<double> * [N];
	for (int j = 0; j < N; j++)
	{
		ifft[j] = new complex<double>[N];
		memset(ifft[j], 0, sizeof(complex<double>) * N);
	}
	for (int i = 0; i < N; i++)
	{
		for (int k = 0; k < N; k++)
		{
			ifft[i][k] = fft[i][k];
		}
		calc_1DIFFT(ifft[i], N);
	}
	
	ifft = calc_transpose(ifft, N);

	for (int i = 0; i < N; i++)
	{
		calc_1DIFFT(ifft[i], N);
	}

	ifft = calc_transpose(ifft, N);
	return ifft;
}

complex<double>** calc_FFT(Mat img)
{
	int rows = img.rows;
	int cols = img.cols;
	complex<double>** fft = new complex<double> * [rows];
	for (int j = 0; j < cols; j++)
	{
		fft[j] = new complex<double>[cols];
		memset(fft[j], 0, sizeof(complex<double>) * cols);
	}
	for (int i = 0; i < rows; i++)
	{
		uchar* row = img.ptr<uchar>(i);
		for (int k = 0; k < img.cols; k++)
		{
			fft[i][k] = complex<double>(row[k], 0);
		}
		calc_1DFFT(fft[i], cols);
	}

	fft = calc_transpose(fft, rows);
	
	for (int i = 0; i < cols; i++)
	{
		calc_1DFFT(fft[i], rows);
	}

	fft = calc_transpose(fft, cols);
	return fft;
}
void handleIdeal_LPF(Mat img)
{
	complex<double>** fft = calc_FFT(img);
	cout << "fft calculated" << endl;

	Mat input_fft = normalize(fft, img.rows, 255);
	cout << "fft normalized" << endl;
	
	shift_FFT(input_fft);
	namedWindow("inputfft", WINDOW_NORMAL);
	imshow("inputfft", input_fft);

	float distance = 0.1f * ((float)select_cutoff);

	int N = img.rows;
	float d = img.rows * img.cols;
	Mat filter_fft = Mat(img.rows, img.cols, CV_32F, Scalar(0));

	for (int i = 0; i < N / 2; i++) {
		float* p = filter_fft.ptr<float>(i);
		float* p1 = filter_fft.ptr<float>(N - 1 - i);
		for (int j = 0; j < N / 2; j++)
		{
			float f = ((i * i) / d) + ((j * j) / d);
			if (f > distance*distance)
			{
				fft[i][j] = 0;
				fft[N - 1 - i][N - 1 - j] = 0;
				fft[N - 1 - i][j] = 0;
				fft[i][N - 1 - j] = 0;
			}
			else {
				p[j] = 1;
				p[N - 1 - j] = 1;
				p1[N - 1 - j] = 1;
				p1[j] = 1;
			}
		}
	}

	complex<double>** ifft = calc_IFFT(fft, N);
	Mat dst = normalize(ifft, N, 1);

	shift_FFT(filter_fft);
	namedWindow("filter", WINDOW_NORMAL);
	imshow("filter", filter_fft);

	/*Mat output_fft = normalize(fft, N, 255);
	shift_FFT(output_fft);
	namedWindow("opfft", WINDOW_NORMAL);
	imshow("opfft", output_fft);*/

	imshow("output", dst);

	destroy(fft,N);
	destroy(ifft,N);
	return;
}

void handleIdeal_HPF(Mat img)
{
	complex<double>** fft = calc_FFT(img);
	cout << "fft calculated" << endl;

	Mat input_fft = normalize(fft, img.rows, 255);
	cout << "fft normalized" << endl;
	shift_FFT(input_fft);
	namedWindow("inputfft", WINDOW_NORMAL);
	imshow("inputfft", input_fft);

	float distance = 0.1f * ((float)select_cutoff);

	int N = img.rows;
	float d = img.rows * img.cols;
	Mat filter_fft = Mat(img.rows, img.cols, CV_32F, Scalar(0));

	for (int i = 0; i < N / 2; i++) {
		float* p = filter_fft.ptr<float>(i);
		float* p1 = filter_fft.ptr<float>(N - 1 - i);
		for (int j = 0; j < N / 2; j++)
		{
			float f = ((i * i) / d) + ((j * j) / d);
			if (f <= distance * distance)
			{
				fft[i][j] = 0;
				fft[N - 1 - i][N - 1 - j] = 0;
				fft[N - 1 - i][j] = 0;
				fft[i][N - 1 - j] = 0;
			}
			else {
				p[j] = 1;
				p[N - 1 - j] = 1;
				p1[N - 1 - j] = 1;
				p1[j] = 1;
			}
		}
	}

	complex<double>** ifft = calc_IFFT(fft, N);
	Mat dst = normalize(ifft, N, 1);

	shift_FFT(filter_fft);
	namedWindow("filter", WINDOW_NORMAL);
	imshow("filter", filter_fft);

	/*Mat output_fft = normalize(fft, N, 1);
	shift_FFT(output_fft);
	namedWindow("opfft", WINDOW_NORMAL);
	imshow("opfft", output_fft);*/

	imshow("output", dst);

	destroy(fft, N);
	destroy(ifft, N);

	return;
}

void handleGaussian_LPF(Mat img)
{
	complex<double>** fft = calc_FFT(img);
	cout << "fft calculated" << endl;

	Mat input_fft = normalize(fft, img.rows, 255);
	cout << "fft normalized" << endl;
	shift_FFT(input_fft);
	namedWindow("inputfft", WINDOW_NORMAL);
	imshow("inputfft", input_fft);

	float distance = 10.0f * ((float)select_cutoff);

	int N = img.rows;
	float d = img.rows * img.cols;
	Mat filter_fft = Mat(img.rows, img.cols, CV_32F, Scalar(0));

	for (int i = 0; i < N / 2; i++) 
	{
		float* p = filter_fft.ptr<float>(i);
		float* p1 = filter_fft.ptr<float>(N - 1 - i);
		for (int j = 0; j < N / 2; j++)
		{
			double f = pow((2 * i * PI), 2) + pow((2 * j * PI), 2);
			double fil = exp(-(f / N) / (2 * distance * distance));
			fft[i][j] = fft[i][j] * fil;
			fft[N - 1 - i][N - 1 - j] = fft[N - 1 - i][N - 1 - j] * fil;
			fft[N - 1 - i][j] = fft[N - 1 - i][j] * fil;
			fft[i][N - 1 - j] = fft[i][N - 1 - j] * fil;
			p[j] = fil;
			p[N - 1 - j] = fil;
			p1[N - 1 - j] = fil;
			p1[j] = fil;
		}
	}

	complex<double>** ifft = calc_IFFT(fft, N);
	Mat dst = normalize(ifft, N, 1);

	shift_FFT(filter_fft);
	namedWindow("filter", WINDOW_NORMAL);
	imshow("filter", filter_fft);

	/*Mat output_fft = normalize(fft, N, 1);
	shift_FFT(output_fft);
	namedWindow("opfft", WINDOW_NORMAL);
	imshow("opfft", output_fft);*/

	imshow("output", dst);

	destroy(fft, N);
	destroy(ifft, N);

	return;
}

void handleGaussian_HPF(Mat img)
{
	complex<double>** fft = calc_FFT(img);
	cout << "fft calculated" << endl;

	Mat input_fft = normalize(fft, img.rows, 255);
	cout << "fft normalized" << endl;
	shift_FFT(input_fft);
	namedWindow("inputfft", WINDOW_NORMAL);
	imshow("inputfft", input_fft);

	float distance = 10.0f * ((float)select_cutoff);

	int N = img.rows;
	float d = img.rows * img.cols;
	Mat filter_fft = Mat(img.rows, img.cols, CV_32F, Scalar(0));

	for (int i = 0; i < N / 2; i++)
	{
		float* p = filter_fft.ptr<float>(i);
		float* p1 = filter_fft.ptr<float>(N - 1 - i);
		for (int j = 0; j < N / 2; j++)
		{
			double f = pow((2 * i * PI), 2) + pow((2 * j * PI), 2);
			double fil = 1 - (exp(-(f / N) / (2 * distance * distance)));
			fft[i][j] = fft[i][j] * fil;
			fft[N - 1 - i][N - 1 - j] = fft[N - 1 - i][N - 1 - j] * fil;
			fft[N - 1 - i][j] = fft[N - 1 - i][j] * fil;
			fft[i][N - 1 - j] = fft[i][N - 1 - j] * fil;
			p[j] = fil;
			p[N - 1 - j] = fil;
			p1[N - 1 - j] = fil;
			p1[j] = fil;
		}
	}

	complex<double>** ifft = calc_IFFT(fft, N);
	Mat dst = normalize(ifft, N, 1);

	shift_FFT(filter_fft);
	namedWindow("filter", WINDOW_NORMAL);
	imshow("filter", filter_fft);

	/*Mat output_fft = normalize(fft, N, 1);
	shift_FFT(output_fft);
	namedWindow("opfft", WINDOW_NORMAL);
	imshow("opfft", output_fft);*/

	imshow("output", dst);

	destroy(fft, N);
	destroy(ifft, N);

	return;
}

void handleButterworth_LPF(Mat img)
{
	complex<double>** fft = calc_FFT(img);
	cout << "fft calculated" << endl;

	Mat input_fft = normalize(fft, img.rows, 255);
	cout << "fft normalized" << endl;
	shift_FFT(input_fft);
	namedWindow("inputfft", WINDOW_NORMAL);
	imshow("inputfft", input_fft);

	float distance = 10.0f * ((float)select_cutoff);

	int N = img.rows;
	float d = img.rows * img.cols;
	float cutoff = pow(distance * distance, 2);
	Mat filter_fft = Mat(img.rows, img.cols, CV_32F, Scalar(0));

	for (int i = 0; i < N / 2; i++)
	{
		float* p = filter_fft.ptr<float>(i);
		float* p1 = filter_fft.ptr<float>(N - 1 - i);
		for (int j = 0; j < N / 2; j++)
		{
			double f = pow(2 * PI * i, 2) + pow(2 * PI * j, 2);
			double fil = 1 / (1 + pow((f / cutoff), 2 ));
			fft[i][j] = fft[i][j] * fil;
			fft[N - 1 - i][N - 1 - j] = fft[N - 1 - i][N - 1 - j] * fil;
			fft[N - 1 - i][j] = fft[N - 1 - i][j] * fil;
			fft[i][N - 1 - j] = fft[i][N - 1 - j] * fil;
			p[j] = fil;
			p[N - 1 - j] = fil;
			p1[N - 1 - j] = fil;
			p1[j] = fil;
		}
	}

	complex<double>** ifft = calc_IFFT(fft, N);
	Mat dst = normalize(ifft, N, 1);

	shift_FFT(filter_fft);
	namedWindow("filter", WINDOW_NORMAL);
	imshow("filter", filter_fft);

	/*Mat output_fft = normalize(fft, N, 1);
	shift_FFT(output_fft);
	namedWindow("opfft", WINDOW_NORMAL);
	imshow("opfft", output_fft);*/

	imshow("output", dst);

	destroy(fft, N);
	destroy(ifft, N);

	return;
}

void handleButterworth_HPF(Mat img)
{
	complex<double>** fft = calc_FFT(img);
	cout << "fft calculated" << endl;

	Mat input_fft = normalize(fft, img.rows, 255);
	cout << "fft normalized" << endl;
	shift_FFT(input_fft);
	namedWindow("inputfft", WINDOW_NORMAL);
	imshow("inputfft", input_fft);

	float distance = 10.0f * ((float)select_cutoff);

	int N = img.rows;
	float d = img.rows * img.cols;
	float cutoff = pow(distance * distance, 2);
	Mat filter_fft = Mat(img.rows, img.cols, CV_32F, Scalar(0));

	for (int i = 0; i < N / 2; i++)
	{
		float* p = filter_fft.ptr<float>(i);
		float* p1 = filter_fft.ptr<float>(N - 1 - i);
		for (int j = 0; j < N / 2; j++)
		{
			double f = pow(2 * PI * i, 2) + pow(2 * PI * j, 2);
			double fil = 1 / (1 + pow((cutoff / f), 2));
			fft[i][j] = fft[i][j] * fil;
			fft[N - 1 - i][N - 1 - j] = fft[N - 1 - i][N - 1 - j] * fil;
			fft[N - 1 - i][j] = fft[N - 1 - i][j] * fil;
			fft[i][N - 1 - j] = fft[i][N - 1 - j] * fil;
			p[j] = fil;
			p[N - 1 - j] = fil;
			p1[N - 1 - j] = fil;
			p1[j] = fil;
		}
	}

	complex<double>** ifft = calc_IFFT(fft, N);
	Mat dst = normalize(ifft, N, 1);

	shift_FFT(filter_fft);
	namedWindow("filter", WINDOW_NORMAL);
	imshow("filter", filter_fft);

	/*Mat output_fft = normalize(fft, N, 1);
	shift_FFT(output_fft);
	namedWindow("opfft", WINDOW_NORMAL);
	imshow("opfft", output_fft);*/

	imshow("output", dst);

	destroy(fft, N);
	destroy(ifft, N);
	return;
}

void OnSelection(int, void*)
{
	cout << "image selected:" << input_file[select_image] << endl;
	cout << "filter selected:" << filter_name[select_filter] << endl;
	cout << "Cutoff: " << select_cutoff << endl;
	Mat img = imread(input_file[select_image], -1);
	if (img.empty())
	{
		cerr << "Imread failed" << endl;
		exit(1);
	}

	cv::imshow("Input Image", img);

	switch (select_filter)
	{
	case Ideal_LPF:
		handleIdeal_LPF(img);
		break;
	case Ideal_HPF:
		handleIdeal_HPF(img);
		break;
	case Gaussian_LPF:
		handleGaussian_LPF(img);
		break;
	case Gaussian_HPF:
		handleGaussian_HPF(img);
		break;
	case Butterworth_LPF:
		handleButterworth_LPF(img);
		break;
	case Butterworth_HPF:
		handleButterworth_HPF(img);
		break;
	default:
		break;
	}
	return;
}


int main()
{
	namedWindow("frequency filtering");
	resizeWindow("frequency filtering", 320, 240);
	createTrackbar("ImageID", "frequency filtering", &select_image, NO_OF_IMAGES, OnSelection);
	createTrackbar("FilterID", "frequency filtering", &select_filter, NO_OF_FILTERS, OnSelection);
	createTrackbar("CutoffID", "frequency filtering", &select_cutoff, NO_OF_CUTOFF_FREQ, OnSelection);

	waitKey(0);
	return 0;
}