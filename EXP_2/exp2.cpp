/* Implementation of histogram equalization and histogram matching */
#include <iostream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define NUM_OF_BINS 256

/* Utility function to create normalized histogram */
void Create_Histogram_pdf(Mat img, float* hist)
{
    int no_pixels = img.rows * img.cols;
    for (int i = 0; i < NUM_OF_BINS; i++)
    {
        hist[i] = 0.0f;
    }

    for (int r = 0; r < img.rows; r++)
    {
        for (int c = 0; c < img.cols; c++)
        {
            hist[(int)img.at<uchar>(r, c)]++;
        }
    }

    for (int j = 0; j < NUM_OF_BINS; j++) {
        hist[j] = hist[j] / no_pixels;
    }

    return;
}

/* utility function to create tansfer function i.e. cdf */
void Create_Histogram_cdf(float* hist, float* hist_cdf)
{
    hist_cdf[0] = hist[0];
    for (int i = 1; i < NUM_OF_BINS; i++)
        hist_cdf[i] = hist[i] + hist_cdf[i - 1];

    return;
}

/* utility function to calculate mapping of pixels */
void hist_eq_mapping(float* hist, uchar* out)
{

    for (int i = 0; i < NUM_OF_BINS; i++)
    {
        out[i] = saturate_cast<uchar>(floor(hist[i] * (NUM_OF_BINS - 1)));
    }

    return;
}

/* utility function to match target histogram */
void Match_target_histogram(float* ih_cdf, float* th_cdf, float* out_hist)
{
    for (int i = 0; i < NUM_OF_BINS; i++)
    {
        int j = 0;
        while ((j < NUM_OF_BINS) && (th_cdf[j] < ih_cdf[i]))
        {
            j++;
        }

        if (j == 0)
        {
            out_hist[i] = (float)j;
        }
        else
        {
            if ((th_cdf[j] - ih_cdf[i]) > (ih_cdf[i] - th_cdf[j - 1]))
            {
                out_hist[i] = (float)(j-1);
            }
            else
            {
                out_hist[i] = (float)j;
            }
        }
    }
    return;
}

/* utility function to display a histogram */
void display_histogram(Mat img,string histtype) 
{
    Mat histogram;     
    Mat canvas;		   
    int hmax = 0;      

    histogram = Mat::zeros(1, NUM_OF_BINS, CV_32SC1);

    for (int r = 0; r < img.rows; r++)
        for (int c = 0; c < img.cols; c++) {
            uchar uc = img.at<uchar>(r, c);
            histogram.at<int>(uc) += 1;
        }

    for (int i = 0; i < NUM_OF_BINS - 1; i++)
        hmax = histogram.at<int>(i) > hmax ? histogram.at<int>(i) : hmax;

    canvas = Mat::ones(256, NUM_OF_BINS, CV_8UC3);

    for (int j = 0, rows = canvas.rows; j < NUM_OF_BINS - 1; j++)
        line(canvas, Point(j, rows), Point(j, rows - (histogram.at<int>(j) * rows / hmax)), Scalar(255, 255, 255), 1, 8, 0);

    imshow(histtype, canvas);

}

int main()
{
    string inputfile,targetfile;
    int consoleinput = 0;
    cout << "Select 1 for Histogram Equalisation \n" << endl;
    cout << "Select 2 for Histogram Matching \n" << endl;
    cin >> consoleinput;
    if (consoleinput == 1)
    {
        cout << "Histogram Equalisation opted" << endl;
        cout << "Enter file name\n";
        cin >> inputfile;
        Mat image = imread(inputfile, -1);
        if (image.empty()) 
        {
            cerr << "Imread failed" << endl;
            exit(1);
        }

        if (image.channels() == 1) 
        {
            float histogram[NUM_OF_BINS];
            float histogram_cdf[NUM_OF_BINS];
            uchar out_histogram[NUM_OF_BINS];
            Create_Histogram_pdf(image, histogram);
            Create_Histogram_cdf(histogram, histogram_cdf);
            hist_eq_mapping(histogram_cdf, out_histogram);

            Mat out_image = image.clone();
            for (int r = 0; r < image.rows; r++) 
            {
                for (int c = 0; c < image.cols; c++) 
                {
                    out_image.at<uchar>(r, c) = saturate_cast<uchar>(out_histogram[image.at<uchar>(r, c)]);
                }
            }

            imshow("Input Image", image);
            display_histogram(image, " input histogram");

            imshow("histogram equilized image", out_image);
            display_histogram(out_image, " equalized histogram");

            waitKey();
        }
        else if (image.channels() == 3)
        {
            Mat hsv_image,out_image;
            cvtColor(image, hsv_image, COLOR_BGR2HSV);
            vector<Mat> hsv_planes;
            split(hsv_image, hsv_planes);
            Mat v_image = hsv_planes[2];

            float histogram[NUM_OF_BINS];
            float histogram_cdf[NUM_OF_BINS];
            uchar out_histogram[NUM_OF_BINS];
            Create_Histogram_pdf(v_image, histogram);
            Create_Histogram_cdf(histogram, histogram_cdf);
            hist_eq_mapping(histogram_cdf, out_histogram);

            for (int r = 0; r < v_image.rows; r++)
            {
                for (int c = 0; c < v_image.cols; c++)
                {
                    Vec3b& hsv = hsv_image.at<Vec3b>(r, c);
                    hsv[2] = saturate_cast<uchar>(out_histogram[(int)v_image.at<uchar>(r,c)]);
                }
            }

            cvtColor(hsv_image, out_image, COLOR_HSV2BGR);

            imshow("Input Image", image);
            display_histogram(v_image, " input histogram");

            vector<Mat> hsv_out_planes;
            split(hsv_image, hsv_out_planes);
            Mat out_v_image = hsv_out_planes[2];

            imshow("histogram equilized image", out_image);
            display_histogram(out_v_image, " equalized histogram");

            waitKey();
        }

    }

    else if (consoleinput == 2) {
        cout << "Histogram matching opted:" << endl;
        cout << "keep image channels same for input and target image";
        cout << "enter input file name:" << endl;
        cin >> inputfile;
        Mat image = imread(inputfile, -1);
        if (image.empty()) 
        {
            cerr << "imread failed" << endl;
            exit(1);
        }

        cout << "enter targetfile name:"<< endl;
        cin >> targetfile;
        Mat tgt_image = imread(targetfile, -1);
        if (tgt_image.empty()) {
            cerr << "Imread failed" << endl;
            exit(1);
        }

        if (image.channels() == 1) 
        {
            float input_histogram[NUM_OF_BINS];
            float target_histogram[NUM_OF_BINS];
            float input_histogram_cdf[NUM_OF_BINS];
            float target_histogram_cdf[NUM_OF_BINS];
            float out_histogram[NUM_OF_BINS];
            
            Create_Histogram_pdf(image, input_histogram);
            Create_Histogram_cdf(input_histogram, input_histogram_cdf);
            Create_Histogram_pdf(tgt_image, target_histogram);
            Create_Histogram_cdf(target_histogram, target_histogram_cdf);
            Match_target_histogram(input_histogram_cdf, target_histogram_cdf, out_histogram);

            Mat out_image = image.clone();

            for (int r = 0; r < image.rows; r++)
                for (int c = 0; c < image.cols; c++)
                    out_image.at<uchar>(r, c) = (int)(out_histogram[image.at<uchar>(r, c)]);

            imshow("Input Image", image);
            display_histogram(image, "input histogram");

            imshow("target image", tgt_image);
            display_histogram(tgt_image, " target histogram");

            imshow("matched image", out_image);
            display_histogram(out_image, " matched histogram");

            waitKey();

        }
        else if (image.channels() == 3)
        {
            float input_histogram[NUM_OF_BINS];
            float target_histogram[NUM_OF_BINS];
            float input_histogram_cdf[NUM_OF_BINS];
            float target_histogram_cdf[NUM_OF_BINS];
            float out_histogram[NUM_OF_BINS];

            Mat hsv_image, tgt_hsv_image, out_image;
            cvtColor(image, hsv_image, COLOR_BGR2HSV);
            vector<Mat> hsv_planes;
            split(hsv_image, hsv_planes);
            Mat v_image = hsv_planes[2];

            cvtColor(tgt_image, tgt_hsv_image, COLOR_BGR2HSV);
            vector<Mat> tgt_hsv_planes;
            split(tgt_hsv_image, tgt_hsv_planes);
            Mat tgt_v_image = tgt_hsv_planes[2];

            Create_Histogram_pdf(v_image, input_histogram);
            Create_Histogram_cdf(input_histogram, input_histogram_cdf);
            Create_Histogram_pdf(tgt_v_image, target_histogram);
            Create_Histogram_cdf(target_histogram, target_histogram_cdf);
            Match_target_histogram(input_histogram_cdf, target_histogram_cdf, out_histogram);

            for (int r = 0; r < v_image.rows; r++)
            {
                for (int c = 0; c < v_image.cols; c++)
                {
                    Vec3b& hsv = hsv_image.at<Vec3b>(r, c);
                    hsv[2] = (int)out_histogram[(int)v_image.at<uchar>(r, c)];
                }
            }

            cvtColor(hsv_image, out_image, COLOR_HSV2BGR);

            imshow("Input Image", image);
            display_histogram(v_image, "input histogram");

            imshow("target image", tgt_image);
            display_histogram(tgt_v_image, " target histogram");

            vector<Mat> hsv_out_planes;
            split(hsv_image, hsv_out_planes);
            Mat out_v_image = hsv_out_planes[2];

            imshow("matched image", out_image);
            display_histogram(out_v_image, " matched histogram");

            waitKey();

        }

    }

    return 0;
}
