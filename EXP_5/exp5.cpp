#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void printmask(int** mask, int n)
{
    cout << "chosen mask:" << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << " " << mask[i][j];
        }
        cout << endl;
    }
}

int** CreateSqMask(int n)
{
    int** mask;
    mask = new int* [n];
    for (int i = 0; i < n; i++)
    {
        mask[i] = new int[n];
        memset(mask[i], 0, n * sizeof(int));
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mask[i][j] = 1;
        }
    }

    return mask;
}

int** CreateDiaMask(int n)
{
    int** mask;
    mask = new int* [n];
    for (int i = 0; i < n; i++)
    {
        mask[i] = new int[n];
        memset(mask[i], 0, n * sizeof(int));
    }

    int space = n / 2;
    int i = 0;

    for (i = 0; i <= n / 2; i++)
    {
        int j = 0;
        int start = 0;
        for (j = 0; j < space; j++)
        {
            mask[i][j] = 0;
        }

        start = j;
        for (; j < (start + (2 * i + 1)); j++)
        {
            mask[i][j] = 1;
        }

        start = j;
        for (; j < n; j++)
        {
            mask[i][j] = 0;
        }

        space--;
    }

    space = 1;
    for (i = n / 2 + 1; i < n; i++)
    {
        int j = 0;
        int start = 0;
        for (j = 0; j < space; j++)
        {
            mask[i][j] = 0;
        }

        start = j;
        for (; j < (start + (2 * (n - 1 - i) + 1)); j++)
        {
            mask[i][j] = 1;
        }

        start = j;
        for (; j < n; j++)
        {
            mask[i][j] = 0;
        }
        space++;
    }

    return mask;
}

Mat ErodeBinary(Mat img, int** mask, int size)
{
    cout << "Erodebinary" << endl;

    int min = 0;
    Mat outimg = img.clone();
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            min = 255;
            for (int k = -(int)size / 2, m = 0; k <= (int)size / 2; k++, m++)
            {
                if (((i + k) >= 0) && ((i + k) < img.rows))
                {
                    for (int l = -(int)size / 2, n = 0; l <= (int)size / 2; l++, n++)
                    {
                        if (((j + l) >= 0) && ((j + l) < img.cols))
                        {
                            if (mask[m][n] == 1)
                            {
                                if ((int)img.at<uchar>(i + k, j + l) == 0)
                                    min = 0;
                            }
                        }
                    }
                }
            }
            if (min == 0)
            {
                outimg.at<uchar>(i, j) = 0;
            }
        }
    }

    cout << endl;
    imshow("input image", img);
    imshow("output image", outimg);

    return outimg;
}

Mat DilateBinary(Mat img, int** mask, int size)
{
    cout << "dilatebinary" << endl;

    int max = 0;
    Mat outimg = img.clone();
    cout << "image cloned" << endl;
    cout << "rows: " << img.rows << "cols: " << img.cols << endl;
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            max = 0;
            for (int k = -(int)size / 2, m = 0; k <= (int)size / 2; k++, m++)
            {
                if (((i + k) >= 0) && ((i + k) < img.rows))
                {
                    for (int l = -(int)size / 2, n = 0; l <= (int)size / 2; l++, n++)
                    {
                        if (((j + l) >= 0) && ((j + l) < img.cols))
                        {
                            if (mask[m][n] == 1)
                            {
                                if ((int)img.at<uchar>(i + k, j + l) == 255)
                                    max = 255;
                            }
                        }
                    }
                }
            }
            if (max == 255)
            {
                outimg.at<uchar>(i, j) = 255;
            }
        }
    }

    cout << endl;
    imshow("input image", img);
    imshow("output image", outimg);
    return outimg;
}


void OpeningBinary(Mat img, int** mask, int size)
{
    Mat erode = ErodeBinary(img, mask, size);
    Mat opening = DilateBinary(erode, mask, size);
    imshow("input image", img);
    imshow("output image", opening);
}

void ClosingBinary(Mat img, int** mask, int size)
{
    Mat dilate = DilateBinary(img, mask, size);
    Mat closing = ErodeBinary(dilate, mask, size);
    imshow("input image", img);
    imshow("output image", closing);
}

int main()
{
    int choice_op = 0;
    int choice_ker = 0;
    int** mask = NULL;
    int size = 0;
    Mat output;
    string input_file = "ricegrains.bmp";
    //string input_file = "lego.tif";
    Mat img_gray = imread(input_file, 0);

    Mat img(img_gray.size(), img_gray.type());

    threshold(img_gray, img, 127, 255, THRESH_BINARY);

    cout << "Choose operation to be done" << endl;
    cout << "Enter 1 for DILATION" << endl;
    cout << "Enter 2 for EROSION" << endl;
    cout << "Enter 3 for OPENING" << endl;
    cout << "Enter 4 for CLOSING" << endl;
    cin >> choice_op;

    cout << "Choose kernel for morphological operation" << endl;
    cout << "Enter 1 for 3*3 square" << endl;
    cout << "Enter 2 for 5*5 square" << endl;
    cout << "Enter 3 for 7*7 square" << endl;
    cout << "Enter 4 for 3*3 diamond" << endl;
    cin >> choice_ker;

    switch (choice_ker)
    {
    case 1:
        mask = CreateSqMask(3);
        size = 3;
        break;
    case 2:
        mask = CreateSqMask(5);
        size = 5;
        break;
    case 3:
        mask = CreateSqMask(7);
        size = 7;
        break;
    case 4:
        mask = CreateDiaMask(3);
        size = 3;
        break;
    default:
        cerr << "invalid choice";
    }

    printmask(mask, size);

    switch (choice_op)
    {
    case 1:
        output = DilateBinary(img, mask, size);
        break;
    case 2:
        output = ErodeBinary(img, mask, size);
        break;
    case 3:
        OpeningBinary(img, mask, size);
        break;
    case 4:
        ClosingBinary(img, mask, size);
        break;
    default:
        cerr << "invalid choice";
    }

    waitKey(0);
    return 0;
}

