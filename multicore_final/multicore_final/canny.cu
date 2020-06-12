#include <iostream>
#include <cmath>
#include <vector>
#include "DS_timer.h"
#include "canny.cuh"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

__global__ void sobel_kernel()
{

}

__global__ void non_max_kernel(float* angles, uchar* nonMaxSupped, uchar* diffValue, int  sizeX, int sizeY)
{
	int rowIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int colIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	if (rowIndex >= sizeY || colIndex >= sizeX)
	{
		return;
	}
	float Tangent = angles[rowIndex * sizeX + colIndex];
	//printf("%f\n", Tangent);
	//printf("%d\n", diffValue[rowIndex * sizeX + colIndex]);

	nonMaxSupped[rowIndex * sizeX + colIndex] = diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)];
	//printf("%d\n", diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)]);
	//Horizontal Edge
	if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
	{
		if ((diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[(rowIndex + 1) * sizeX + (colIndex + 2)]) || (diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[(rowIndex + 1) * sizeX + colIndex]))
			nonMaxSupped[rowIndex * sizeX + colIndex] = 0;
	}
	//Vertical Edge
	if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
	{
		if ((diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[(rowIndex + 2) * sizeX + (colIndex + 1)]) || (diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[(rowIndex)*sizeX + (colIndex + 1)]))
			nonMaxSupped[rowIndex * sizeX + colIndex] = 0;
	}

	//-45 Degree Edge
	if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
	{
		if ((diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[rowIndex * sizeX + (colIndex + 2)]) || (diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[(rowIndex + 2) * sizeX + colIndex]))
			nonMaxSupped[rowIndex * sizeX + colIndex] = 0;
	}

	//45 Degree Edge
	if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
	{
		if ((diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[(rowIndex + 2) * sizeX + (colIndex + 2)]) || (diffValue[(rowIndex + 1) * sizeX + (colIndex + 1)] < diffValue[rowIndex * sizeX + colIndex]))
			nonMaxSupped[rowIndex * sizeX + colIndex] = 0;
	}

}
canny::canny(String filename)
{
	img = imread(filename);

	if (!img.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;

	}
	else
	{

		vector<vector<double>> filter = createFilter(3, 3, 1);

		//Print filter
		for (int i = 0; i < filter.size(); i++)
		{
			for (int j = 0; j < filter[i].size(); j++)
			{
				cout << filter[i][j] << " ";
			}
		}
		cvtColor(img, grayscaled, COLOR_RGB2GRAY); //Grayscale the image
		gFiltered = Mat(useFilter(grayscaled, filter)); //Gaussian Filter
		sFiltered = Mat(sobel()); //Sobel Filter

		non = Mat(nonMaxSupp()); //Non-Maxima Suppression
		thres = Mat(threshold(non, 20, 40)); //Double Threshold and Finalize


		imshow("Original", img);
		imshow("GrayScaled", grayscaled);
		imshow("Gaussian Blur", gFiltered);
		imshow("Sobel Filtered", sFiltered);
		imshow("Non-Maxima Supp.", non);
		imshow("Final", thres);

	}
}

Mat canny::toGrayScale()
{
	grayscaled = Mat(img.rows, img.cols, CV_8UC1); //To one channel
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			int b = img.at<Vec3b>(i, j)[0];
			int g = img.at<Vec3b>(i, j)[1];
			int r = img.at<Vec3b>(i, j)[2];

			double newValue = (r * 0.2126 + g * 0.7152 + b * 0.0722);
			grayscaled.at<uchar>(i, j) = newValue;

		}
	return grayscaled;
}

vector<vector<double>> canny::createFilter(int row, int column, double sigmaIn)
{
	vector<vector<double>> filter;

	for (int i = 0; i < row; i++)
	{
		vector<double> col;
		for (int j = 0; j < column; j++)
		{
			col.push_back(-1);
		}
		filter.push_back(col);
	}

	float coordSum = 0;
	float constant = 2.0 * sigmaIn * sigmaIn;

	// Sum is for normalization
	float sum = 0.0;

	for (int x = -row / 2; x <= row / 2; x++)
	{
		for (int y = -column / 2; y <= column / 2; y++)
		{
			coordSum = (x * x + y * y);
			filter[x + row / 2][y + column / 2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
			sum += filter[x + row / 2][y + column / 2];
		}
	}

	// Normalize the Filter
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			filter[i][j] /= sum;

	return filter;

}

Mat canny::useFilter(Mat img_in, vector<vector<double>> filterIn)
{
	int size = (int)filterIn.size() / 2;
	Mat filteredImg = Mat(img_in.rows - 2 * size, img_in.cols - 2 * size, CV_8UC1);
	for (int i = size; i < img_in.rows - size; i++)
	{
		for (int j = size; j < img_in.cols - size; j++)
		{
			double sum = 0;

			for (int x = 0; x < filterIn.size(); x++)
				for (int y = 0; y < filterIn.size(); y++)
				{
					sum += filterIn[x][y] * (double)(img_in.at<uchar>(i + x - size, j + y - size));
				}

			filteredImg.at<uchar>(i - size, j - size) = sum;
		}

	}
	return filteredImg;
}

Mat canny::sobel()
{

	//Sobel X Filter
	double x1[] = { -1.0, 0, 1.0 };
	double x2[] = { -2.0, 0, 2.0 };
	double x3[] = { -1.0, 0, 1.0 };

	vector<vector<double>> xFilter(3);
	xFilter[0].assign(x1, x1 + 3);
	xFilter[1].assign(x2, x2 + 3);
	xFilter[2].assign(x3, x3 + 3);

	//Sobel Y Filter
	double y1[] = { 1.0, 2.0, 1.0 };
	double y2[] = { 0, 0, 0 };
	double y3[] = { -1.0, -2.0, -1.0 };

	vector<vector<double>> yFilter(3);
	yFilter[0].assign(y1, y1 + 3);
	yFilter[1].assign(y2, y2 + 3);
	yFilter[2].assign(y3, y3 + 3);

	//Limit Size
	int size = (int)xFilter.size() / 2;

	Mat filteredImg = Mat(gFiltered.rows - 2 * size, gFiltered.cols - 2 * size, CV_8UC1);

	angles = Mat(gFiltered.rows - 2 * size, gFiltered.cols - 2 * size, CV_32FC1); //AngleMap

	for (int i = size; i < gFiltered.rows - size; i++)
	{
		for (int j = size; j < gFiltered.cols - size; j++)
		{
			double sumx = 0;
			double sumy = 0;

			for (int x = 0; x < xFilter.size(); x++)
				for (int y = 0; y < xFilter.size(); y++)
				{
					sumx += xFilter[x][y] * (double)(gFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_X Filter Value
					sumy += yFilter[x][y] * (double)(gFiltered.at<uchar>(i + x - size, j + y - size)); //Sobel_Y Filter Value
				}
			double sumxsq = sumx * sumx;
			double sumysq = sumy * sumy;

			double sq2 = sqrt(sumxsq + sumysq);

			if (sq2 > 255) //Unsigned Char Fix
				sq2 = 255;
			filteredImg.at<uchar>(i - size, j - size) = sq2;

			if (sumx == 0) //Arctan Fix
				angles.at<float>(i - size, j - size) = 90;
			else
				angles.at<float>(i - size, j - size) = atan(sumy / sumx);
		}
	}

	return filteredImg;
}


Mat canny::nonMaxSupp()
{
	Mat nonMaxSupped = Mat(sFiltered.rows - 2, sFiltered.cols - 2, CV_8UC1);
	Mat cu_nonMaxSupped = Mat(sFiltered.rows, sFiltered.cols, CV_8UC1);
	Mat test_Mat = Mat(sFiltered.rows, sFiltered.cols, CV_8UC1);
	DS_timer timer(2);
	timer.setTimerName(0, "CPU");
	timer.setTimerName(1, "GPU");
	uchar* g_nonMaxSupped, *testImage;
	float* angleData;
	uchar* diffValue;

	cudaMalloc(&g_nonMaxSupped, (sizeof(uchar) * (sFiltered.rows - 2) * (sFiltered.cols - 2)));
	cudaMalloc(&angleData, (sizeof(float) * angles.rows * angles.cols));
	cudaMalloc(&diffValue, (sizeof(uchar) * sFiltered.rows * sFiltered.cols));




	dim3 grid(ceil((float)(sFiltered.cols - 2) / BLOCK_SIZE_X), ceil((float)(sFiltered.rows - 2) / BLOCK_SIZE_Y));
	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	timer.onTimer(1);
	cudaMemcpy(angleData, angles.ptr<float>(0, 0), angles.rows * angles.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(diffValue, sFiltered.data, sFiltered.rows * sFiltered.cols * sizeof(uchar), cudaMemcpyHostToDevice);
	non_max_kernel << <grid, block >> > (angleData, g_nonMaxSupped, diffValue, sFiltered.cols - 2, sFiltered.rows - 2);
	cudaThreadSynchronize();
	cudaMemcpy(cu_nonMaxSupped.data, g_nonMaxSupped, sizeof(uchar) * (sFiltered.rows - 2) * (sFiltered.cols - 2), cudaMemcpyDeviceToHost);
	timer.offTimer(1);
	timer.onTimer(0);
	for (int i = 1; i < sFiltered.rows - 1; i++) {
		for (int j = 1; j < sFiltered.cols - 1; j++) {
			float Tangent = angles.at<float>(i, j);

			nonMaxSupped.at<uchar>(i - 1, j - 1) = sFiltered.at<uchar>(i, j);
			//Horizontal Edge
			if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
			{
				if ((sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i, j + 1)) || (sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i, j - 1)))
					nonMaxSupped.at<uchar>(i - 1, j - 1) = 0;
			}
			//Vertical Edge
			if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
			{
				if ((sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i + 1, j)) || (sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i - 1, j)))
					nonMaxSupped.at<uchar>(i - 1, j - 1) = 0;
			}

			//-45 Degree Edge
			if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
			{
				if ((sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i - 1, j + 1)) || (sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i + 1, j - 1)))
					nonMaxSupped.at<uchar>(i - 1, j - 1) = 0;
			}

			//45 Degree Edge
			if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
			{
				if ((sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i + 1, j + 1)) || (sFiltered.at<uchar>(i, j) < sFiltered.at<uchar>(i - 1, j - 1)))
					nonMaxSupped.at<uchar>(i - 1, j - 1) = 0;
			}
		}
	}
	timer.offTimer(0);
	timer.printTimer();
	return nonMaxSupped;
}

Mat canny::threshold(Mat imgin, int low, int high)
{
	if (low > 255)
		low = 255;
	if (high > 255)
		high = 255;

	Mat EdgeMat = Mat(imgin.rows, imgin.cols, imgin.type());

	for (int i = 0; i < imgin.rows; i++)
	{
		for (int j = 0; j < imgin.cols; j++)
		{
			EdgeMat.at<uchar>(i, j) = imgin.at<uchar>(i, j);
			if (EdgeMat.at<uchar>(i, j) > high)
				EdgeMat.at<uchar>(i, j) = 255;
			else if (EdgeMat.at<uchar>(i, j) < low)
				EdgeMat.at<uchar>(i, j) = 0;
			else
			{
				bool anyHigh = false;
				bool anyBetween = false;
				for (int x = i - 1; x < i + 2; x++)
				{
					for (int y = j - 1; y < j + 2; y++)
					{
						if (x <= 0 || y <= 0 || EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
							continue;
						else
						{
							if (EdgeMat.at<uchar>(x, y) > high)
							{
								EdgeMat.at<uchar>(i, j) = 255;
								anyHigh = true;
								break;
							}
							else if (EdgeMat.at<uchar>(x, y) <= high && EdgeMat.at<uchar>(x, y) >= low)
								anyBetween = true;
						}
					}
					if (anyHigh)
						break;
				}
				if (!anyHigh && anyBetween)
					for (int x = i - 2; x < i + 3; x++)
					{
						for (int y = j - 1; y < j + 3; y++)
						{
							if (x < 0 || y < 0 || x > EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
								continue;
							else
							{
								if (EdgeMat.at<uchar>(x, y) > high)
								{
									EdgeMat.at<uchar>(i, j) = 255;
									anyHigh = true;
									break;
								}
							}
						}
						if (anyHigh)
							break;
					}
				if (!anyHigh)
					EdgeMat.at<uchar>(i, j) = 0;
			}
		}
	}
	return EdgeMat;
}