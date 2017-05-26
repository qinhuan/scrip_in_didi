#include "laneDetection.h"
#include "caffe/layers/memory_data_layer.hpp"

#include <iostream>
using namespace std;

#define NetF float

typedef struct KeypointType
{
    Point oriPt;
    Point newPt;
    bool bProcessed;
}KeypointType;

void LaneDetector::getResults(vector< vector<Point> > &vvResult, vector<vector<float> > &vvMsk)
{
    vector<int> vLeft, vRight, vTop;
    getScanLine(vLeft, vRight, vTop);

    removeDuplicate(vvMsk);

    // get new key points
    vector<vector<KeypointType> > vKeyPts;
    for(int idx = 0; idx < m_slcImgs.size(); idx++)
    {
    vKeyPts.push_back(vector<KeypointType>());
        for(int start = 0; start < 128; start++)
        {
        if(vvMsk[idx][start] > 0)
        {
        KeypointType kpt;
        kpt.oriPt.x = start/128.0 * (vRight[idx]-vLeft[idx]+1) + vLeft[idx];
        kpt.oriPt.y = vTop[idx];
        if(vLeft[idx] > 0)
        {
            kpt.newPt.x = start;
        }
        else
        {
            int delta = vLeft[0] - vLeft[1];
            int curLeft = vLeft[0] - idx * delta;
                    int scanLineWid = (m_IMG_WID/2 - curLeft) * 2;
            kpt.newPt.x = (start/128.0 * m_IMG_WID - curLeft) / scanLineWid * 128;
        }
        kpt.newPt.y = idx;
        kpt.bProcessed = false;
        vKeyPts[idx].push_back(kpt);
        }
    }
    }

    // get rough line
    vector<vector<KeypointType> > vLines;
    for(int i = vKeyPts.size()-1; i >= 0; i--)
    {
        for(int j = 0; j < vKeyPts[i].size(); j++)
        {
        int bestLineID = -1;
        int minDist = INT_MAX;
        bool bFound = false;

        for(int m = 0; m < vLines.size(); m++)
        {
        if(vLines[m].size() > 0)
        {
            if(vLines[m][vLines[m].size()-1].newPt.y == vKeyPts[i][j].newPt.y)
            {
            if(vLines[m].size() > 1)
            {
                int prepreX = vLines[m][vLines[m].size()-2].newPt.x;
                int preX = vLines[m][vLines[m].size()-1].newPt.x;
                int curX = vKeyPts[i][j].newPt.x;
                if(abs(curX - prepreX) < abs(preX - prepreX))
                {
                vector<KeypointType> vtmp;
                        vtmp.push_back(vLines[m][vLines[m].size()-1]);
                        vLines.push_back(vtmp);
                
                vLines[m][vLines[m].size()-1] = vKeyPts[i][j];
                bFound = true;
                }
            }
            }
            else
            {
                KeypointType comparePt = vLines[m][vLines[m].size()-1];
                if(abs(comparePt.newPt.x - vKeyPts[i][j].newPt.x) < 10)
                {
                if(abs(comparePt.newPt.x - vKeyPts[i][j].newPt.x) < minDist)
                {
                    minDist = abs(comparePt.newPt.x - vKeyPts[i][j].newPt.x);
                        bestLineID = m;
                }
                bFound = true;
                //break;
                }
            }
        }
        }
        if(bFound == false)
        {
        vector<KeypointType> vtmp;
        vtmp.push_back(vKeyPts[i][j]);
        vLines.push_back(vtmp);
        }
        else if(bestLineID >= 0)
        {
        vLines[bestLineID].push_back(vKeyPts[i][j]);
        }
    }
    }

    // clean key point in line
    for(int m = 0; m < vLines.size(); m++)
    {
    if(vLines[m].size() < 3)
        continue;
    float aveDeltaX = 0;
        for(int n = 1; n < vLines[m].size(); n++)
    {
        aveDeltaX += (vLines[m][n].newPt.x - vLines[m][n-1].newPt.x);
    }
    aveDeltaX /= (vLines[m].size() - 1);

        for(int n = 1; n < vLines[m].size(); n++)
        {
            int deltaX = (vLines[m][n].newPt.x - vLines[m][n-1].newPt.x);
        if(abs(deltaX - aveDeltaX) > 5)
        {
        for(; n < vLines[m].size(); n++)
            vLines[m][n].newPt.x = -1;
        break;
        }
        }
    }

    // draw new key points in eye view image
    Mat ipm(m_slcImgs.size()*10, 128, CV_8UC3);
    for(int y = 0; y < ipm.rows; y++)
        for(int x = 0; x < ipm.cols*3; x++)
            ipm.at<unsigned char>(y, x) = 0;
    for(int i = 0; i < vKeyPts.size(); i++)
    {
    for(int j = 0; j < vKeyPts[i].size(); j++)
    {
        circle(ipm, Point(vKeyPts[i][j].newPt.x, vKeyPts[i][j].newPt.y*10), 5, CV_RGB(255,0,0));
    }
    }
    imshow("a", ipm);

    // convert to result
    for(int m = 0; m < vLines.size(); m++)
    {
    if(vLines[m].size() < 3)
        continue;
    vector<Point> line;
    for(int n = 0; n < vLines[m].size(); n++)
    {
        if(vLines[m][n].newPt.x == -1)
                continue;
        line.push_back(vLines[m][n].oriPt);
    }
    vvResult.push_back(line);
    }
}

void LaneDetector::removeDuplicate(vector<vector<float> > &vvMsk)
{
    vector<int> vLeft, vRight, vTop;
    getScanLine(vLeft, vRight, vTop);
    
    int gap = 1;
    for(int idx = 0; idx < m_slcImgs.size(); idx++)
    {
        for(int start = 0; start < 128; start++)
        {
            if(vvMsk[idx][start] > 0.002)
            {
                for(int j = start - gap; j <= start + gap; j++)
                {
                    if(j >= 0 && j < 128)
                    {
                        if(vvMsk[idx][j] == 0)
                        vvMsk[idx][j] = 0.001;
                    }
                }
            }
        }
    }
    for(int idx = 0; idx < m_slcImgs.size(); idx++)
    {
        for(int start = 0; start < 128; start++)
        {
            if(vvMsk[idx][start] > 0)
            {
                int end = start + 1;
                for(; end < 128; end++)
                {
                    if(vvMsk[idx][end] < 0.0001)
                    break;
                }
                int maxIdx = (start + end) / 2;
                
                for(;start<end && start < 128;start++)
                vvMsk[idx][start] = 0;
                vvMsk[idx][maxIdx] = 1;
                start = end;
            }
        }
    }
}

void LaneDetector::process(vector< vector<Point> > &vvResult, Mat &rgbImg)
{
    get_proposals(rgbImg);
    
    vector<vector<float> > vvMsk;
    vvMsk.resize(m_slcImgs.size());
    for(int i = 0; i < vvMsk.size(); i++)
    vvMsk[i].resize(128, 0);
    
    int input_channels_ = 3;
    int input_height_ = 16;
    int input_width_ = 128;
    for(int idx = 0; idx < m_slcImgs.size(); idx++)
    {
        float* input_data = _net->input_blobs()[0]->mutable_cpu_data();
        for (int i = 0; i < input_channels_; ++ i) {
            for (int h = 0; h < input_height_; ++ h) {
                for (int w = 0; w < input_width_; ++ w) {
                    input_data[h * input_width_ + w] = m_slcImgs[idx].at<cv::Vec3b>(h, w)[i]/255.0;
                }
            }
            input_data += input_height_ * input_width_;
        }
        _net->Forward();
        boost::shared_ptr<caffe::Blob<float> > presult = _net->blob_by_name("output");
        const float* pstart = presult->cpu_data();
        //vvResult.push_back(vector<Point>());
        float minVal = FLT_MAX;
        float maxVal = FLT_MIN;
        for(int j = 0; j < 128; j++)
        {
            if(pstart[j] > maxVal)
            maxVal = pstart[j];
            if(pstart[j] < minVal)
            minVal = pstart[j];
            if(pstart[j] > 0.9)
            {
                vvMsk[idx][j] = pstart[j];
                //int x = j/128.0 * (vRight[idx]-vLeft[idx]+1) + vLeft[idx];
                //vvResult[idx].push_back(cvPoint(x, vTop[idx]));
            }
        }
    }
    
    getResults(vvResult, vvMsk);
}

void LaneDetector::initScanlinePos()
{
    for(int index = 0; index < m_NUM_SLICE_IMG; index++)
    {
        int deltaY = m_LINE_STEP;
        int left = 0;
        if (index == 0)
            left = 180;
        else if(index == 1)
            left = 150;
        else if(index == 2)
            left = 120;
        else if(index == 3)
            left = 90;
        else if(index == 4)
            left = 60;
        else if(index == 5)
            left = 30;
        m_vLeft.push_back(left);
        m_vRight.push_back(m_IMG_WID - left - 1);
        //m_vTop.push_back(m_LINE_TOP + index * m_LINE_STEP);
        m_vTop.push_back(m_LINE_TOP + index * deltaY);
    }
}

void LaneDetector::initKalman()
{
    for(int i = 0; i < m_NUM_SLICE_IMG*2; i++)
    {
        m_vkf.push_back(KalmanFilter(2,1,0));
        m_vState.push_back(Mat(2, 1, CV_32F));
        m_vMeasurement.push_back(Mat::zeros(1, 1, CV_32F));
    }
    for(int i = 0; i < m_NUM_SLICE_IMG*2; i++)
    {
        randn( m_vState[i], Scalar::all(0), Scalar::all(0.1) );
        m_vkf[i].transitionMatrix = *(Mat_<float>(2, 2) << 1, 1, 0, 1);
        setIdentity(m_vkf[i].measurementMatrix);
        setIdentity(m_vkf[i].processNoiseCov, Scalar::all(1e-5));
        setIdentity(m_vkf[i].measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(m_vkf[i].errorCovPost, Scalar::all(1));
        randn(m_vkf[i].statePost, Scalar::all(0), Scalar::all(0.1));
    }
}

void LaneDetector::setProposalPos(vector<float> &vResult)
{
    for(int index = m_vLeft.size()-1; index >= 1; index--)
    {
        int result_left = vResult[index*2]*m_IMG_SLICE_WID/2;
        int result_right = vResult[index*2+1]*m_IMG_SLICE_WID/2;
        if(result_left > 5 && result_right > 5 && 
           result_left < m_IMG_SLICE_WID/2 - 5 && result_right < m_IMG_SLICE_WID/2 - 5)
        {
            result_left = result_left/(float)m_IMG_SLICE_WID * (m_vRight[index] - m_vLeft[index]) + m_vLeft[index];
            result_right = result_right/(float)m_IMG_SLICE_WID * (m_vRight[index] - m_vLeft[index]) + m_vLeft[index] + (m_vRight[index] - m_vLeft[index])/2;
            int delta = (m_vLeft[index-1] + m_vRight[index-1])/2 - (result_left + result_right)/2;
            m_vLeft[index-1] -= delta;
            m_vRight[index-1] -= delta;
            if(m_vLeft[index-1]<0) m_vLeft[index-1] = 0;
            if(m_vLeft[index-1]>=m_IMG_WID) m_vLeft[index-1] = m_IMG_WID-1;
            if(m_vRight[index-1]<0) m_vRight[index-1] = 0;
            if(m_vRight[index-1]>=m_IMG_WID) m_vRight[index-1] = m_IMG_WID-1;
        }
    }
}

void LaneDetector::drawScanline(Mat &rgbImg)
{
    vector<int> vLeft, vRight, vTop;
    getScanLine(vLeft, vRight, vTop);
    for(int i = 0; i < vLeft.size(); i++)
    {
        //circle(rgbImg, Point((vLeft[i] + vRight[i])/2, vTop[i]), 3, Scalar(255,255,0));
        line(rgbImg, Point(vLeft[i], vTop[i]), Point(vRight[i], vTop[i]), Scalar(255,0,0), 1);
    }
}
  
void LaneDetector::getScanLine(vector<int> &vLeft, vector<int> &vRight, vector<int> &vTop)
{
    for(int i = 0; i < m_NUM_SLICE_IMG; i++)
    {
        int left, right, curY, singleWid, sliceID;
        getProposalPos(i, left, right, curY, singleWid, sliceID);
        vLeft.push_back(left);
        vRight.push_back(right);
        vTop.push_back(curY);
    }
}
    	
void LaneDetector::getProposalPos(int index, int &left, int &right, int &curY, int &singleWid, int &sliceID)
{
    left = m_vLeft[index];
    right = m_vRight[index];
    curY = m_vTop[index];
    singleWid = right - left + 1;
    sliceID = index;
}

void LaneDetector::prepare_img_memory()
{
    m_vsliceImg_mem.reserve(m_NUM_SLICE_IMG);
    for (int i = 0; i < m_NUM_SLICE_IMG; i++) {
	Mat img(m_PROPOSAL_DURATION, m_SLICE_IMG_WIDTH, CV_8UC3);        
        m_vsliceImg_mem.push_back(img);
            
        //img = cvCreateImage(cvSize(m_IMG_ORI_WID, m_IMG_ORI_HGT), IPL_DEPTH_8U, 3);
        //m_oriImgs.push_back(img);
            
        //img = cvCreateImage(cvSize(m_IMG_SLICE_WID, m_IMG_SLICE_HGT), IPL_DEPTH_8U, 3);
        Mat img1(m_IMG_SLICE_HGT, m_IMG_SLICE_WID, CV_8UC3);
        m_slcImgs.push_back(img1);
    }
}
    	
void LaneDetector::get_slice_img(Mat& sliceImg, Mat& rgbImg, int y)
{
    // copy image
    for(int yy = 1; yy < sliceImg.rows; yy++)
    {
        sliceImg.row(yy-1) = sliceImg.row(yy) + Scalar(0,0,0);
    }
    sliceImg.row(sliceImg.rows-1) = rgbImg.row(y) + Scalar(0,0,0);
}

void LaneDetector::get_proposals(Mat& rgbImg)
{
    Rect rect;
        
    for (int i = 0; i < m_vsliceImg_mem.size(); i++) 
    {
        int left, right, curY, singleWid, sliceID;
        getProposalPos(i, left, right, curY, singleWid, sliceID);
            
        get_slice_img(m_vsliceImg_mem[i], rgbImg, curY);
/*            
        rect.x = left;
        rect.width = right - left + 1;
        rect.y = curY - 10;
        rect.height = 20;
        cvSetImageROI(rgbImg, rect);
        cvResize(rgbImg, m_oriImgs[i]);
        cvResetImageROI(rgbImg);
*/
        rect.x = left;
        rect.width = right - left + 1;
        rect.y = 0;
        rect.height = m_PROPOSAL_DURATION;
        Mat slcmat = m_vsliceImg_mem[i](rect);
        resize(slcmat, m_slcImgs[i], Size(m_IMG_SLICE_WID, m_IMG_SLICE_HGT));
    }
}
