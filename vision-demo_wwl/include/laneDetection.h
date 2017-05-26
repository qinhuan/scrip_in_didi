#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using namespace cv;

class LaneDetector{
public:
    // scanlineGap: 7
    // scanLineNum: 6
    // startX: 180
    // deltaX: 30
    LaneDetector(int imgWid, int imgHgt, 
            int scanlineGap, int scanlineNum, int startX, int deltaX, 
            string model_file, string trained_file):
		m_IMG_WID(imgWid), m_SLICE_IMG_WIDTH(imgWid)
    {
        m_LINE_TOP = imgHgt - 3 - (scanlineNum-1) * scanlineGap; //125;
        m_NUM_SLICE_IMG = scanlineNum;
        m_LINE_STEP   = scanlineGap;

        for(int index = 0; index < m_NUM_SLICE_IMG; index++)
        {
            int deltaY = m_LINE_STEP;
            int left = startX - index * deltaX;

            if(left < 0)
                left = 0;
            m_vLeft.push_back(left);
            m_vRight.push_back(m_IMG_WID - left - 1);
            //m_vTop.push_back(m_LINE_TOP + index * m_LINE_STEP);
            m_vTop.push_back(m_LINE_TOP + index * deltaY);
        }

        // initial proposal pos
        //initScanlinePos();
        initKalman();
        
        prepare_img_memory();
        _net.reset(new Net<float>(model_file, caffe::TEST));
        _net->CopyTrainedLayersFrom(trained_file);
    }
    void process(vector< vector<Point> > &vvResult, Mat &rgbImg);
    void drawScanline(Mat &rgbImg);
private:
    static const int m_IMG_CHANNEL = 3;
    static const int m_IMG_ORI_WID = 128;
    static const int m_IMG_ORI_HGT = 20;
    static const int m_IMG_SLICE_WID = 128;
    static const int m_IMG_SLICE_HGT = 16;
    
    static const int m_PROPOSAL_DURATION = 64;
    
    int m_LINE_TOP;
    int m_NUM_SLICE_IMG;
    int m_LINE_STEP;

    int m_IMG_WID;
    int m_SLICE_IMG_WIDTH;
 
    vector<Mat> m_vsliceImg_mem;
    vector<Mat> m_oriImgs;
    vector<Mat> m_slcImgs;
    
    std::vector<Mat> dv_1;
    std::vector<Mat> dv_2;
    shared_ptr<Net<float> > _net;
  
    vector<int> m_vLeft;
    vector<int> m_vRight;
    vector<int> m_vTop;

    // kalman staff
    vector<KalmanFilter> m_vkf;
    vector<Mat> m_vState;
    vector<Mat> m_vMeasurement;

    void initKalman();
    void initScanlinePos();
    void getScanLine(vector<int> &vLeft, vector<int> &vRight, vector<int> &vTop);
    void getProposalPos(int index, int &left, int &right, int &curY, int &singleWid, int &sliceID);
    void setProposalPos(vector<float> &vvResult);
    void prepare_img_memory();
    void get_slice_img(Mat &sliceImg, Mat &rgbImg, int y);
    void get_proposals(Mat &rgbImg);
    void removeDuplicate(vector<vector<float> > &vvMsk);
    void getResults(vector< vector<Point> > &vvResult, vector<vector<float> > &vvMsk);
};
#endif
