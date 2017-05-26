#ifndef STEREO_H
#define STEREO_H

#include "box.h"

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;

class Stereo {
 public:
  // Get stereo depth image.
  static cv::Mat GetDepth(const shared_ptr<Blob<float> > ste_blob);

  // Concat source image with depth image.
  static void Draw(cv::Mat &image, const cv::Mat &depth);
  
  // Caculate distance of an object.
  static float CalcDist(const cv::Mat &depth, const Box &box); 
};

#endif // STEREO_H
