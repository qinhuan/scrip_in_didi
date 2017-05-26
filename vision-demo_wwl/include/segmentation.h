#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;

class Segmentation {
 public:
  // Get segmentation masks.
  static cv::Mat GetMasks(const shared_ptr<Blob<float> > seg_blob);
  
  // Draw masks on source images with default color green.
  static void Draw(cv::Mat &image, const cv::Mat &masks); 
};

#endif // SEGMENTATION_H
