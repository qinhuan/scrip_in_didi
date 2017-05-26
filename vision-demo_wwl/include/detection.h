#ifndef DETECTION_H 
#define DETECTION_H
/**
 * Process detection output.
 * */
#include "box.h"

#include <caffe/caffe.hpp>
#include <vector>

class Detection {
 public:
  static void GetDetectedBoxes(const shared_ptr<Blob<float> > det_blob, vector<Box> &boxes);
  static void Draw(cv::Mat &image, const vector<Box> &boxes);
  static void FilterBoxes(vector<Box> &prev_boxes, vector<Box> &boxes, 
      vector<Box> &show_boxes);
  static float threshod[10]; 
  static cv::Scalar color[10]; 
  // Todo threshold, color by label, 
};
float Detection::threshod[10] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.95, 0.95, 0.95, 0.5, 0.5};
cv::Scalar Detection::color[10] = {
  cv::Scalar(255, 0, 0), 
  cv::Scalar(0, 255, 0),
  cv::Scalar(0, 0, 255),
  cv::Scalar(0, 255, 255),
  cv::Scalar(255, 0, 255),
  cv::Scalar(255, 255, 0),
};


#endif // DETECTION_H

