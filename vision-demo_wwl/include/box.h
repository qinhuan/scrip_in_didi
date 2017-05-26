#ifndef BOX_H
#define BOX_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

using namespace caffe;

class Box {
 public:
  Box() {}

  Box(const vector<float> &box_info, const string &type) {
    if (type == "xywh") {
      x = box_info[0];    
      y = box_info[1];    
      w = box_info[2];    
      h = box_info[3];    
    }
    else if (type == "xyxy") {
      x = box_info[0];    
      y = box_info[1];    
      w = box_info[2] - x;    
      h = box_info[3] - y;    
    }
    else {
      LOG(ERROR)<<"Unknow box type \"" << type << "\".";
    } 
    label = 0;
    score = 0;
    if (box_info.size() > 4) {
      score = box_info[4];
    }
    if (box_info.size() > 5) {
      label = int(box_info[5]);
    }
  }
  
  ~Box() {}
  
  cv::Point topLeft() const {
    return cv::Point(int(x), int(y));
  }
  
  cv::Point bottomRight() const {
    return cv::Point(int(x + w), int(y + h));
  }

  void draw(cv::Mat &image, bool show_score=false, 
      cv::Scalar color=cv::Scalar(255, 0, 0), int thickness=2) {
    char score_str[10];
    sprintf(score_str, "%.4f", score);
    putText(image, score_str, topLeft(), 0, 0.7, color, thickness);
    cv::rectangle(image, topLeft(), bottomRight(), color, thickness); 
  }
  
  Box scale(int height, int width) const {
    Box s_box = *this;
    s_box.x = x * width;
    s_box.y = y * height;
    s_box.w = w * width;
    s_box.h = h * height;
    return s_box;
  }

  float area() const {
    return w * h;
  }
  
  Box cross(const Box& box) const {
    Box crs_box;
    crs_box.x = std::max(x, box.x);
    crs_box.y = std::max(y, box.y);
    crs_box.w = std::max(0.0f, std::min(x + w, box.x + box.w) - crs_box.x);
    crs_box.h = std::max(0.0f, std::min(y + h, box.y + box.h) - crs_box.y);
    return crs_box;
  }
  
  float iou(const Box& box) const {
    Box inter_box = cross(box);
    float area_i = inter_box.area();
    float area_u = box.area() + area() - area_i;
    CHECK_GT(area_u, 0.0) << "The union area need to be positive!";
    return area_i / area_u; 
  }

  void crop(const cv::Mat &image, cv::Mat &roi_image) {
    //Todo
  }
  
  int label;
  float x, y, h, w;
  float score;
};

#endif // BOX_H
