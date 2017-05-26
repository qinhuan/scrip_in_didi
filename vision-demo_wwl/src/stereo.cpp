#include "stereo.h"
#include <opencv2/contrib/contrib.hpp>

cv::Mat Stereo::GetDepth(const shared_ptr<Blob<float> > ste_blob) {
  int height = ste_blob->height();
  int width = ste_blob->width();
  const float* ste_data = ste_blob->cpu_data();
  cv:: Mat depth = cv::Mat::zeros(height, width, CV_32F);
  memcpy(depth.data, ste_data, height *width *sizeof(float));
  return depth;
}

void Stereo::Draw(cv::Mat &image, const cv::Mat &depth) {
  // get color map 
  cv::Mat depth_show;
  double vmin, vmax, alpha;
  cv::minMaxLoc(depth, &vmin, &vmax);
  alpha = 255.0 / (vmax - vmin);
  depth.convertTo(depth_show, CV_8U, alpha, -vmin * alpha);
  cv::applyColorMap(depth_show, depth_show, cv::COLORMAP_OCEAN);

  int height = image.rows;
  int width = image.cols;
  
  cv::resize(depth_show, depth_show, image.size());
  
  cv::Mat frame = cv::Mat::zeros(height * 2, width, CV_8UC3);
  image.copyTo(frame(cv::Rect(0, 0, width, height)));
  depth_show.copyTo(frame(cv::Rect(0, height, width, height)));
  image = frame;
}

float Stereo::CalcDist(const cv::Mat &depth, const Box &box) {
  Box s_box = box.scale(depth.rows, depth.cols); 
  
  float dep = 0;
  int cnt = 0;
  vector<float> deps;
  for (int x = std::max(0, s_box.topLeft().x); x < std::min(s_box.bottomRight().x, depth.cols); ++x) {
    for (int y = std::max(0, s_box.topLeft().y); y < std::min(s_box.bottomRight().y, depth.rows); ++ y) {
      deps.push_back(depth.at<float>(y, x));
    }
  }
  if (deps.size() == 0) return -1;
  sort(deps.begin(), deps.end());
  float val = deps[deps.size() / 2];
  val = 1.0 / std::max(val, 1.0f / 1000) * 60;
  return val;
}
