#include "segmentation.h"

cv::Mat Segmentation::GetMasks(const shared_ptr<Blob<float> > seg_blob) {
  int height = seg_blob->height();
  int width = seg_blob->width();
  const float* seg_data = seg_blob->cpu_data();
  cv:: Mat masks = cv::Mat::zeros(height, width, CV_32F);
  memcpy(masks.data, seg_data, height *width *sizeof(float));
  return masks;
}

void Segmentation::Draw(cv::Mat &image, const cv::Mat &masks) {
  // Resize mask image
  cv::Mat masks_resize;
  cv::resize(masks, masks_resize, image.size());
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float index = masks_resize.at<float>(i,j);
      if (index < 0.5)
        image.at<cv::Vec3b>(i,j)[1] = 128; 
    }
  }
}
