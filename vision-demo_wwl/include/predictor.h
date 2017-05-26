#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;

class Predictor {
 public:
  Predictor(const string& model_def, const string& model_weights,
      const string& norm_type="MeanValue");
  ~Predictor() {}

  void Process(const cv::Mat& image);

  
  shared_ptr<Blob<float> > Output(const string& name);

 private:
  void FeedIn(const cv::Mat& image);

 protected:
  shared_ptr<Net<float> > net_;
  int input_height_;
  int input_width_;
  int input_channels_;
  
  vector<int> mean_values_;
  float scale_;
  cv::Mat mean_;
};

#endif //PREDICTOR_H
