// c++ interface of caffe networks
#include "predictor.h"

Predictor::Predictor(const string& model_def,
    const string& model_weights, const string& norm_type) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  LOG(INFO) << "RUN IN CPU MODE!";
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  // Load the networks.
  net_.reset(new Net<float>(model_def, TEST));
  net_->CopyTrainedLayersFrom(model_weights);
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  input_channels_ = input_layer->channels();
  input_height_ = input_layer->height();
  input_width_ = input_layer->width();
  CHECK(input_channels_ == 3)<< "Input layer should be 3 channels.";

  // Set mean value.
  mean_values_.resize(3, 0);
  scale_ = 1.0;
  if (norm_type == "MeanValue") {
    mean_values_[0] = 114; 
    mean_values_[1] = 115; 
    mean_values_[2] = 108; 
  }
  else if (norm_type == "Scale") {
    scale_ = 1.0/255;
  }
  else {
    LOG(ERROR)<<"Unkown norm type: "<<norm_type;
  }
}

void Predictor::Process(const cv::Mat& image) {
  CHECK(image.channels() == 3) << "Input layer should be 3 channels."; 
  net_->Reshape();
  FeedIn(image);
  net_->Forward();
}

shared_ptr<Blob<float> > Predictor::Output(const string& name) {
  return net_->blob_by_name(name); 
}

void Predictor::FeedIn(const cv::Mat& image) {
  cv::Mat sample = image;
  if (image.rows != input_height_ || image.cols != input_width_) {
    cv::resize(image, sample, cv::Size(input_width_, input_height_));
  }
  // Sub mean value
  float* input_data = net_->input_blobs()[0]->mutable_cpu_data();
  for (int i = 0; i < input_channels_; ++ i) {
    for (int h = 0; h < input_height_; ++ h) {
      for (int w = 0; w < input_width_; ++ w) {
        input_data[h * input_width_ + w] = 
            (sample.at<cv::Vec3b>(h, w)[i] - mean_values_[i]) / scale_; 
      }
    }
    input_data += input_height_ * input_width_;
  } 
}

