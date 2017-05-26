#include "predictor.h"
#include "segmentation.h"

#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <vector>

const int SHOW_HEIGHT = 375;
const int SHOW_WIDTH = 1242;
const int SHIFT_WIDTH = 0;
const int SHIFT_HEIGHT = 85;

struct Argument {
  void init(int argc, char* argv[]) {
    CHECK_GE(argc, 4) << usage;
    model_def = argv[1];
    model_weights = argv[2];
    video_path = argv[3]; 
    if (argc > 4) {
      save_path = argv[4];
    }
    else {
      save_path = "/tmp/save.avi";
    }
  }
  const string usage; 
  string model_def, model_weights;
  string video_path, save_path;
} Parser;

int loadFrames(cv::VideoCapture cap, cv::Mat &frame) {
    cap.read(frame);
    frame = frame(cv::Rect(SHIFT_WIDTH, SHIFT_HEIGHT, 640, 195));
    return 1;
}

int saveFrames(cv::VideoWriter writer, cv::Mat &frame) {
  cv::resize(frame, frame, cv::Size(SHOW_WIDTH, SHOW_HEIGHT));
  writer<<frame;
  return 1;
}

int main(int argc, char* argv[]) {
  Parser.init(argc, argv);
  string model_def = Parser.model_def;
  string model_weights = Parser.model_weights;
  Predictor pred(model_def, model_weights); 

  cv::VideoCapture cap(Parser.video_path);
  cv::VideoWriter writer(Parser.save_path,
      CV_FOURCC('M', 'J', 'P', 'G'),
      25.0,
      cv::Size(SHOW_WIDTH, SHOW_HEIGHT));
  cv::Mat image;
  while (loadFrames(cap, image)) {
    pred.Process(image);
    
    cv::Mat masks = Segmentation::GetMasks(pred.Output("score")); 
    // Show segmentation
    Segmentation::Draw(image, masks);
    
    cv::imshow("Vision System", image);
    saveFrames(writer, image);
    cv::waitKey(1);
  }
  return 0;
}
