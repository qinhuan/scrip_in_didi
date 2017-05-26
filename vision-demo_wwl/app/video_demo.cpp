#include "predictor.h"
#include "laneDetection.h"
#include "detection.h"
#include "segmentation.h"
#include "stereo.h"

#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <vector>

const int SHOW_HEIGHT = 375;
const int SHOW_WIDTH = 1242;
const int SHIFT_WIDTH = 0;
const int SHIFT_HEIGHT = 85;
const int LANE_INPUT_WIDTH = 640;

struct Argument {
  void init(int argc, char* argv[]) {
    CHECK_GE(argc, 5) << usage;
    version = argv[1];
    mode = atoi(argv[2]);
    video_path = argv[3]; 
    start_frames = atoi(argv[4]);
    if (argc > 5) {
      save_path = argv[5];
    }
    else {
      save_path = "/tmp/save.avi";
    }
  }
  const string usage; 
  string version;
  int mode, start_frames;
  string video_path, save_path;
} Parser;

int loadFrames(cv::VideoCapture cap, cv::Mat &frame) {
  cap.read(frame);
  //frame = frame(cv::Rect(SHIFT_WIDTH, SHIFT_HEIGHT, 640, 195));
  //LOG(INFO)<<frame.rows<<" "<<frame.cols;
  cv::resize(frame, frame, cv::Size(0, 0),  LANE_INPUT_WIDTH * 1.0 / frame.cols, 
      LANE_INPUT_WIDTH * 1.0 / frame.cols);
  //LOG(INFO)<<frame.rows<<" "<<frame.cols;
  return 1;
}

int saveFrames(cv::VideoWriter writer, cv::Mat &frame) {
  cv::resize(frame, frame, cv::Size(SHOW_WIDTH, SHOW_HEIGHT * (Parser.mode & 4 ? 1 : 2)));
  writer<<frame;
  return 1;
}

void drawLanes(Mat &frame, vector<vector<cv::Point> > &lanes) {
  for(int i = 0; i < lanes.size(); i++) {
    for(int j = 0; j < lanes[i].size(); j++) {
      cv::circle(frame, lanes[i][j], 3, cv::Scalar(0, 0, 255));
    }
  }        
  for(int i = 0; i < lanes.size(); i++) {
    for(int j = 1; j < lanes[i].size(); j++) {
      cv::line(frame, lanes[i][j-1], lanes[i][j], cv::Scalar(0, 0, 255), 2);
    }
  }
}


int main(int argc, char* argv[]) {
  Parser.init(argc, argv);
  string version_path = "../models/" + Parser.version + "/";
  string model_def = version_path + Parser.version + ".prototxt";
  string model_weights = version_path + Parser.version + ".caffemodel";
  string model_def_lane = version_path + Parser.version + "_lane.prototxt"; 
  string model_weights_lane = version_path + Parser.version + "_lane.caffemodel"; 
  Predictor pred(model_def, model_weights); 
  LaneDetector lane_pred(LANE_INPUT_WIDTH, model_def_lane, model_weights_lane);

  cv::VideoCapture cap(Parser.video_path);
  LOG(INFO) << "Start from " << Parser.start_frames << " frames.";
  cap.set(CV_CAP_PROP_POS_FRAMES, Parser.start_frames);
  cv::VideoWriter writer(Parser.save_path,
      CV_FOURCC('M', 'J', 'P', 'G'),
      25.0,
      cv::Size(SHOW_WIDTH, SHOW_HEIGHT * (Parser.mode & 4 ? 1 : 2)));
  cv::Mat image;
  while (loadFrames(cap, image)) {
    pred.Process(image);
    vector<vector<cv::Point> > lanes;
    if ((Parser.mode & 8) == 0) {
  //    cv::Mat resize_image;
  //    int rw = 480;
  //    int rh = rw * 1.0 /image.cols * image.rows;
  //    cv::resize(image, resize_image, cv::Size(rh, rw));
      lane_pred.process(lanes, image);
    }
    
    cv::Mat masks = Segmentation::GetMasks(pred.Output("seg_score")); 
    cv::Mat depth = Stereo::GetDepth(pred.Output("ste_disp1"));
    vector<Box> boxes;
    Detection::GetDetectedBoxes(pred.Output("detection_out"), boxes); 
    // Get depth of bounding boxes
    for (int i = 0; i < boxes.size(); ++ i) {
      boxes[i].score = Stereo::CalcDist(depth, boxes[i]);
    }
    cv::Mat show_image = image;
    // Show detection
    if ((Parser.mode & 1) == 0) {
      Detection::Draw(show_image, boxes);
    }
    // Show segmentation
    if ((Parser.mode & 2) == 0) {
      Segmentation::Draw(show_image, masks);
    }
    // Show stereo
    if ((Parser.mode & 4) == 0) {
      Stereo::Draw(show_image, depth);
    } 
    // Show lane
    if ((Parser.mode & 8) == 0) {
      lane_pred.drawScanline(show_image);
      drawLanes(show_image, lanes);
    }
    
    cv::resize(show_image, show_image, 
            cv::Size(SHOW_WIDTH, SHOW_HEIGHT * (Parser.mode & 4 ? 1 : 2)));
    cv::imshow("Vision System", show_image);
    saveFrames(writer, show_image);
    cv::waitKey(1);
  }
  return 0;
}
