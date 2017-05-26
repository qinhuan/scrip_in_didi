#include "detection.h"

void Detection::GetDetectedBoxes(const shared_ptr<Blob<float> > det_blob, 
    vector<Box> &boxes) {
  const float* det_data = det_blob->mutable_cpu_data();
  int num_boxes = det_blob->height();
  boxes.clear();
  for (int i = 0; i < num_boxes; ++ i) {
    // Skip invalid detection.
    if (det_data[0] == -1) {
      det_data += 7;
      continue;
    }
    vector<float> box_info(6, 0);
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    box_info[4] = det_data[2]; // score 
    box_info[5] = det_data[1] - 1; // label 
    for (int i = 0; i < 4; ++ i) {
      box_info[i] = det_data[i + 3];
    }

    det_data += 7;
    Box b = Box(box_info, "xyxy");
    if (b.score < threshod[b.label]) {
      continue;
    }
    boxes.push_back(b);
  }
  LOG(INFO) << "Number of boxes: " << boxes.size();
}

void Detection::FilterBoxes(vector<Box> &prev_boxes, vector<Box> &boxes, 
    vector<Box> &show_boxes) {
  for (int i = 0; i < boxes.size(); ++ i) {
    for (int j = 0; j < prev_boxes.size(); ++ j) {
      if (prev_boxes[j].iou(boxes[i]) > 0.5) {
        show_boxes.push_back(boxes[i]);
        break;
      }
    }
  }
}

void Detection::Draw(cv::Mat &image, const vector<Box> &boxes) {
  for (int i = 0; i < boxes.size(); ++ i) {
    Box box = boxes[i].scale(image.rows, image.cols);
    box.draw(image, true, color[box.label]);
  }
}

