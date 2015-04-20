// Copyright (c) 2010-2014, The Video Segmentation Project
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the The Video Segmentation Project nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ---

#include "base/base_impl.h"
#include <cmath>

#include <gflags/gflags.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

DEFINE_string(input_video, "", "The input video (.mp4) REQUIRED");
DEFINE_bool(logging, false, "If set output various logging information.");
DEFINE_string(output_file, "", "Output directory for scoring text files. REQUIRED");

struct RGBPoint {
  int r, g, b;
};

struct ColorBin {
  // Index to which color this bin is.
  int color_idx;
  // Point with the maximum distance away from the color_idx.
  float radius = 0.f;
  // The weight of this bin.
  int weight;
  // List of RGB points that belong to this bin.
  int num_points = 0;
};

float dist2D(float dX0, float dY0, float dX1, float dY1) {
    return sqrt((dX1 - dX0)*(dX1 - dX0) + (dY1 - dY0)*(dY1 - dY0));
}

float dist3D(float dX0, float dY0, float dZ0, float dX1, float dY1, float dZ1) {
    return sqrt((dX1 - dX0)*(dX1 - dX0) + (dY1 - dY0)*(dY1 - dY0) + (dZ1 - dZ0)*(dZ1 - dZ0));
}

// Assumes point is three dimensional, from [0,1].
void rgb_to_lab(std::vector<float> point, std::vector<float>* lab_point) {
  for (int idx = 0; idx < point.size(); idx++) {
    if (point[idx] > 0.04045) {
      point[idx] = pow((point[idx] + 0.055) / 1.055, 2.4);
    } else {
      point[idx] /= 12.92;
    }
    point[idx] *= 100;
  }

  float x_point = point[0] * 0.4124 + point[1] * 0.3576 + point[2] * 0.1805;
  float y_point = point[0] * 0.2126 + point[1] * 0.7152 + point[2] * 0.0722;
  float z_point = point[0] * 0.0193 + point[1] * 0.1192 + point[2] * 0.9505;

  x_point /= 95.047;
  y_point /= 100.000;
  z_point /= 108.883;

  std::vector<float> xyz_point = {x_point, y_point, z_point};

  for (int idx = 0; idx < xyz_point.size(); idx++) {
    if (xyz_point[idx] > 0.008856) {
      xyz_point[idx] = pow(xyz_point[idx], 1.0f / 3.0f);
    } else {
      xyz_point[idx] = xyz_point[idx] * 7.87 + (16.0f / 116.0f);
    }
  }

  (*lab_point)[0] = 116 * xyz_point[1] - 16;
  (*lab_point)[1] = 500 * (xyz_point[0] - xyz_point[1]);
  (*lab_point)[2] = 200 * (xyz_point[1] - xyz_point[2]);
}

float rgb_distance(int r1, int g1, int b1, int r2, int g2, int b2) {
  std::vector<float> p1 = {r1 / 255.0f, g1 / 255.0f, b1 / 255.0f};
  std::vector<float> p2 = {r2 / 255.0f, g2 / 255.0f, b2 / 255.0f};
  std::vector<float> p1_lab = {0.f, 0.f, 0.f};
  std::vector<float> p2_lab = {0.f, 0.f, 0.f};

  rgb_to_lab(p1, &p1_lab);
  rgb_to_lab(p2, &p2_lab);

  return dist3D(p1_lab[0], p1_lab[1], p1_lab[2], p2_lab[0], p2_lab[1], p2_lab[2]);
}

int main(int argc, char** argv) {
  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_logging) {
    FLAGS_logtostderr = 1;
  }

  if (FLAGS_input_video.empty()) {
    std::cerr << "Input file not specified. Specify via -input_video. \n";
    return 1;
  }
  if (FLAGS_output_file.empty()) {
    std::cerr << "Please specify the path to an output file.\n";
    return 1;
  }

  // Denote color weights for segment relevance algorithm.
  std::unordered_map<int*, int> color_weights;
  int white[] = {255, 255, 255};
  int yellow[] = {255, 255, 0};
  int yellow_green[] = {173, 255, 0};
  int green[] = {0, 255, 0};
  int blue_green[] = {0, 173, 255};
  int blue[] = {0, 0, 255};
  int violet[] = {255, 173, 255};
  int red[] = {255, 0, 0};
  int orange[] = {255, 173, 0};
  int magenta[] = {255, 0, 255};
  int cyan[] = {0, 255, 255};
  int black[] = {0, 0, 0};

  color_weights[white] = 1;
  color_weights[yellow] = 9;
  color_weights[yellow_green] = 7;
  color_weights[green] = 6;
  color_weights[blue_green] = 5;
  color_weights[blue] = 4;
  color_weights[violet] = 3;
  color_weights[red] = 6;
  color_weights[orange] = 8;
  color_weights[magenta] = 6;
  color_weights[cyan] = 4;
  color_weights[black] = 1;

  std::vector<int*> colors = {white, yellow, yellow_green, green, blue_green, blue, violet, red,
                              orange, magenta, cyan, black};



  // Input video filename
  std::string vid_filename = FLAGS_input_video;

  // Read in video -- make sure it is read in properly.
  cv::VideoCapture capture(vid_filename);
  if (!capture.isOpened()) {
    std::cout << "Video file " << vid_filename << " cannot be opened." << std::endl;
    return -1;
  }

  LOG(INFO) << "Frame Width: " << capture.get(3);
  LOG(INFO) << "Frame Height: " << capture.get(4);

  int frame_width = capture.get(3);
  int frame_height = capture.get(4);

  std::unordered_map<int, float> scores;

  // For each frame.
  int frame_idx = 0;
  for (;;) {
    LOG(INFO) << "Processing frame " << frame_idx << std::endl;
    // Get video.
    cv::Mat current_frame(frame_width, frame_height, CV_8UC3);
    
    capture >> current_frame;
    // LOG(INFO) << resized_frame.cols << "," << resized_frame.rows << std::endl;
    // break;
    // Breakpoint.
    if (current_frame.empty()) {
      LOG(INFO) << "Current frame is empty, exiting for loop.";
      break;
    }
    cv::Mat resized_frame;
    cv::resize(current_frame, resized_frame, cv::Size(), 0.1, 0.1);
    frame_width = resized_frame.cols;
    frame_height = resized_frame.rows;

    LOG(INFO) << "Resized Frame Width: " << frame_width;
    LOG(INFO) << "Resized Frame Height: " << frame_height;

    // Create bins.
    std::unordered_map<int, ColorBin> bins;
    for (int pt_idx = 0; pt_idx < colors.size(); pt_idx++) {
      ColorBin bin;
      bin.color_idx = pt_idx;
      bin.weight = color_weights[colors[pt_idx]];
      bins[pt_idx] = bin;
    }

    // Iterate image for colors.
    uchar* output_ptr;
    float color_distance;
    int color_idx;
    for (int y_idx = 0; y_idx < resized_frame.rows; y_idx++) {
      output_ptr = resized_frame.ptr<uchar>(y_idx);
      for (int x_idx = 0; x_idx < 3 * resized_frame.cols; x_idx += 3) {
        // Find color bin.
        color_distance = -1;
        color_idx = -1;
        for (int pt_idx = 0; pt_idx < colors.size(); pt_idx++) {
          float calc_distance = rgb_distance(int(output_ptr[x_idx + 2]),
                                             int(output_ptr[x_idx + 1]),
                                             int(output_ptr[x_idx]),
                                             colors[pt_idx][0],
                                             colors[pt_idx][1],
                                             colors[pt_idx][2]);
          if (color_distance == -1) {
            color_distance = calc_distance;
            color_idx = pt_idx;
          }
          else if (calc_distance < color_distance) {
            color_distance = calc_distance;
            color_idx = pt_idx;
          }
        }
        // Add to radius sum, and num_points.
        bins[color_idx].radius += color_distance;
        bins[color_idx].num_points++;
      }
    } // Finish image iteration.

    float score = 0;
    // Calculate the score based on color bins.
    for (auto& bin : bins) {
      LOG(INFO) << "##########################################################";
      LOG(INFO) << " ";
      LOG(INFO) << "Bin: " << bin.first;
      LOG(INFO) << "  Radius: " << bin.second.radius;
      LOG(INFO) << "  Weight: " << bin.second.weight;
      LOG(INFO) << "  Points: " << bin.second.num_points;
      LOG(INFO) << "  Scale: " << float(bin.second.num_points) / 
                                 float(frame_width * frame_height);
      LOG(INFO) << "  Radius Scale: " << float(bin.second.radius) / 
                                         float(frame_width * frame_height);
      LOG(INFO) << "Bin Score: " << (bin.second.radius * bin.second.weight) /
                                    float(frame_width * frame_height);
      LOG(INFO) << " ";
      LOG(INFO) << "##########################################################";
      // This looks weird because num_points ends up cancelling itself out.
      // Explanation:
      // sum( (d_sum / num_points) * weight * (num_points) / (total_points)
      // d_sum / num_points is the average radius. 
      // num_points / total_points is the scale based on pixel points.
      // Final formula looks simpler for that reason:
      //        d_sum * weight / total_points.
      score += (bin.second.radius * bin.second.weight) /
               float(frame_width * frame_height);
    }
    LOG(INFO) << "Total points: " << frame_width * frame_height;
    LOG(INFO) << "Total score: " << score;

    scores[frame_idx] = score;
    frame_idx++;
  } // Closes for loop that iterates each frame.

  // Output file.
  std::string curr_file = FLAGS_output_file;
  std::ofstream output_stream(curr_file, std::ios_base::out);

  for (int idx = 0; idx < scores.size(); idx++) {
    output_stream << idx << "," << scores[idx] << std::endl;
  }

  return 0;
}

