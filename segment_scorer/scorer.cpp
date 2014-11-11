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

#include "segment_util/segmentation_io.h"
#include "segment_util/segmentation_render.h"
#include "segment_util/segmentation_util.h"

DEFINE_double(hierarchy_level, 0.8, "[0, 1] percent of hierarchy, 0 is lowest, 1 is highest.");
DEFINE_string(input_video, "", "The input video (.mp4) REQUIRED");
DEFINE_string(input_segmentation, "", "The input segmentation protobuffer (.pb). REQUIRED");
DEFINE_bool(logging, false, "If set output various logging information.");
DEFINE_string(output_file, "", "Output directory for scoring text files. REQUIRED");

using namespace segmentation;

typedef SegmentationDesc::Region2D SegRegion;
typedef SegmentationDesc::Polygon Polygon;
typedef SegmentationDesc::CompoundRegion CompRegion;

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

  if (FLAGS_input_video.empty() || FLAGS_input_segmentation.empty()) {
    std::cerr << "Input files not specified. Specify via -input_video and -input_segmentation.\n";
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

  color_weights[white] = 10;
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

  int frame_width = 1920;
  int frame_height = 1080;
  int top_left[] = {frame_width / 3, frame_height / 3};
  int top_right[] = {2 * frame_width / 3, frame_height / 3};
  int bottom_left[] = {frame_width / 3, 2 * frame_height / 3};
  int bottom_right[] = {2 * frame_width / 3, 2 * frame_height / 3};
  std::vector<int*> rot_points = {top_left, top_right, bottom_left, bottom_right};


  // Input segmentation filename
  std::string seg_filename = FLAGS_input_segmentation;
  std::string vid_filename = FLAGS_input_video;

  // Read in video -- make sure it is read in properly.
  cv::VideoCapture capture(vid_filename);
  if (!capture.isOpened()) {
    std::cout << "Video file " << vid_filename << " cannot be opened." << std::endl;
    return -1;
  }

  // Read segmentation file.
  SegmentationReader segment_reader(seg_filename);
  segment_reader.OpenFileAndReadHeaders();
  std::vector<int> segment_headers = segment_reader.GetHeaderFlags();

  LOG(INFO) << "Segmentation file " << seg_filename << " contains "
            << segment_reader.NumFrames() << " frames.\n";

  Hierarchy hierarchy;
  int absolute_level = -1;

  std::map<int, float> scores;
  float max_score = 0;
  float min_score = -1;

  // For each frame.
  for (int frame_idx = 0; frame_idx < segment_reader.NumFrames(); ++frame_idx) {

    // Read the segmentation from file.
    SegmentationDesc segmentation;
    segment_reader.ReadNextFrame(&segmentation);
    segment_reader.SeekToFrame(frame_idx);

    // Get frame width & height. This should be outside for loop for optimization.
    frame_width = segmentation.frame_width();
    frame_height = segmentation.frame_height();
    
    // Get video.
    cv::Mat current_frame(frame_width, frame_height, CV_8UC3);
    capture >> current_frame;

    // Maps a parent segment id to its number of pixels.
    std::map<int, int> segment_sizes;
    std::map<int, std::vector<int>> segment_avg_color;
    std::map<int, std::vector<int>> segment_bounding_box;

    // Setup the hierarchy.
    if (segmentation.hierarchy_size() > 0) {
      hierarchy.Clear();
      hierarchy.MergeFrom(segmentation.hierarchy());

      // Convert fractional to constant absolute level.
      absolute_level = FLAGS_hierarchy_level * (float)hierarchy.size();
      LOG(INFO) << "Selecting level " << absolute_level << " of " << hierarchy.size()
                << std::endl;
    }

    // Iterate through regions in the current frame.
    for (const auto& region : segmentation.region()) {
      // Get id of the correct hierarchy level.
      int region_id = region.id();
      if (absolute_level != 0) {
        region_id = GetParentId(region_id, 0, absolute_level, hierarchy);
      }

      if (segment_avg_color.find(region_id) == segment_avg_color.end()) {
        segment_avg_color[region_id] = {0, 0, 0};
      }

      if (segment_bounding_box.find(region_id) == segment_bounding_box.end()) {
        segment_bounding_box[region_id] = {frame_width, 0, frame_height, 0};
      }

      // Pointer to the average colors.
      std::vector<int>* avg_colors = &segment_avg_color[region_id];
      std::vector<int>* bounding_box = &segment_bounding_box[region_id];

      int size = 0;

      uchar* output_ptr;
      for (const auto& scan : region.raster().scan_inter()) {
        // Obtain / Update bounding box values for that region.
        (*bounding_box)[0] = std::min((*bounding_box)[0], scan.left_x());
        (*bounding_box)[1] = std::max((*bounding_box)[1], scan.right_x());
        (*bounding_box)[2] = std::min((*bounding_box)[2], scan.y());
        (*bounding_box)[3] = std::max((*bounding_box)[3], scan.y());

        output_ptr = current_frame.ptr<uchar>(scan.y()) + 3 * scan.left_x();
        for (int j = 0, len = 3 * (scan.right_x() - scan.left_x() + 1); j < len; j += 3) {
          // BGR to RGB conversion.
          (*avg_colors)[2] += int(output_ptr[j]);
          (*avg_colors)[1] += int(output_ptr[j + 1]);
          (*avg_colors)[0] += int(output_ptr[j + 2]);
          size++;
        }
      }

      if (segment_sizes.find(region_id) == segment_sizes.end()) {
        segment_sizes[region_id] = size;
      } else {
        segment_sizes[region_id] += size;
      }
    }

    std::map<int, float> temp_scores;
    // Iterate through segments, and choose the rule of thirds
    for (int pt_idx = 0; pt_idx < rot_points.size(); pt_idx++) {
      temp_scores[pt_idx] = 0;
      for (const auto& segment: segment_sizes) {
        // Calculate the distance for each segment.
        std::vector<int>* bounding_box = &segment_bounding_box[segment.first];
        int center_x = (*bounding_box)[1] - (*bounding_box)[0];
        int center_y = (*bounding_box)[3] - (*bounding_box)[2];
        float calc_distance = dist2D(center_x,
                                     center_y,
                                     rot_points[pt_idx][0],
                                     rot_points[pt_idx][1]);
        temp_scores[pt_idx] += 1.0f / calc_distance;
      }
    }

    float scoreMax = 0.f;
    int thirds_index = -1;
    for (const auto& temp_score : temp_scores) {
      if (temp_score.second > scoreMax) {
        scoreMax = temp_score.second;
        thirds_index = temp_score.first;
      }
    }

    // After iterating through the segments, we divide the summed color by the size of the
    // segments to get the average color.
    float score = 0;
    for (const auto& segment : segment_sizes) {
      std::vector<int>* avg_colors = &segment_avg_color[segment.first];
      std::vector<int>* bounding_box = &segment_bounding_box[segment.first];
      int center_x = (*bounding_box)[1] - (*bounding_box)[0];
      int center_y = (*bounding_box)[3] - (*bounding_box)[2];
      
      if (segment.second != 0) {
        (*avg_colors)[0] /= segment.second;
        (*avg_colors)[1] /= segment.second;
        (*avg_colors)[2] /= segment.second;
      }

      // Calculate shortest distance based on max score.
      float distance = dist2D(center_x,
                              center_y,
                              rot_points[thirds_index][0],
                              rot_points[thirds_index][1]);

      // Iterate through colors to find closest color.
      float color_distance = -1;
      int color_idx = -1;
      for (int pt_idx = 0; pt_idx < colors.size(); pt_idx++) {
        float calc_distance = rgb_distance((*avg_colors)[0], (*avg_colors)[1], (*avg_colors)[2],
                                           colors[pt_idx][0], colors[pt_idx][1], colors[pt_idx][2]);
        if (color_distance == -1) {
          color_distance = calc_distance;
          color_idx = pt_idx;
        }
        else if (calc_distance < color_distance) {
          color_distance = calc_distance;
          color_idx = pt_idx;
        }
      }

      // segment.second stores the size.
      score += segment.second * color_weights[colors[color_idx]] / distance;
    }
    score /= segment_sizes.size();

    if (min_score == -1 || score < min_score) {
      min_score = score;
    }
    if (score > max_score) {
      max_score = score;
    }

    scores[frame_idx] = score;
  } // Closes for loop that iterates each frame.

  // Output file.
  std::string curr_file = FLAGS_output_file;
  std::ofstream output_stream(curr_file, std::ios_base::out);

  // Normalize scores and write to file.
  for (int idx = 0; idx < scores.size(); idx++) {
    // Don't normalize.
    // scores[idx] = (scores[idx] - min_score) / (max_score - min_score);
    output_stream << idx << "," << scores[idx] << std::endl;
  }

  segment_reader.CloseFile();
  return 0;
}

