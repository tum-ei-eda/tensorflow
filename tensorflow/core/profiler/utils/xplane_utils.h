/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

// Returns the plane with the given name or nullptr if not found.
const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name);
XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name);

// Returns the plane with the given name in the container. If necessary, adds a
// new plane to the container.
XPlane* FindOrAddMutablePlaneWithName(XSpace* space, absl::string_view name);

// Returns all the planes with a given prefix.
std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix);
std::vector<XPlane*> FindMutablePlanesWithPrefix(XSpace* space,
                                                 absl::string_view prefix);

// Returns true if event is nested by parent.
bool IsNested(const tensorflow::profiler::XEvent& event,
              const tensorflow::profiler::XEvent& parent);

void AddOrUpdateIntStat(int64 metadata_id, int64 value,
                        tensorflow::profiler::XEvent* event);

void AddOrUpdateStrStat(int64 metadata_id, absl::string_view value,
                        tensorflow::profiler::XEvent* event);

void RemovePlaneWithName(XSpace* space, absl::string_view name);
void RemoveEmptyPlanes(XSpace* space);
void RemoveEmptyLines(XPlane* plane);

// Sort lines in plane with a provided comparator.
template <class Compare>
void SortXLinesBy(XPlane* plane, Compare comp) {
  std::sort(plane->mutable_lines()->pointer_begin(),
            plane->mutable_lines()->pointer_end(), comp);
}

class XLinesComparatorByName {
 public:
  bool operator()(const XLine* a, const XLine* b) const {
    auto& line_a = a->display_name().empty() ? a->name() : a->display_name();
    auto& line_b = b->display_name().empty() ? b->name() : b->display_name();
    return line_a < line_b;
  }
};

// Sorts each XLine's XEvents by offset_ps (ascending) and duration_ps
// (descending) so nested events are sorted from outer to innermost.
void SortXPlane(XPlane* plane);
// Sorts each plane of the XSpace.
void SortXSpace(XSpace* space);

// Normalize timestamps by time-shifting to start_time_ns_ as origin.
void NormalizeTimestamps(XPlane* plane, uint64 start_time_ns);
void NormalizeTimestamps(XSpace* space, uint64 start_time_ns);

// Merge Xplane src_plane into Xplane dst_plane, both plane level stats, lines,
// events and event level stats are merged; If src_plane and dst_plane both have
// the same line, which have different start timestamps, we will normalize the
// events offset timestamp correspondingly.
void MergePlanes(const XPlane& src_plane, XPlane* dst_plane);

// Plane's start timestamp is defined as the minimum of all lines' start
// timestamps. If zero line exists, return 0;
uint64 GetStartTimestampNs(const XPlane& plane);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
