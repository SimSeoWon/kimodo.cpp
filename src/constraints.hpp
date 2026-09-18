#pragma once

#include <kimodo/kimodo.hpp>

#include <expected>
#include <span>
#include <string>
#include <vector>

namespace kimodo::detail {
struct skeleton_spec;

struct pose_condition {
    std::vector<float> observed;
    std::vector<float> observed_mask;
    float first_heading = 0.F;
    bool heading_constrained = false;
};

std::expected<pose_condition, std::string> build_pose_condition(
    const skeleton_spec &skeleton, std::span<const pose_constraint> constraints,
    std::size_t frames, std::span<const float> global_mean,
    std::span<const float> global_std, std::span<const float> body_mean,
    std::span<const float> body_std);
}
