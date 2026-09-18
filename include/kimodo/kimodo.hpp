#pragma once

#include <kimodo/kimodo_capi.h>

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kimodo {

inline constexpr unsigned embedding_width = 4096;

struct motion_data {
    unsigned frames = 0;
    unsigned joints = 0;
    std::vector<float> local_rotations_xyzw;
    std::vector<float> root_positions;
};

struct prompt_segment {
    std::string prompt;
    unsigned frames = 0;
};

// Sparse, time-localized pose control in world space. Joint indices follow the
// loaded model skeleton (SOMA30 in the Web UI). Position and rotation can be
// enabled independently; unconstrained features remain free for diffusion.
struct pose_constraint {
    unsigned frame = 0;
    unsigned joint = 0;
    bool constrain_position = false;
    std::array<float, 3> world_position{};
    bool constrain_rotation = false;
    std::array<float, 4> world_rotation_xyzw{0.F, 0.F, 0.F, 1.F};
};

class KIMODO_API model {
public:
    static std::expected<std::unique_ptr<model>, std::string> load(
        std::string_view motion_gguf, std::string_view text_bundle = {});
    // `negative_embedding`/`negative_prompt` are optional (empty by default, reproducing prior
    // behavior exactly). When given, they replace the zero vector the CFG "unconditional"
    // branch otherwise uses — the standard diffusion negative-prompt technique. This is a
    // kimodo.cpp addition, not part of upstream NVIDIA Kimodo's published API.
    std::expected<motion_data, std::string> generate_embedding(
        const std::array<float, embedding_width> &embedding,
        unsigned frames, unsigned steps, std::uint64_t seed,
        float text_cfg, float constraint_cfg,
        std::span<const float> negative_embedding = {}) const;
    std::expected<motion_data, std::string> generate_text(
        std::string_view utf8_prompt, unsigned frames, unsigned steps, std::uint64_t seed,
        float text_cfg, float constraint_cfg,
        std::string_view negative_prompt = {}) const;
    std::expected<motion_data, std::string> generate_text_constrained(
        std::string_view utf8_prompt, unsigned frames, unsigned steps, std::uint64_t seed,
        float text_cfg, float constraint_cfg, std::span<const pose_constraint> constraints,
        std::string_view negative_prompt = {}) const;
    std::expected<motion_data, std::string> generate_text_sequence(
        std::span<const prompt_segment> segments, unsigned transition_frames,
        unsigned steps, std::uint64_t seed, float text_cfg, float constraint_cfg) const;
    std::expected<motion_data, std::string> generate_text_sequence_constrained(
        std::span<const prompt_segment> segments, unsigned transition_frames,
        unsigned steps, std::uint64_t seed, float text_cfg, float constraint_cfg,
        std::span<const pose_constraint> constraints) const;
    ~model();
    model(const model &) = delete;
    model &operator=(const model &) = delete;
private:
    struct impl;
    explicit model(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> impl_;
};

} // namespace kimodo
