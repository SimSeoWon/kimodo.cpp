#include <kimodo/kimodo.hpp>
#include "gguf.hpp"
#include "skeleton.hpp"
#include "constraints.hpp"
#ifdef KIMODO_HAVE_GGML
#include "ggml_weights.hpp"
#include "denoiser.hpp"
#include "motion_decode.hpp"
#include "llm_text_encoder.hpp"
#include "profile.hpp"
#endif

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <random>

namespace kimodo {
struct model::impl {
    detail::gguf_file motion;
    std::string motion_path;
    const detail::skeleton_spec *skeleton = nullptr;
#ifdef KIMODO_HAVE_GGML
    mutable std::unique_ptr<detail::ggml_motion_weights> weights;
    std::unique_ptr<detail::llm_text_encoder> text;
    mutable std::mutex inference_mutex;
#endif
};
model::model(std::unique_ptr<impl> state) : impl_(std::move(state)) {}
model::~model() = default;

std::expected<std::unique_ptr<model>, std::string> model::load(std::string_view motion_path, std::string_view text_path) {
    auto file = detail::read_gguf_header(motion_path);
    if (!file) return std::unexpected(file.error());
    if (auto valid = detail::validate_motion_gguf(*file); !valid) return std::unexpected(valid.error());
    auto state = std::make_unique<impl>();
    state->motion = std::move(*file);
    state->motion_path = std::string(motion_path);
    state->skeleton = detail::find_skeleton(state->motion.strings.at("kimodo.skeleton"));
#ifdef KIMODO_HAVE_GGML
    if (!text_path.empty()) {
        auto text = detail::llm_text_encoder::load(text_path);
        if (!text) return std::unexpected(text.error());
        state->text = std::move(*text);
    }
#else
    if (!text_path.empty()) return std::unexpected("Kimodo was built without GGML support");
#endif
    return std::unique_ptr<model>(new model(std::move(state)));
}

std::expected<motion_data, std::string> model::generate_text(
    std::string_view utf8_prompt, unsigned frames, unsigned steps, std::uint64_t seed,
    float text_cfg, float constraint_cfg, std::string_view negative_prompt) const {
#ifdef KIMODO_HAVE_GGML
    if (!impl_->text) return std::unexpected("model was loaded without a native text bundle");
    auto embedding = impl_->text->encode(utf8_prompt);
    if (!embedding) return std::unexpected(embedding.error());
    if (negative_prompt.empty())
        return generate_embedding(*embedding, frames, steps, seed, text_cfg, constraint_cfg);
    auto negative_embedding = impl_->text->encode(negative_prompt);
    if (!negative_embedding) return std::unexpected(negative_embedding.error());
    return generate_embedding(*embedding, frames, steps, seed, text_cfg, constraint_cfg, *negative_embedding);
#else
    (void) utf8_prompt; (void) frames; (void) steps; (void) seed; (void) text_cfg; (void) constraint_cfg;
    (void) negative_prompt;
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_text_constrained(
    std::string_view utf8_prompt, unsigned frames, unsigned steps, std::uint64_t seed,
    float text_cfg, float constraint_cfg, std::span<const pose_constraint> constraints,
    std::string_view negative_prompt) const {
#ifdef KIMODO_HAVE_GGML
    if (!impl_->text) return std::unexpected("model was loaded without a native text bundle");
    if (constraints.empty()) return generate_text(utf8_prompt,frames,steps,seed,text_cfg,constraint_cfg,negative_prompt);
    if (frames == 0 || frames > 10000 || steps == 0 || steps > 1000)
        return std::unexpected("invalid constrained generation dimensions");
    auto embedding=impl_->text->encode(utf8_prompt);if(!embedding)return std::unexpected(embedding.error());
    std::array<float,embedding_width> negative{};std::span<const float> negative_span;
    if(!negative_prompt.empty()){auto encoded=impl_->text->encode(negative_prompt);if(!encoded)return std::unexpected(encoded.error());negative=*encoded;negative_span=negative;}
    if(!impl_->weights){auto loaded=detail::ggml_motion_weights::load(impl_->motion_path);if(!loaded)return std::unexpected(loaded.error());impl_->weights=std::move(*loaded);}
    auto gm=impl_->weights->f32_values("stats.global_root.mean"),gs=impl_->weights->f32_values("stats.global_root.std");
    auto bm=impl_->weights->f32_values("stats.body.mean"),bs=impl_->weights->f32_values("stats.body.std");
    if(!gm||!gs||!bm||!bs)return std::unexpected("motion GGUF lacks normalization statistics");
    auto condition=detail::build_pose_condition(*impl_->skeleton,constraints,frames,*gm,*gs,*bm,*bs);
    if(!condition)return std::unexpected(condition.error());
    std::mt19937_64 rng(seed);std::normal_distribution<float> normal(0.F,1.F);
    std::vector<float> noise(static_cast<std::size_t>(frames)*impl_->skeleton->motion_dim());for(float&value:noise)value=normal(rng);
    auto sampled=detail::sample_motion_from_noise_conditioned(*impl_->weights,noise,*embedding,condition->observed,
        condition->observed_mask,condition->first_heading,frames,steps,text_cfg,constraint_cfg,negative_span);
    if(!sampled)return std::unexpected(sampled.error());
    auto decoded=detail::decode_motion(*sampled,frames,*impl_->skeleton,*gm,*gs,*bm,*bs);if(!decoded)return std::unexpected(decoded.error());
    motion_data result;result.frames=frames;result.joints=static_cast<unsigned>(impl_->skeleton->joints());
    result.local_rotations_xyzw=std::move(decoded->local_xyzw);result.root_positions=std::move(decoded->root_positions);return result;
#else
    (void)utf8_prompt;(void)frames;(void)steps;(void)seed;(void)text_cfg;(void)constraint_cfg;(void)constraints;(void)negative_prompt;
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_embedding(
    const std::array<float, embedding_width> &embedding, unsigned frames, unsigned steps,
    std::uint64_t seed, float text_cfg, float constraint_cfg,
    std::span<const float> negative_embedding) const {
    if (frames == 0 || frames > 10000) return std::unexpected("frames must be in 1..10000");
    if (steps == 0 || steps > 1000) return std::unexpected("diffusion_steps must be in 1..1000");
    if (!std::isfinite(text_cfg) || !std::isfinite(constraint_cfg)) return std::unexpected("CFG weights must be finite");
    for (float value : embedding) if (!std::isfinite(value)) return std::unexpected("embedding contains a non-finite value");
    if (!negative_embedding.empty()) {
        if (negative_embedding.size() != embedding_width)
            return std::unexpected("negative_embedding must be empty or exactly embedding_width values");
        for (float value : negative_embedding)
            if (!std::isfinite(value)) return std::unexpected("negative_embedding contains a non-finite value");
    }
#ifdef KIMODO_HAVE_GGML
    const std::lock_guard inference_lock(impl_->inference_mutex);
    const auto generate_started = std::chrono::steady_clock::now();
    // Weight residency is deferred until inference so model-load stays a
    // bounded metadata operation.  The graph integration consumes this exact
    // session; no separate unchecked tensor loader exists in the runtime.
    if (!impl_->weights) {
        const auto weights_started = std::chrono::steady_clock::now();
        auto loaded = detail::ggml_motion_weights::load(impl_->motion_path);
        if (!loaded) return std::unexpected(loaded.error());
        impl_->weights = std::move(*loaded);
        if (detail::profile_enabled())
            std::fprintf(stderr, "profile motion.weights_ready_ms=%.3f\n", detail::profile_elapsed_ms(weights_started));
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    const size_t motion_dim=impl_->skeleton->motion_dim();
    std::vector<float> noise(static_cast<size_t>(frames)*motion_dim);
    for (float &value : noise) value = normal(rng);
    const auto sampling_started = std::chrono::steady_clock::now();
    auto sampled = detail::sample_motion_from_noise(*impl_->weights, noise, embedding, frames, steps, text_cfg, constraint_cfg, negative_embedding);
    if (!sampled) return std::unexpected(sampled.error());
    const double sampling_ms = detail::profile_elapsed_ms(sampling_started);
    auto global_mean=impl_->weights->f32_values("stats.global_root.mean"), global_std=impl_->weights->f32_values("stats.global_root.std");
    auto body_mean=impl_->weights->f32_values("stats.body.mean"), body_std=impl_->weights->f32_values("stats.body.std");
    if (!global_mean) return std::unexpected(global_mean.error());
    if (!global_std) return std::unexpected(global_std.error());
    if (!body_mean) return std::unexpected(body_mean.error());
    if (!body_std) return std::unexpected(body_std.error());
    const auto decode_started = std::chrono::steady_clock::now();
    auto decoded=detail::decode_motion(*sampled,frames,*impl_->skeleton,*global_mean,*global_std,*body_mean,*body_std);
    if (!decoded) return std::unexpected(decoded.error());
    motion_data result;
    result.frames=frames; result.joints=static_cast<unsigned>(impl_->skeleton->joints());
    result.local_rotations_xyzw=std::move(decoded->local_xyzw);
    result.root_positions=std::move(decoded->root_positions);
    if (detail::profile_enabled())
        std::fprintf(stderr, "profile motion.generate sampling_ms=%.3f decode_ms=%.3f total_ms=%.3f\n",
                     sampling_ms, detail::profile_elapsed_ms(decode_started),
                     detail::profile_elapsed_ms(generate_started));
    return result;
#else
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_text_sequence(
    std::span<const prompt_segment> segments, unsigned transition_frames,
    unsigned steps, std::uint64_t seed, float text_cfg, float constraint_cfg) const {
#ifdef KIMODO_HAVE_GGML
    if (!impl_->text) return std::unexpected("model was loaded without a native text bundle");
    if (segments.empty() || segments.size() > 16) return std::unexpected("sequence requires 1..16 prompt segments");
    if (steps == 0 || steps > 1000 || transition_frames == 0 || transition_frames > 60)
        return std::unexpected("invalid sequence sampling parameters");
    std::vector<std::array<float, embedding_width>> embeddings;
    embeddings.reserve(segments.size());
    for (size_t index=0; index<segments.size(); ++index) {
        const auto &segment=segments[index];
        if (segment.prompt.empty() || segment.frames < 2 || segment.frames > 300)
            return std::unexpected("each sequence segment must contain a prompt and have 2..300 frames");
        if (index && transition_frames >= segment.frames) return std::unexpected("transition must be shorter than every following segment");
        auto embedding=impl_->text->encode(segment.prompt);
        if (!embedding) return std::unexpected(embedding.error());
        embeddings.push_back(*embedding);
    }
    const std::lock_guard inference_lock(impl_->inference_mutex);
    // Initialize and warm the quantized text backend before the F32 motion
    // backend applies its process-wide Vulkan parity flags. This preserves
    // cooperative-matrix text kernels in persistent sequence workers.
    if (!impl_->weights) {
        auto loaded = detail::ggml_motion_weights::load(impl_->motion_path);
        if (!loaded) return std::unexpected(loaded.error());
        impl_->weights = std::move(*loaded);
    }
    auto bm=impl_->weights->f32_values("stats.body.mean"), bs=impl_->weights->f32_values("stats.body.std");
    auto gm=impl_->weights->f32_values("stats.global_root.mean"), gs=impl_->weights->f32_values("stats.global_root.std");
    if (!gm || !gs || !bm || !bs) return std::unexpected("motion GGUF lacks normalization statistics");
    std::mt19937_64 rng(seed); std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<std::vector<float>> noise;
    std::vector<detail::sampled_sequence_segment> sampled;
    noise.reserve(segments.size()); sampled.reserve(segments.size());
    for (size_t index=0; index<segments.size(); ++index) {
        const auto &segment=segments[index];
        const auto sampled_frames = static_cast<size_t>(segment.frames) +
            (index == 0 ? 0 : transition_frames);
        noise.emplace_back(sampled_frames*impl_->skeleton->motion_dim());
        for (float &value : noise.back()) value=normal(rng);
        sampled.push_back({embeddings[index], noise.back(), segment.frames, {}, {}, 0.F, false});
    }
    auto joined=detail::sample_motion_sequence_from_noise(*impl_->weights,sampled,transition_frames,steps,text_cfg,constraint_cfg);
    if (!joined) return std::unexpected(joined.error());
    const size_t motion_dim=impl_->skeleton->motion_dim(), body_dim=impl_->skeleton->body_dim();
    const auto frames=static_cast<unsigned>(joined->size()/motion_dim);
    auto normalized=*joined;
    for (size_t row=0; row<frames; ++row) {
        auto *value=normalized.data()+row*motion_dim;
        for (size_t d=0; d<5; ++d) value[d]=(value[d]-(*gm)[d])/std::sqrt((*gs)[d]*(*gs)[d]+1.e-5F);
        for (size_t d=0; d<body_dim; ++d) value[5+d]=(value[5+d]-(*bm)[d])/std::sqrt((*bs)[d]*(*bs)[d]+1.e-5F);
    }
    auto decoded=detail::decode_motion(normalized,frames,*impl_->skeleton,*gm,*gs,*bm,*bs);
    if (!decoded) return std::unexpected(decoded.error());
    motion_data result; result.frames=frames; result.joints=static_cast<unsigned>(impl_->skeleton->joints());
    result.local_rotations_xyzw=std::move(decoded->local_xyzw); result.root_positions=std::move(decoded->root_positions);
    return result;
#else
    (void) segments; (void) transition_frames; (void) steps; (void) seed; (void) text_cfg; (void) constraint_cfg;
    return std::unexpected("Kimodo was built without GGML support");
#endif
}

std::expected<motion_data, std::string> model::generate_text_sequence_constrained(
    std::span<const prompt_segment> segments, unsigned transition_frames,
    unsigned steps, std::uint64_t seed, float text_cfg, float constraint_cfg,
    std::span<const pose_constraint> constraints) const {
#ifdef KIMODO_HAVE_GGML
    if(constraints.empty())return generate_text_sequence(segments,transition_frames,steps,seed,text_cfg,constraint_cfg);
    if(!impl_->text)return std::unexpected("model was loaded without a native text bundle");
    if(segments.empty()||segments.size()>16||!transition_frames||transition_frames>60)return std::unexpected("invalid constrained sequence");
    if(!impl_->weights){auto loaded=detail::ggml_motion_weights::load(impl_->motion_path);if(!loaded)return std::unexpected(loaded.error());impl_->weights=std::move(*loaded);}
    auto gm=impl_->weights->f32_values("stats.global_root.mean"),gs=impl_->weights->f32_values("stats.global_root.std");auto bm=impl_->weights->f32_values("stats.body.mean"),bs=impl_->weights->f32_values("stats.body.std");if(!gm||!gs||!bm||!bs)return std::unexpected("motion GGUF lacks normalization statistics");
    std::size_t total_frames=0;for(const auto&segment:segments)total_frames+=segment.frames;for(const auto&constraint:constraints)if(constraint.frame>=total_frames)return std::unexpected("sequence constraint frame is out of range");
    std::mt19937_64 rng(seed);std::normal_distribution<float> normal(0.F,1.F);std::vector<std::array<float,embedding_width>> embeddings;std::vector<std::vector<float>> noise,observed,masks;std::vector<detail::sampled_sequence_segment> sampled;embeddings.reserve(segments.size());noise.reserve(segments.size());observed.reserve(segments.size());masks.reserve(segments.size());sampled.reserve(segments.size());
    std::size_t segment_start=0;for(std::size_t index=0;index<segments.size();++index){const auto&segment=segments[index];if(segment.prompt.empty()||segment.frames<2||segment.frames>300||(index&&transition_frames>=segment.frames))return std::unexpected("invalid constrained sequence segment");auto embedding=impl_->text->encode(segment.prompt);if(!embedding)return std::unexpected(embedding.error());embeddings.push_back(*embedding);const std::size_t prefix=index?transition_frames:0,sampled_frames=segment.frames+prefix;noise.emplace_back(sampled_frames*impl_->skeleton->motion_dim());for(float&value:noise.back())value=normal(rng);std::vector<pose_constraint> local;for(const auto&constraint:constraints)if(constraint.frame>=segment_start&&constraint.frame<segment_start+segment.frames){auto copy=constraint;copy.frame=static_cast<unsigned>(constraint.frame-segment_start+prefix);local.push_back(copy);}float heading=0.F;bool heading_constrained=false;if(!local.empty()){auto condition=detail::build_pose_condition(*impl_->skeleton,local,sampled_frames,*gm,*gs,*bm,*bs);if(!condition)return std::unexpected(condition.error());observed.push_back(std::move(condition->observed));masks.push_back(std::move(condition->observed_mask));heading=condition->first_heading;heading_constrained=condition->heading_constrained;}else{observed.emplace_back();masks.emplace_back();}sampled.push_back({embeddings.back(),noise.back(),segment.frames,observed.back(),masks.back(),heading,heading_constrained});segment_start+=segment.frames;}
    auto joined=detail::sample_motion_sequence_from_noise(*impl_->weights,sampled,transition_frames,steps,text_cfg,constraint_cfg);if(!joined)return std::unexpected(joined.error());const std::size_t D=impl_->skeleton->motion_dim(),body=impl_->skeleton->body_dim();const auto frames=static_cast<unsigned>(joined->size()/D);auto normalized=*joined;for(std::size_t row=0;row<frames;++row){auto*value=normalized.data()+row*D;for(std::size_t d=0;d<5;++d)value[d]=(value[d]-(*gm)[d])/std::sqrt((*gs)[d]*(*gs)[d]+1.e-5F);for(std::size_t d=0;d<body;++d)value[5+d]=(value[5+d]-(*bm)[d])/std::sqrt((*bs)[d]*(*bs)[d]+1.e-5F);}auto decoded=detail::decode_motion(normalized,frames,*impl_->skeleton,*gm,*gs,*bm,*bs);if(!decoded)return std::unexpected(decoded.error());motion_data result;result.frames=frames;result.joints=static_cast<unsigned>(impl_->skeleton->joints());result.local_rotations_xyzw=std::move(decoded->local_xyzw);result.root_positions=std::move(decoded->root_positions);return result;
#else
    (void)segments;(void)transition_frames;(void)steps;(void)seed;(void)text_cfg;(void)constraint_cfg;(void)constraints;return std::unexpected("Kimodo was built without GGML support");
#endif
}
} // namespace kimodo
