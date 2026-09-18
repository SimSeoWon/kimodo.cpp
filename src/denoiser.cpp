#include "denoiser.hpp"
#include "ggml_weights.hpp"
#include "motion_rep.hpp"
#include "motion_graph_cache.hpp"
#include "diffusion.hpp"
#include "profile.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace kimodo::detail {
namespace {
constexpr int width=1024, heads=8, head_width=128, text_tokens=50, prefix_tokens=52;
constexpr size_t cached_context_bytes = 1ULL*1024*1024;
thread_local std::vector<std::pair<ggml_tensor *, std::span<const float>>> inputs;
struct execution_profile {
    unsigned calls = 0;
    double graph_ms = 0, allocation_ms = 0, upload_ms = 0, compute_ms = 0, download_ms = 0;
};
thread_local execution_profile *active_profile = nullptr;
struct active_profile_guard {
    execution_profile *previous;
    explicit active_profile_guard(execution_profile *current) : previous(active_profile) {
        active_profile = current;
    }
    ~active_profile_guard() { active_profile = previous; }
};
bool packed_motion_attention() noexcept {
    const char *value = std::getenv("KIMODO_MOTION_PACKED_ATTENTION");
    return !value || std::string_view(value) != "0";
}
bool motion_graph_cache_enabled() noexcept {
    const char *value = std::getenv("KIMODO_MOTION_GRAPH_CACHE");
    return packed_motion_attention() && (!value || std::string_view(value) != "0");
}
int motion_layer_chunk_size() noexcept {
    const int fallback = packed_motion_attention() ? 8 : 4;
    const int maximum = packed_motion_attention() ? 16 : 4;
    const char *value = std::getenv("KIMODO_MOTION_LAYER_CHUNK");
    if (!value) return fallback;
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 1 || parsed > maximum)
        return fallback;
    return static_cast<int>(parsed);
}
ggml_tensor *input(ggml_context *ctx, std::span<const float> values, int a, int b, int c) {
    auto *r=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,a,b,c); inputs.emplace_back(r, values); return r;
}
ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *bias) {
    auto *y=ggml_mul_mat(ctx,w,x);
    // F32 parity takes precedence over Tensor Core throughput.  In
    // particular, do not let a Vulkan backend lower the accumulation
    // precision for the reference model.
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return ggml_add(ctx,y,ggml_repeat(ctx,bias,y));
}
ggml_tensor *norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *scale, ggml_tensor *bias) {
    auto *n=ggml_norm(ctx,x,1.e-5f); return ggml_add(ctx,ggml_mul(ctx,n,ggml_repeat(ctx,scale,n)),ggml_repeat(ctx,bias,n));
}
std::expected<std::vector<float>,std::string> execute(ggml_context *ctx, ggml_tensor *out, size_t values, const ggml_motion_weights &weights) {
    auto started = std::chrono::steady_clock::now();
    auto *graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
    if (active_profile) { active_profile->calls++; active_profile->graph_ms += profile_elapsed_ms(started); }
    started = std::chrono::steady_clock::now();
    auto *alloc=weights.allocator();
    if(!alloc || !ggml_gallocr_reserve(alloc,graph) || !ggml_gallocr_alloc_graph(alloc,graph))
        return std::unexpected("GGML graph allocation failed");
    if (active_profile) active_profile->allocation_ms += profile_elapsed_ms(started);
    started = std::chrono::steady_clock::now();
    for(const auto &[t,data]:inputs) ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(float));
    inputs.clear();
    if (active_profile) active_profile->upload_ms += profile_elapsed_ms(started);
    started = std::chrono::steady_clock::now();
    if(ggml_backend_graph_compute(weights.backend(),graph)!=GGML_STATUS_SUCCESS) return std::unexpected("GGML graph execution failed");
    if (active_profile) active_profile->compute_ms += profile_elapsed_ms(started);
    started = std::chrono::steady_clock::now();
    std::vector<float> r(values); ggml_backend_tensor_get(out,r.data(),0,r.size()*sizeof(float));
    if (active_profile) active_profile->download_ms += profile_elapsed_ms(started);
    return r;
}
std::expected<std::unique_ptr<cached_motion_graph>, std::string> cache_graph(
    ggml_context *ctx, ggml_tensor *out, size_t values,
    std::initializer_list<bool> retain_inputs, const ggml_motion_weights &weights) {
    if (retain_inputs.size() != inputs.size()) {
        inputs.clear();
        ggml_free(ctx);
        return std::unexpected("GGML cached graph retention count changed");
    }
    auto result = std::make_unique<cached_motion_graph>();
    result->context = ctx;
    result->output = out;
    result->output_values = values;
    result->inputs.reserve(inputs.size());
    result->retained_inputs.resize(inputs.size());
    auto retain_it = retain_inputs.begin();
    size_t input_index = 0;
    for (const auto &[tensor, data] : inputs) {
        result->inputs.push_back(tensor);
        if (*retain_it++)
            result->retained_inputs[input_index].assign(data.begin(), data.end());
        ++input_index;
    }

    auto started = std::chrono::steady_clock::now();
    result->graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(result->graph, out);
    if (active_profile) active_profile->graph_ms += profile_elapsed_ms(started);

    started = std::chrono::steady_clock::now();
    result->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.backend()));
    if (!result->allocator || !ggml_gallocr_reserve(result->allocator, result->graph) ||
        !ggml_gallocr_alloc_graph(result->allocator, result->graph)) {
        inputs.clear();
        return std::unexpected("GGML cached graph allocation failed");
    }
    result->scratch_bytes = ggml_gallocr_get_buffer_size(result->allocator, 0);
    if (active_profile) active_profile->allocation_ms += profile_elapsed_ms(started);
    if (profile_enabled())
        std::fprintf(stderr,
                     "profile motion.graph_cache nodes=%d metadata_kib=%.1f scratch_mib=%.2f\n",
                     ggml_graph_n_nodes(result->graph),
                     static_cast<double>(ggml_used_mem(ctx))/1024.0,
                     static_cast<double>(result->scratch_bytes)/(1024.0*1024.0));
    return result;
}
void bind_inputs(cached_motion_graph &graph,
                 std::initializer_list<std::span<const float>> values,
                 std::initializer_list<bool> changed) {
    if (values.size() != graph.inputs.size() || changed.size() != graph.inputs.size())
        throw std::runtime_error("cached GGML graph input count changed");
    inputs.clear();
    size_t index = 0;
    auto changed_it = changed.begin();
    for (auto value : values) {
        auto *tensor = graph.inputs[index];
        if (*changed_it++) {
            if (static_cast<size_t>(ggml_nelements(tensor)) != value.size())
                throw std::runtime_error("cached GGML graph input shape changed");
            if (!graph.retained_inputs[index].empty())
                graph.retained_inputs[index].assign(value.begin(), value.end());
            inputs.emplace_back(tensor, value);
        } else {
            if (graph.retained_inputs[index].empty())
                throw std::runtime_error("cached GGML graph input was not retained");
            inputs.emplace_back(tensor, graph.retained_inputs[index]);
        }
        ++index;
    }
}
std::expected<std::vector<float>,std::string> execute(cached_motion_graph &cached,
                                                       const ggml_motion_weights &weights) {
    if (inputs.size() != cached.inputs.size()) {
        inputs.clear();
        return std::unexpected("cached GGML graph inputs are incomplete");
    }
    if (active_profile) active_profile->calls++;
    auto started = std::chrono::steady_clock::now();
    for (size_t index = 0; index < inputs.size(); ++index) {
        const auto &[tensor, data] = inputs[index];
        if (tensor != cached.inputs[index]) {
            inputs.clear();
            return std::unexpected("cached GGML graph inputs changed");
        }
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size()*sizeof(float));
    }
    inputs.clear();
    if (active_profile) active_profile->upload_ms += profile_elapsed_ms(started);
    started = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(weights.backend(), cached.graph) != GGML_STATUS_SUCCESS)
        return std::unexpected("GGML cached graph execution failed");
    if (active_profile) active_profile->compute_ms += profile_elapsed_ms(started);
    started = std::chrono::steady_clock::now();
    std::vector<float> result(cached.output_values);
    ggml_backend_tensor_get(cached.output, result.data(), 0, result.size()*sizeof(float));
    if (active_profile) active_profile->download_ms += profile_elapsed_ms(started);
    return result;
}
cached_motion_transformer *transformer_cache(const ggml_motion_weights &weights,
                                             std::string_view prefix, size_t motion_dim,
                                             size_t batch, size_t frames, int layer_chunk) {
    auto *cache = weights.graph_cache();
    if (!cache || cache->batch != batch || cache->frames != frames ||
        cache->layer_chunk != layer_chunk) {
        auto replacement = std::make_unique<motion_graph_cache>();
        replacement->batch = batch;
        replacement->frames = frames;
        replacement->layer_chunk = layer_chunk;
        weights.graph_cache(std::move(replacement));
        cache = weights.graph_cache();
    }
    for (auto &candidate : cache->transformers)
        if (candidate->prefix == prefix && candidate->motion_dim == motion_dim)
            return candidate.get();
    auto result = std::make_unique<cached_motion_transformer>();
    result->prefix = prefix;
    result->motion_dim = motion_dim;
    result->stages.resize(static_cast<size_t>(2 + (16 + layer_chunk - 1) / layer_chunk));
    auto *value = result.get();
    cache->transformers.push_back(std::move(result));
    return value;
}
ggml_tensor *weight(const ggml_motion_weights&w,std::string_view n) { auto*t=w.tensor(n); if(!t) throw std::runtime_error("missing GGML tensor: "+std::string(n)); return t; }
ggml_tensor *layer(ggml_context *ctx,ggml_tensor*x,const ggml_motion_weights&w,std::string_view p,int seq,int batch) {
    const std::string s(p); auto*qkv=linear(ctx,x,weight(w,s+"self_attn.in_proj_weight"),weight(w,s+"self_attn.in_proj_bias"));
    ggml_tensor *a = nullptr;
    if (packed_motion_attention()) {
        auto pack = [&](int index) {
            auto *value = ggml_view_4d(ctx, qkv, head_width, heads, seq, batch,
                                      static_cast<size_t>(head_width) * sizeof(float),
                                      qkv->nb[1], qkv->nb[2],
                                      static_cast<size_t>(index * width) * sizeof(float));
            return ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3));
        };
        auto *q = pack(0), *k = pack(1), *v = pack(2);
        auto *scores = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        auto *probability = ggml_soft_max(ctx, ggml_scale(ctx, scores, 1.f/std::sqrt(float(head_width))));
        auto *values = ggml_mul_mat(ctx, probability, ggml_cont(ctx, ggml_transpose(ctx, v)));
        ggml_mul_mat_set_prec(values, GGML_PREC_F32);
        values = ggml_cont(ctx, ggml_transpose(ctx, values));
        values = ggml_cont(ctx, ggml_permute(ctx, values, 0, 2, 1, 3));
        a = ggml_reshape_3d(ctx, values, width, seq, batch);
    } else {
        // The explicit [head, batch] graph is the byte-stable reference path.
        auto head = [&](int block, int h, int b) {
            return ggml_view_2d(ctx, qkv, head_width, seq, qkv->nb[1],
                static_cast<size_t>(block*width)*sizeof(float) +
                static_cast<size_t>(b)*qkv->nb[2] +
                static_cast<size_t>(h*head_width)*sizeof(float));
        };
        std::vector<ggml_tensor *> batches;
        batches.reserve(static_cast<size_t>(batch));
        for (int b=0; b<batch; ++b) {
            std::vector<ggml_tensor *> joined_heads;
            joined_heads.reserve(heads);
            for (int h=0; h<heads; ++h) {
                auto *q=ggml_cont(ctx,head(0,h,b)), *k=ggml_cont(ctx,head(1,h,b)), *v=ggml_cont(ctx,head(2,h,b));
                auto *scores=ggml_mul_mat(ctx,k,q);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                auto *prob=ggml_soft_max(ctx,ggml_scale(ctx,scores,1.f/std::sqrt(float(head_width))));
                auto *value_product=ggml_mul_mat(ctx,prob,ggml_cont(ctx,ggml_transpose(ctx,v)));
                ggml_mul_mat_set_prec(value_product, GGML_PREC_F32);
                joined_heads.push_back(ggml_transpose(ctx,value_product));
            }
            auto *joined=joined_heads.front();
            for (int h=1; h<heads; ++h) joined=ggml_concat(ctx,joined,joined_heads[static_cast<size_t>(h)],0);
            batches.push_back(ggml_reshape_3d(ctx,joined,width,seq,1));
        }
        a=batches.front();
        for (int b=1; b<batch; ++b) a=ggml_concat(ctx,a,batches[static_cast<size_t>(b)],2);
    }
    a=linear(ctx,a,weight(w,s+"self_attn.out_proj.weight"),weight(w,s+"self_attn.out_proj.bias")); x=norm(ctx,ggml_add(ctx,x,a),weight(w,s+"norm1.weight"),weight(w,s+"norm1.bias")); auto*ff=linear(ctx,x,weight(w,s+"linear1.weight"),weight(w,s+"linear1.bias")); ff=ggml_gelu_erf(ctx,ff); ff=linear(ctx,ff,weight(w,s+"linear2.weight"),weight(w,s+"linear2.bias")); return norm(ctx,ggml_add(ctx,x,ff),weight(w,s+"norm2.weight"),weight(w,s+"norm2.bias"));
}
}
std::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_conditioned(
    const ggml_motion_weights &, std::span<const float>, std::span<const float>,
    std::span<const float>, std::span<const float>, float, float, float, float, std::size_t,
    std::span<const float> = {});
std::expected<std::vector<float>, std::string> run_motion_transformer(
    const ggml_motion_weights &w, std::string_view prefix,
    std::span<const float> motion, size_t motion_dim,
    std::span<const float> embedding, std::span<const float> timesteps,
    std::span<const float> headings, size_t batch, size_t frames) try {
    const auto profile_started = std::chrono::steady_clock::now();
    execution_profile profile;
    active_profile_guard profile_guard(profile_enabled() ? &profile : nullptr);
    if (!batch || !frames || motion.size() != batch*frames*motion_dim ||
        embedding.size() != batch*4096 || timesteps.size() != batch ||
        headings.size() != batch)
        return std::unexpected("invalid Transformer input dimensions");

    const int seq = prefix_tokens + static_cast<int>(frames);
    const std::string p(prefix);
    const int layer_chunk = motion_layer_chunk_size();
    cached_motion_transformer *cached = nullptr;
    if (motion_graph_cache_enabled()) {
        cached = transformer_cache(w, prefix, motion_dim, batch, frames, layer_chunk);
    } else if (w.graph_cache()) {
        w.graph_cache({});
    }

    const bool setup_cached = cached && cached->stages[0];
    const bool embedding_changed = !setup_cached ||
        cached->last_embedding.size() != embedding.size() ||
        !std::equal(cached->last_embedding.begin(), cached->last_embedding.end(), embedding.begin());
    const bool headings_changed = !setup_cached ||
        cached->last_headings.size() != headings.size() ||
        !std::equal(cached->last_headings.begin(), cached->last_headings.end(), headings.begin());
    std::vector<float> text, angle, position;
    std::vector<float> time(batch*width);
    if (embedding_changed) text.resize(batch*text_tokens*4096);
    if (headings_changed) angle.resize(batch*2);
    if (!setup_cached) position.resize(size_t(seq)*width);
    for (size_t b=0; b<batch; ++b) {
        if (embedding_changed)
            std::memcpy(text.data()+b*text_tokens*4096, embedding.data()+b*4096,
                        4096*sizeof(float));
        for (int d=0; d<width; d+=2) {
            const float value = timesteps[b]*std::pow(10000.f, -float(d)/width);
            time[b*width+d] = std::sin(value);
            time[b*width+d+1] = std::cos(value);
        }
        if (headings_changed) {
            angle[2*b] = std::cos(headings[b]);
            angle[2*b+1] = std::sin(headings[b]);
        }
    }
    if (!setup_cached) {
        for (int s=0; s<seq; ++s) for (int d=0; d<width; d+=2) {
            const float value = float(s)*std::pow(10000.f, -float(d)/width);
            position[size_t(s)*width+d] = std::sin(value);
            position[size_t(s)*width+d+1] = std::cos(value);
        }
    }

    auto run_graph = [&](size_t stage, size_t context_bytes, size_t output_values,
                         std::initializer_list<std::span<const float>> values,
                         std::initializer_list<bool> changed,
                         std::initializer_list<bool> retain,
                         auto &&build) -> std::expected<std::vector<float>, std::string> {
        if (cached && cached->stages[stage]) {
            bind_inputs(*cached->stages[stage], values, changed);
            return execute(*cached->stages[stage], w);
        }
        inputs.clear();
        auto *ctx = ggml_init({context_bytes, nullptr, true});
        if (!ctx) return std::unexpected("GGML context allocation failed");
        auto *out = build(ctx);
        if (cached) {
            auto built = cache_graph(ctx, out, output_values, retain, w);
            if (!built) return std::unexpected(built.error());
            cached->stages[stage] = std::move(*built);
            return execute(*cached->stages[stage], w);
        }
        auto result = execute(ctx, out, output_values, w);
        ggml_free(ctx);
        return result;
    };

    const size_t setup_context = cached ? cached_context_bytes : 128ULL*1024*1024;
    auto state_result = run_graph(0, setup_context, size_t(width)*seq*batch,
        {motion, text, time, angle, position},
        {true, embedding_changed, true, headings_changed, !setup_cached},
        {false, true, false, true, true},
        [&](ggml_context *ctx) {
            auto *m = linear(ctx, input(ctx,motion,int(motion_dim),int(frames),int(batch)),
                             weight(w,p+"input_linear.weight"), weight(w,p+"input_linear.bias"));
            auto *te = linear(ctx, input(ctx,text,4096,text_tokens,int(batch)),
                              weight(w,p+"embed_text.weight"), weight(w,p+"embed_text.bias"));
            auto *ti = linear(ctx, input(ctx,time,width,1,int(batch)),
                              weight(w,p+"embed_timestep.time_embed.0.weight"),
                              weight(w,p+"embed_timestep.time_embed.0.bias"));
            ti = linear(ctx, ggml_silu(ctx,ti),
                        weight(w,p+"embed_timestep.time_embed.2.weight"),
                        weight(w,p+"embed_timestep.time_embed.2.bias"));
            auto *he = linear(ctx, input(ctx,angle,2,1,int(batch)),
                              weight(w,p+"linear_first_heading_angle.weight"),
                              weight(w,p+"linear_first_heading_angle.bias"));
            auto *x = ggml_concat(ctx,ggml_concat(ctx,ggml_concat(ctx,te,ti,1),he,1),m,1);
            auto *pos = input(ctx,position,width,seq,1);
            return ggml_add(ctx,x,ggml_repeat(ctx,pos,x));
        });
    if (!state_result) return std::unexpected(state_result.error());
    if (cached) {
        if (embedding_changed) cached->last_embedding.assign(embedding.begin(), embedding.end());
        if (headings_changed) cached->last_headings.assign(headings.begin(), headings.end());
    }
    std::vector<float> state = std::move(*state_result);

    size_t stage = 1;
    for (int first=0; first<16; first+=layer_chunk, ++stage) {
        const size_t layer_context = cached ? cached_context_bytes : 256ULL*1024*1024;
        auto next = run_graph(stage, layer_context, size_t(width)*seq*batch,
            {state}, {true}, {false},
            [&](ggml_context *ctx) {
                auto *x = input(ctx,state,width,seq,int(batch));
                for (int i=first; i<std::min(first+layer_chunk,16); ++i)
                    x=layer(ctx,x,w,p+"seqTransEncoder.layers."+std::to_string(i)+".",seq,int(batch));
                return x;
            });
        if (!next) return std::unexpected(next.error());
        state = std::move(*next);
    }

    const size_t outdim = size_t(weight(w,p+"output_linear.bias")->ne[0]);
    const size_t output_context = cached ? cached_context_bytes : 32ULL*1024*1024;
    auto result = run_graph(stage, output_context, outdim*frames*batch,
        {state}, {true}, {false},
        [&](ggml_context *ctx) {
            auto *all = input(ctx,state,width,seq,int(batch));
            auto *part = ggml_view_3d(ctx,all,width,frames,batch,all->nb[1],all->nb[2],
                                      size_t(prefix_tokens)*width*sizeof(float));
            part = ggml_cont(ctx,part);
            return linear(ctx,part,weight(w,p+"output_linear.weight"),
                          weight(w,p+"output_linear.bias"));
        });
    if (profile_enabled()) {
        std::fprintf(stderr,
                     "profile motion.transformer stage=%.*s calls=%u graph_ms=%.3f allocation_ms=%.3f upload_ms=%.3f compute_ms=%.3f download_ms=%.3f total_ms=%.3f\n",
                     static_cast<int>(prefix.size() - 1), prefix.data(), profile.calls, profile.graph_ms,
                     profile.allocation_ms, profile.upload_ms, profile.compute_ms, profile.download_ms,
                     profile_elapsed_ms(profile_started));
    }
    return result;
} catch(const std::exception&e){inputs.clear();return std::unexpected(e.what());}

std::expected<std::vector<float>, std::string> run_two_stage_denoiser(
    const ggml_motion_weights &weights, std::span<const float> x,
    std::span<const float> embedding, std::span<const float> timesteps,
    std::span<const float> headings, std::span<const float> mask,
    std::size_t batch, std::size_t frames) {
    const size_t dim=weights.motion_dim(), root_input_dim=2*dim, body_input_dim=2*dim-1;
    if (!batch || !frames || !dim || x.size()!=batch*frames*root_input_dim || mask.size()!=batch*frames)
        return std::unexpected("invalid two-stage denoiser input dimensions");
    auto root=run_motion_transformer(weights,"root_model.",x,root_input_dim,embedding,timesteps,headings,batch,frames);
    if(!root)return std::unexpected(root.error());
    auto gm=weights.f32_values("stats.global_root.mean"), gs=weights.f32_values("stats.global_root.std"), lm=weights.f32_values("stats.local_root.mean"), ls=weights.f32_values("stats.local_root.std");
    if(!gm)return std::unexpected(gm.error());
    if(!gs)return std::unexpected(gs.error());
    if(!lm)return std::unexpected(lm.error());
    if(!ls)return std::unexpected(ls.error());
    auto local=global_root_to_local_root(*root,mask,batch,frames,*gm,*gs,*lm,*ls);
    if(!local)return std::unexpected(local.error());
    std::vector<float> body_input(batch*frames*body_input_dim);
    for(std::size_t b=0;b<batch;++b) for(std::size_t t=0;t<frames;++t) {
        const auto src=(b*frames+t)*root_input_dim, dst=(b*frames+t)*body_input_dim;
        std::memcpy(body_input.data()+dst,local->data()+(b*frames+t)*4,4*sizeof(float));
        std::memcpy(body_input.data()+dst+4,x.data()+src+5,(root_input_dim-5)*sizeof(float));
    }
    auto body=run_motion_transformer(weights,"body_model.",body_input,body_input_dim,embedding,timesteps,headings,batch,frames);
    if(!body)return std::unexpected(body.error());
    std::vector<float> output(batch*frames*dim);
    for(std::size_t b=0;b<batch;++b)for(std::size_t t=0;t<frames;++t){const auto r=(b*frames+t)*5, q=(b*frames+t)*(dim-5), o=(b*frames+t)*dim;std::memcpy(output.data()+o,root->data()+r,5*sizeof(float));std::memcpy(output.data()+o+5,body->data()+q,(dim-5)*sizeof(float));}
    return output;
}

std::expected<std::vector<float>, std::string> run_separated_cfg_denoiser(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, float timestep, float text_weight,
    float constraint_weight, std::size_t frames,
    std::span<const float> negative_embedding) {
    const std::vector<float> empty(frames*weights.motion_dim(), 0.f);
    return run_separated_cfg_denoiser_conditioned(weights, motion, embedding, empty, empty,
                                                  timestep, 0.f, text_weight, constraint_weight, frames,
                                                  negative_embedding);
}

std::expected<std::vector<float>, std::string> run_separated_cfg_denoiser_conditioned(
    const ggml_motion_weights &weights, std::span<const float> motion,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float timestep, float heading,
    float text_weight, float constraint_weight, std::size_t frames,
    std::span<const float> negative_embedding) {
    const size_t dim=weights.motion_dim();
    if (!dim || motion.size()!=frames*dim || embedding.size()!=4096 || !std::isfinite(timestep) || !std::isfinite(text_weight) || !std::isfinite(constraint_weight))
        return std::unexpected("invalid separated CFG denoiser input");
    if (observed.size()!=frames*dim || observed_mask.size()!=frames*dim || !std::isfinite(heading))
        return std::unexpected("invalid separated CFG condition dimensions");
    if (!negative_embedding.empty() && negative_embedding.size()!=4096)
        return std::unexpected("negative_embedding must be empty or exactly 4096 values");
    constexpr size_t cfg_batch=3; std::vector<float> extended(cfg_batch*frames*2*dim), text(cfg_batch*4096), times(cfg_batch,timestep), headings(cfg_batch,heading), mask(cfg_batch*frames,1.f);
    for(size_t b=0;b<cfg_batch;++b) for(size_t t=0;t<frames;++t) {
        auto *dst=extended.data()+(b*frames+t)*2*dim;
        std::memcpy(dst,motion.data()+t*dim,dim*sizeof(float));
        // Upstream separated CFG is [text, constraint, unconditional]. Only
        // the constraint branch receives observed motion and its feature mask.
        if (b==1) for (size_t d=0;d<dim;++d) dst[d]=motion[t*dim+d]*(1.f-observed_mask[t*dim+d])+observed[t*dim+d]*observed_mask[t*dim+d];
        if (b==1) std::memcpy(dst+dim,observed_mask.data()+t*dim,dim*sizeof(float));
    }
    // Only branch zero has text. Branch one is constraint-only; branch two is
    // unconditional. This is the upstream separated-CFG batch order.
    std::memcpy(text.data(),embedding.data(),4096*sizeof(float));
    // Standard diffusion negative-prompt technique: fill the "unconditional"
    // branch's text slot with the negative-prompt embedding instead of
    // leaving it zero, so CFG pushes away from that concept specifically
    // rather than away from nothing. Left zero (upstream behavior) when no
    // negative prompt was given.
    if (!negative_embedding.empty())
        std::memcpy(text.data()+2*4096,negative_embedding.data(),4096*sizeof(float));
    auto all=run_two_stage_denoiser(weights,extended,text,times,headings,mask,cfg_batch,frames);
    if(!all)return std::unexpected(all.error());
    std::vector<float> result(frames*dim);
    for(size_t i=0;i<result.size();++i) result[i]=(*all)[2*result.size()+i]+text_weight*((*all)[i]-(*all)[2*result.size()+i])+constraint_weight*((*all)[result.size()+i]-(*all)[2*result.size()+i]);
    return result;
}

std::expected<std::vector<float>, std::string> sample_motion_from_noise(
    const ggml_motion_weights &weights, std::span<const float> initial,
    std::span<const float> embedding, std::size_t frames, unsigned steps,
    float text_weight, float constraint_weight,
    std::span<const float> negative_embedding) {
    if(initial.size()!=frames*weights.motion_dim()) return std::unexpected("invalid initial motion noise dimensions");
    auto schedule=make_cosine_schedule(1000,steps); if(!schedule)return std::unexpected(schedule.error());
    std::vector<float> state(initial.begin(),initial.end()), next(state.size());
    for(unsigned i=steps;i-->0;) {
        const auto step_started = std::chrono::steady_clock::now();
        auto clean=run_separated_cfg_denoiser(weights,state,embedding,float(schedule->use_timesteps[i]),text_weight,constraint_weight,frames,negative_embedding);
        if(!clean)return std::unexpected(clean.error());
        auto stepped=ddim_step(*schedule,i,state.data(),clean->data(),next.data(),state.size());
        if(!stepped)return std::unexpected(stepped.error());
        state.swap(next);
        if (profile_enabled())
            std::fprintf(stderr, "profile motion.diffusion_step index=%u ms=%.3f\n", i, profile_elapsed_ms(step_started));
    }
    return state;
}

std::expected<std::vector<float>, std::string> sample_motion_from_noise_conditioned(
    const ggml_motion_weights &weights, std::span<const float> initial,
    std::span<const float> embedding, std::span<const float> observed,
    std::span<const float> observed_mask, float heading, std::size_t frames,
    unsigned steps, float text_weight, float constraint_weight,
    std::span<const float> negative_embedding) {
    if(initial.size()!=frames*weights.motion_dim() || observed.size()!=initial.size() || observed_mask.size()!=initial.size())
        return std::unexpected("invalid conditioned motion noise dimensions");
    auto schedule=make_cosine_schedule(1000,steps); if(!schedule)return std::unexpected(schedule.error());
    std::vector<float> state(initial.begin(),initial.end()), next(state.size());
    for(unsigned i=steps;i-->0;) {
        auto clean=run_separated_cfg_denoiser_conditioned(weights,state,embedding,observed,observed_mask,
                                                           float(schedule->use_timesteps[i]),heading,text_weight,constraint_weight,frames,
                                                           negative_embedding);
        if(!clean)return std::unexpected(clean.error());
        auto stepped=ddim_step(*schedule,i,state.data(),clean->data(),next.data(),state.size());
        if(!stepped)return std::unexpected(stepped.error());
        state.swap(next);
    }
    return state;
}
}
