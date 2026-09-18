#include "constraints.hpp"
#include "skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace kimodo::detail {
namespace {
float scale(float value) { return std::sqrt(value * value + 1.e-5F); }

bool finite(std::span<const float> values) {
    return std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
}

std::array<float, 6> quaternion_to_six(std::array<float, 4> q) {
    const float length=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for(float &value:q)value/=length;
    const auto [x,y,z,w]=q;
    return {
        1.F-2.F*(y*y+z*z), 2.F*(x*y+z*w), 2.F*(x*z-y*w),
        2.F*(x*y-z*w), 1.F-2.F*(x*x+z*z), 2.F*(y*z+x*w),
    };
}

float quaternion_heading(std::array<float, 4> q) {
    const float length=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for(float &value:q)value/=length;
    const auto [x,y,z,w]=q;
    const float forward_x=2.F*(x*z+w*y);
    const float forward_z=1.F-2.F*(x*x+y*y);
    return std::atan2(forward_x,forward_z);
}
}

std::expected<pose_condition, std::string> build_pose_condition(
    const skeleton_spec &s, std::span<const pose_constraint> constraints,
    std::size_t frames, std::span<const float> gm, std::span<const float> gs,
    std::span<const float> bm, std::span<const float> bs) {
    const std::size_t D=s.motion_dim(),J=s.joints(),rotation_begin=5+3*J,body=D-5;
    if(!frames||gm.size()!=5||gs.size()!=5||bm.size()!=body||bs.size()!=body||
       !finite(gm)||!finite(gs)||!finite(bm)||!finite(bs))
        return std::unexpected("invalid pose-condition statistics");
    pose_condition result;
    result.observed.assign(frames*D,0.F);
    result.observed_mask.assign(frames*D,0.F);
    std::vector<std::array<float,3>> roots(frames);
    std::vector<bool> root_known(frames,false);
    for(const auto &constraint:constraints) {
        if(constraint.frame>=frames||constraint.joint>=J||
           (!constraint.constrain_position&&!constraint.constrain_rotation))
            return std::unexpected("invalid pose constraint index or empty constraint");
        if(constraint.constrain_position&&!finite(constraint.world_position))
            return std::unexpected("pose constraint position must be finite");
        if(constraint.constrain_rotation) {
            if(!finite(constraint.world_rotation_xyzw))
                return std::unexpected("pose constraint rotation must be finite");
            float length=0.F;for(float value:constraint.world_rotation_xyzw)length+=value*value;
            if(length<1.e-12F)return std::unexpected("pose constraint rotation must not be zero");
        }
        if(constraint.joint==0&&constraint.constrain_position) {
            roots[constraint.frame]=constraint.world_position;
            root_known[constraint.frame]=true;
        }
    }
    bool have_heading=false;
    for(const auto &constraint:constraints) {
        const auto frame=static_cast<std::size_t>(constraint.frame),joint=static_cast<std::size_t>(constraint.joint);
        const auto base=frame*D;
        if(constraint.constrain_position) {
            const auto root=root_known[frame]?roots[frame]:std::array<float,3>{};
            const std::array<float,3> represented{
                constraint.world_position[0]-root[0], constraint.world_position[1],
                constraint.world_position[2]-root[2]};
            for(std::size_t axis=0;axis<3;++axis) {
                const auto body_index=joint*3+axis, index=base+5+body_index;
                result.observed[index]=(represented[axis]-bm[body_index])/scale(bs[body_index]);
                result.observed_mask[index]=1.F;
            }
            if(joint==0)for(std::size_t axis=0;axis<3;++axis) {
                const auto index=base+axis;
                result.observed[index]=(constraint.world_position[axis]-gm[axis])/scale(gs[axis]);
                result.observed_mask[index]=1.F;
            }
        }
        if(constraint.constrain_rotation) {
            const auto six=quaternion_to_six(constraint.world_rotation_xyzw);
            for(std::size_t component=0;component<6;++component) {
                const auto body_index=(rotation_begin-5)+joint*6+component;
                const auto index=base+rotation_begin+joint*6+component;
                result.observed[index]=(six[component]-bm[body_index])/scale(bs[body_index]);
                result.observed_mask[index]=1.F;
            }
            if(joint==0) {
                const float heading=quaternion_heading(constraint.world_rotation_xyzw);
                for(std::size_t component=3;component<5;++component) {
                    const float value=component==3?std::cos(heading):std::sin(heading);
                    result.observed[base+component]=(value-gm[component])/scale(gs[component]);
                    result.observed_mask[base+component]=1.F;
                }
                if(!have_heading){result.first_heading=heading;result.heading_constrained=true;have_heading=true;}
            }
        }
    }
    return result;
}
}
