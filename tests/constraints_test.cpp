#include "constraints.hpp"
#include "skeleton.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const auto &s=kimodo::detail::soma30_spec;
    std::vector<float> gm(5,0.F),gs(5,1.F),bm(s.body_dim(),0.F),bs(s.body_dim(),1.F);
    std::vector<kimodo::pose_constraint> constraints{
        {2,0,true,{1.F,0.9F,2.F},true,{0.F,0.F,0.F,1.F}},
        {2,13,true,{1.4F,1.2F,2.6F},true,{0.F,0.F,0.F,1.F}},
    };
    auto condition=kimodo::detail::build_pose_condition(s,constraints,5,gm,gs,bm,bs);
    if(!condition){std::cerr<<condition.error()<<'\n';return 1;}
    if(!condition->heading_constrained)return 7;
    const std::size_t D=s.motion_dim(),base=2*D,rotation_begin=5+3*s.joints();
    // Root xyz + heading, pelvis position/rotation, and hand position/rotation.
    for(std::size_t index:{base,base+1,base+2,base+3,base+4})if(condition->observed_mask[index]!=1.F)return 2;
    for(std::size_t axis=0;axis<3;++axis)if(condition->observed_mask[base+5+13*3+axis]!=1.F)return 3;
    for(std::size_t component=0;component<6;++component)if(condition->observed_mask[base+rotation_begin+13*6+component]!=1.F)return 4;
    // Hand positions are represented relative to the constrained root in X/Z.
    const float normalization=std::sqrt(1.F+1.e-5F);
    const std::array<float,3> expected{.4F/normalization,1.2F/normalization,.6F/normalization};
    for(std::size_t axis=0;axis<3;++axis)if(std::abs(condition->observed[base+5+13*3+axis]-expected[axis])>1.e-5F)return 5;
    // An unconstrained frame remains completely free.
    for(std::size_t index=0;index<D;++index)if(condition->observed_mask[index]!=0.F)return 6;
    return 0;
}
