// Copyright (c) 2026. Licensed under the repository LICENSE.
// Same target-only ND def template as the tree verify operator.
#include "register/op_def_registry.h"
namespace ops {
class TreeGdnCompactReplayV310 : public OpDef {
public:
    explicit TreeGdnCompactReplayV310(const char *name) : OpDef(name) {
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("initial_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("records").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("accepted_states").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Attr("parents").AttrType(REQUIRED).ListInt();
        this->Attr("path").AttrType(REQUIRED).ListInt();
        this->Attr("v_tile").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend310p", config);
    }
};
OP_ADD(TreeGdnCompactReplayV310);
} // namespace ops
