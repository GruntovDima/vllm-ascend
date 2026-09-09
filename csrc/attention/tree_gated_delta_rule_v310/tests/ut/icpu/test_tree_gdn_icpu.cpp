// ICPU functional sanity only. Actual silicon evidence is the separate ACLNN suite.
// Adapted from the P6 test.cpp.tmpl; CTest replaces a hard-coded gtest dependency.
#include "tikicpulib.h"
#include "../../../op_kernel/tree_gated_delta_rule_v310.h"
#include "../../../op_host/tree_gdn_plan.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

TreeGDN310::TreeGatedDeltaRuleV310TilingData PrepareIcpuTiling(
    const std::vector<int64_t>&, uint32_t, uint32_t, uint32_t, float);

extern "C" __global__ __aicore__ void tree_gdn_icpu(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initial, GM_ADDR g,
    GM_ADDR out, GM_ADDR states, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    TreeGDN310::TreeGDNKernel kernel;
    const auto *td=reinterpret_cast<const TreeGDN310::TreeGatedDeltaRuleV310TilingData *>(tiling);
    kernel.Init(q,k,v,beta,initial,g,out,states,td,&pipe);
    kernel.Process();
}
namespace {
struct Gm {
    uint8_t *data; size_t bytes;
    explicit Gm(size_t n):data(static_cast<uint8_t*>(AscendC::GmAlloc(n))),bytes(n) {
        if (!data) throw std::runtime_error("GmAlloc");
        std::memset(data,0,n);
    }
    ~Gm(){AscendC::GmFree(data);}
    Gm(const Gm&)=delete;
};
void Load(Gm &dst,const std::string &path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if (!in || static_cast<size_t>(in.tellg())!=dst.bytes) throw std::runtime_error("input size: "+path);
    in.seekg(0);in.read(reinterpret_cast<char*>(dst.data),dst.bytes);
    if (!in) throw std::runtime_error("input read: "+path);
}
void Compare(const Gm &actual,const std::string &path,uint32_t nodes) {
    Gm expected(actual.bytes);Load(expected,path);
    const half *a=reinterpret_cast<const half*>(actual.data);
    const half *b=reinterpret_cast<const half*>(expected.data);
    const size_t perNode=actual.bytes/sizeof(half)/nodes;
    double worst=1;size_t changed=0;
    for(uint32_t node=0;node<nodes;++node) {
        double dot=0,aa=0,bb=0;
        for(size_t i=node*perNode;i<(node+1)*perNode;++i) {
            const double x=static_cast<float>(a[i]),y=static_cast<float>(b[i]);
            if(!std::isfinite(x)||!std::isfinite(y)) throw std::runtime_error("nonfinite");
            dot+=x*y;aa+=x*x;bb+=y*y;
            changed += std::memcmp(a+i,b+i,sizeof(half))!=0;
        }
        const double cosine=(aa==0 || bb==0)?(aa==bb?1:0):dot/std::sqrt(aa*bb);
        worst=std::min(worst,cosine);
    }
    std::cout.precision(12);
    std::cout<<"ICPU cosine: "<<worst<<" changed="<<changed<<" reference="<<path<<"\n";
    if(worst<0.999999) throw std::runtime_error("ICPU numerical mismatch");
}
}
int main(int argc,char **argv) {
    try {
        using namespace TreeGDN310;
        if(argc!=2)throw std::runtime_error("usage: test_tree_gdn_icpu CASE_DIR");
        const std::string dir=argv[1];
        uint32_t n,hk,hv,r;float scale;
        std::ifstream params(dir+"/params.txt");
        if(!(params>>n>>hk>>hv>>r>>scale))throw std::runtime_error("params");
        std::vector<int64_t> parents(n);
        for(auto &p:parents)if(!(params>>p))throw std::runtime_error("parents");
        const auto td=PrepareIcpuTiling(parents,hk,hv,r,scale);
        Gm q(n*hk*128*2),k(q.bytes),v(n*hv*128*2),beta(n*hv*2),
           initial(hv*128*128*2),g(n*hv*4),out(v.bytes),states(n*initial.bytes),tiling(sizeof(td));
        Load(q,dir+"/input_query.bin");Load(k,dir+"/input_key.bin");
        Load(v,dir+"/input_value.bin");Load(beta,dir+"/input_beta.bin");
        Load(initial,dir+"/input_initial_state.bin");Load(g,dir+"/input_g.bin");
        std::memcpy(tiling.data,&td,sizeof(td));
        ICPU_RUN_KF(tree_gdn_icpu,td.coreCount,q.data,k.data,v.data,beta.data,
                    initial.data,g.data,out.data,states.data,tiling.data);
        Compare(out,dir+"/golden_out.bin",n);Compare(states,dir+"/golden_snapshots.bin",n);
        std::cout<<"OK: ICPU functional test, not device acceptance\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
