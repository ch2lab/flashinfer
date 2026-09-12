#!/bin/bash
# SM70(V100) 增量编译 flashinfer 源码 csrc 内核
# 用法: ./build_flashinfer_sm70.sh [verbose]
#
# flashinfer 用 ninja 增量编译 csrc/*.cu → ~/.cache/flashinfer/0.0.0+unknown/
# 改 .cuh/.cu 后重跑本脚本,只有改动的内核重编(ninja 依赖跟踪,天然增量)。
# 目标架构 = V100/CC7.0(FLASHINFER_CUDA_ARCH_LIST=7.0 + FLASHINFER_ALLOW_SM70=1)。

set -e
cd /data/src/flashinfer

# 目标架构 = V100/SM70;放开 SM70 闸;并行编译数;JIT 日志
export FLASHINFER_ALLOW_SM70=1
export FLASHINFER_CUDA_ARCH_LIST="7.0"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
export FLASHINFER_JIT_VERBOSE="${FLASHINFER_JIT_VERBOSE:-1}"

# 确保 submodule + editable 数据目录(符号链接,幂等;JIT 从 data/ 读模板/头文件)
git submodule update --init --recursive >/dev/null 2>&1 || true
mkdir -p flashinfer/data
ln -sfn "$PWD/csrc"             flashinfer/data/csrc
ln -sfn "$PWD/include"          flashinfer/data/include
ln -sfn "$PWD/3rdparty/cutlass" flashinfer/data/cutlass
ln -sfn "$PWD/3rdparty/spdlog"  flashinfer/data/spdlog
ln -sfn "$PWD/3rdparty/cccl"    flashinfer/data/cccl

python3 - <<'PY'
import sys
from flashinfer.compilation_context import CompilationContext
from flashinfer.jit import jit_spec_registry, build_jit_specs
from flashinfer.aot import register_default_modules

TARGET = "sm_70"


# 软 patch: 匹配不到目标 arch 时不再 raise(原函数会 raise "No supported CUDA
# architectures"),包一层拦截 → 返回无 gencode 的 flags,让 gen_all_modules 能走完。
_orig_get_nvcc_flags = CompilationContext.get_nvcc_flags_list


def _soft_get_nvcc_flags(self, *a, **k):
    try:
        return _orig_get_nvcc_flags(self, *a, **k)
    except RuntimeError:
        return list(self.COMMON_NVCC_FLAGS)


CompilationContext.get_nvcc_flags_list = _soft_get_nvcc_flags

if not jit_spec_registry.get_all_statuses():
    register_default_modules()

all_specs = list(jit_spec_registry.get_all_specs().values())


# 关键:16-bit 分页 attention(decode + prefill)模块不写显式 -gencode(走默认
# arch)。本构建 FLASHINFER_CUDA_ARCH_LIST=7.0,默认 arch 即 sm_70,因此它们实际
# 会被编成 sm_70 —— 但旧启发式只看 stored flags 里是否含 "sm_70" 字样,导致 0 命中。
# 这里按 模块名 + 16-bit KV 类型 + head_dim 精确挑选目标模块,既保证编到 16-bit
# 分页 decode/prefill,又避免把 sm90/smel120 专用 kernel(编不了 sm_70)也拉进来。
def _is_16bit_paged_target(s):
    n = s.name
    if not (
        n.startswith("batch_decode_with_kv_cache")
        or n.startswith("batch_prefill_with_kv_cache")
    ):
        return False
    # 只要 16-bit KV(bf16/f16);排除 fp8(e4m3/e5m2)与 nvfp4/fp4
    if (
        "_dtype_kv_e4m3" in n
        or "_dtype_kv_e5m2" in n
        or "_dtype_kv_fp4" in n
        or "_dtype_kv_nvfp4" in n
    ):
        return False
    if "_dtype_kv_bf16" not in n and "_dtype_kv_f16" not in n:
        return False
    # head_dim_qk=128(模型配置);64/256 走同一 kernel 路径,需要时放开此过滤
    if "_head_dim_qk_128_" not in n:
        return False
    return True


specs = [s for s in all_specs if _is_16bit_paged_target(s)]
st0 = jit_spec_registry.get_stats()
print(f"=== SM70 目标模块: {len(specs)} / 共 {st0['total']} ===")

verbose = "verbose" in sys.argv
if specs:
    try:
        build_jit_specs(specs, verbose=verbose, skip_prebuilt=True)
    except Exception as e:  # 组合 ninja 失败 → 逐个编译并报告
        print(f"[组合 ninja 失败: {e!r}] → 逐个:")
        ok, fail = 0, []
        for s in specs:
            try:
                if not s.aot_path.exists():
                    s.build(verbose=verbose, need_lock=True)
                ok += 1
            except Exception as ee:
                fail.append((s.name, repr(ee)))
        print(f"built={ok} failed={len(fail)}")
        for n, err in fail[:20]:
            print(f"  FAIL {n}: {err[:140]}")
else:
    print("(当前无模块声明支持 sm_70 —— 下一步: 把 7.0 加进各模块 supported_major_versions)")

st = jit_spec_registry.get_stats()
print(f"=== stats: total={st['total']} compiled={st['compiled']} not={st['not_compiled']} ===")
PY
