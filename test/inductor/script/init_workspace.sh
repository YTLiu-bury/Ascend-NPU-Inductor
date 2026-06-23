#!/bin/bash
# Initialize a torch-npu + triton-ascend build workspace (AscendNPU-IR -> torch-npu -> triton-ascend)
# Idempotent: fresh workspace does git clone; existing workspace does git pull + rebuild
# Usage: bash init_workspace.sh <workspace_path> [--daily]
#   --daily  Also checkout & pull latest 3.2.2_dev branch for triton-ascend
# Run in background:
#   nohup bash /path/to/init_workspace.sh <workspace_path> &>init.log &

set -e

BASE=${1:?"Usage: $0 <workspace_path> [--daily]"}
DAILY=false
if [ "$2" = "--daily" ]; then
    DAILY=true
fi
REPO=$BASE/pytorch
TRITON=$REPO/third_party/triton-ascend
ASCENDNPU_IR=$TRITON/third_party/ascend/AscendNPU-IR

# 1. Source CANN environment
source /root/Ascend/ascend-toolkit/set_env.sh
source /root/Ascend/cann-9.0.0/set_env.sh
unset PYTHONPATH

# 2. Create workspace and venv
mkdir -p $BASE
if [ ! -f "$BASE/venv/bin/activate" ]; then
    python3.11 -m venv $BASE/venv
fi
source $BASE/venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.7.1 pyyaml cmake ninja pybind11 expecttest pytest==8.3.2 pytest-xdist

# 3. Clone or update torch-npu
if [ -d "$REPO/.git" ]; then
    cd $REPO && git pull origin v2.7.1-26.0.0
else
    cd $BASE && git clone -b v2.7.1-26.0.0 https://gitcode.com/huyuchao/pytorch.git
fi

# 4. Init submodules (skip torch-mlir)
cd $REPO
git config submodule."third_party/torch-mlir".update none
git config submodule.alternateErrorStrategy info
for sub in Tensorpipe dvm/dvm fmt googletest nlohmann "op-plugin" "torchair/torchair"; do
    git submodule update --init --depth 1 --recursive "third_party/$sub"
done
git submodule update --init third_party/triton-ascend

# 5. Init triton-ascend submodules
cd $TRITON
if $DAILY; then
    git checkout 3.2.2_dev
    git pull origin 3.2.2_dev
fi
git submodule update --init third_party/ascend/AscendNPU-IR

# 6. Build AscendNPU-IR (bishengir-compile)
cd $ASCENDNPU_IR
if $DAILY; then
    git checkout master
    git pull origin master
fi
# Use pre-patched third-party from local archive (avoid git submodule init + cp)
THIRD_PARTY_LOCAL=/models/torch-inductor/g00929853/thrid_party_patch
./build-tools/build.sh -o ./build --build-type Release -t \
    --bisheng-compiler /root/Ascend/cann-9.0.0/bin/ \
    --llvm-source-dir "$THIRD_PARTY_LOCAL/llvm-project" \
    --torch-mlir-source-dir "$THIRD_PARTY_LOCAL/torch-mlir"
export PATH="$ASCENDNPU_IR/build/bin:$PATH"

# 7. Sync triton-ascend submodule pointer to latest AscendNPU-IR master
# Prevents triton-ascend build (setup.py BackendInstaller) from resetting
# AscendNPU-IR to stale pinned commit during "git submodule update --init ascend"
if $DAILY; then
    cd $TRITON
    LATEST_IR_COMMIT=$(cd $ASCENDNPU_IR && git rev-parse HEAD)
    git update-index --cacheinfo 160000,$LATEST_IR_COMMIT,third_party/ascend/AscendNPU-IR
fi

# 8. Build torch-npu
cd $REPO
rm -rf dist/torch_npu-*.whl
bash ci/build.sh --python=3.11
pip install dist/torch_npu-*.whl --force-reinstall

# 9. Build triton-ascend
cd $TRITON/python
rm -rf dist/triton_ascend-*.whl
pip uninstall -y triton-ascend triton 2>/dev/null || true
python setup.py bdist_wheel
pip install dist/triton_ascend-*.whl --force-reinstall

# 10. Verify
cd $BASE
python -c "import torch; import torch_npu; print(f'NPUs: {torch.npu.device_count()}')" || echo "WARNING: Quick verify failed, try sourcing venv manually"
echo "Workspace initialized at $BASE"
