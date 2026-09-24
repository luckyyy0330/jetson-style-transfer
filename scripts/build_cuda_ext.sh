#!/bin/bash
# 构建 style_cuda pybind11 CUDA 扩展
#
# 用法:
#   bash scripts/build_cuda_ext.sh
#
# 前置条件:
#   - CUDA Toolkit (nvcc)
#   - pybind11 (pip install pybind11)
#   - CMake >= 3.18
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/cuda/build"

echo "构建 style_cuda CUDA 扩展..."
echo "  项目目录: ${PROJECT_DIR}"
echo "  构建目录: ${BUILD_DIR}"

# 确保 pybind11 可用
python3 -c "import pybind11; print(f'pybind11: {pybind11.__version__}')" 2>/dev/null || {
    echo "pybind11 未安装，正在安装..."
    pip install pybind11
}

PYBIND11_CMAKE=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
echo "  pybind11 CMake: ${PYBIND11_CMAKE}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="${PYBIND11_CMAKE}" \
    -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build . -j$(nproc)

echo ""
echo "构建完成: app/cuda_ext/style_cuda*.so"
echo "验证: python3 -c \"import sys; sys.path.insert(0, 'app/cuda_ext'); import style_cuda; print('ok', style_cuda.version())\""
