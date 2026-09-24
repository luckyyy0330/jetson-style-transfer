"""设备初始化脚本

在 Jetson 上运行，配置系统环境
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_jetson():
    """检查是否在 Jetson 上运行"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read()
            if 'nvidia' in content.lower():
                return True
    except FileNotFoundError:
        pass
    return False


def check_cuda():
    """检查 CUDA 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass

    # 尝试 nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"CUDA 已安装")
            return True
    except FileNotFoundError:
        pass

    return False


def check_tensorrt():
    """检查 TensorRT 是否可用"""
    try:
        import tensorrt
        logger.info(f"TensorRT 版本: {tensorrt.__version__}")
        return True
    except ImportError:
        return False


def install_python_packages():
    """安装 Python 依赖"""
    logger.info("安装 Python 依赖...")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_file = os.path.join(project_dir, "requirements.txt")

    if os.path.exists(requirements_file):
        # 先强制重新安装核心依赖（解决版本冲突）
        # pip 默认不会降级已安装的包，需要 --force-reinstall
        logger.info("强制重新安装核心依赖（修复版本冲突）...")
        core_cmd = [
            sys.executable, "-m", "pip", "install",
            "--force-reinstall",
            "numpy<2.0",
        ]
        subprocess.run(core_cmd, check=False)

        # 安装其余依赖
        cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
        subprocess.run(cmd, check=False)
    else:
        logger.warning(f"requirements.txt 不存在: {requirements_file}")


def configure_system():
    """配置系统设置"""
    logger.info("配置系统...")

    # 设置性能模式
    try:
        # 只设置 nvpmodel 为 MaxN Super 模式（ID=2）
        # 注意：不要使用 jetson_clocks，它会锁定所有时钟频率，
        # 可能导致 CSI 摄像头 VI 管线无法初始化
        subprocess.run(
            ['sudo', 'nvpmodel', '-m', '2'],  # MAXN_SUPER mode
            capture_output=True
        )
        logger.info("已设置 MAXN_SUPER 性能模式")
        logger.info("注意：未使用 jetson_clocks 避免摄像头冲突")
    except Exception as e:
        logger.warning(f"设置性能模式失败: {e}")

    # 设置风扇速度
    try:
        fan_path = "/sys/devices/pwm-fan/target_duty_cycle"
        if os.path.exists(fan_path):
            with open(fan_path, 'w') as f:
                f.write('200')  # 80% 风扇速度
            logger.info("已设置风扇速度: 80%")
    except Exception as e:
        logger.warning(f"设置风扇速度失败: {e}")


def create_directories():
    """创建必要的目录"""
    dirs = [
        "assets/captures",
        "models/trt",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"目录已创建: {d}")


def main():
    logger.info("=" * 50)
    logger.info("Jetson 风格转换设备 - 初始化")
    logger.info("=" * 50)

    # 检查环境
    is_jetson = check_jetson()
    logger.info(f"Jetson 设备: {'是' if is_jetson else '否（开发模式）'}")

    cuda_available = check_cuda()
    logger.info(f"CUDA: {'可用' if cuda_available else '不可用'}")

    tensorrt_available = check_tensorrt()
    logger.info(f"TensorRT: {'可用' if tensorrt_available else '不可用'}")

    # 创建目录
    create_directories()

    # 配置系统（仅在 Jetson 上）
    if is_jetson:
        configure_system()

    # 安装依赖
    install_python_packages()

    # 打印结果
    logger.info("\n" + "=" * 50)
    logger.info("初始化完成!")
    logger.info("=" * 50)
    logger.info("下一步:")
    logger.info("1. 运行 download_models.py 下载模型")
    logger.info("2. 运行 convert_tensorrt.py 转换为 TensorRT 引擎")
    logger.info("3. 运行 python -m app.main 启动应用")


if __name__ == "__main__":
    main()
