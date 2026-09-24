"""下载风格模型

根据 config/styles.json 配置，下载所有风格所需的 GAN ONNX 模型。
支持按风格下载或一次性下载所有模型（自动去重）。

用法:
  # 下载所有风格所需的模型（去重）
  python scripts/download_models.py --all

  # 下载指定风格的模型
  python scripts/download_models.py --style 1

  # 使用 HuggingFace 镜像（国内加速）
  HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py --all
"""

import os
import sys
import argparse
import logging
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.style.style_manager import StyleManager


def download_model(url: str, output_path: str) -> bool:
    """
    下载模型文件

    Args:
        url: 下载地址
        output_path: 本地保存路径

    Returns:
        是否成功
    """
    if not url:
        logger.warning(f"未配置下载地址: {output_path}")
        return False

    # 如果文件已存在，跳过
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        logger.info(f"模型已存在，跳过: {output_path}")
        return True

    try:
        logger.info(f"下载模型: {url}")
        logger.info(f"保存到: {output_path}")

        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 下载文件（支持 HuggingFace 等需要自定义 UA 的源）
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                print(f"\r  下载进度: {percent}% ({downloaded // 1024}KB / {total_size // 1024}KB)", end="", flush=True)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response, open(output_path, 'wb') as out_file:
            total = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = min(100, downloaded * 100 // total)
                    print(f"\r  下载进度: {percent}% ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
        print()  # 换行

        # 验证文件
        file_size = os.path.getsize(output_path)
        if file_size < 1024:
            logger.error(f"下载的文件太小 ({file_size} bytes)，可能不是有效的模型文件")
            os.remove(output_path)
            return False

        logger.info(f"下载完成: {output_path} ({file_size // 1024}KB)")
        return True

    except Exception as e:
        logger.error(f"下载失败: {e}")
        # 清理不完整的文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def download_pt_model(url: str, output_path: str) -> bool:
    """
    下载 PyTorch .pt 模型并提示需要转换为 ONNX

    Args:
        url: 下载地址
        output_path: 本地保存路径（.onnx）

    Returns:
        是否成功
    """
    if not url.endswith('.pt'):
        return download_model(url, output_path)

    # .pt 文件需要先下载再转换
    pt_path = output_path.replace('.onnx', '.pt')
    success = download_model(url, pt_path)
    if success:
        logger.info(f"已下载 PyTorch 模型: {pt_path}")
        logger.info("请手动转换为 ONNX 格式，或使用支持 ONNX 的模型")
    return success


def download_style_models(
    style_manager: StyleManager,
    style_id: int = None,
    models_dir: str = "models"
) -> bool:
    """
    下载风格所需的模型

    Args:
        style_manager: 风格管理器
        style_id: 指定风格 ID，None 则下载所有
        models_dir: 模型目录

    Returns:
        是否全部成功
    """
    all_success = True

    if style_id is not None:
        # 下载指定风格
        style = style_manager.get_style(style_id)
        if style is None:
            logger.error(f"风格 {style_id} 不存在")
            return False

        model_path = style_manager.resolve_model_path(style)
        model_url = style_manager.get_model_url(style)
        logger.info(f"下载风格 {style.name} 的模型...")

        if model_url.endswith('.pt'):
            success = download_pt_model(model_url, model_path)
        else:
            success = download_model(model_url, model_path)
        if not success:
            all_success = False

    else:
        # 下载所有风格的模型（去重）
        unique_models = style_manager.get_unique_models()
        logger.info(f"共 {len(unique_models)} 个唯一模型需要下载")

        for model_path, style_ids in unique_models.items():
            style_names = [style_manager.get_style(sid).name for sid in style_ids]
            first_style = style_manager.get_style(style_ids[0])
            model_url = style_manager.get_model_url(first_style)

            logger.info(f"模型 {model_url} -> {model_path} (用于: {', '.join(style_names)})")

            if model_url.endswith('.pt'):
                success = download_pt_model(model_url, model_path)
            else:
                success = download_model(model_url, model_path)
            if not success:
                all_success = False

    return all_success


def main():
    parser = argparse.ArgumentParser(description="下载 GAN 风格模型")
    parser.add_argument("--all", action="store_true", help="下载所有风格所需的模型")
    parser.add_argument("--style", type=int, help="下载指定风格 ID 的模型")
    parser.add_argument("--models", default="models", help="模型目录")
    parser.add_argument("--config", default="config/styles.json", help="风格配置文件")

    args = parser.parse_args()

    # 加载风格配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_path = os.path.join(project_dir, args.config)

    style_manager = StyleManager(
        config_path=config_path,
        models_dir=os.path.join(project_dir, args.models)
    )

    if not style_manager.styles:
        logger.error("没有找到风格配置")
        return

    if args.all:
        success = download_style_models(style_manager, models_dir=args.models)
    elif args.style:
        success = download_style_models(style_manager, style_id=args.style, models_dir=args.models)
    else:
        logger.error("请指定 --all 或 --style <ID>")
        return

    if success:
        logger.info("下载完成!")
        logger.info("下一步:")
        logger.info("  python -m app.main")
    else:
        logger.warning("部分模型下载失败，请检查日志")
        logger.info("提示: 某些模型可能需要手动下载，请查看 config/styles.json 中的 model_url")


if __name__ == "__main__":
    main()
