"""存储管理模块

管理 SD 卡和 USB 存储设备的文件保存和导出
"""

import os
import shutil
import time
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class StorageManager:
    """存储管理器"""

    def __init__(
        self,
        sd_card_path: str = "/media/sdcard",
        usb_path: str = "/media/usb",
        captures_dir: str = "assets/captures",
        max_captures: int = 1000
    ):
        """
        初始化存储管理器

        Args:
            sd_card_path: SD 卡挂载路径
            usb_path: USB 存储挂载路径
            captures_dir: 本地捕获目录
            max_captures: 最大捕获文件数
        """
        self.sd_card_path = sd_card_path
        self.usb_path = usb_path
        self.captures_dir = captures_dir
        self.max_captures = max_captures

        # 确保目录存在
        os.makedirs(captures_dir, exist_ok=True)

    def save_image(self, image, filename: Optional[str] = None) -> str:
        """
        保存图片

        Args:
            image: numpy 数组格式的图片
            filename: 文件名，None 则自动生成

        Returns:
            保存的文件路径
        """
        import cv2

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"

        # 保存到本地
        local_path = os.path.join(self.captures_dir, filename)
        cv2.imwrite(local_path, image)
        logger.info(f"图片已保存: {local_path}")

        # 尝试同步到外部存储
        self._sync_to_external(filename)

        # 清理旧文件
        self._cleanup_old_files()

        return local_path

    def save_video(self, video_path: str, filename: Optional[str] = None) -> str:
        """
        保存视频

        Args:
            video_path: 视频文件路径
            filename: 目标文件名

        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(video_path)[1]
            filename = f"video_{timestamp}{ext}"

        # 复制到本地
        local_path = os.path.join(self.captures_dir, filename)
        shutil.copy2(video_path, local_path)
        logger.info(f"视频已保存: {local_path}")

        # 尝试同步到外部存储
        self._sync_to_external(filename)

        return local_path

    def _sync_to_external(self, filename: str):
        """同步文件到外部存储"""
        # 尝试 SD 卡
        if os.path.exists(self.sd_card_path):
            try:
                dest = os.path.join(self.sd_card_path, filename)
                src = os.path.join(self.captures_dir, filename)
                shutil.copy2(src, dest)
                logger.info(f"已同步到 SD 卡: {dest}")
                return
            except Exception as e:
                logger.warning(f"同步到 SD 卡失败: {e}")

        # 尝试 USB
        if os.path.exists(self.usb_path):
            try:
                dest = os.path.join(self.usb_path, filename)
                src = os.path.join(self.captures_dir, filename)
                shutil.copy2(src, dest)
                logger.info(f"已同步到 USB: {dest}")
                return
            except Exception as e:
                logger.warning(f"同步到 USB 失败: {e}")

    def _cleanup_old_files(self):
        """清理旧文件，保持文件数在限制内"""
        try:
            files = sorted(
                [f for f in os.listdir(self.captures_dir)
                 if os.path.isfile(os.path.join(self.captures_dir, f))],
                key=lambda f: os.path.getmtime(os.path.join(self.captures_dir, f))
            )

            if len(files) > self.max_captures:
                # 删除最旧的文件
                files_to_remove = files[:len(files) - self.max_captures]
                for f in files_to_remove:
                    filepath = os.path.join(self.captures_dir, f)
                    os.remove(filepath)
                    logger.debug(f"已删除旧文件: {filepath}")

        except Exception as e:
            logger.warning(f"清理旧文件失败: {e}")

    def get_available_storage(self) -> dict:
        """获取可用存储信息"""
        info = {
            "local": self._get_disk_info(self.captures_dir),
            "sd_card": None,
            "usb": None
        }

        if os.path.exists(self.sd_card_path):
            info["sd_card"] = self._get_disk_info(self.sd_card_path)

        if os.path.exists(self.usb_path):
            info["usb"] = self._get_disk_info(self.usb_path)

        return info

    def _get_disk_info(self, path: str) -> Optional[dict]:
        """获取磁盘信息"""
        try:
            usage = shutil.disk_usage(path)
            return {
                "total_gb": usage.total / (1024 ** 3),
                "used_gb": usage.used / (1024 ** 3),
                "free_gb": usage.free / (1024 ** 3),
                "percent": (usage.used / usage.total) * 100
            }
        except Exception as e:
            logger.warning(f"获取磁盘信息失败: {e}")
            return None

    def get_capture_count(self) -> int:
        """获取当前捕获文件数量"""
        try:
            return len([
                f for f in os.listdir(self.captures_dir)
                if os.path.isfile(os.path.join(self.captures_dir, f))
            ])
        except Exception:
            return 0

    def export_all(self, target_path: str) -> bool:
        """
        导出所有捕获文件到指定路径

        Args:
            target_path: 目标路径

        Returns:
            是否成功
        """
        try:
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)

            # 复制所有文件
            for filename in os.listdir(self.captures_dir):
                src = os.path.join(self.captures_dir, filename)
                if os.path.isfile(src):
                    dst = os.path.join(target_path, filename)
                    shutil.copy2(src, dst)

            logger.info(f"已导出到: {target_path}")
            return True

        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
