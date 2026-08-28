"""风格预设管理器

管理 GAN 风格模型配置，每个风格对应独立的 ONNX 模型文件
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Style:
    """风格配置"""
    id: int
    name: str
    icon: str
    base_model: str              # ONNX 模型文件名（相对于 models/）
    model_url: str = ""          # 模型下载地址
    input_size: int = 512        # 模型输入尺寸
    normalize: str = "minus_one_to_one"  # 归一化方式


class StyleManager:
    """风格预设管理器

    从 config/styles.json 加载风格配置
    每个风格对应独立的 GAN ONNX 模型
    """

    def __init__(
        self,
        config_path: str = "config/styles.json",
        models_dir: str = "models"
    ):
        """
        初始化风格管理器

        Args:
            config_path: 风格配置文件路径
            models_dir: 模型基础目录
        """
        self.config_path = config_path
        self.models_dir = models_dir
        self.styles: Dict[int, Style] = {}
        self.settings: dict = {}
        self.current_style_id: Optional[int] = None

        self._load_config()

    def _load_config(self):
        """加载风格配置"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}")
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            for style_data in config.get('styles', []):
                style = Style(
                    id=style_data['id'],
                    name=style_data['name'],
                    icon=style_data.get('icon', ''),
                    base_model=style_data['base_model'],
                    model_url=style_data.get('model_url', ''),
                    input_size=style_data.get('input_size', 512),
                    normalize=style_data.get('normalize', 'minus_one_to_one'),
                )
                self.styles[style.id] = style

            self.settings = config.get('settings', {})
            default_id = self.settings.get('default_style_id', 1)
            if default_id in self.styles:
                self.current_style_id = default_id

            logger.info(f"已加载 {len(self.styles)} 种风格配置")

        except Exception as e:
            logger.error(f"加载风格配置失败: {e}")

    def get_style(self, style_id: int) -> Optional[Style]:
        """获取指定风格"""
        return self.styles.get(style_id)

    def get_all_styles(self) -> List[Style]:
        """获取所有风格"""
        return list(self.styles.values())

    def get_current_style(self) -> Optional[Style]:
        """获取当前风格"""
        if self.current_style_id is not None:
            return self.styles.get(self.current_style_id)
        return None

    def set_current_style(self, style_id: int) -> bool:
        """设置当前风格"""
        if style_id in self.styles:
            self.current_style_id = style_id
            logger.info(f"当前风格: {self.styles[style_id].name}")
            return True
        logger.warning(f"风格 {style_id} 不存在")
        return False

    def resolve_model_path(self, style: Style) -> str:
        """
        解析模型本地路径

        Args:
            style: 风格配置

        Returns:
            模型本地绝对路径
        """
        # 绝对路径直接返回
        if os.path.isabs(style.base_model):
            return style.base_model

        # 相对于 models_dir
        return os.path.join(self.models_dir, style.base_model)

    def get_model_url(self, style: Style) -> str:
        """
        获取模型下载地址

        Args:
            style: 风格配置

        Returns:
            模型下载 URL，如果没有配置则返回空字符串
        """
        return style.model_url

    def validate_models(self) -> Dict[int, bool]:
        """
        验证所有风格的模型是否存在

        Returns:
            {style_id: exists}
        """
        results = {}
        for style_id, style in self.styles.items():
            model_path = self.resolve_model_path(style)
            exists = os.path.exists(model_path)
            results[style_id] = exists
            if not exists:
                logger.warning(f"模型不存在: {style.name} -> {model_path}")
        return results

    def get_unique_models(self) -> Dict[str, List[int]]:
        """
        获取去重后的模型列表

        Returns:
            {model_path: [style_ids]}
        """
        model_map: Dict[str, List[int]] = {}
        for style in self.styles.values():
            path = self.resolve_model_path(style)
            if path not in model_map:
                model_map[path] = []
            model_map[path].append(style.id)
        return model_map
