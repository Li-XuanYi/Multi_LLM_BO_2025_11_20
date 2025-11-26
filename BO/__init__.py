"""
BO Package - 贝叶斯优化主包
统一管理所有子模块的导入

Author: Research Team
Date: 2025-01-19
"""

# 导出核心模块
from .llmbo_core import (
    MultiObjectiveEvaluator,
    LLMEnhancedBO,
    LLMEnhancedEI,
    LLMEnhancedMultiObjectiveBO,
    SPM_Sensitivity,
    PyBaMMSensitivityComputer
)

__all__ = [
    'MultiObjectiveEvaluator',
    'LLMEnhancedBO',
    'LLMEnhancedEI',
    'LLMEnhancedMultiObjectiveBO',
    'SPM_Sensitivity',
    'PyBaMMSensitivityComputer'
]

__version__ = '2.0.0'
