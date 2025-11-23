"""
LLMBO Core Module
LLM增强的多目标贝叶斯优化核心模块

使用懒加载避免不必要的导入延迟
"""

__version__ = '1.0.0'

# ============================================================
# 懒加载实现 - 只在实际使用时才导入模块
# ============================================================

def __getattr__(name):
    """
    懒加载：只在访问时才导入对应模块
    这避免了导入 llmbo_core 时加载所有子模块
    """
    
    # 定义模块映射
    _module_map = {
        'SPM_Sensitivity': '.SPM',
        'PyBaMMSensitivityComputer': '.PybammSensitivity',
        'MultiObjectiveEvaluator': '.multi_objective_evaluator',
        'LLMEnhancedBO': '.LLM_enhanced_surrogate_modeling',
        'LLMEnhancedEI': '.LLM_Enhanced_Expected_Improvement',
    }
    
    # 常规模块
    if name in _module_map:
        import importlib
        module = importlib.import_module(_module_map[name], package='llmbo_core')
        globals()[name] = getattr(module, name)
        return globals()[name]
    
    # 特殊处理：带连字符的模块
    if name == 'LLMEnhancedMultiObjectiveBO':
        import importlib
        module = importlib.import_module(
            '.LLM_Enhanced_Multi-Objective_Bayesian_Optimization', 
            package='llmbo_core'
        )
        globals()[name] = module.LLMEnhancedMultiObjectiveBO
        return globals()[name]
    
    raise AttributeError(f"module 'llmbo_core' has no attribute '{name}'")


# 定义 __all__ 以支持 from llmbo_core import *
__all__ = [
    'SPM_Sensitivity',
    'PyBaMMSensitivityComputer',
    'MultiObjectiveEvaluator',
    'LLMEnhancedBO',
    'LLMEnhancedEI',
    'LLMEnhancedMultiObjectiveBO'
]


# ============================================================
# 可选：提供便捷的直接导入函数（用于需要多个类的场景）
# ============================================================

def load_all():
    """
    显式加载所有模块（用于需要全部功能的场景）
    
    用法:
        from llmbo_core import load_all
        load_all()
    """
    for name in __all__:
        globals()[name] = __getattr__(name)
    print("✓ 所有 llmbo_core 模块已加载")