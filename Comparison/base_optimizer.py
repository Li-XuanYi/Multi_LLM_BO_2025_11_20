"""
Base Optimizer Interface
统一优化器接口，用于公平对比不同优化算法

所有优化算法必须实现此接口以确保：
1. 统一的评估次数控制
2. 一致的历史记录格式
3. 可重复的随机性
4. 标准化的结果输出

Author: Research Team
Date: 2025-01-19
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class BaseOptimizer(ABC):
    """
    优化器基类
    
    所有优化算法（BO、GA、PSO等）必须继承此类并实现抽象方法
    """
    
    def __init__(
        self,
        evaluator,
        pbounds: Dict[str, tuple],
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        初始化优化器
        
        参数：
            evaluator: MultiObjectiveEvaluator实例
            pbounds: 参数边界 {'current1': (3.0, 6.0), ...}
            random_state: 随机种子
            verbose: 是否打印详细信息
        """
        self.evaluator = evaluator
        self.pbounds = pbounds
        self.random_state = random_state
        self.verbose = verbose
        
        # 历史记录
        self.optimization_history = []
        self.best_solution = None
        self.elapsed_time = 0.0
        
        # 设置随机种子
        np.random.seed(random_state)
    
    @abstractmethod
    def optimize(
        self,
        n_iterations: int = 50,
        n_random_init: int = 10
    ) -> Dict:
        """
        执行优化
        
        参数：
            n_iterations: 优化迭代次数
            n_random_init: 随机初始化点数量
        
        返回：
            results: {
                'best_solution': {...},
                'optimization_history': [...],
                'elapsed_time': float,
                'algorithm': str
            }
        """
        pass
    
    @abstractmethod
    def get_history(self) -> List[Dict]:
        """
        获取优化历史记录
        
        返回：
            List[Dict]: 标准化的历史记录列表
            [
                {
                    'iteration': int,
                    'params': {'current1': ..., 'charging_number': ..., 'current2': ...},
                    'scalarized': float,
                    'objectives': {'time': ..., 'temp': ..., 'aging': ...},
                    'valid': bool
                },
                ...
            ]
        """
        pass
    
    def _objective_function(self, **params) -> float:
        """
        统一的目标函数接口
        
        将参数传递给评估器并返回标量化值
        注意：所有算法都应最小化此函数
        """
        scalarized = self.evaluator.evaluate(**params)
        return scalarized
    
    def _get_best_solution(self) -> Dict:
        """从评估器获取当前最优解"""
        return self.evaluator.get_best_solution()
    
    def _compile_results(self, elapsed_time: float) -> Dict:
        """
        编译标准化的结果字典
        
        参数：
            elapsed_time: 运行时间（秒）
        
        返回：
            标准化结果字典
        """
        best_solution = self._get_best_solution()
        
        results = {
            'best_solution': best_solution,
            'optimization_history': self.get_history(),
            'elapsed_time': elapsed_time,
            'algorithm': self.__class__.__name__,
            'statistics': self.evaluator.get_statistics(),
            'config': {
                'random_state': self.random_state,
                'pbounds': self.pbounds
            }
        }
        
        return results
    
    def reset(self):
        """重置优化器状态（用于多次运行）"""
        self.optimization_history = []
        self.best_solution = None
        self.elapsed_time = 0.0
        
        # 注意：不重置evaluator，因为它可能在外部被重置
    
    def __repr__(self):
        return f"{self.__class__.__name__}(random_state={self.random_state})"


class OptimizerFactory:
    """优化器工厂类，用于创建不同的优化器实例"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, optimizer_class):
        """注册优化器类"""
        cls._registry[name] = optimizer_class
    
    @classmethod
    def create(
        cls,
        name: str,
        evaluator,
        pbounds: Dict[str, tuple],
        **kwargs
    ):
        """
        创建优化器实例
        
        参数：
            name: 优化器名称 ('BO', 'GA', 'PSO', 'LLMBO')
            evaluator: 评估器实例
            pbounds: 参数边界
            **kwargs: 额外的优化器特定参数
        
        返回：
            优化器实例
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown optimizer: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        return cls._registry[name](evaluator, pbounds, **kwargs)
    
    @classmethod
    def available_optimizers(cls) -> List[str]:
        """返回可用的优化器列表"""
        return list(cls._registry.keys())


if __name__ == "__main__":
    print("✓ Base Optimizer Interface 已创建")
    print(f"  可用方法: optimize(), get_history(), reset()")
    print(f"  工厂模式: OptimizerFactory.create()")