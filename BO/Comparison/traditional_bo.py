"""
Traditional Bayesian Optimization Wrapper - FIXED VERSION
传统贝叶斯优化算法包装器（修复版）

修复内容：
1. ✅ 统一导入路径
2. ✅ 确保注册代码在模块级别执行
3. ✅ 移除 if __name__ 条件
4. ✅ 使用旧版SPM（无自动微分优化）

Author: Research Team
Date: 2025-01-19 (修复版)
"""

import numpy as np
import time
from typing import Dict, List

# ✅ 修复1: 使用统一的导入路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Comparison.base_optimizer import BaseOptimizer, OptimizerFactory
from bayes_opt import BayesianOptimization

# 导入旧版SPM（没有自动微分优化）
try:
    from llmbo_core.SPM import SPM_Sensitivity as SPM_Legacy
except ImportError:
    try:
        from BO.llmbo_core.SPM import SPM_Sensitivity as SPM_Legacy
    except ImportError:
        print("警告: 无法导入旧版SPM，将使用新版SPM_v3")
        try:
            from llmbo_core.SPM_v3 import SPM_Sensitivity as SPM_Legacy
        except ImportError:
            from BO.llmbo_core.SPM_v3 import SPM_Sensitivity as SPM_Legacy


class LegacyEvaluator:
    """
    使用旧版SPM的评估器（用于传统BO对比）
    简化版评估器，只提供基本的评估功能，不使用新的自动微分优化
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        temp_max: float = 309.0,
        max_steps: int = 300,
        verbose: bool = False
    ):
        """
        初始化旧版评估器
        
        参数：
            weights: 各目标权重
            temp_max: 温度约束上限[K]
            max_steps: 单次充电最大步数限制
            verbose: 是否打印详细日志
        """
        self.weights = weights or {
            'time': 0.4,
            'temp': 0.35,
            'aging': 0.25
        }
        self.temp_max = temp_max
        self.max_steps = max_steps
        self.verbose = verbose
        
        # 简化的历史记录
        self.history = {
            'time': [],
            'temp': [],
            'aging': [],
            'valid': [],
            'params': [],
            'scalarized': []
        }
        
        self.eval_count = 0
        
        # 临时固定边界（用于归一化）
        self.bounds = {
            'time': {'best': 10, 'worst': 150},
            'temp': {'best': 298.0, 'worst': temp_max},
            'aging': {'best': 0.0, 'worst': 0.5}
        }
        
        if self.verbose:
            print(f"[Legacy Evaluator] 初始化完成，使用旧版SPM（无自动微分）")
    
    def evaluate(
        self,
        current1: float,
        charging_number: int,
        current2: float
    ) -> float:
        """
        评估充电策略（返回标量化值）
        
        参数：
            current1: 第一阶段电流[A]
            charging_number: 第一阶段步数
            current2: 第二阶段电流[A]
        
        返回：
            scalarized: 标量化目标值（越小越好）
        """
        self.eval_count += 1
        
        # 运行充电仿真（使用旧版SPM）
        result = self._run_charging_simulation(current1, charging_number, current2)
        
        if not result['valid']:
            # 约束违反：返回惩罚值
            scalarized = 1e6
        else:
            # 归一化
            normalized = self._normalize_objectives(result['objectives'])
            
            # 标量化（增强切比雪夫）
            scalarized = self._scalarize(normalized)
        
        # 记录历史
        params = {
            'current1': current1,
            'charging_number': charging_number,
            'current2': current2
        }
        
        self.history['time'].append(result['objectives']['time'])
        self.history['temp'].append(result['objectives']['temp'])
        self.history['aging'].append(result['objectives']['aging'])
        self.history['valid'].append(result['valid'])
        self.history['params'].append(params)
        self.history['scalarized'].append(scalarized)
        
        return scalarized
    
    def _run_charging_simulation(
        self,
        current1: float,
        charging_number: int,
        current2: float
    ) -> Dict:
        """
        运行充电仿真（使用旧版SPM）
        """
        try:
            # 创建旧版SPM实例
            env = SPM_Legacy(init_v=3.0, init_t=298, enable_sensitivities=False)
            
            # 运行两阶段充电
            result = env.run_two_stage_charging(
                current1=current1,
                charging_number=int(charging_number),
                current2=current2,
                return_sensitivities=False  # 不计算梯度
            )
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"[警告] 仿真失败: {e}")
            
            return {
                'objectives': {'time': 1e6, 'temp': 1e6, 'aging': 1e6},
                'valid': False
            }
    
    def _normalize_objectives(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """归一化目标值到[0,1]"""
        normalized = {}
        
        for obj_name in ['time', 'temp', 'aging']:
            value = objectives[obj_name]
            best = self.bounds[obj_name]['best']
            worst = self.bounds[obj_name]['worst']
            
            # 线性归一化
            normalized[obj_name] = (value - best) / (worst - best)
            
            # 裁剪到[0,1]
            normalized[obj_name] = np.clip(normalized[obj_name], 0.0, 1.0)
        
        return normalized
    
    def _scalarize(self, normalized: Dict[str, float]) -> float:
        """增强切比雪夫标量化"""
        weighted_objectives = []
        
        for obj_name in ['time', 'temp', 'aging']:
            weight = self.weights[obj_name]
            value = normalized[obj_name]
            weighted_objectives.append(weight * value)
        
        # 切比雪夫：max + 0.05*sum
        tcheby = np.max(weighted_objectives)
        augment = 0.05 * np.sum(weighted_objectives)
        
        return tcheby + augment
    
    def get_best_solution(self) -> Dict:
        """获取最优解"""
        if not self.history['scalarized']:
            return None
        
        best_idx = np.argmin(self.history['scalarized'])
        
        return {
            'params': self.history['params'][best_idx],
            'objectives': {
                'time': self.history['time'][best_idx],
                'temp': self.history['temp'][best_idx],
                'aging': self.history['aging'][best_idx]
            },
            'scalarized': self.history['scalarized'][best_idx],
            'valid': self.history['valid'][best_idx]
        }
    
    def export_database(self) -> List[Dict]:
        """导出评估历史"""
        database = []
        
        for i in range(len(self.history['scalarized'])):
            database.append({
                'eval_id': i,
                'params': self.history['params'][i],
                'objectives': {
                    'time': self.history['time'][i],
                    'temp': self.history['temp'][i],
                    'aging': self.history['aging'][i]
                },
                'scalarized': self.history['scalarized'][i],
                'valid': self.history['valid'][i]
            })
        
        return database
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.history['scalarized']:
            return {}
        
        valid_indices = [i for i, v in enumerate(self.history['valid']) if v]
        
        if not valid_indices:
            return {
                'n_evaluations': self.eval_count,
                'n_valid': 0,
                'n_invalid': self.eval_count
            }
        
        valid_scalarized = [self.history['scalarized'][i] for i in valid_indices]
        
        return {
            'n_evaluations': self.eval_count,
            'n_valid': len(valid_indices),
            'n_invalid': self.eval_count - len(valid_indices),
            'best_scalarized': float(np.min(valid_scalarized)),
            'mean_scalarized': float(np.mean(valid_scalarized)),
            'std_scalarized': float(np.std(valid_scalarized))
        }


class TraditionalBO(BaseOptimizer):
    """
    传统贝叶斯优化包装器
    
    使用bayes_opt库实现，无任何LLM增强
    使用旧版SPM（无自动微分优化）以确保公平对比
    """
    
    def __init__(
        self,
        evaluator,
        pbounds: Dict[str, tuple],
        random_state: int = 42,
        verbose: bool = True,
        acquisition_kappa: float = 2.576,
        use_legacy_spm: bool = True  # 新增：是否使用旧版SPM
    ):
        """
        初始化Traditional BO
        
        参数：
            evaluator: MultiObjectiveEvaluator实例（如果use_legacy_spm=False）
            pbounds: 参数边界
            random_state: 随机种子
            verbose: 是否打印详细信息
            acquisition_kappa: UCB参数（如果使用UCB采集函数）
            use_legacy_spm: 是否使用旧版SPM（推荐True，确保公平对比）
        """
        # 如果使用旧版SPM，替换evaluator
        if use_legacy_spm:
            original_weights = getattr(evaluator, 'weights', {
                'time': 0.4, 'temp': 0.35, 'aging': 0.25
            })
            evaluator = LegacyEvaluator(
                weights=original_weights,
                verbose=verbose
            )
            if verbose:
                print("[Traditional BO] 使用旧版SPM（无自动微分优化）")
        
        super().__init__(evaluator, pbounds, random_state, verbose)
        
        self.acquisition_kappa = acquisition_kappa
        
        # 创建优化器实例
        # 注意：bayes_opt是最大化，我们需要包装成最小化
        self.optimizer = BayesianOptimization(
            f=None,  # 我们会手动调用probe
            pbounds=pbounds,
            random_state=random_state,
            verbose=0,  # 关闭内部输出，我们自己控制
            allow_duplicate_points=False
        )
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("Traditional Bayesian Optimization 已初始化")
            print("=" * 70)
            print(f"参数边界: {pbounds}")
            print(f"随机种子: {random_state}")
            print(f"采集函数: Expected Improvement")
            print(f"SPM版本: {'Legacy (v2.1)' if use_legacy_spm else 'v3.0 (Auto-Diff)'}")
            print("=" * 70)
    
    def optimize(
        self,
        n_iterations: int = 50,
        n_random_init: int = 10
    ) -> Dict:
        """
        执行传统BO优化
        
        参数：
            n_iterations: 总迭代次数
            n_random_init: 随机初始化点数量
        
        返回：
            results: 标准化结果字典
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"开始Traditional BO优化")
            print(f"随机初始化: {n_random_init} 点")
            print(f"BO迭代: {n_iterations} 次")
            print("=" * 70)
        
        # 阶段1: 随机初始化
        if self.verbose:
            print(f"\n阶段1: 随机初始化 ({n_random_init} 点)")
        
        for i in range(n_random_init):
            # 生成随机参数
            params = {
                key: np.random.uniform(low, high)
                for key, (low, high) in self.pbounds.items()
            }
            # 确保charging_number是整数
            params['charging_number'] = int(params['charging_number'])
            
            # 评估
            scalarized = self._objective_function(**params)
            
            # 注册到bayes_opt（注意：转换为最大化）
            self.optimizer.register(
                params=params,
                target=-scalarized  # 负号：最小化→最大化
            )
            
            # 记录历史
            self.optimization_history.append({
                'iteration': i,
                'params': params,
                'scalarized': scalarized,
                'source': 'random_init'
            })
            
            if self.verbose and (i + 1) % 5 == 0:
                print(f"  随机点 {i+1}/{n_random_init}: f = {scalarized:.4f}")
        
        # 阶段2: BO迭代
        if self.verbose:
            print(f"\n阶段2: BO迭代 ({n_iterations} 次)")
        
        for i in range(n_iterations):
            # 使用BO建议下一个点
            next_point = self.optimizer.suggest()
            
            # 确保charging_number是整数
            next_point['charging_number'] = int(round(next_point['charging_number']))
            
            # 评估
            scalarized = self._objective_function(**next_point)
            
            # 注册到bayes_opt
            self.optimizer.register(
                params=next_point,
                target=-scalarized  # 负号：最小化→最大化
            )
            
            # 记录历史
            self.optimization_history.append({
                'iteration': n_random_init + i,
                'params': next_point,
                'scalarized': scalarized,
                'source': 'bo_iteration'
            })
            
            if self.verbose and (i + 1) % 10 == 0:
                current_best = self._get_best_solution()
                print(f"\n[进度] 完成 {i+1}/{n_iterations} 迭代")
                print(f"  当前最优: f = {current_best['scalarized']:.4f}")
                print(f"  I1={next_point['current1']:.2f}A, "
                      f"t1={next_point['charging_number']}, "
                      f"I2={next_point['current2']:.2f}A")
        
        elapsed_time = time.time() - start_time
        
        # 编译结果
        results = self._compile_results(elapsed_time)
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def get_history(self) -> List[Dict]:
        """
        获取标准化的优化历史
        
        返回：
            List[Dict]: 历史记录列表
        """
        # 从evaluator获取完整信息
        database = self.evaluator.export_database()
        
        # 标准化格式
        history = []
        for record in database:
            history.append({
                'iteration': record['eval_id'],
                'params': record['params'],
                'scalarized': record['scalarized'],
                'objectives': record['objectives'],
                'valid': record['valid']
            })
        
        return history
    
    def _print_summary(self, results: Dict):
        """打印优化摘要"""
        print("\n" + "=" * 70)
        print("优化完成！")
        print("=" * 70)
        
        best = results['best_solution']
        print(f"\n【最优解】")
        print(f"  充电策略:")
        print(f"    - I1: {best['params']['current1']:.2f} A")
        print(f"    - t1: {best['params']['charging_number']}")
        print(f"    - I2: {best['params']['current2']:.2f} A")
        print(f"\n  性能指标:")
        print(f"    - 充电时间: {best['objectives']['time']:.1f} 步")
        print(f"    - 峰值温度: {best['objectives']['temp']:.2f} K")
        print(f"    - 容量衰减: {best['objectives']['aging']:.6f} %")
        print(f"    - 标量化值: {best['scalarized']:.4f}")
        
        print(f"\n【统计信息】")
        stats = results['statistics']
        print(f"  总评估次数: {stats.get('n_evaluations', 'N/A')}")
        print(f"  有效评估: {stats.get('n_valid', 'N/A')}")
        print(f"  无效评估: {stats.get('n_invalid', 'N/A')}")
        print(f"  运行时间: {results['elapsed_time']:.1f} 秒")
        
        print("\n" + "=" * 70)


# ============================================================
# ✅ 修复2: 确保注册代码在模块级别执行（无条件）
# ============================================================

print(f"[traditional_bo.py] 正在注册 TraditionalBO...")
OptimizerFactory.register('BO', TraditionalBO)
print(f"[traditional_bo.py] ✓ TraditionalBO 已注册")
print(f"[traditional_bo.py] 当前注册表: {OptimizerFactory.available_optimizers()}")


# ============================================================
# 测试代码（仅在直接运行时执行）
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("测试 Traditional BO Wrapper")
    print("=" * 70)
    
    print("\n✓ TraditionalBO 类已创建")
    print("✓ 已注册到 OptimizerFactory")
    print(f"✓ 可用优化器: {OptimizerFactory.available_optimizers()}")
    print("\n方法:")
    print("  - optimize(n_iterations, n_random_init)")
    print("  - get_history()")
    print("  - reset()")
    
    print("\n" + "=" * 70)
    print("准备就绪，可与MultiObjectiveEvaluator集成测试")
    print("=" * 70)