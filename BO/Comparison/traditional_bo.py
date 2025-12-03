"""
Traditional Bayesian Optimization Wrapper - UNIFIED VERSION
传统贝叶斯优化算法包装器（统一物理内核版）

修复内容：
1. ✅ 统一导入路径
2. ✅ 确保注册代码在模块级别执行
3. ✅ 移除 if __name__ 条件
4. ✅ 统一使用 MultiObjectiveEvaluator (SPM_v3)

Author: Research Team
Date: 2025-12-02 (统一内核版)
"""

import numpy as np
import time
from typing import Dict, List

# ✅ 修复1: 使用统一的导入路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Comparison.base_optimizer import BaseOptimizer, OptimizerFactory
from llmbo_core.multi_objective_evaluator import MultiObjectiveEvaluator
from bayes_opt import BayesianOptimization


class ScalarOnlyEvaluatorWrapper:
    """
    标量值包装器：将MultiObjectiveEvaluator包装为只返回标量的评估器
    
    用于传统算法（BO/GA/PSO），统一使用SPM_v3物理内核，但忽略梯度信息
    """
    
    def __init__(
        self,
        base_evaluator: MultiObjectiveEvaluator,
        verbose: bool = False
    ):
        """
        初始化包装器
        
        参数：
            base_evaluator: 底层MultiObjectiveEvaluator实例（使用SPM_v3）
            verbose: 是否打印详细日志
        """
        self.base_evaluator = base_evaluator
        self.verbose = verbose
        
        if self.verbose:
            print(f"[ScalarOnlyWrapper] 已包装MultiObjectiveEvaluator（使用SPM_v3）")
            print(f"[ScalarOnlyWrapper] 权重: {base_evaluator.weights}")
    
    @property
    def weights(self):
        """获取权重（代理到基础评估器）"""
        return self.base_evaluator.weights
    
    def evaluate(
        self,
        current1: float,
        charging_number: int,
        current2: float
    ) -> float:
        """
        评估充电策略（只返回标量化值，忽略梯度）
        
        参数：
            current1: 第一阶段电流[A]
            charging_number: 第一阶段步数
            current2: 第二阶段电流[A]
        
        返回：
            scalarized: 标量化目标值（越小越好）
        """
        # 调用基础评估器（内部使用SPM_v3）
        # MultiObjectiveEvaluator.evaluate() 直接返回标量值
        scalarized = self.base_evaluator.evaluate(
            current1=current1,
            charging_number=int(charging_number),
            current2=current2
        )
        
        # 直接返回标量值
        return scalarized
    
    def get_best_solution(self) -> Dict:
        """获取最优解（代理到基础评估器）"""
        return self.base_evaluator.get_best_solution()
    
    def export_database(self) -> List[Dict]:
        """导出评估历史（代理到基础评估器）"""
        return self.base_evaluator.export_database()
    
    def get_statistics(self) -> Dict:
        """获取统计信息（代理到基础评估器）"""
        return self.base_evaluator.get_statistics()


class TraditionalBO(BaseOptimizer):
    """
    传统贝叶斯优化包装器
    
    使用bayes_opt库实现，无任何LLM增强
    使用MultiObjectiveEvaluator（SPM_v3）但仅使用标量值
    """
    
    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        pbounds: Dict[str, tuple],
        random_state: int = 42,
        verbose: bool = True,
        acquisition_kappa: float = 2.576
    ):
        """
        初始化Traditional BO
        
        参数：
            evaluator: MultiObjectiveEvaluator实例（使用SPM_v3）
            pbounds: 参数边界
            random_state: 随机种子
            verbose: 是否打印详细信息
            acquisition_kappa: UCB参数（如果使用UCB采集函数）
        """
        # 包装evaluator为标量值接口
        wrapped_evaluator = ScalarOnlyEvaluatorWrapper(
            base_evaluator=evaluator,
            verbose=verbose
        )
        
        super().__init__(wrapped_evaluator, pbounds, random_state, verbose)
        
        self.acquisition_kappa = acquisition_kappa
        
        if self.verbose:
            print("[Traditional BO] 使用SPM_v3（通过包装器，只使用标量值）")
        
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
            print(f"SPM版本: v3.0 (统一物理内核)")
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