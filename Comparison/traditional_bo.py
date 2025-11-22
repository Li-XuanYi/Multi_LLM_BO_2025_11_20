"""
Traditional Bayesian Optimization Wrapper
传统贝叶斯优化算法包装器

使用bayes_opt库实现标准BO，作为LLMBO的baseline对比

关键特性：
- 标准高斯过程代理模型
- Expected Improvement采集函数
- 无LLM增强
- 公平对比设置

Author: Research Team
Date: 2025-01-19
"""

import numpy as np
import time
from typing import Dict, List
from base_optimizer import BaseOptimizer, OptimizerFactory
from BO.bayes_opt.bayesian_optimization import BayesianOptimization


class TraditionalBO(BaseOptimizer):
    """
    传统贝叶斯优化包装器
    
    使用bayes_opt库实现，无任何LLM增强
    """
    
    def __init__(
        self,
        evaluator,
        pbounds: Dict[str, tuple],
        random_state: int = 42,
        verbose: bool = True,
        acquisition_kappa: float = 2.576
    ):
        """
        初始化Traditional BO
        
        参数：
            evaluator: MultiObjectiveEvaluator实例
            pbounds: 参数边界
            random_state: 随机种子
            verbose: 是否打印详细信息
            acquisition_kappa: UCB参数（如果使用UCB采集函数）
        """
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
        print(f"  总评估次数: {stats['total_evaluations']}")
        print(f"  有效评估: {stats['valid_evaluations']}")
        print(f"  运行时间: {results['elapsed_time']:.1f} 秒")
        
        print("\n" + "=" * 70)


# 注册到工厂
OptimizerFactory.register('BO', TraditionalBO)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("测试 Traditional BO Wrapper")
    print("=" * 70)
    
    # 注意：这里只是结构测试，不运行实际优化
    print("\n✓ TraditionalBO 类已创建")
    print("✓ 已注册到 OptimizerFactory")
    print("\n方法:")
    print("  - optimize(n_iterations, n_random_init)")
    print("  - get_history()")
    print("  - reset()")
    
    print("\n" + "=" * 70)
    print("准备就绪，可与MultiObjectiveEvaluator集成测试")
    print("=" * 70)