"""
Particle Swarm Optimization (PSO) Wrapper
粒子群优化算法包装器

使用pyswarms库实现标准PSO，作为baseline对比

算法参数：
- 粒子数: 30
- 惯性权重: 0.7
- 认知系数: 1.5
- 社会系数: 1.5
- 速度限制: 20% of search space

Author: Research Team
Date: 2025-01-19
"""

import numpy as np
import time
from typing import Dict, List
from base_optimizer import BaseOptimizer, OptimizerFactory

try:
    from pyswarms.single import GlobalBestPSO
    PYSWARMS_AVAILABLE = True
except ImportError:
    PYSWARMS_AVAILABLE = False
    print("警告: pyswarms库未安装，请运行: pip install pyswarms")


class ParticleSwarmOptimization(BaseOptimizer):
    """
    粒子群优化器包装器
    
    使用pyswarms库实现标准PSO
    """
    
    def __init__(
        self,
        evaluator,
        pbounds: Dict[str, tuple],
        random_state: int = 42,
        verbose: bool = True,
        n_particles: int = 30,
        w: float = 0.7,  # 惯性权重
        c1: float = 1.5,  # 认知系数
        c2: float = 1.5   # 社会系数
    ):
        """
        初始化PSO优化器
        
        参数：
            evaluator: MultiObjectiveEvaluator实例
            pbounds: 参数边界
            random_state: 随机种子
            verbose: 是否打印详细信息
            n_particles: 粒子数量
            w: 惯性权重
            c1: 认知系数（个体最优影响）
            c2: 社会系数（全局最优影响）
        """
        if not PYSWARMS_AVAILABLE:
            raise ImportError("pyswarms库未安装，请运行: pip install pyswarms")
        
        super().__init__(evaluator, pbounds, random_state, verbose)
        
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # 准备PSO参数
        self.param_names = list(pbounds.keys())
        self.bounds = (
            np.array([pbounds[k][0] for k in self.param_names]),  # lower
            np.array([pbounds[k][1] for k in self.param_names])   # upper
        )
        
        # 速度限制（搜索空间的20%）
        self.velocity_clamp = 0.2 * (self.bounds[1] - self.bounds[0])
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("Particle Swarm Optimization 已初始化")
            print("=" * 70)
            print(f"粒子数量: {n_particles}")
            print(f"惯性权重: {w}")
            print(f"认知系数: {c1}")
            print(f"社会系数: {c2}")
            print(f"速度限制: {self.velocity_clamp}")
            print(f"参数边界: {pbounds}")
            print("=" * 70)
    
    def _evaluate_swarm(self, positions: np.ndarray) -> np.ndarray:
        """
        评估粒子群
        
        参数：
            positions: (n_particles, n_dims) 粒子位置矩阵
        
        返回：
            costs: (n_particles,) 适应度数组
        """
        costs = []
        
        for position in positions:
            # 转换为参数字典
            params = {
                self.param_names[i]: float(position[i])
                for i in range(len(self.param_names))
            }
            
            # 确保charging_number是整数
            params['charging_number'] = int(round(params['charging_number']))
            
            # 评估
            scalarized = self._objective_function(**params)
            costs.append(scalarized)
        
        return np.array(costs)
    
    def optimize(
        self,
        n_iterations: int = 50,
        n_random_init: int = 10
    ) -> Dict:
        """
        执行PSO优化
        
        参数：
            n_iterations: 总迭代次数
            n_random_init: 初始粒子数（使用n_particles）
        
        返回：
            results: 标准化结果字典
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"开始Particle Swarm Optimization")
            print(f"初始粒子: {self.n_particles} 个")
            print(f"迭代次数: {n_iterations}")
            print("=" * 70)
        
        # PSO选项
        options = {
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w
        }
        
        # 创建PSO优化器
        # 注意：pyswarms默认就是最小化
        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(self.param_names),
            options=options,
            bounds=self.bounds,
            velocity_clamp=tuple(self.velocity_clamp.tolist())
        )
        
        # 创建评估函数的包装器（用于记录历史）
        iteration_counter = [0]  # 使用列表避免闭包问题
        
        def objective_with_logging(positions):
            """评估并记录历史"""
            costs = self._evaluate_swarm(positions)
            
            # 记录每个粒子的评估
            for i, (position, cost) in enumerate(zip(positions, costs)):
                params = {
                    self.param_names[j]: float(position[j])
                    for j in range(len(self.param_names))
                }
                params['charging_number'] = int(round(params['charging_number']))
                
                self.optimization_history.append({
                    'iteration': iteration_counter[0],
                    'params': params,
                    'scalarized': cost,
                    'source': f'pso_particle_{i}'
                })
            
            iteration_counter[0] += 1
            
            # 打印进度
            if self.verbose and iteration_counter[0] % 10 == 0:
                best_cost = np.min(costs)
                current_best = self._get_best_solution()
                
                print(f"\n[迭代 {iteration_counter[0]}/{n_iterations}]")
                print(f"  粒子群最优: {best_cost:.4f}")
                print(f"  全局最优: {current_best['scalarized']:.4f}")
            
            return costs
        
        # 运行优化
        try:
            cost, pos = optimizer.optimize(
                objective_with_logging,
                iters=n_iterations,
                verbose=False  # 我们自己控制输出
            )
        except Exception as e:
            print(f"PSO优化出错: {e}")
            raise
        
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
        print("PSO优化完成！")
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
        print(f"  运行时间: {results['elapsed_time']:.1f} 秒")
        
        print("\n" + "=" * 70)


# 注册到工厂
if PYSWARMS_AVAILABLE:
    OptimizerFactory.register('PSO', ParticleSwarmOptimization)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("测试 Particle Swarm Optimization Wrapper")
    print("=" * 70)
    
    if PYSWARMS_AVAILABLE:
        print("\n✓ ParticleSwarmOptimization 类已创建")
        print("✓ 已注册到 OptimizerFactory")
        print("\n方法:")
        print("  - optimize(n_iterations)")
        print("  - get_history()")
        print("  - reset()")
        
        print("\n算法参数:")
        print("  - 粒子数: 30")
        print("  - 惯性权重: 0.7")
        print("  - 认知系数: 1.5")
        print("  - 社会系数: 1.5")
        print("  - 速度限制: 20% of search space")
    else:
        print("\n✗ pyswarms库未安装")
        print("  请运行: pip install pyswarms")
    
    print("\n" + "=" * 70)