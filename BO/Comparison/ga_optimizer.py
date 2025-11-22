"""
Genetic Algorithm (GA) Optimizer Wrapper
遗传算法优化器包装器

使用DEAP库实现标准遗传算法，作为baseline对比

算法参数：
- 种群大小: 50
- 交叉率: 0.8
- 变异率: 0.2
- 选择方法: 锦标赛选择
- 交叉方法: 模拟二进制交叉(SBX)
- 变异方法: 多项式变异

Author: Research Team
Date: 2025-01-19
"""

import numpy as np
import time
import random
from typing import Dict, List
from base_optimizer import BaseOptimizer, OptimizerFactory

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("警告: DEAP库未安装，请运行: pip install deap")


class GeneticAlgorithm(BaseOptimizer):
    """
    遗传算法优化器包装器
    
    使用DEAP库实现标准GA
    """
    
    def __init__(
        self,
        evaluator,
        pbounds: Dict[str, tuple],
        random_state: int = 42,
        verbose: bool = True,
        population_size: int = 50,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        tournament_size: int = 3
    ):
        """
        初始化GA优化器
        
        参数：
            evaluator: MultiObjectiveEvaluator实例
            pbounds: 参数边界
            random_state: 随机种子
            verbose: 是否打印详细信息
            population_size: 种群大小
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            tournament_size: 锦标赛大小
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP库未安装，请运行: pip install deap")
        
        super().__init__(evaluator, pbounds, random_state, verbose)
        
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
        # 设置DEAP随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 设置DEAP框架
        self._setup_deap()
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("Genetic Algorithm 已初始化")
            print("=" * 70)
            print(f"种群大小: {population_size}")
            print(f"交叉概率: {crossover_prob}")
            print(f"变异概率: {mutation_prob}")
            print(f"锦标赛大小: {tournament_size}")
            print(f"参数边界: {pbounds}")
            print("=" * 70)
    
    def _setup_deap(self):
        """设置DEAP框架"""
        # 清除之前的定义（如果有）
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # 创建适应度类（最小化）
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        
        # 创建个体类
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # 创建工具箱
        self.toolbox = base.Toolbox()
        
        # 注册参数生成函数
        param_names = list(self.pbounds.keys())
        for i, param_name in enumerate(param_names):
            low, high = self.pbounds[param_name]
            self.toolbox.register(
                f"attr_{i}",
                random.uniform,
                low,
                high
            )
        
        # 注册个体和种群生成函数
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [getattr(self.toolbox, f"attr_{i}") for i in range(len(param_names))],
            n=1
        )
        
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # 注册遗传算子
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                              low=[self.pbounds[k][0] for k in param_names],
                              up=[self.pbounds[k][1] for k in param_names],
                              eta=20.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                              low=[self.pbounds[k][0] for k in param_names],
                              up=[self.pbounds[k][1] for k in param_names],
                              eta=20.0,
                              indpb=1.0/len(param_names))
        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.tournament_size)
    
    def _evaluate_individual(self, individual: List[float]) -> tuple:
        """
        评估单个个体
        
        参数：
            individual: 个体（参数向量）
        
        返回：
            (fitness,): 适应度元组
        """
        param_names = list(self.pbounds.keys())
        params = {
            param_names[i]: individual[i]
            for i in range(len(param_names))
        }
        
        # 确保charging_number是整数
        params['charging_number'] = int(round(params['charging_number']))
        
        # 评估
        scalarized = self._objective_function(**params)
        
        return (scalarized,)
    
    def optimize(
        self,
        n_iterations: int = 50,
        n_random_init: int = 10
    ) -> Dict:
        """
        执行GA优化
        
        参数：
            n_iterations: 总代数
            n_random_init: 初始种群大小（使用population_size）
        
        返回：
            results: 标准化结果字典
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"开始Genetic Algorithm优化")
            print(f"初始种群: {self.population_size} 个体")
            print(f"进化代数: {n_iterations}")
            print("=" * 70)
        
        # 创建初始种群
        population = self.toolbox.population(n=self.population_size)
        
        # 评估初始种群
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # 记录初始种群
        param_names = list(self.pbounds.keys())
        for i, ind in enumerate(population):
            params = {
                param_names[j]: ind[j]
                for j in range(len(param_names))
            }
            params['charging_number'] = int(round(params['charging_number']))
            
            self.optimization_history.append({
                'iteration': 0,
                'params': params,
                'scalarized': ind.fitness.values[0],
                'source': 'initial_population'
            })
        
        # 进化循环
        for gen in range(1, n_iterations + 1):
            # 选择
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # 变异
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 评估需要重新计算适应度的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 记录新个体
            for ind in invalid_ind:
                params = {
                    param_names[j]: ind[j]
                    for j in range(len(param_names))
                }
                params['charging_number'] = int(round(params['charging_number']))
                
                self.optimization_history.append({
                    'iteration': gen,
                    'params': params,
                    'scalarized': ind.fitness.values[0],
                    'source': 'ga_offspring'
                })
            
            # 更新种群
            population[:] = offspring
            
            # 打印进度
            if self.verbose and gen % 10 == 0:
                fits = [ind.fitness.values[0] for ind in population]
                best_fit = min(fits)
                current_best = self._get_best_solution()
                
                print(f"\n[代数 {gen}/{n_iterations}]")
                print(f"  种群最优: {best_fit:.4f}")
                print(f"  全局最优: {current_best['scalarized']:.4f}")
        
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
        print("GA优化完成！")
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
if DEAP_AVAILABLE:
    OptimizerFactory.register('GA', GeneticAlgorithm)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("测试 Genetic Algorithm Wrapper")
    print("=" * 70)
    
    if DEAP_AVAILABLE:
        print("\n✓ GeneticAlgorithm 类已创建")
        print("✓ 已注册到 OptimizerFactory")
        print("\n方法:")
        print("  - optimize(n_iterations)")
        print("  - get_history()")
        print("  - reset()")
        
        print("\n算法参数:")
        print("  - 种群大小: 50")
        print("  - 交叉概率: 0.8")
        print("  - 变异概率: 0.2")
        print("  - 选择: 锦标赛选择(size=3)")
        print("  - 交叉: SBX")
        print("  - 变异: 多项式变异")
    else:
        print("\n✗ DEAP库未安装")
        print("  请运行: pip install deap")
    
    print("\n" + "=" * 70)