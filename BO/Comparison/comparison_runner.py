"""
Algorithm Comparison Runner
算法对比运行框架

执行公平对比实验：
- 运行15次重复实验
- 固定随机种子保证可重复性
- 标准化评估次数
- 收集详细统计数据

Author: Research Team
Date: 2025-01-19
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入评估器和优化器
# ✅ 新的导入方式
import sys
from pathlib import Path
# 将 BO 目录添加到 Python 路径（从 Comparison 目录往上一级就是 BO 目录）
sys.path.insert(0, str(Path(__file__).parent.parent))


from llmbo_core import MultiObjectiveEvaluator
from base_optimizer import OptimizerFactory

# 导入所有优化器（自动注册）
from traditional_bo import TraditionalBO
from ga_optimizer import GeneticAlgorithm
from pso_optimizer import ParticleSwarmOptimization


class ComparisonRunner:
    """
    算法对比运行器
    
    执行多次重复实验并收集统计数据
    """
    
    def __init__(
        self,
        algorithms: List[str],
        n_trials: int = 15,
        n_iterations: int = 50,
        n_random_init: int = 10,
        random_seed: int = 42,
        save_dir: str = './comparison_results',
        verbose: bool = True
    ):
        """
        初始化对比运行器
        
        参数：
            algorithms: 算法列表 ['BO', 'GA', 'PSO']
            n_trials: 重复实验次数
            n_iterations: 每次实验的迭代次数
            n_random_init: 随机初始化点数
            random_seed: 基础随机种子
            save_dir: 结果保存目录
            verbose: 是否打印详细信息
        """
        self.algorithms = algorithms
        self.n_trials = n_trials
        self.n_iterations = n_iterations
        self.n_random_init = n_random_init
        self.random_seed = random_seed
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        
        # 参数边界（与LLMBO一致）
        self.pbounds = {
            'current1': (3.0, 6.0),
            'charging_number': (5, 25),
            'current2': (1.0, 4.0)
        }
        
        # 目标权重（与LLMBO一致）
        self.objective_weights = {
            'time': 0.4,
            'temp': 0.35,
            'aging': 0.25
        }
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果存储
        self.all_results = {alg: [] for alg in algorithms}
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("Algorithm Comparison Runner 已初始化")
            print("=" * 80)
            print(f"对比算法: {algorithms}")
            print(f"重复次数: {n_trials} trials")
            print(f"迭代次数: {n_iterations} iterations per trial")
            print(f"随机初始化: {n_random_init} points")
            print(f"基础随机种子: {random_seed}")
            print(f"结果保存至: {self.save_dir}")
            print("=" * 80)
    
    def run_all_comparisons(self):
        """运行所有算法的对比实验"""
        start_time = time.time()
        
        for algorithm in self.algorithms:
            if self.verbose:
                print(f"\n{'=' * 80}")
                print(f"开始运行算法: {algorithm}")
                print(f"{'=' * 80}")
            
            self.run_algorithm_trials(algorithm)
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"所有对比实验完成！")
            print(f"总运行时间: {total_time:.1f} 秒")
            print(f"{'=' * 80}")
        
        # 保存结果
        self.save_results()
    
    def run_algorithm_trials(self, algorithm: str):
        """
        运行单个算法的多次实验
        
        参数：
            algorithm: 算法名称 ('BO', 'GA', 'PSO')
        """
        algorithm_results = []
        
        for trial in range(self.n_trials):
            if self.verbose:
                print(f"\n--- Trial {trial + 1}/{self.n_trials} ---")
            
            # 每次trial使用不同的随机种子
            trial_seed = self.random_seed + trial
            
            try:
                # 运行单次实验
                result = self.run_single_trial(
                    algorithm=algorithm,
                    trial_id=trial,
                    random_seed=trial_seed
                )
                
                algorithm_results.append(result)
                
                if self.verbose:
                    print(f"  ✓ Trial {trial + 1} 完成")
                    print(f"    最优值: {result['best_solution']['scalarized']:.4f}")
                    print(f"    运行时间: {result['elapsed_time']:.1f}s")
            
            except Exception as e:
                print(f"  ✗ Trial {trial + 1} 失败: {e}")
                continue
        
        # 存储结果
        self.all_results[algorithm] = algorithm_results
    
    def run_single_trial(
        self,
        algorithm: str,
        trial_id: int,
        random_seed: int
    ) -> Dict:
        """
        运行单次实验
        
        参数：
            algorithm: 算法名称
            trial_id: 实验ID
            random_seed: 随机种子
        
        返回：
            result: 实验结果字典
        """
        # 创建新的评估器（每次trial都重新创建）
        evaluator = MultiObjectiveEvaluator(
            weights=self.objective_weights,
            verbose=False  # 关闭评估器的输出
        )
        
        # 创建优化器
        optimizer = OptimizerFactory.create(
            name=algorithm,
            evaluator=evaluator,
            pbounds=self.pbounds,
            random_state=random_seed,
            verbose=False  # 关闭优化器的输出
        )
        
        # 运行优化
        result = optimizer.optimize(
            n_iterations=self.n_iterations,
            n_random_init=self.n_random_init
        )
        
        # 添加trial信息
        result['trial_id'] = trial_id
        result['random_seed'] = random_seed
        
        return result
    
    def save_results(self):
        """保存所有结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果（包含完整历史）
        detailed_file = self.save_dir / f"detailed_results_{timestamp}.json"
        
        # 准备可序列化的数据
        serializable_results = {}
        for alg, trials in self.all_results.items():
            serializable_results[alg] = []
            for trial in trials:
                # 移除不可序列化的对象
                trial_copy = trial.copy()
                if 'config' in trial_copy:
                    trial_copy.pop('config', None)
                serializable_results[alg].append(trial_copy)
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存: {detailed_file}")
        
        # 保存统计摘要
        summary_file = self.save_dir / f"summary_{timestamp}.json"
        summary = self.compute_summary_statistics()
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"统计摘要已保存: {summary_file}")
    
    def compute_summary_statistics(self) -> Dict:
        """计算统计摘要"""
        summary = {}
        
        for alg, trials in self.all_results.items():
            if not trials:
                continue
            
            # 提取最优值
            best_values = [t['best_solution']['scalarized'] for t in trials]
            
            # 提取运行时间
            run_times = [t['elapsed_time'] for t in trials]
            
            # 提取目标值
            objectives_time = [t['best_solution']['objectives']['time'] for t in trials]
            objectives_temp = [t['best_solution']['objectives']['temp'] for t in trials]
            objectives_aging = [t['best_solution']['objectives']['aging'] for t in trials]
            
            summary[alg] = {
                'scalarized': {
                    'best': float(np.min(best_values)),
                    'mean': float(np.mean(best_values)),
                    'std': float(np.std(best_values)),
                    'median': float(np.median(best_values)),
                    'all_values': best_values
                },
                'runtime': {
                    'mean': float(np.mean(run_times)),
                    'std': float(np.std(run_times)),
                    'all_values': run_times
                },
                'objectives': {
                    'time': {
                        'mean': float(np.mean(objectives_time)),
                        'std': float(np.std(objectives_time))
                    },
                    'temp': {
                        'mean': float(np.mean(objectives_temp)),
                        'std': float(np.std(objectives_temp))
                    },
                    'aging': {
                        'mean': float(np.mean(objectives_aging)),
                        'std': float(np.std(objectives_aging))
                    }
                },
                'n_trials': len(trials)
            }
        
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.compute_summary_statistics()
        
        print("\n" + "=" * 80)
        print("统计摘要（基于 {} 次运行）".format(self.n_trials))
        print("=" * 80)
        
        for alg in self.algorithms:
            if alg not in summary:
                continue
            
            stats = summary[alg]
            
            print(f"\n【{alg}】")
            print(f"  标量化目标值:")
            print(f"    Best:   {stats['scalarized']['best']:.4f}")
            print(f"    Mean:   {stats['scalarized']['mean']:.4f}")
            print(f"    Std:    {stats['scalarized']['std']:.4f}")
            print(f"    Median: {stats['scalarized']['median']:.4f}")
            
            print(f"  运行时间:")
            print(f"    Mean:   {stats['runtime']['mean']:.1f}s")
            print(f"    Std:    {stats['runtime']['std']:.1f}s")
            
            print(f"  目标值（平均）:")
            print(f"    Time:   {stats['objectives']['time']['mean']:.1f} ± {stats['objectives']['time']['std']:.1f}")
            print(f"    Temp:   {stats['objectives']['temp']['mean']:.2f} ± {stats['objectives']['temp']['std']:.2f}")
            print(f"    Aging:  {stats['objectives']['aging']['mean']:.6f} ± {stats['objectives']['aging']['std']:.6f}")
        
        print("\n" + "=" * 80)


# ============================================================
# 主函数 - 运行对比实验
# ============================================================

def main():
    """运行完整的对比实验"""
    
    # 配置
    algorithms = ['BO', 'GA', 'PSO']  # 可以添加'LLMBO'
    n_trials = 15
    n_iterations = 50
    n_random_init = 10
    
    # 创建运行器
    runner = ComparisonRunner(
        algorithms=algorithms,
        n_trials=n_trials,
        n_iterations=n_iterations,
        n_random_init=n_random_init,
        random_seed=42,
        save_dir='./comparison_results',
        verbose=True
    )
    
    # 运行所有对比
    runner.run_all_comparisons()
    
    # 打印摘要
    runner.print_summary()


if __name__ == "__main__":
    main()