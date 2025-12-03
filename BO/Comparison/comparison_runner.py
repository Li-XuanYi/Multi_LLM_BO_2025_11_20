"""
Algorithm Comparison Runner - UNIFIED KERNEL VERSION
算法对比运行框架 (统一物理内核版)

修改内容:
1. 修复导入路径错误
2. 添加依赖检查和诊断
3. 确保OptimizerFactory注册正确
4. 统一物理内核：所有算法使用MultiObjectiveEvaluator（SPM_v3）
5. 传统算法通过ScalarOnlyEvaluatorWrapper只使用标量值

Author: Research Team
Date: 2025-12-02 (统一物理内核版)
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 修复1: 正确的导入路径和诊断
# ============================================================

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent  # 假设在 BO/Comparison/ 目录下
sys.path.insert(0, str(project_root))

print("\n" + "=" * 80)
print("检查依赖和导入...")
print("=" * 80)

# 检查关键依赖
dependencies_ok = True

# 检查 DEAP (用于 GA)
try:
    import deap
    print("✓ DEAP 已安装")
    DEAP_AVAILABLE = True
except ImportError:
    print("✗ DEAP 未安装 - GA 算法不可用")
    print("  安装: pip install deap")
    DEAP_AVAILABLE = False
    dependencies_ok = False

# 检查 pyswarms (用于 PSO)
try:
    import pyswarms
    print("✓ pyswarms 已安装")
    PYSWARMS_AVAILABLE = True
except ImportError:
    print("✗ pyswarms 未安装 - PSO 算法不可用")
    print("  安装: pip install pyswarms")
    PYSWARMS_AVAILABLE = False
    dependencies_ok = False

# 检查 bayes_opt (用于 BO)
try:
    import bayes_opt
    print("✓ bayes-opt 已安装")
    BAYESOPT_AVAILABLE = True
except ImportError:
    print("✗ bayes-opt 未安装 - BO 算法不可用")
    print("  安装: pip install bayesian-optimization")
    BAYESOPT_AVAILABLE = False
    dependencies_ok = False

print("=" * 80)

# 导入核心模块
try:
    # 方式1: 如果在 BO/Comparison/ 目录下运行
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llmbo_core import MultiObjectiveEvaluator
    from Comparison.base_optimizer import OptimizerFactory
    print("✓ 使用相对导入路径 (方式1)")
except ImportError:
    try:
        # 方式2: 如果在项目根目录运行
        from BO.llmbo_core import MultiObjectiveEvaluator
        from BO.Comparison.base_optimizer import OptimizerFactory
        print("✓ 使用绝对导入路径 (方式2)")
    except ImportError:
        print("✗ 无法导入核心模块")
        print("  请确保在正确的目录运行此脚本")
        print("  或调整 PYTHONPATH")
        sys.exit(1)

# ============================================================
# 修复2: 条件导入优化器（带详细错误信息）
# ============================================================

print("\n导入优化器...")

# Traditional BO
try:
    try:
        from Comparison.traditional_bo import TraditionalBO
    except ImportError:
        from BO.Comparison.traditional_bo import TraditionalBO
    print("✓ Traditional BO 已导入")
except ImportError as e:
    print(f"✗ Traditional BO 导入失败: {e}")

# GA
if DEAP_AVAILABLE:
    try:
        try:
            from Comparison.ga_optimizer import GeneticAlgorithm
        except ImportError:
            from BO.Comparison.ga_optimizer import GeneticAlgorithm
        print("✓ GA 已导入")
    except ImportError as e:
        print(f"✗ GA 导入失败: {e}")
else:
    print("⊘ GA 跳过（DEAP 未安装）")

# PSO
if PYSWARMS_AVAILABLE:
    try:
        try:
            from Comparison.pso_optimizer import ParticleSwarmOptimization
        except ImportError:
            from BO.Comparison.pso_optimizer import ParticleSwarmOptimization
        print("✓ PSO 已导入")
    except ImportError as e:
        print(f"✗ PSO 导入失败: {e}")
else:
    print("⊘ PSO 跳过（pyswarms 未安装）")

# ============================================================
# 修复3: 验证 OptimizerFactory 注册
# ============================================================

print("\n检查 OptimizerFactory 注册...")
available_optimizers = OptimizerFactory.available_optimizers()

if len(available_optimizers) == 0:
    print("✗ OptimizerFactory 注册表为空！")
    print("\n可能的原因:")
    print("1. 优化器模块导入失败")
    print("2. 优化器文件中的注册代码未执行")
    print("3. 缺少必要的依赖库")
    print("\n请检查上述错误信息")
    sys.exit(1)
else:
    print(f"✓ 已注册的优化器: {available_optimizers}")

print("=" * 80)


# ============================================================
# 原有的 ComparisonRunner 类（保持不变）
# ============================================================

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
        # 验证算法可用性
        available = OptimizerFactory.available_optimizers()
        invalid_algorithms = [alg for alg in algorithms if alg not in available]
        
        if invalid_algorithms:
            raise ValueError(
                f"以下算法不可用: {invalid_algorithms}\n"
                f"可用算法: {available}\n"
                f"请检查依赖库是否安装"
            )
        
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
            print("Algorithm Comparison Runner Initialized")
            print("=" * 80)
            print(f"Algorithms: {algorithms}")
            print(f"Trials: {n_trials} trials")
            print(f"Iterations: {n_iterations} iterations per trial")
            print(f"Random initialization: {n_random_init} points")
            print(f"Base random seed: {random_seed}")
            print(f"Results saved to: {self.save_dir}")
            print("=" * 80)
    
    def run_all_comparisons(self):
        """运行所有算法的对比实验"""
        start_time = time.time()
        
        for algorithm in self.algorithms:
            if self.verbose:
                print(f"\n{'=' * 80}")
                print(f"Running algorithm: {algorithm}")
                print(f"{'=' * 80}")
            
            self.run_algorithm_trials(algorithm)
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"All comparison experiments completed!")
            print(f"Total runtime: {total_time:.1f} seconds")
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
                    print(f"  [OK] Trial {trial + 1} completed")
                    print(f"    Best value: {result['best_solution']['scalarized']:.4f}")
                    print(f"    Runtime: {result['elapsed_time']:.1f}s")
            
            except Exception as e:
                print(f"  [FAIL] Trial {trial + 1} failed: {e}")
                import traceback
                traceback.print_exc()
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
        
        print(f"\nDetailed results saved: {detailed_file}")
        
        # Save statistical summary
        summary_file = self.save_dir / f"summary_{timestamp}.json"
        summary = self.compute_summary_statistics()
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Statistical summary saved: {summary_file}")
    
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
        """Print statistical summary"""
        summary = self.compute_summary_statistics()
        
        print("\n" + "=" * 80)
        print("Statistical Summary (based on {} runs)".format(self.n_trials))
        print("=" * 80)
        
        for alg in self.algorithms:
            if alg not in summary:
                continue
            
            stats = summary[alg]
            
            print(f"\n[{alg}]")
            print(f"  Scalarized objective:")
            print(f"    Best:   {stats['scalarized']['best']:.4f}")
            print(f"    Mean:   {stats['scalarized']['mean']:.4f}")
            print(f"    Std:    {stats['scalarized']['std']:.4f}")
            print(f"    Median: {stats['scalarized']['median']:.4f}")
            
            print(f"  Runtime:")
            print(f"    Mean:   {stats['runtime']['mean']:.1f}s")
            print(f"    Std:    {stats['runtime']['std']:.1f}s")
            
            print(f"  Objectives (average):")
            print(f"    Time:   {stats['objectives']['time']['mean']:.1f} +/- {stats['objectives']['time']['std']:.1f}")
            print(f"    Temp:   {stats['objectives']['temp']['mean']:.2f} +/- {stats['objectives']['temp']['std']:.2f}")
            print(f"    Aging:  {stats['objectives']['aging']['mean']:.6f} +/- {stats['objectives']['aging']['std']:.6f}")
        
        print("\n" + "=" * 80)


# ============================================================
# 主函数 - 运行对比实验
# ============================================================

def main():
    """Run complete comparison experiments"""
    
    # 获取可用算法
    available = OptimizerFactory.available_optimizers()
    
    if not available:
        print("\n✗ 没有可用的优化器！")
        print("请安装所需的依赖库")
        return
    
    # Configuration
    algorithms = available  # 使用所有可用的算法
    n_trials = 3  # 快速测试：3次，完整实验：15次
    n_iterations = 20  # 快速测试：20次，完整实验：50次
    n_random_init = 5  # 快速测试：5个，完整实验：10个
    
    print(f"\n将运行以下算法: {algorithms}")
    print(f"配置: {n_trials} trials × {n_iterations} iterations")
    
    # Create runner
    try:
        runner = ComparisonRunner(
            algorithms=algorithms,
            n_trials=n_trials,
            n_iterations=n_iterations,
            n_random_init=n_random_init,
            random_seed=42,
            save_dir='./comparison_results',
            verbose=True
        )
        
        # Run all comparisons
        runner.run_all_comparisons()
        
        # Print summary
        runner.print_summary()
        
    except Exception as e:
        print(f"\n✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()