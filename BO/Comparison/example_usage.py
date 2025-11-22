"""
Example Usage Script
示例用法脚本

演示如何使用算法对比框架
"""

import sys
sys.path.insert(0, '/mnt/project')

from llmbo_core import MultiObjectiveEvaluator
from base_optimizer import OptimizerFactory


def example_1_single_algorithm():
    """示例1：运行单个算法"""
    print("\n" + "=" * 70)
    print("示例1: 运行单个算法 (GA)")
    print("=" * 70)
    
    # 创建评估器
    evaluator = MultiObjectiveEvaluator(
        weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
        verbose=False
    )
    
    # 创建GA优化器
    optimizer = OptimizerFactory.create(
        name='GA',
        evaluator=evaluator,
        pbounds={
            'current1': (3.0, 6.0),
            'charging_number': (5, 25),
            'current2': (1.0, 4.0)
        },
        random_state=42,
        verbose=True
    )
    
    # 运行优化（快速测试：10迭代）
    results = optimizer.optimize(n_iterations=10, n_random_init=5)
    
    # 打印结果
    best = results['best_solution']
    print(f"\n最优解:")
    print(f"  I1={best['params']['current1']:.2f}A, "
          f"t1={best['params']['charging_number']}, "
          f"I2={best['params']['current2']:.2f}A")
    print(f"  标量化值: {best['scalarized']:.4f}")
    print(f"  运行时间: {results['elapsed_time']:.1f}s")


def example_2_compare_algorithms():
    """示例2：对比多个算法（快速版）"""
    print("\n" + "=" * 70)
    print("示例2: 对比多个算法（快速版：3次×10迭代）")
    print("=" * 70)
    
    from comparison_runner import ComparisonRunner
    
    runner = ComparisonRunner(
        algorithms=['BO', 'GA', 'PSO'],
        n_trials=3,           # 快速测试：3次
        n_iterations=10,      # 快速测试：10迭代
        n_random_init=5,
        random_seed=42,
        save_dir='./test_results',
        verbose=True
    )
    
    runner.run_all_comparisons()
    runner.print_summary()


def example_3_analyze_results():
    """示例3：分析已有结果"""
    print("\n" + "=" * 70)
    print("示例3: 分析结果并生成图表")
    print("=" * 70)
    
    # 注意：需要先运行example_2生成结果文件
    from pathlib import Path
    
    results_dir = Path('./test_results')
    result_files = sorted(results_dir.glob('detailed_results_*.json'))
    
    if not result_files:
        print("未找到结果文件，请先运行 example_2")
        return
    
    latest_result = result_files[-1]
    print(f"分析结果文件: {latest_result}")
    
    from results_analyzer import ResultsAnalyzer
    
    analyzer = ResultsAnalyzer(
        results_file=str(latest_result),
        save_dir='./test_figures'
    )
    
    # 生成所有图表
    analyzer.generate_all_figures()


def example_4_custom_optimizer():
    """示例4：自定义优化器"""
    print("\n" + "=" * 70)
    print("示例4: 自定义优化器参数")
    print("=" * 70)
    
    from multi_objective_evaluator import MultiObjectiveEvaluator
    from pso_optimizer import ParticleSwarmOptimization
    
    evaluator = MultiObjectiveEvaluator(
        weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
        verbose=False
    )
    
    # 创建自定义PSO（修改参数）
    pso = ParticleSwarmOptimization(
        evaluator=evaluator,
        pbounds={
            'current1': (3.0, 6.0),
            'charging_number': (5, 25),
            'current2': (1.0, 4.0)
        },
        random_state=42,
        n_particles=20,    # 自定义：粒子数
        w=0.6,             # 自定义：惯性权重
        c1=2.0,            # 自定义：认知系数
        c2=2.0,            # 自定义：社会系数
        verbose=True
    )
    
    results = pso.optimize(n_iterations=10)
    
    best = results['best_solution']
    print(f"\n最优解: {best['scalarized']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Algorithm Comparison Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help='选择示例编号 (1-4)'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_single_algorithm,
        2: example_2_compare_algorithms,
        3: example_3_analyze_results,
        4: example_4_custom_optimizer
    }
    
    examples[args.example]()
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)