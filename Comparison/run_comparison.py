"""
Main Execution Script
主执行脚本 - 一键运行完整对比实验

执行流程：
1. 运行Traditional BO、GA、PSO对比实验
2. 收集统计数据
3. 生成专业图表
4. 输出结果报告

Author: Research Team  
Date: 2025-01-19
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/mnt/project')

from comparison_runner import ComparisonRunner
from results_analyzer import ResultsAnalyzer


def run_comparison_experiments():
    """运行对比实验"""
    print("\n" + "=" * 80)
    print("🚀 开始运行算法对比实验")
    print("=" * 80)
    
    # 配置参数
    algorithms = ['BO', 'GA', 'PSO']  # 可以添加'LLMBO'
    n_trials = 3  # 快速测试用3次，正式实验用15次
    n_iterations = 20  # 快速测试用20次，正式实验用50次
    n_random_init = 5  # 快速测试用5个，正式实验用10个
    
    print(f"\n配置:")
    print(f"  算法: {algorithms}")
    print(f"  重复次数: {n_trials}")
    print(f"  迭代次数: {n_iterations}")
    print(f"  随机初始化: {n_random_init}")
    print()
    
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
    
    # 运行对比
    try:
        runner.run_all_comparisons()
        runner.print_summary()
        
        # 获取最新的结果文件
        results_dir = Path('./comparison_results')
        result_files = sorted(results_dir.glob('detailed_results_*.json'))
        
        if result_files:
            latest_result = result_files[-1]
            print(f"\n✓ 实验完成！结果文件: {latest_result}")
            return str(latest_result)
        else:
            print("\n✗ 未找到结果文件")
            return None
            
    except Exception as e:
        print(f"\n✗ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_results(results_file: str):
    """分析结果并生成图表"""
    print("\n" + "=" * 80)
    print("📊 开始生成可视化图表")
    print("=" * 80)
    
    try:
        # 创建分析器
        analyzer = ResultsAnalyzer(
            results_file=results_file,
            save_dir='./figures'
        )
        
        # 生成所有图表
        analyzer.generate_all_figures()
        
        print("\n✓ 图表生成完成！")
        
    except Exception as e:
        print(f"\n✗ 图表生成失败: {e}")
        import traceback
        traceback.print_exc()


def generate_report(results_file: str):
    """生成文本报告"""
    print("\n" + "=" * 80)
    print("📝 生成结果报告")
    print("=" * 80)
    
    # 加载结果
    with open(results_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    # 计算统计量
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ALGORITHM COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\nResults file: {results_file}")
    report_lines.append("\n" + "=" * 80)
    
    for alg, trials in all_results.items():
        if not trials:
            continue
        
        report_lines.append(f"\n【{alg}】")
        report_lines.append("-" * 40)
        
        # 最优值统计
        best_values = [t['best_solution']['scalarized'] for t in trials]
        report_lines.append(f"\nScalarized Objective Value:")
        report_lines.append(f"  Best:   {min(best_values):.4f}")
        report_lines.append(f"  Mean:   {sum(best_values)/len(best_values):.4f}")
        report_lines.append(f"  Worst:  {max(best_values):.4f}")
        report_lines.append(f"  Std:    {(sum((x-sum(best_values)/len(best_values))**2 for x in best_values)/len(best_values))**0.5:.4f}")
        
        # 运行时间
        run_times = [t['elapsed_time'] for t in trials]
        report_lines.append(f"\nRuntime:")
        report_lines.append(f"  Mean:   {sum(run_times)/len(run_times):.1f}s")
        report_lines.append(f"  Std:    {(sum((x-sum(run_times)/len(run_times))**2 for x in run_times)/len(run_times))**0.5:.1f}s")
        
        # 最优解的参数
        best_trial = min(trials, key=lambda x: x['best_solution']['scalarized'])
        best_params = best_trial['best_solution']['params']
        report_lines.append(f"\nBest Solution Parameters:")
        report_lines.append(f"  I1: {best_params['current1']:.2f} A")
        report_lines.append(f"  t1: {best_params['charging_number']}")
        report_lines.append(f"  I2: {best_params['current2']:.2f} A")
        
        # 目标值
        best_obj = best_trial['best_solution']['objectives']
        report_lines.append(f"\nObjective Values (Best Solution):")
        report_lines.append(f"  Time:   {best_obj['time']:.2f} steps")
        report_lines.append(f"  Temp:   {best_obj['temp']:.2f} K")
        report_lines.append(f"  Aging:  {best_obj['aging']:.6f} %")
    
    report_lines.append("\n" + "=" * 80)
    
    # 保存报告
    report_file = Path('./comparison_results') / 'report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 打印报告
    print('\n'.join(report_lines))
    print(f"\n✓ 报告已保存: {report_file}")


def main():
    """主流程"""
    print("\n" + "=" * 80)
    print("🔬 Battery Charging Optimization - Algorithm Comparison")
    print("=" * 80)
    
    # 步骤1: 运行对比实验
    print("\n步骤 1/3: 运行对比实验...")
    results_file = run_comparison_experiments()
    
    if results_file is None:
        print("\n❌ 实验失败，终止流程")
        return
    
    # 步骤2: 生成图表
    print("\n步骤 2/3: 生成可视化图表...")
    analyze_results(results_file)
    
    # 步骤3: 生成报告
    print("\n步骤 3/3: 生成结果报告...")
    generate_report(results_file)
    
    # 完成
    print("\n" + "=" * 80)
    print("✅ 所有任务完成！")
    print("=" * 80)
    print(f"\n结果位置:")
    print(f"  - 数据: ./comparison_results/")
    print(f"  - 图表: ./figures/")
    print(f"  - 报告: ./comparison_results/report.txt")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()