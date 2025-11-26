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

# -*- coding: utf-8 -*-
import sys
import os

# 设置stdout编码为utf-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/mnt/project')

from comparison_runner import ComparisonRunner
from results_analyzer import ResultsAnalyzer


def run_comparison_experiments():
    """Run comparison experiments"""
    print("\n" + "=" * 80)
    print("[START] Running Algorithm Comparison Experiments")
    print("=" * 80)
    
    # Configuration
    algorithms = ['BO', 'GA', 'PSO']  # Can add 'LLMBO'
    n_trials = 3  # Quick test: 3, Full experiment: 15
    n_iterations = 20  # Quick test: 20, Full experiment: 50
    n_random_init = 5  # Quick test: 5, Full experiment: 10
    
    print(f"\nConfiguration:")
    print(f"  Algorithms: {algorithms}")
    print(f"  Trials: {n_trials}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Random init: {n_random_init}")
    print()
    
    # Create runner
    runner = ComparisonRunner(
        algorithms=algorithms,
        n_trials=n_trials,
        n_iterations=n_iterations,
        n_random_init=n_random_init,
        random_seed=42,
        save_dir='./comparison_results',
        verbose=True
    )
    
    # Run comparisons
    try:
        runner.run_all_comparisons()
        runner.print_summary()
        
        # Get latest result file
        results_dir = Path('./comparison_results')
        result_files = sorted(results_dir.glob('detailed_results_*.json'))
        
        if result_files:
            latest_result = result_files[-1]
            print(f"\n[OK] Experiment completed! Result file: {latest_result}")
            return str(latest_result)
        else:
            print("\n[FAIL] No result files found")
            return None
            
    except Exception as e:
        print(f"\n[FAIL] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_results(results_file: str):
    """Analyze results and generate figures"""
    print("\n" + "=" * 80)
    print("[CHARTS] Generating Visualization Figures")
    print("=" * 80)
    
    try:
        # Create analyzer
        analyzer = ResultsAnalyzer(
            results_file=results_file,
            save_dir='./figures'
        )
        
        # Generate all figures
        analyzer.generate_all_figures()
        
        print("\n[OK] Figures generated successfully!")
        
    except Exception as e:
        print(f"\n[FAIL] Figure generation failed: {e}")
        import traceback
        traceback.print_exc()


def generate_report(results_file: str):
    """Generate text report"""
    print("\n" + "=" * 80)
    print("[REPORT] Generating Results Report")
    print("=" * 80)
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    # Compute statistics
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
    
    # Print report
    print('\n'.join(report_lines))
    print(f"\n[OK] Report saved: {report_file}")


def main():
    """Main workflow"""
    print("\n" + "=" * 80)
    print("[Battery Charging Optimization - Algorithm Comparison]")
    print("=" * 80)
    
    # Step 1: Run comparison experiments
    print("\nStep 1/3: Running comparison experiments...")
    results_file = run_comparison_experiments()
    
    if results_file is None:
        print("\n[X] Experiment failed, terminating workflow")
        return
    
    # Step 2: Generate charts
    print("\nStep 2/3: Generating visualization charts...")
    analyze_results(results_file)
    
    # Step 3: Generate report
    print("\nStep 3/3: Generating results report...")
    generate_report(results_file)
    
    # Complete
    print("\n" + "=" * 80)
    print("[OK] All tasks completed!")
    print("=" * 80)
    print(f"\nResults location:")
    print(f"  - Data: ./comparison_results/")
    print(f"  - Figures: ./figures/")
    print(f"  - Report: ./comparison_results/report.txt")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()