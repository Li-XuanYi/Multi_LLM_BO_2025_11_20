"""
Results Analyzer and Visualizer
结果分析与可视化工具

生成专业的学术图表（英文标签）：
1. 收敛曲线对比图
2. 箱型图（显示稳定性）
3. 统计表格
4. 3D Pareto前沿图
5. 雷达图（多目标性能对比）

Author: Research Team
Date: 2025-01-19
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from scipy import stats


class ResultsAnalyzer:
    """
    结果分析器
    
    加载对比实验结果并生成可视化图表
    """
    
    def __init__(
        self,
        results_file: str,
        save_dir: str = './figures'
    ):
        """
        初始化分析器
        
        参数：
            results_file: 结果JSON文件路径
            save_dir: 图表保存目录
        """
        self.results_file = Path(results_file)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载结果
        with open(results_file, 'r', encoding='utf-8') as f:
            self.all_results = json.load(f)
        
        self.algorithms = list(self.all_results.keys())
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 150
        
        print(f"\n✓ 已加载结果文件: {results_file}")
        print(f"  算法: {self.algorithms}")
        print(f"  图表保存至: {self.save_dir}")
    
    def plot_convergence_curves(self, save_name: str = 'convergence_curves.png'):
        """
        绘制收敛曲线对比图
        
        X轴: Evaluation Number
        Y轴: Best Scalarized Objective
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 颜色方案
        colors = {
            'BO': '#1f77b4',
            'GA': '#ff7f0e',
            'PSO': '#2ca02c',
            'LLMBO': '#d62728'
        }
        
        for alg in self.algorithms:
            trials = self.all_results[alg]
            
            if not trials:
                continue
            
            # 收集所有trial的收敛曲线
            all_curves = []
            max_len = 0
            
            for trial in trials:
                history = trial['optimization_history']
                
                # 提取累积最优值
                scalarized_values = [h['scalarized'] for h in history]
                cumulative_best = []
                best_so_far = float('inf')
                
                for val in scalarized_values:
                    if val < best_so_far:
                        best_so_far = val
                    cumulative_best.append(best_so_far)
                
                all_curves.append(cumulative_best)
                max_len = max(max_len, len(cumulative_best))
            
            # 对齐长度（用最后一个值填充）
            aligned_curves = []
            for curve in all_curves:
                if len(curve) < max_len:
                    aligned = curve + [curve[-1]] * (max_len - len(curve))
                else:
                    aligned = curve[:max_len]
                aligned_curves.append(aligned)
            
            # 计算平均值和标准差
            mean_curve = np.mean(aligned_curves, axis=0)
            std_curve = np.std(aligned_curves, axis=0)
            
            iterations = np.arange(1, max_len + 1)
            
            # 绘制平均曲线
            color = colors.get(alg, 'gray')
            ax.plot(iterations, mean_curve, label=alg, color=color, linewidth=2)
            
            # 绘制置信区间
            ax.fill_between(
                iterations,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=color,
                alpha=0.2
            )
        
        ax.set_xlabel('Evaluation Number', fontsize=12)
        ax.set_ylabel('Best Scalarized Objective', fontsize=12)
        ax.set_title('Convergence Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 收敛曲线已保存: {save_path}")
        plt.close()
    
    def plot_boxplots(self, save_name: str = 'boxplots.png'):
        """
        绘制箱型图（显示15次运行的分布）
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 收集数据
        data = []
        labels = []
        
        for alg in self.algorithms:
            trials = self.all_results[alg]
            if not trials:
                continue
            
            best_values = [t['best_solution']['scalarized'] for t in trials]
            data.append(best_values)
            labels.append(alg)
        
        # 绘制箱型图
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
        )
        
        # 设置颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Best Scalarized Objective', fontsize=12)
        ax.set_title('Distribution of Best Solutions (15 Trials)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 箱型图已保存: {save_path}")
        plt.close()
    
    def generate_statistics_table(self, save_name: str = 'statistics_table.csv'):
        """生成统计表格"""
        
        stats_data = []
        
        for alg in self.algorithms:
            trials = self.all_results[alg]
            if not trials:
                continue
            
            # 提取数据
            best_values = [t['best_solution']['scalarized'] for t in trials]
            run_times = [t['elapsed_time'] for t in trials]
            
            objectives_time = [t['best_solution']['objectives']['time'] for t in trials]
            objectives_temp = [t['best_solution']['objectives']['temp'] for t in trials]
            objectives_aging = [t['best_solution']['objectives']['aging'] for t in trials]
            
            # 计算统计量
            stats_data.append({
                'Algorithm': alg,
                'Best': f"{np.min(best_values):.4f}",
                'Mean': f"{np.mean(best_values):.4f}",
                'Std': f"{np.std(best_values):.4f}",
                'Median': f"{np.median(best_values):.4f}",
                'Time (mean)': f"{np.mean(objectives_time):.2f}",
                'Temp (mean)': f"{np.mean(objectives_temp):.2f}",
                'Aging (mean)': f"{np.mean(objectives_aging):.6f}",
                'Runtime (s)': f"{np.mean(run_times):.1f} ± {np.std(run_times):.1f}"
            })
        
        # 创建DataFrame
        df = pd.DataFrame(stats_data)
        
        # 保存为CSV
        save_path = self.save_dir / save_name
        df.to_csv(save_path, index=False)
        print(f"✓ 统计表格已保存: {save_path}")
        
        # 打印到控制台
        print("\n统计表格:")
        print(df.to_string(index=False))
        
        return df
    
    def plot_3d_pareto_front(self, save_name: str = 'pareto_front_3d.png'):
        """绘制3D Pareto前沿"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {
            'BO': '#1f77b4',
            'GA': '#ff7f0e',
            'PSO': '#2ca02c',
            'LLMBO': '#d62728'
        }
        
        for alg in self.algorithms:
            trials = self.all_results[alg]
            if not trials:
                continue
            
            # 收集所有评估点的目标值
            all_time = []
            all_temp = []
            all_aging = []
            
            for trial in trials:
                history = trial['optimization_history']
                for h in history:
                    # 从evaluator的database获取目标值
                    # 注意：这里需要重新构建，因为optimization_history可能不包含objectives
                    # 我们使用best_solution作为代表
                    pass
            
            # 使用最优解作为代表点
            time_vals = [t['best_solution']['objectives']['time'] for t in trials]
            temp_vals = [t['best_solution']['objectives']['temp'] for t in trials]
            aging_vals = [t['best_solution']['objectives']['aging'] for t in trials]
            
            color = colors.get(alg, 'gray')
            ax.scatter(
                time_vals,
                temp_vals,
                aging_vals,
                c=color,
                marker='o',
                s=100,
                alpha=0.6,
                label=alg
            )
        
        ax.set_xlabel('Charging Time (steps)', fontsize=11)
        ax.set_ylabel('Peak Temperature (K)', fontsize=11)
        ax.set_zlabel('Capacity Fade (%)', fontsize=11)
        ax.set_title('Best Solutions in Objective Space (3D)', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 3D Pareto前沿图已保存: {save_path}")
        plt.close()
    
    def plot_radar_chart(self, save_name: str = 'radar_chart.png'):
        """绘制雷达图（多目标性能对比）"""
        from math import pi
        
        # 准备数据
        categories = ['Time\n(lower better)', 'Temp\n(lower better)', 
                     'Aging\n(lower better)', 'Convergence\n(lower better)']
        N = len(categories)
        
        # 计算归一化指标
        normalized_data = {}
        
        for alg in self.algorithms:
            trials = self.all_results[alg]
            if not trials:
                continue
            
            # 平均目标值
            time_mean = np.mean([t['best_solution']['objectives']['time'] for t in trials])
            temp_mean = np.mean([t['best_solution']['objectives']['temp'] for t in trials])
            aging_mean = np.mean([t['best_solution']['objectives']['aging'] for t in trials])
            scalarized_mean = np.mean([t['best_solution']['scalarized'] for t in trials])
            
            normalized_data[alg] = [time_mean, temp_mean, aging_mean, scalarized_mean]
        
        # 归一化到[0, 1]（所有指标都是越小越好）
        all_values = np.array(list(normalized_data.values()))
        min_vals = all_values.min(axis=0)
        max_vals = all_values.max(axis=0)
        
        for alg in normalized_data:
            normalized_data[alg] = [
                (normalized_data[alg][i] - min_vals[i]) / (max_vals[i] - min_vals[i] + 1e-10)
                for i in range(N)
            ]
        
        # 绘制雷达图
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        colors = {
            'BO': '#1f77b4',
            'GA': '#ff7f0e',
            'PSO': '#2ca02c',
            'LLMBO': '#d62728'
        }
        
        for alg in self.algorithms:
            if alg not in normalized_data:
                continue
            
            values = normalized_data[alg]
            values += values[:1]
            
            color = colors.get(alg, 'gray')
            ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.title('Multi-Objective Performance Comparison\n(Normalized, Lower is Better)',
                 size=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 雷达图已保存: {save_path}")
        plt.close()
    
    def perform_statistical_tests(self):
        """执行统计显著性检验（Wilcoxon符号秩检验）"""
        print("\n" + "=" * 70)
        print("统计显著性检验（Wilcoxon Signed-Rank Test）")
        print("=" * 70)
        
        # 收集所有算法的最优值
        algorithm_data = {}
        for alg in self.algorithms:
            trials = self.all_results[alg]
            if not trials:
                continue
            algorithm_data[alg] = [t['best_solution']['scalarized'] for t in trials]
        
        # 两两比较
        algorithms_list = list(algorithm_data.keys())
        
        for i in range(len(algorithms_list)):
            for j in range(i + 1, len(algorithms_list)):
                alg1 = algorithms_list[i]
                alg2 = algorithms_list[j]
                
                data1 = algorithm_data[alg1]
                data2 = algorithm_data[alg2]
                
                # Wilcoxon检验
                statistic, p_value = stats.wilcoxon(data1, data2)
                
                # 判断显著性
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "n.s."
                
                print(f"\n{alg1} vs {alg2}:")
                print(f"  Statistic: {statistic:.4f}")
                print(f"  P-value: {p_value:.4f} {significance}")
                
                # 计算效应量(mean difference)
                mean_diff = np.mean(data1) - np.mean(data2)
                print(f"  Mean difference: {mean_diff:.4f}")
        
        print("\n" + "=" * 70)
        print("显著性标记: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
        print("=" * 70)
    
    def generate_all_figures(self):
        """生成所有图表"""
        print("\n" + "=" * 70)
        print("生成所有可视化图表...")
        print("=" * 70)
        
        self.plot_convergence_curves()
        self.plot_boxplots()
        self.generate_statistics_table()
        self.plot_3d_pareto_front()
        self.plot_radar_chart()
        self.perform_statistical_tests()
        
        print("\n" + "=" * 70)
        print("所有图表已生成完毕！")
        print(f"保存位置: {self.save_dir}")
        print("=" * 70)


# ============================================================
# 主函数
# ============================================================

def main():
    """示例：分析结果并生成图表"""
    
    # 指定结果文件
    results_file = './comparison_results/detailed_results_20250119_123456.json'
    
    # 创建分析器
    analyzer = ResultsAnalyzer(
        results_file=results_file,
        save_dir='./figures'
    )
    
    # 生成所有图表
    analyzer.generate_all_figures()


if __name__ == "__main__":
    print("✓ Results Analyzer 已创建")
    print("\n使用方法:")
    print("  analyzer = ResultsAnalyzer('results.json')")
    print("  analyzer.generate_all_figures()")