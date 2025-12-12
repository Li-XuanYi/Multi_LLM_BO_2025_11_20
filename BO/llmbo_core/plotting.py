"""
Unified Plotting Module for LLM-Enhanced Multi-Objective Optimization
统一可视化模块 - 英文标签，专业配色

Features:
1. Single algorithm visualization
   - Convergence curves (4 objectives)
   - 3D Pareto front
   - 2D projections

2. Multi-algorithm comparison
   - Convergence comparison with mean/std
   - Boxplots (4 objectives)
   - 3D Pareto front comparison

Color Scheme:
- BO: blue (#1f77b4)
- GA: orange (#ff7f0e)
- PSO: green (#2ca02c)
- LLMBO/LLM_MOBO: red (#d62728) / purple (#9467bd)

Author: Research Team
Date: 2025-12-11
Version: 1.0
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from mpl_toolkits.mplot3d import Axes3D

# 忽略matplotlib警告
warnings.filterwarnings('ignore')


class OptimizationPlotter:
    """
    Unified plotting class for optimization results
    """
    
    def __init__(self, save_dir: str = './figures'):
        """
        Initialize plotter
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for algorithms
        self.colors = {
            'BO': '#1f77b4',      # blue
            'GA': '#ff7f0e',      # orange
            'PSO': '#2ca02c',     # green
            'LLMBO': '#d62728',   # red
            'LLM_MOBO': '#9467bd' # purple
        }
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 300
        
        print(f"[Plotter] Initialized, figures will be saved to: {self.save_dir}")
    
    # ================================================================
    # Single Algorithm Visualization
    # ================================================================
    
    def plot_convergence(
        self,
        database: List[Dict],
        save_name: str = 'convergence.png',
        show: bool = False
    ) -> None:
        """
        Plot convergence curves for a single algorithm
        
        Creates 4 subplots:
        1. Scalarized objective value
        2. Charging time
        3. Peak temperature
        4. Capacity fade
        
        Args:
            database: List of evaluation records
            save_name: Filename to save
            show: Whether to display the plot
        """
        if len(database) == 0:
            print("[Warning] Empty database, skipping convergence plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        iterations = [d['eval_id'] for d in database]
        scalarized = [d['scalarized'] for d in database]
        time_vals = [d['objectives']['time'] for d in database]
        temp_vals = [d['objectives']['temp'] for d in database]
        aging_vals = [d['objectives']['aging'] for d in database]
        
        # Compute cumulative best
        cumulative_best = []
        best_so_far = float('inf')
        for s in scalarized:
            best_so_far = min(best_so_far, s)
            cumulative_best.append(best_so_far)
        
        # Subplot 1: Scalarized objective
        axes[0, 0].plot(iterations, scalarized, 'o-', alpha=0.6, 
                       markersize=4, label='Evaluated Values')
        axes[0, 0].plot(iterations, cumulative_best, 'r-', 
                       linewidth=2, label='Best So Far')
        axes[0, 0].set_xlabel('Evaluation Number')
        axes[0, 0].set_ylabel('Scalarized Objective Value')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Subplot 2: Charging time
        axes[0, 1].plot(iterations, time_vals, 'o-', 
                       alpha=0.6, color='#1f77b4', markersize=4)
        axes[0, 1].set_xlabel('Evaluation Number')
        axes[0, 1].set_ylabel('Charging Time (steps)')
        axes[0, 1].set_title('Charging Time Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Subplot 3: Peak temperature
        axes[1, 0].plot(iterations, temp_vals, 'o-', 
                       alpha=0.6, color='#d62728', markersize=4)
        axes[1, 0].axhline(y=309, color='r', linestyle='--', 
                          linewidth=1.5, label='Temperature Limit (309K)')
        axes[1, 0].set_xlabel('Evaluation Number')
        axes[1, 0].set_ylabel('Peak Temperature (K)')
        axes[1, 0].set_title('Peak Temperature Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: Capacity fade
        axes[1, 1].plot(iterations, aging_vals, 'o-', 
                       alpha=0.6, color='#2ca02c', markersize=4)
        axes[1, 1].set_xlabel('Evaluation Number')
        axes[1, 1].set_ylabel('Capacity Fade (%)')
        axes[1, 1].set_title('Capacity Fade Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotter] Convergence plot saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_pareto_front_3d(
        self,
        database: List[Dict],
        pareto_front: List[Dict],
        save_name: str = 'pareto_front_3d.png',
        show: bool = False
    ) -> None:
        """
        Plot 3D Pareto front
        
        Args:
            database: All evaluation records
            pareto_front: Pareto optimal solutions
            save_name: Filename to save
            show: Whether to display the plot
        """
        if len(pareto_front) == 0:
            print("[Warning] Empty Pareto front, skipping 3D plot")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Pareto optimal points (red, large)
        time_vals = [p['objectives']['time'] for p in pareto_front]
        temp_vals = [p['objectives']['temp'] for p in pareto_front]
        aging_vals = [p['objectives']['aging'] for p in pareto_front]
        
        ax.scatter(time_vals, temp_vals, aging_vals,
                  c='red', marker='o', s=100, 
                  label='Pareto Optimal Solutions', 
                  edgecolors='black', linewidths=1.5)
        
        # All evaluated points (gray, small)
        all_time = [d['objectives']['time'] for d in database]
        all_temp = [d['objectives']['temp'] for d in database]
        all_aging = [d['objectives']['aging'] for d in database]
        
        ax.scatter(all_time, all_temp, all_aging,
                  c='gray', marker='.', s=20, alpha=0.3,
                  label='All Evaluated Points')
        
        ax.set_xlabel('Charging Time (steps)', fontsize=11)
        ax.set_ylabel('Peak Temperature (K)', fontsize=11)
        ax.set_zlabel('Capacity Fade (%)', fontsize=11)
        ax.set_title('Pareto Front (3D Objective Space)', fontsize=12)
        ax.legend(fontsize=10)
        
        # Save
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotter] 3D Pareto front saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_pareto_front_2d(
        self,
        pareto_front: List[Dict],
        save_name: str = 'pareto_front_2d.png',
        show: bool = False
    ) -> None:
        """
        Plot 2D projections of Pareto front
        
        Creates 3 subplots:
        1. Time vs Temperature
        2. Time vs Aging
        3. Temperature vs Aging
        
        Args:
            pareto_front: Pareto optimal solutions
            save_name: Filename to save
            show: Whether to display the plot
        """
        if len(pareto_front) == 0:
            print("[Warning] Empty Pareto front, skipping 2D projections")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Extract data
        time_vals = [p['objectives']['time'] for p in pareto_front]
        temp_vals = [p['objectives']['temp'] for p in pareto_front]
        aging_vals = [p['objectives']['aging'] for p in pareto_front]
        
        # Subplot 1: Time vs Temperature
        axes[0].scatter(time_vals, temp_vals, c='red', s=60, 
                       alpha=0.7, edgecolors='black', linewidths=1)
        axes[0].set_xlabel('Charging Time (steps)')
        axes[0].set_ylabel('Peak Temperature (K)')
        axes[0].set_title('Time vs Temperature')
        axes[0].grid(True, alpha=0.3)
        
        # Subplot 2: Time vs Aging
        axes[1].scatter(time_vals, aging_vals, c='red', s=60,
                       alpha=0.7, edgecolors='black', linewidths=1)
        axes[1].set_xlabel('Charging Time (steps)')
        axes[1].set_ylabel('Capacity Fade (%)')
        axes[1].set_title('Time vs Capacity Fade')
        axes[1].grid(True, alpha=0.3)
        
        # Subplot 3: Temperature vs Aging
        axes[2].scatter(temp_vals, aging_vals, c='red', s=60,
                       alpha=0.7, edgecolors='black', linewidths=1)
        axes[2].set_xlabel('Peak Temperature (K)')
        axes[2].set_ylabel('Capacity Fade (%)')
        axes[2].set_title('Temperature vs Capacity Fade')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotter] 2D Pareto projections saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # ================================================================
    # Multi-Algorithm Comparison Visualization
    # ================================================================
    
    def plot_algorithm_comparison_convergence(
        self,
        all_results: Dict[str, List[Dict]],
        save_name: str = 'comparison_convergence.png',
        show: bool = False
    ) -> None:
        """
        Plot convergence comparison across algorithms
        
        Shows mean convergence curve with std deviation bands
        
        Args:
            all_results: Dict of {algorithm_name: [trial_results]}
            save_name: Filename to save
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for alg_name, trials in all_results.items():
            if len(trials) == 0:
                continue
            
            # Extract convergence curves
            all_curves = []
            max_len = 0
            
            for trial in trials:
                # Get database from trial
                if 'optimization_history' in trial:
                    history = trial['optimization_history']
                elif 'database' in trial:
                    history = trial['database']
                else:
                    continue
                
                # Compute cumulative best
                cumulative_best = []
                best_so_far = float('inf')
                for record in history:
                    s = record.get('scalarized', float('inf'))
                    best_so_far = min(best_so_far, s)
                    cumulative_best.append(best_so_far)
                
                all_curves.append(cumulative_best)
                max_len = max(max_len, len(cumulative_best))
            
            if len(all_curves) == 0:
                continue
            
            # Pad curves to same length (use last value)
            padded_curves = []
            for curve in all_curves:
                if len(curve) < max_len:
                    curve = curve + [curve[-1]] * (max_len - len(curve))
                padded_curves.append(curve)
            
            # Compute mean and std
            curves_array = np.array(padded_curves)
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            
            # Plot
            x = np.arange(1, max_len + 1)
            color = self.colors.get(alg_name, '#000000')
            
            ax.plot(x, mean_curve, color=color, linewidth=2, label=alg_name)
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Evaluation Number', fontsize=11)
        ax.set_ylabel('Best Scalarized Objective', fontsize=11)
        ax.set_title('Algorithm Convergence Comparison', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotter] Convergence comparison saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_algorithm_comparison_boxplot(
        self,
        all_results: Dict[str, List[Dict]],
        save_name: str = 'comparison_boxplot.png',
        show: bool = False
    ) -> None:
        """
        Plot boxplot comparison for 4 objectives
        
        Creates 4 subplots showing distribution of:
        1. Scalarized objective
        2. Charging time
        3. Peak temperature
        4. Capacity fade
        
        Args:
            all_results: Dict of {algorithm_name: [trial_results]}
            save_name: Filename to save
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algorithms = list(all_results.keys())
        colors_list = [self.colors.get(alg, '#000000') for alg in algorithms]
        
        # Collect data
        scalarized_data = []
        time_data = []
        temp_data = []
        aging_data = []
        
        for alg in algorithms:
            trials = all_results[alg]
            
            scalarized_vals = [t['best_solution']['scalarized'] for t in trials]
            time_vals = [t['best_solution']['objectives']['time'] for t in trials]
            temp_vals = [t['best_solution']['objectives']['temp'] for t in trials]
            aging_vals = [t['best_solution']['objectives']['aging'] for t in trials]
            
            scalarized_data.append(scalarized_vals)
            time_data.append(time_vals)
            temp_data.append(temp_vals)
            aging_data.append(aging_vals)
        
        # Boxplot 1: Scalarized
        bp1 = axes[0, 0].boxplot(scalarized_data, labels=algorithms,
                                 patch_artist=True, showmeans=True)
        for patch, color in zip(bp1['boxes'], colors_list):
            patch.set_facecolor(color)
        axes[0, 0].set_ylabel('Scalarized Objective Value')
        axes[0, 0].set_title('Scalarized Objective Distribution')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Boxplot 2: Time
        bp2 = axes[0, 1].boxplot(time_data, labels=algorithms,
                                 patch_artist=True, showmeans=True)
        for patch, color in zip(bp2['boxes'], colors_list):
            patch.set_facecolor(color)
        axes[0, 1].set_ylabel('Charging Time (steps)')
        axes[0, 1].set_title('Charging Time Distribution')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Boxplot 3: Temperature
        bp3 = axes[1, 0].boxplot(temp_data, labels=algorithms,
                                 patch_artist=True, showmeans=True)
        for patch, color in zip(bp3['boxes'], colors_list):
            patch.set_facecolor(color)
        axes[1, 0].set_ylabel('Peak Temperature (K)')
        axes[1, 0].set_title('Peak Temperature Distribution')
        axes[1, 0].axhline(y=309, color='r', linestyle='--', 
                          linewidth=1.5, label='Limit (309K)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Boxplot 4: Aging
        bp4 = axes[1, 1].boxplot(aging_data, labels=algorithms,
                                 patch_artist=True, showmeans=True)
        for patch, color in zip(bp4['boxes'], colors_list):
            patch.set_facecolor(color)
        axes[1, 1].set_ylabel('Capacity Fade (%)')
        axes[1, 1].set_title('Capacity Fade Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotter] Boxplot comparison saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_algorithm_comparison_pareto_3d(
        self,
        all_results: Dict[str, List[Dict]],
        save_name: str = 'comparison_pareto_3d.png',
        show: bool = False
    ) -> None:
        """
        Plot 3D Pareto front comparison
        
        Shows best solutions from all algorithms in 3D objective space
        
        Args:
            all_results: Dict of {algorithm_name: [trial_results]}
            save_name: Filename to save
            show: Whether to display the plot
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for alg_name, trials in all_results.items():
            if len(trials) == 0:
                continue
            
            # Collect all best solutions
            time_vals = []
            temp_vals = []
            aging_vals = []
            
            for trial in trials:
                best = trial['best_solution']
                time_vals.append(best['objectives']['time'])
                temp_vals.append(best['objectives']['temp'])
                aging_vals.append(best['objectives']['aging'])
            
            # Plot
            color = self.colors.get(alg_name, '#000000')
            ax.scatter(time_vals, temp_vals, aging_vals,
                      c=color, marker='o', s=60, alpha=0.7,
                      label=alg_name, edgecolors='black', linewidths=1)
        
        ax.set_xlabel('Charging Time (steps)', fontsize=11)
        ax.set_ylabel('Peak Temperature (K)', fontsize=11)
        ax.set_zlabel('Capacity Fade (%)', fontsize=11)
        ax.set_title('Algorithm Comparison (3D Objective Space)', fontsize=12)
        ax.legend(fontsize=10)
        
        # Save
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plotter] 3D comparison saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # ================================================================
    # Convenience Methods
    # ================================================================
    
    def generate_all_single_algorithm_plots(
        self,
        database: List[Dict],
        pareto_front: List[Dict],
        prefix: str = '',
        show: bool = False
    ) -> None:
        """
        Generate all plots for a single algorithm run
        
        Args:
            database: Complete evaluation database
            pareto_front: Pareto optimal solutions
            prefix: Filename prefix
            show: Whether to display plots
        """
        print(f"\n[Plotter] Generating all plots with prefix '{prefix}'...")
        
        # Convergence
        self.plot_convergence(
            database=database,
            save_name=f'{prefix}convergence.png',
            show=show
        )
        
        # 3D Pareto
        self.plot_pareto_front_3d(
            database=database,
            pareto_front=pareto_front,
            save_name=f'{prefix}pareto_3d.png',
            show=show
        )
        
        # 2D projections
        self.plot_pareto_front_2d(
            pareto_front=pareto_front,
            save_name=f'{prefix}pareto_2d.png',
            show=show
        )
        
        print(f"[Plotter] All single-algorithm plots saved to: {self.save_dir}")
    
    def generate_all_comparison_plots(
        self,
        all_results: Dict[str, List[Dict]],
        prefix: str = '',
        show: bool = False
    ) -> None:
        """
        Generate all comparison plots
        
        Args:
            all_results: Dict of {algorithm_name: [trial_results]}
            prefix: Filename prefix
            show: Whether to display plots
        """
        print(f"\n[Plotter] Generating all comparison plots with prefix '{prefix}'...")
        
        # Convergence comparison
        self.plot_algorithm_comparison_convergence(
            all_results=all_results,
            save_name=f'{prefix}convergence.png',
            show=show
        )
        
        # Boxplot comparison
        self.plot_algorithm_comparison_boxplot(
            all_results=all_results,
            save_name=f'{prefix}boxplot.png',
            show=show
        )
        
        # 3D comparison
        self.plot_algorithm_comparison_pareto_3d(
            all_results=all_results,
            save_name=f'{prefix}pareto_3d.png',
            show=show
        )
        
        print(f"[Plotter] All comparison plots saved to: {self.save_dir}")
