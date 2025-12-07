"""
LLM-Enhanced Multi-Objective Bayesian Optimization
for Battery Fast-Charging Strategy Optimization

完整集成系统

=============================================================================
系统架构
=============================================================================
1. MultiObjectiveEvaluator - 多目标评估器
   └── LLM Warm Start (异步初始化)

2. LLMEnhancedBO - LLM增强的代理模型
   ├── 自动微分梯度计算
   ├── 数据驱动的耦合矩阵估计
   ├── LLM物理解释顾问
   ├── 复合核函数 (RBF + Coupling)
   └── 动态γ调整

3. LLMEnhancedEI - LLM增强的采集函数
   ├── 优化状态检测
   ├── LLM采样策略顾问
   ├── 数据驱动的μ/σ计算
   └── 高斯权重函数

4. BayesianOptimization - 优化循环

=============================================================================
使用方法
=============================================================================
```python
# 初始化优化器
optimizer = LLMEnhancedMultiObjectiveBO(
    llm_api_key='your-api-key',
    n_warmstart=5,
    n_iterations=50,
    verbose=True
)

# 运行优化（异步）
results = await optimizer.optimize_async()

# 或同步运行
results = optimizer.optimize()

# 分析结果
optimizer.plot_optimization_history()
optimizer.export_results('results.json')
```

=============================================================================
作者: Research Team
日期: 2025-01-12
版本: v1.0 - 完整集成版
=============================================================================
"""

import numpy as np
import asyncio
import json
import warnings
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.stats import norm
# 导入自定义模块（支持包导入和直接运行）
try:
    from .multi_objective_evaluator import MultiObjectiveEvaluator
    from .LLM_enhanced_surrogate_modeling import LLMEnhancedBO
    from .LLM_Enhanced_Expected_Improvement import LLMEnhancedEI
    from .result_manager import ResultManager  # 新增
except ImportError:
    from multi_objective_evaluator import MultiObjectiveEvaluator
    from LLM_enhanced_surrogate_modeling import LLMEnhancedBO
    from LLM_Enhanced_Expected_Improvement import LLMEnhancedEI
    from result_manager import ResultManager  # 新增

# 导入bayes_opt
from bayes_opt.bayesian_optimization import BayesianOptimization
# from bayes_opt.util import acq_max


# ============================================================
# 主优化器类
# ============================================================

class LLMEnhancedMultiObjectiveBO:
    """
    LLM增强的多目标贝叶斯优化器
    
    完整集成：
    - LLM Warm Start
    - LLM-Enhanced Surrogate Model
    - LLM-Enhanced Expected Improvement
    """
    
    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str = 'https://api.nuwaapi.com/v1',
        llm_model: str = "gpt-3.5-turbo",
        n_warmstart: int = 5,
        n_iterations: int = 50,
        n_random_init: int = 10,
        objective_weights: Optional[Dict[str, float]] = None,
        enable_llm_warmstart: bool = True,
        enable_llm_surrogate: bool = True,
        enable_llm_ei: bool = True,
        verbose: bool = True,
        save_dir: str = './results'
    ):
        """
        初始化LLM增强的多目标贝叶斯优化器
        
        参数：
            llm_api_key: LLM API密钥
            llm_base_url: API基础URL
            llm_model: 使用的LLM模型
            n_warmstart: Warm Start策略数量
            n_iterations: BO迭代次数
            n_random_init: 随机初始化点数量
            objective_weights: 目标权重
            enable_llm_warmstart: 启用LLM Warm Start
            enable_llm_surrogate: 启用LLM增强代理模型
            enable_llm_ei: 启用LLM增强EI
            verbose: 详细输出
            save_dir: 结果保存目录
        """
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.n_warmstart = n_warmstart
        self.n_iterations = n_iterations
        self.n_random_init = n_random_init
        self.verbose = verbose
        self.save_dir = save_dir
        
        # 功能开关
        self.enable_llm_warmstart = enable_llm_warmstart
        self.enable_llm_surrogate = enable_llm_surrogate
        self.enable_llm_ei = enable_llm_ei
        
        # 参数边界
        self.pbounds = {
            'current1': (3.0, 6.0),
            'charging_number': (5, 25),
            'current2': (1.0, 4.0)
        }
        
        # 初始化评估器
        self.evaluator = MultiObjectiveEvaluator(
            weights=objective_weights or {'time': 0.4, 'temp': 0.35, 'aging': 0.25},
            update_interval=10,
            temp_max=309.0,
            max_steps=300,
            verbose=verbose
        )
        
        # 初始化LLM增强的代理模型
        if enable_llm_surrogate:
            self.llm_surrogate = LLMEnhancedBO(
                evaluator=self.evaluator,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                enable_llm_advisor=True,
                initial_gamma=0.1,
                update_coupling_every=5,
                verbose=verbose
            )
        else:
            self.llm_surrogate = None
        
        # 初始化LLM增强的EI
        if enable_llm_ei:
            self.llm_ei = LLMEnhancedEI(
                evaluator=self.evaluator,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                enable_llm_advisor=True,
                update_strategy_every=3,
                verbose=verbose
            )
        else:
            self.llm_ei = None
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # ====== 新增: 初始化ResultManager ======
        self.result_manager = ResultManager(save_dir=save_dir)
        
        # 优化历史
        self.optimization_history = []
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("LLM-Enhanced Multi-Objective Bayesian Optimization")
            print("=" * 80)
            print(f"LLM模型: {llm_model}")
            print(f"Warm Start: {'启用' if enable_llm_warmstart else '禁用'} ({n_warmstart} 策略)")
            print(f"增强代理模型: {'启用' if enable_llm_surrogate else '禁用'}")
            print(f"增强EI: {'启用' if enable_llm_ei else '禁用'}")
            print(f"随机初始化: {n_random_init} 点")
            print(f"BO迭代: {n_iterations} 次")
            print(f"目标权重: {self.evaluator.weights}")
            print("=" * 80)
    
    async def _warmstart_async(self) -> List[Dict]:
        """LLM Warm Start（异步）"""
        if not self.enable_llm_warmstart:
            if self.verbose:
                print("\n[跳过 Warm Start]")
            return []
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("阶段 1: LLM-Enhanced Warm Start")
            print("=" * 80)
        
        warmstart_results = await self.evaluator.initialize_with_llm_warmstart(
            n_strategies=self.n_warmstart,
            llm_api_key=self.llm_api_key,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model
        )
        
        return warmstart_results
    
    def _random_initialization(self, n_points: int) -> List[Dict]:
        """随机初始化"""
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"阶段 2: 随机初始化 ({n_points} 点)")
            print("=" * 80)
        
        results = []
        for i in range(n_points):
            params = {
                'current1': np.random.uniform(*self.pbounds['current1']),
                'charging_number': int(np.random.uniform(*self.pbounds['charging_number'])),
                'current2': np.random.uniform(*self.pbounds['current2'])
            }
            
            scalarized = self.evaluator.evaluate(**params)
            
            results.append({
                'iteration': i,
                'params': params,
                'scalarized': scalarized,
                'source': 'random_init'
            })
            
            if self.verbose and (i + 1) % 3 == 0:
                print(f"  随机点 {i+1}/{n_points}: f = {scalarized:.4f}")
        
        return results
    
    async def _bo_iteration_async(self, iteration: int) -> Dict:
        """单次BO迭代（异步）"""
        history = self.evaluator.export_database()
        
        # 更新LLM增强的代理模型
        if self.enable_llm_surrogate:
            gp = await self.llm_surrogate.fit_surrogate_async(history)
        else:
            # 使用标准GP
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            
            X = np.array([[h['params']['current1'], 
                          h['params']['charging_number'], 
                          h['params']['current2']] 
                         for h in history if h['valid']])
            y = np.array([h['scalarized'] for h in history if h['valid']])
            
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
            gp.fit(X, y)
        
        # 更新LLM增强的EI策略
        if self.enable_llm_ei:
            await self.llm_ei.update_sampling_strategy_async(history)
        
        # 寻找下一个采样点
        next_point = self._find_next_point(gp, history)
        
        # 评估
        scalarized = self.evaluator.evaluate(**next_point)
        
        # 更新γ（如果启用代理模型增强）
        if self.enable_llm_surrogate:
            current_best = min([h['scalarized'] for h in history if h['valid']])
            self.llm_surrogate.update_gamma(current_best)
        
        result = {
            'iteration': iteration,
            'params': next_point,
            'scalarized': scalarized,
            'source': 'bo_iteration'
        }
        
        if self.verbose:
            print(f"\n[迭代 {iteration}] f = {scalarized:.4f} | "
                  f"I1={next_point['current1']:.2f}A, "
                  f"t1={next_point['charging_number']}, "
                  f"I2={next_point['current2']:.2f}A")
        
        return result
    
    def _find_next_point(self, gp, history) -> Dict[str, float]:
        """
        寻找下一个采样点
        
        ✨ v2.1更新:
        1. 适配数据标准化（归一化输入给GP，反归一化输出）
        2. 添加LLM权重的迭代衰减机制
        3. 使用LLM增强的EI或标准EI
        """
        # 获取当前最优
        valid_history = [h for h in history if h['valid']]
        y_max = -min([h['scalarized'] for h in valid_history])  # 转为最大化
        
        # ✨ 计算迭代进度 (用于LLM权重衰减)
        total_evals = len(history)
        expected_total = self.n_warmstart + self.n_random_init + self.n_iterations
        iter_ratio = min(1.0, total_evals / expected_total)
        
        # 定义采集函数
        def acquisition(x_normalized):
            """
            EI采集函数（可选LLM增强）
            
            参数:
                x_normalized: 归一化后的参数向量 [0, 1]^3
            
            返回:
                ei: 期望改进值（越大越好）
            """
            x_normalized = np.atleast_2d(x_normalized)
            
            # GP预测（输入已经是归一化的）
            mean, std = gp.predict(x_normalized, return_std=True)
            mean = -mean  # 转回最大化
            
            # 标准EI
            with np.errstate(divide='warn', invalid='warn'):
                z = (mean - y_max) / (std + 1e-10)
                ei = (mean - y_max) * norm.cdf(z) + std * norm.pdf(z)
                ei[std == 0.0] = 0.0
            
            # ✨ 如果启用LLM增强 + 添加衰减机制
            if self.enable_llm_ei and self.llm_ei.weight_function is not None:
                # 反归一化到原始参数空间（用于计算LLM权重）
                x_raw = self.llm_surrogate.scaler.inverse_transform(x_normalized)
                
                # 计算LLM权重
                weights = self.llm_ei.compute_llm_weights(x_raw)
                
                # ✨ 动态衰减：decay从1.0（开始）→0.0（结束）
                decay = max(0.0, 1.0 - iter_ratio)
                
                # 混合策略: ei * (weights ^ decay)
                # - 开始: decay=1, weights^1 = weights (强LLM引导)
                # - 结束: decay=0, weights^0 = 1 (纯EI)
                weighted_ei = ei * (weights ** decay)
                
                # if self.verbose and total_evals % 10 == 0:  # 每10次迭代打印一次
                #     print(f"  [LLM衰减] 进度={iter_ratio:.1%}, decay={decay:.2f}, "
                #           f"权重范围=[{weights.min():.3f}, {weights.max():.3f}]")
                
                return weighted_ei
            
            return ei
        
        # ✨ 使用scipy优化寻找最大EI点（在归一化空间中搜索）
        from scipy.optimize import minimize
        
        best_ei = -np.inf
        best_x_normalized = None
        
        # 多起点优化
        n_restarts = 10
        for _ in range(n_restarts):
            # 在归一化空间中随机初始化 [0, 1]^3
            x0_normalized = np.random.uniform(0, 1, 3)
            
            # 归一化空间的边界
            bounds_normalized = [(0.0, 1.0)] * 3
            
            # 最大化EI = 最小化-EI
            res = minimize(
                lambda x: -acquisition(x)[0],
                x0_normalized,
                bounds=bounds_normalized,
                method='L-BFGS-B'
            )
            
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x_normalized = res.x
        
        # ✨ 反归一化到原始参数空间
        best_x_raw = self.llm_surrogate.scaler.inverse_transform(
            best_x_normalized.reshape(1, -1)
        )[0]
        
        # 转换为参数字典
        next_point = {
            'current1': float(best_x_raw[0]),
            'charging_number': int(round(best_x_raw[1])),  # 确保是整数
            'current2': float(best_x_raw[2])
        }
        
        # 边界检查（防止数值误差导致越界）
        next_point['current1'] = np.clip(
            next_point['current1'], 
            self.pbounds['current1'][0], 
            self.pbounds['current1'][1]
        )
        next_point['charging_number'] = int(np.clip(
            next_point['charging_number'],
            self.pbounds['charging_number'][0],
            self.pbounds['charging_number'][1]
        ))
        next_point['current2'] = np.clip(
            next_point['current2'],
            self.pbounds['current2'][0],
            self.pbounds['current2'][1]
        )
        
        return next_point
    
    async def optimize_async(self) -> Dict:
        """
        执行完整的优化流程（异步）
        
        返回：
            results: 优化结果字典
        """
        start_time = datetime.now()
        
        # 阶段1: LLM Warm Start
        warmstart_results = await self._warmstart_async()
        self.optimization_history.extend(warmstart_results)
        
        # 阶段2: 随机初始化
        if self.n_random_init > 0:
            random_results = self._random_initialization(self.n_random_init)
            self.optimization_history.extend(random_results)
        
        # 阶段3: BO迭代
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"阶段 3: LLM-Enhanced Bayesian Optimization ({self.n_iterations} 迭代)")
            print("=" * 80)
        
        for i in range(self.n_iterations):
            result = await self._bo_iteration_async(i + 1)
            self.optimization_history.append(result)
            
            # 每10次迭代打印进度
            if self.verbose and (i + 1) % 10 == 0:
                current_best = self.evaluator.get_best_solution()
                print(f"\n[进度] 完成 {i+1}/{self.n_iterations} 迭代")
                print(f"  当前最优: f = {current_best['scalarized']:.4f}")
                print(f"  时间={current_best['objectives']['time']}, "
                      f"温度={current_best['objectives']['temp']:.2f}K, "
                      f"老化={current_best['objectives']['aging']:.6f}%")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # 收集最终结果
        results = self._compile_results(elapsed)
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def optimize(self) -> Dict:
        """执行优化（同步包装）"""
        return asyncio.run(self.optimize_async())
    
    def _compile_results(self, elapsed_time: float) -> Dict:
        """编译优化结果"""
        best_solution = self.evaluator.get_best_solution()
        pareto_front = self.evaluator.get_pareto_front()
        database = self.evaluator.export_database()
        
        results = {
            'best_solution': best_solution,
            'pareto_front': pareto_front,
            'optimization_history': self.optimization_history,
            'database': database,
            'statistics': self.evaluator.get_statistics(),
            'elapsed_time': elapsed_time,
            'config': {
                'llm_model': self.llm_model,
                'n_warmstart': self.n_warmstart,
                'n_random_init': self.n_random_init,
                'n_iterations': self.n_iterations,
                'enable_llm_warmstart': self.enable_llm_warmstart,
                'enable_llm_surrogate': self.enable_llm_surrogate,
                'enable_llm_ei': self.enable_llm_ei,
                'objective_weights': self.evaluator.weights
            }
        }
        
        # 如果启用了代理模型增强，记录γ历史
        if self.enable_llm_surrogate:
            fmin_history, gamma_history = self.llm_surrogate.get_gamma_history()
            results['gamma_history'] = {
                'fmin': fmin_history,
                'gamma': gamma_history
            }
        
        return results
    
    def _print_summary(self, results: Dict) -> None:
        """打印优化摘要"""
        print("\n" + "=" * 80)
        print("优化完成！")
        print("=" * 80)
        
        best = results['best_solution']
        print(f"\n【最优解】")
        print(f"  充电策略:")
        print(f"    - 第一阶段电流 (I1): {best['params']['current1']:.2f} A")
        print(f"    - 转换步数 (t1):     {best['params']['charging_number']}")
        print(f"    - 第二阶段电流 (I2): {best['params']['current2']:.2f} A")
        print(f"\n  性能指标:")
        print(f"    - 充电时间:   {best['objectives']['time']} 步")
        print(f"    - 峰值温度:   {best['objectives']['temp']:.2f} K")
        print(f"    - 容量衰减:   {best['objectives']['aging']:.6f} %")
        print(f"    - 标量化值:   {best['scalarized']:.4f}")
        
        print(f"\n【统计信息】")
        stats = results['statistics']
        print(f"  总评估次数: {stats['total_evaluations']}")
        print(f"  有效评估:   {stats['valid_evaluations']}")
        print(f"  帕累托解:   {len(results['pareto_front'])} 个")
        print(f"  运行时间:   {results['elapsed_time']:.1f} 秒")
        
        print("\n" + "=" * 80)
    
    def export_results(self, filename: str = None) -> str:
        """
        导出结果到JSON文件 - 使用ResultManager保存完整数据
        
        改进:
        1. 使用ResultManager.save_optimization_run保存完整数据
        2. 不仅保存最优解,还保存所有评估点
        3. 包含详细统计分析
        
        返回:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"llm_mobo_{timestamp}"
        else:
            run_id = filename.replace('.json', '')
        
        # 获取完整数据
        database = self.evaluator.export_database()
        best_solution = self.evaluator.get_best_solution()
        pareto_front = self.evaluator.get_pareto_front()
        statistics = self.evaluator.get_statistics()
        
        # 配置信息
        config = {
            'llm_model': self.llm_model,
            'n_warmstart': self.n_warmstart,
            'n_random_init': self.n_random_init,
            'n_iterations': self.n_iterations,
            'objective_weights': self.evaluator.weights,
            'enable_llm_warmstart': self.enable_llm_warmstart,
            'enable_llm_surrogate': self.enable_llm_surrogate,
            'enable_llm_ei': self.enable_llm_ei
        }
        
        # ====== 使用ResultManager保存 ======
        filepath = self.result_manager.save_optimization_run(
            run_id=run_id,
            database=database,
            best_solution=best_solution,
            pareto_front=pareto_front,
            config=config,
            statistics=statistics,
            metadata={
                'elapsed_time': self.optimization_history[-1].get('elapsed_time', 0) if self.optimization_history else 0
            }
        )
        
        print(f"\n[OK] 完整优化结果已保存")
        return filepath
    
    def plot_optimization_history(self, save_path: str = None) -> None:
        """绘制优化历史"""
        database = self.evaluator.export_database()
        
        if len(database) == 0:
            print("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 提取数据
        iterations = [d['eval_id'] for d in database]
        scalarized = [d['scalarized'] for d in database]
        time_vals = [d['objectives']['time'] for d in database]
        temp_vals = [d['objectives']['temp'] for d in database]
        aging_vals = [d['objectives']['aging'] for d in database]
        
        # 累积最优
        cumulative_best = []
        best_so_far = float('inf')
        for s in scalarized:
            if s < best_so_far:
                best_so_far = s
            cumulative_best.append(best_so_far)
        
        # 子图1: 标量化值
        axes[0, 0].plot(iterations, scalarized, 'o-', alpha=0.6, label='Evaluated Values')
        axes[0, 0].plot(iterations, cumulative_best, 'r-', linewidth=2, label='Best So Far')
        axes[0, 0].set_xlabel('Evaluation Number')
        axes[0, 0].set_ylabel('Scalarized Objective Value')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: 充电时间
        axes[0, 1].plot(iterations, time_vals, 'o-', alpha=0.6, color='blue')
        axes[0, 1].set_xlabel('Evaluation Number')
        axes[0, 1].set_ylabel('Charging Time (steps)')
        axes[0, 1].set_title('Charging Time Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: 峰值温度
        axes[1, 0].plot(iterations, temp_vals, 'o-', alpha=0.6, color='red')
        axes[1, 0].axhline(y=309, color='r', linestyle='--', label='Temperature Limit')
        axes[1, 0].set_xlabel('Evaluation Number')
        axes[1, 0].set_ylabel('Peak Temperature (K)')
        axes[1, 0].set_title('Peak Temperature Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 容量衰减
        axes[1, 1].plot(iterations, aging_vals, 'o-', alpha=0.6, color='green')
        axes[1, 1].set_xlabel('Evaluation Number')
        axes[1, 1].set_ylabel('Capacity Fade (%)')
        axes[1, 1].set_title('Capacity Fade Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
    
    def plot_pareto_front(self, save_path: str = None) -> None:
        """绘制帕累托前沿（3D）"""
        pareto_front = self.evaluator.get_pareto_front()
        
        if len(pareto_front) == 0:
            print("没有帕累托最优解")
            return
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 帕累托点
        time_vals = [p['objectives']['time'] for p in pareto_front]
        temp_vals = [p['objectives']['temp'] for p in pareto_front]
        aging_vals = [p['objectives']['aging'] for p in pareto_front]
        
        ax.scatter(time_vals, temp_vals, aging_vals, 
                  c='red', marker='o', s=100, label='Pareto Optimal Solutions')
        
        # 所有评估点
        database = self.evaluator.export_database()
        all_time = [d['objectives']['time'] for d in database]
        all_temp = [d['objectives']['temp'] for d in database]
        all_aging = [d['objectives']['aging'] for d in database]
        
        ax.scatter(all_time, all_temp, all_aging,
                  c='gray', marker='.', s=20, alpha=0.3, label='All Evaluated Points')
        
        ax.set_xlabel('Charging Time (steps)')
        ax.set_ylabel('Peak Temperature (K)')
        ax.set_zlabel('Capacity Fade (%)')
        ax.set_title('Pareto Front (3D Objective Space)')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"帕累托前沿图已保存至: {save_path}")
        else:
            plt.show()


# ============================================================
# 完整测试代码
# ============================================================

async def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("LLM-Enhanced Multi-Objective Bayesian Optimization")
    print("电池快充策略优化 - 完整集成测试")
    print("=" * 80)
    
    # 配置
    # API_KEY = 
    
    # 初始化优化器
    optimizer = LLMEnhancedMultiObjectiveBO(
        llm_api_key='sk-Evfy9FZGKZ31bpgdNsDSFfkWMopRE6EN4V4r801oRaIi8jm7',
        llm_base_url='https://api.nuwaapi.com/v1',
        llm_model="gpt-4o",
        n_warmstart=5,
        n_iterations=30,  # 测试用，正式运行建议50+
        n_random_init=0,
        objective_weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
        enable_llm_warmstart=True,
        enable_llm_surrogate=True,
        enable_llm_ei=True,
        verbose=True,
        save_dir='./results'
    )
    
    # 运行优化
    results = await optimizer.optimize_async()
    
    # 导出结果
    optimizer.export_results()
    
    # 绘制图表
    print("\n生成可视化图表...")
    optimizer.plot_optimization_history(save_path='./results/optimization_history.png')
    optimizer.plot_pareto_front(save_path='./results/pareto_front.png')
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())