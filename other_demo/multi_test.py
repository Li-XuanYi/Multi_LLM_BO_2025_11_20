"""
多目标评价器（Multi-Objective Evaluator）v3
完整实现：充电仿真 + 动态分位数边界 + 切比雪夫标量化

作者：Claude AI Assistant
日期：2025-01-15
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from SPM import SPM
import warnings


class MultiObjectiveEvaluator:
    """
    多目标充电策略评价器
    
    功能：
    1. 评估充电策略的三个目标（时间、温度、老化）
    2. 维护历史数据并动态更新分位数边界
    3. 归一化 + 切比雪夫标量化
    4. 约束违反处理
    """
    
    def __init__(
        self, 
        weights: Optional[Dict[str, float]] = None,
        update_interval: int = 10,
        temp_max: float = 309.0,
        max_steps: int = 300,
        verbose: bool = True,
        use_dynamic_bounds: bool = True,
        quantile_low: float = 0.05,  # Q5分位数
        quantile_high: float = 0.95  # Q95分位数
    ):
        """
        初始化评价器
        
        参数：
            weights: 各目标权重，默认 {'time': 0.4, 'temp': 0.35, 'aging': 0.25}
            update_interval: 分位数更新间隔（每N次评估更新一次）
            temp_max: 温度约束上限[K]
            max_steps: 单次充电最大步数限制
            verbose: 是否打印详细日志
            use_dynamic_bounds: 是否使用动态分位数边界
            quantile_low: 低分位数（Q5）
            quantile_high: 高分位数（Q95）
        """
        # 权重设置
        self.weights = weights or {
            'time': 0.4,    # 充电时间权重
            'temp': 0.35,   # 峰值温度权重
            'aging': 0.25   # 容量老化权重
        }
        
        # 验证权重和为1
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"权重之和必须为1.0，当前为 {weight_sum}")
        
        # 历史数据存储
        self.history = {
            'time': [],      # 充电步数
            'temp': [],      # 峰值温度[K]
            'aging': [],     # 容量衰减[%]
            'valid': []      # 是否满足约束
        }
        
        # 评估计数
        self.eval_count = 0
        self.update_interval = update_interval
        self.temp_max = temp_max
        self.max_steps = max_steps
        self.verbose = verbose
        self.use_dynamic_bounds = use_dynamic_bounds
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        
        # 动态分位数边界
        self.bounds = None
        
        # 临时固定边界（前10次使用）
        self.temp_bounds = {
            'time': {'best': 10, 'worst': 200},           # 步数
            'temp': {'best': 298.0, 'worst': temp_max},   # 温度[K]
            'aging': {'best': 0.0, 'worst': 0.5}          # 容量损失[%]
        }
        
        # 详细日志（用于后续分析）
        self.detailed_logs = []
        
        if self.verbose:
            print("=" * 70)
            print("多目标评价器已初始化")
            print("=" * 70)
            print(f"权重设置: {self.weights}")
            print(f"分位数更新间隔: 每 {update_interval} 次评估")
            print(f"温度约束上限: {temp_max} K")
            print(f"最大步数限制: {max_steps} 步")
            print(f"边界类型: {'动态分位数边界' if use_dynamic_bounds else '固定边界'}")
            if use_dynamic_bounds:
                print(f"  低分位数: {quantile_low:.0%}, 高分位数: {quantile_high:.0%}")
            print(f"临时边界（前10次）:")
            print(f"  时间: {self.temp_bounds['time']}")
            print(f"  温度: {self.temp_bounds['temp']}")
            print(f"  老化: {self.temp_bounds['aging']}")
            print("=" * 70)
    
    def evaluate(
        self, 
        current1: float, 
        charging_number: float, 
        current2: float
    ) -> float:
        """
        评估单次充电策略（主接口，BO调用）
        
        参数：
            current1: 第一阶段充电电流[A]
            charging_number: 阶段切换步数
            current2: 第二阶段充电电流[A]
        
        返回：
            scalarized_value: 切比雪夫标量化后的值（用于最小化）
        """
        # 1. 运行充电仿真
        sim_result = self._run_charging_simulation(current1, charging_number, current2)
        
        # 2. 更新历史
        self._update_history(sim_result)
        
        # 3. 每N次更新分位数边界
        if (self.eval_count % self.update_interval == 0 and 
            self.eval_count >= 10 and 
            self.use_dynamic_bounds):
            self._update_bounds()
        
        # 4. 归一化
        objectives_only = {
            'time': sim_result['time'],
            'temp': sim_result['temp'],
            'aging': sim_result['aging']
        }
        normalized = self._normalize(objectives_only)
        
        # 5. 切比雪夫标量化
        scalarized = self._chebyshev_scalarize(normalized)
        
        # 6. 约束违反惩罚（软约束）
        if sim_result['constraint_violation'] > 0:
            penalty = sim_result['constraint_violation'] * 0.5  # 每次违反增加0.5惩罚
            scalarized += penalty
        
        # 7. 记录详细日志
        log_entry = {
            'eval_id': self.eval_count,
            'params': {
                'current1': current1, 
                'charging_number': charging_number, 
                'current2': current2
            },
            'objectives': objectives_only,
            'normalized': normalized,
            'scalarized': scalarized,
            'valid': sim_result['valid'],
            'violations': sim_result['constraint_violation'],
            'termination': sim_result['termination'],
            'bounds_used': self.bounds if self.bounds else self.temp_bounds
        }
        self.detailed_logs.append(log_entry)
        
        # 8. 可选：打印进度
        if self.verbose and self.eval_count % 5 == 0:
            print(f"[Eval {self.eval_count}] 时间={sim_result['time']}, "
                  f"温度={sim_result['temp']:.2f}K, "
                  f"老化={sim_result['aging']:.6f}%, "
                  f"标量化={scalarized:.4f}, "
                  f"约束违反={sim_result['constraint_violation']}")
        
        return scalarized
    
    def _update_history(self, sim_result: Dict) -> None:
        """更新历史数据"""
        self.history['time'].append(sim_result['time'])
        self.history['temp'].append(sim_result['temp'])
        self.history['aging'].append(sim_result['aging'])
        self.history['valid'].append(sim_result['valid'])
        self.eval_count += 1
    
    def _run_charging_simulation(
        self, 
        current1: float, 
        charging_number: float, 
        current2: float
    ) -> Dict:
        """
        运行充电仿真并收集三个目标
        
        充电策略：
        1. 前 charging_number 步使用 current1
        2. 之后使用 current2
        3. 两阶段都应用电压限制（CV过渡策略）
        
        返回：
            {
                'time': 充电步数,
                'temp': 峰值温度[K],
                'aging': 容量衰减[%],
                'valid': 是否满足约束,
                'constraint_violation': 约束违反次数,
                'termination': 终止原因
            }
        """
        # 初始化SPM环境
        env = SPM(init_v=3.0, init_t=298)
        
        done = False
        step_count = 0
        constraint_violations = 0
        
        # 约束限制
        voltage_max = env.sett['constraints voltage max']  # 4.2V
        temp_max = env.sett['constraints temperature max']  # 309K
        
        # 记录峰值温度
        peak_temp = env.temp
        
        while not done and step_count < self.max_steps:
            # 选择充电电流（两阶段策略）
            if step_count < int(charging_number):
                current = current1
            else:
                current = current2
            
            # CV过渡策略：电压接近上限时指数衰减电流
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4.0))
            
            # 执行一步仿真
            _, done, _ = env.step(current)
            step_count += 1
            
            # 更新峰值温度
            if env.temp > peak_temp:
                peak_temp = env.temp
            
            # 检查约束违反（软约束：记录但继续）
            if env.voltage > voltage_max or env.temp > temp_max:
                constraint_violations += 1
        
        # 收集目标值
        objectives = {
            'time': step_count,                          # 充电步数
            'temp': peak_temp,                           # 峰值温度[K]
            'aging': env.capacity_fade_percent,          # 容量衰减[%]（真实值）
            'valid': (constraint_violations == 0),       # 是否满足约束
            'constraint_violation': constraint_violations,
            'termination': 'soc_reached' if done else 'max_steps'
        }
        
        return objectives
    
    def _update_bounds(self) -> None:
        """
        根据历史数据更新分位数边界（Q5/Q95）
        """
        if self.eval_count < 10:
            warnings.warn("历史数据不足10次，无法更新边界")
            return
        
        self.bounds = {}
        
        for obj_name in ['time', 'temp', 'aging']:
            if len(self.history[obj_name]) < 10:
                # 数据不足时使用临时边界
                self.bounds[obj_name] = self.temp_bounds[obj_name]
                continue
            
            # 计算分位数
            data = np.array(self.history[obj_name])
            
            # 排除异常值（超出3个标准差的点）
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:  # 避免除以0
                filtered_data = data[(data >= mean - 3*std) & (data <= mean + 3*std)]
                if len(filtered_data) > 0:
                    data = filtered_data
            
            q_low = np.quantile(data, self.quantile_low)
            q_high = np.quantile(data, self.quantile_high)
            
            # 确保best < worst
            best = min(q_low, q_high)
            worst = max(q_low, q_high)
            
            # 对于time和temp，最小值是best，最大值是worst
            # 对于aging，最小值是best，最大值是worst
            self.bounds[obj_name] = {
                'best': best,
                'worst': worst
            }
        
        if self.verbose:
            print(f"\n[边界更新] 评估次数: {self.eval_count}")
            print(f"  时间边界: best={self.bounds['time']['best']:.1f}, worst={self.bounds['time']['worst']:.1f}")
            print(f"  温度边界: best={self.bounds['temp']['best']:.2f}K, worst={self.bounds['temp']['worst']:.2f}K")
            print(f"  老化边界: best={self.bounds['aging']['best']:.6f}%, worst={self.bounds['aging']['worst']:.6f}%")
    
    def _normalize(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """
        使用当前边界归一化目标值到[0,1]
        
        参数：
            objectives: 原始目标值
        
        返回：
            归一化后的目标值
        """
        normalized = {}
        current_bounds = self.bounds if self.bounds else self.temp_bounds
        
        for obj_name, value in objectives.items():
            bounds = current_bounds[obj_name]
            
            # 处理边界相等的情况
            if np.isclose(bounds['worst'], bounds['best']):
                normalized[obj_name] = 0.0 if value <= bounds['best'] else 1.0
                continue
            
            # 归一化：(value - best) / (worst - best)
            # 确保值在[0, 1]范围内
            norm_value = (value - bounds['best']) / (bounds['worst'] - bounds['best'])
            norm_value = max(0.0, min(1.0, norm_value))  # 裁剪到[0, 1]
            
            normalized[obj_name] = norm_value
        
        return normalized
    
    def _chebyshev_scalarize(self, normalized: Dict[str, float]) -> float:
        """
        切比雪夫标量化
        
        参数：
            normalized: 归一化后的目标值
        
        返回：
            标量化值 = max(w_i × normalized_i)
        """
        # 计算每个目标的加权值
        weighted_values = [self.weights[obj_name] * normalized[obj_name] 
                          for obj_name in normalized]
        
        # 切比雪夫方法：取最大加权值
        scalarized = max(weighted_values)
        
        return scalarized
    
    def get_statistics(self) -> Dict:
        """
        获取当前统计信息（用于调试和分析）
        
        返回：
            包含历史统计的字典
        """
        if self.eval_count == 0:
            return {"message": "尚未进行任何评估"}
        
        # 获取当前使用的边界
        current_bounds = self.bounds if self.bounds else self.temp_bounds
        
        stats = {
            "total_evaluations": self.eval_count,
            "valid_evaluations": sum(self.history['valid']),
            "current_bounds": current_bounds,
            "history_summary": {
                "time": {
                    "min": np.min(self.history['time']),
                    "max": np.max(self.history['time']),
                    "mean": np.mean(self.history['time']),
                    "std": np.std(self.history['time']) if len(self.history['time']) > 1 else 0.0
                },
                "temp": {
                    "min": np.min(self.history['temp']),
                    "max": np.max(self.history['temp']),
                    "mean": np.mean(self.history['temp']),
                    "std": np.std(self.history['temp']) if len(self.history['temp']) > 1 else 0.0
                },
                "aging": {
                    "min": np.min(self.history['aging']),
                    "max": np.max(self.history['aging']),
                    "mean": np.mean(self.history['aging']),
                    "std": np.std(self.history['aging']) if len(self.history['aging']) > 1 else 0.0
                }
            }
        }
        return stats
    
    def get_best_solutions(self, top_n: int = 5) -> List[Dict]:
        """
        获取最佳解决方案（按标量化值排序）
        
        参数：
            top_n: 返回最佳解决方案的数量
        
        返回：
            按标量化值排序的最佳解决方案列表
        """
        if not self.detailed_logs:
            return []
        
        # 按标量化值排序
        sorted_logs = sorted(self.detailed_logs, key=lambda x: x['scalarized'])
        
        # 返回前top_n个
        return sorted_logs[:min(top_n, len(sorted_logs))]
    
    def plot_objectives(self) -> None:
        """
        绘制目标值随时间的变化（需要matplotlib）
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.eval_count == 0:
                print("没有评估数据可绘制")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 时间演化
            eval_ids = [log['eval_id'] for log in self.detailed_logs]
            times = [log['objectives']['time'] for log in self.detailed_logs]
            temps = [log['objectives']['temp'] for log in self.detailed_logs]
            agings = [log['objectives']['aging'] for log in self.detailed_logs]
            scalarized = [log['scalarized'] for log in self.detailed_logs]
            
            ax1.plot(eval_ids, times, 'b-', alpha=0.7)
            ax1.set_title('充电时间演化')
            ax1.set_xlabel('评估次数')
            ax1.set_ylabel('时间步数')
            ax1.grid(True)
            
            ax2.plot(eval_ids, temps, 'r-', alpha=0.7)
            ax2.set_title('峰值温度演化')
            ax2.set_xlabel('评估次数')
            ax2.set_ylabel('温度 [K]')
            ax2.grid(True)
            
            ax3.plot(eval_ids, agings, 'g-', alpha=0.7)
            ax3.set_title('容量衰减演化')
            ax3.set_xlabel('评估次数')
            ax3.set_ylabel('衰减 [%]')
            ax3.grid(True)
            
            ax4.plot(eval_ids, scalarized, 'm-', alpha=0.7)
            ax4.set_title('标量化值演化')
            ax4.set_xlabel('评估次数')
            ax4.set_ylabel('标量化值')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig('multi_objective_evolution.png', dpi=300)
            plt.show()
            
            # 相关性热图
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_data = np.array([
                [log['objectives']['time'] for log in self.detailed_logs],
                [log['objectives']['temp'] for log in self.detailed_logs],
                [log['objectives']['aging'] for log in self.detailed_logs],
                [log['scalarized'] for log in self.detailed_logs]
            ]).T
            
            corr = np.corrcoef(corr_data.T)
            sns.heatmap(corr, 
                        annot=True, 
                        fmt='.2f', 
                        xticklabels=['时间', '温度', '老化', '标量化'],
                        yticklabels=['时间', '温度', '老化', '标量化'],
                        cmap='coolwarm',
                        center=0)
            plt.title('目标值相关性分析')
            plt.tight_layout()
            plt.savefig('correlation_analysis.png', dpi=300)
            plt.show()
            
        except ImportError:
            print("需要安装 matplotlib 和 seaborn 来使用绘图功能")
            print("安装命令: pip install matplotlib seaborn")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("\n【测试】MultiObjectiveEvaluator v3")
    print("完整功能测试\n")
    
    # === 测试1：基础功能测试 ===
    print("\n【测试1】基础功能测试")
    print("-" * 50)
    
    # 初始化评价器
    evaluator = MultiObjectiveEvaluator(
        verbose=True,
        use_dynamic_bounds=True,
        update_interval=5
    )
    
    # 测试多个评估
    test_cases = [
        (4.0, 10, 2.0),
        (5.0, 15, 3.0),
        (6.0, 20, 4.0),
        (3.0, 8, 1.5),
        (7.0, 25, 5.0),
        (4.5, 12, 2.5),
        (5.5, 18, 3.5),
        (6.5, 22, 4.5),
        (3.5, 10, 2.0),
        (7.5, 30, 5.5)
    ]
    
    print("\n运行多次评估...")
    for i, (c1, cn, c2) in enumerate(test_cases):
        result = evaluator.evaluate(
            current1=c1,
            charging_number=cn,
            current2=c2
        )
        print(f"评估 {i+1} - 电流1={c1}A, 切换步数={cn}, 电流2={c2}A -> 标量化值={result:.4f}")
    
    # 获取统计信息
    print("\n【测试2】统计信息")
    stats = evaluator.get_statistics()
    print(f"总评估次数: {stats['total_evaluations']}")
    print(f"有效评估次数: {stats['valid_evaluations']}")
    print("\n历史统计:")
    for obj_name in ['time', 'temp', 'aging']:
        print(f"  {obj_name}: min={stats['history_summary'][obj_name]['min']:.2f}, "
              f"max={stats['history_summary'][obj_name]['max']:.2f}, "
              f"mean={stats['history_summary'][obj_name]['mean']:.2f}")
    
    # === 测试3：最佳解决方案 ===
    print("\n【测试3】最佳解决方案")
    best_solutions = evaluator.get_best_solutions(top_n=3)
    for i, sol in enumerate(best_solutions):
        params = sol['params']
        print(f"\n第{i+1}佳解决方案 (评估ID={sol['eval_id']}):")
        print(f"  参数: current1={params['current1']:.1f}A, "
              f"charging_number={params['charging_number']:.1f}, "
              f"current2={params['current2']:.1f}A")
        print(f"  目标值: time={sol['objectives']['time']}, "
              f"temp={sol['objectives']['temp']:.2f}K, "
              f"aging={sol['objectives']['aging']:.6f}%")
        print(f"  标量化值: {sol['scalarized']:.4f}")
        print(f"  边界使用: {sol['bounds_used']}")
    
    # === 测试4：边界更新测试 ===
    print("\n【测试4】边界更新测试")
    print("运行更多评估以触发边界更新...")
    
    for i in range(11, 21):
        # 使用随机参数
        c1 = np.random.uniform(3.0, 8.0)
        cn = np.random.uniform(5, 30)
        c2 = np.random.uniform(1.0, 6.0)
        
        result = evaluator.evaluate(
            current1=c1,
            charging_number=cn,
            current2=c2
        )
        
        if i % 5 == 0:
            print(f"评估 {i} - 标量化值={result:.4f}")
    
    # 检查更新后的边界
    print("\n更新后的边界:")
    updated_stats = evaluator.get_statistics()
    if updated_stats['current_bounds'] != evaluator.temp_bounds:
        print("✓ 边界已成功更新")
        print(f"当前边界: {updated_stats['current_bounds']}")
    else:
        print("✗ 边界未更新")
    
    # === 测试5：可视化（如果安装了matplotlib） ===
    try:
        print("\n【测试5】可视化结果")
        evaluator.plot_objectives()
        print("✓ 可视化图表已保存")
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 多目标评价器完整版测试通过！")
    print("=" * 70)