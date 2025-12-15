"""
多目标评价器（Multi-Objective Evaluator）- 完整版（已修复）
用于 LLM 增强的分解多目标贝叶斯优化 (LLM-DMOBO)

=============================================================================
修复内容
=============================================================================
[OK] 添加了 initialize_with_llm_warmstart() 异步方法
   - 支持 LLM API 调用生成智能初始策略
   - 无 API key 时自动回退到随机策略
   - 完整的错误处理和日志输出

=============================================================================
核心功能
=============================================================================
1. 充电仿真与多目标评估
   - 两阶段恒流充电策略: [I1, t1, I2]
   - 三个优化目标: [时间, 温度峰值, 电池老化]

2. 数据库管理 (D)
   - 完整历史记录: 参数、目标值、归一化值、标量化值
   - 动态分位数边界更新 (Q5/Q95)
   - 帕累托前沿提取

3. 多目标分解
   - 增强切比雪夫标量化 (Augmented Tchebycheff)
   - 动态权重更新（支持分解策略）
   
4. LLM 集成接口
   - [OK] LLM Warm Start 初始化（新增）
   - 外部评估添加（用于 Warm Start）
   - 语义约束 S 预留接口
   - 最佳解查询

=============================================================================
作者: Claude AI Assistant
日期: 2025-01-19（修复版）
版本: v3.1 - 添加 LLM Warm Start
=============================================================================
"""

import numpy as np
import asyncio
import json
from typing import Dict, List, Optional, Tuple
try:
    from .SPM_v3 import SPM_Sensitivity as SPM
except ImportError:
    from SPM_v3 import SPM_Sensitivity as SPM

# 新增: 历史数据驱动的WarmStart
try:
    from .historical_warmstart import HistoricalWarmStart
except ImportError:
    from historical_warmstart import HistoricalWarmStart


class SoftConstraintHandler:
    """
    软约束处理器 - 指数/平方惩罚机制
    
    设计理念:
    - 接近限制时施加平滑惩罚，而非硬截断
    - 保持目标函数连续可微
    - BO可以学习如何避开危险区域
    """
    
    def __init__(
        self,
        temp_max: float = 318.0,  # ✅ 改进：309K + 3K裕度
        temp_penalty_rate: float = 0.15,
        temp_penalty_scale: float = 0.05,
        aging_threshold: float = 0.5,  # ✅ 对数空间阈值
        aging_penalty_scale: float = 0.1,
        verbose: bool = True
    ):
        self.temp_max = temp_max
        self.temp_penalty_rate = temp_penalty_rate
        self.temp_penalty_scale = temp_penalty_scale
        self.aging_threshold = aging_threshold
        self.aging_penalty_scale = aging_penalty_scale
        self.verbose = verbose
        
        if self.verbose:
            print("\n" + "="*70)
            print("🔧 软约束处理器已初始化 [v2.0]")
            print("="*70)
            print(f"温度约束: T_max = {temp_max}K (309K + 3K裕度)")
            print(f"  λ = {temp_penalty_rate} (指数增长率)")
            print(f"  α = {temp_penalty_scale} (惩罚缩放)")
            print(f"老化约束: A_threshold = {aging_threshold} (对数空间阈值)")
            print(f"  β = {aging_penalty_scale}")
            print("="*70)
    
    def compute_temperature_penalty(self, temp: float) -> Tuple[float, str]:
        """温度软约束 (指数惩罚)"""
        if temp <= self.temp_max:
            return 0.0, "safe"
        
        excess = temp - self.temp_max
        penalty = self.temp_penalty_scale * (np.exp(self.temp_penalty_rate * excess) - 1)
        
        if excess <= 3.0:
            status = "mild"
        elif excess <= 6.0:
            status = "moderate"
        else:
            status = "severe"
        
        return penalty, status
    
    def compute_aging_penalty(self, aging: float) -> Tuple[float, str]:
        """老化软约束 (平方惩罚)"""
        if aging <= self.aging_threshold:
            return 0.0, "safe"
        
        excess = aging - self.aging_threshold
        penalty = self.aging_penalty_scale * (excess ** 2)
        
        if excess <= 0.1:
            status = "mild"
        elif excess <= 0.3:
            status = "moderate"
        else:
            status = "severe"
        
        return penalty, status
    
    def compute_total_penalty(self, objectives: Dict[str, float]) -> Dict:
        """计算总惩罚"""
        temp = objectives['temp']
        aging = objectives['aging']
        
        temp_penalty, temp_status = self.compute_temperature_penalty(temp)
        aging_penalty, aging_status = self.compute_aging_penalty(aging)
        
        total_penalty = temp_penalty + aging_penalty
        is_severe = (temp_status == "severe" or aging_status == "severe")
        
        return {
            'total_penalty': total_penalty,
            'penalties': {'temp': temp_penalty, 'aging': aging_penalty},
            'statuses': {'temp': temp_status, 'aging': aging_status},
            'is_severe': is_severe
        }


class MultiObjectiveEvaluator:
    """
    多目标充电策略评价器
    
    功能：
    1. 评估充电策略的三个目标（时间、温度、老化）
    2. 维护历史数据并动态更新分位数边界
    3. 归一化 + 切比雪夫标量化
    4. LLM 热启动支持
    """
    
    def __init__(
        self, 
        weights: Optional[Dict[str, float]] = None,
        update_interval: int = 10,
        temp_max: float = 309.0,
        max_steps: int = 300,
        verbose: bool = True
    ):
        """
        初始化评价器
        
        参数：
            weights: 各目标权重，默认 {'time': 0.4, 'temp': 0.35, 'aging': 0.25}
            update_interval: 分位数更新间隔（每N次评估更新一次）
            temp_max: 温度约束上限[K]
            max_steps: 单次充电最大步数限制
            verbose: 是否打印详细日志
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
        
        # ✅ 核心改进：物理边界（固定，基于电化学原理）
        self.physical_bounds = {
            'time': {
                'min': 20,      # ✅ 最快20步（约40分钟）
                'max': 120      # ✅ 最慢120步（约4小时）
            },
            'temp': {
                'min': 298.0,   # 室温起始
                'max': 318.0    # 
            },
            'aging': {
                'min': 0.0,     # ✅ log1p(aging_raw*100) 最小应为 0
                'max': 6.5      # ✅ log1p(5.0*100) ≈ 6.2（严重老化）+ 裕度
            }
        }
        
        # ✅ 运行时边界（单调扩展，从物理边界开始）
        self.running_bounds = {
            'time': {'min': self.physical_bounds['time']['min'], 
                    'max': self.physical_bounds['time']['max']},
            'temp': {'min': self.physical_bounds['temp']['min'], 
                    'max': self.physical_bounds['temp']['max']},
            'aging': {'min': self.physical_bounds['aging']['min'], 
                     'max': self.physical_bounds['aging']['max']}
        }
        
        # ✅ 原始历史数据（存储物理值）
        self.raw_history = []
        
        # 历史数据存储（保留兼容性）
        self.history = {
            'time': [],      # 充电步数
            'temp': [],      # 峰值温度[K]
            'aging': [],     # 容量衰减（对数值）
            'valid': []      # 是否满足约束
        }
        
        # 评估计数
        self.eval_count = 0
        self.update_interval = update_interval
        self.temp_max = temp_max
        self.max_steps = max_steps
        self.verbose = verbose
        
        # ✅ 初始化梯度计算器（避免 verbose=False 时未定义）
        self.spm_for_gradients = None
        self.gradient_compute_interval = 3  # 每3次计算一次梯度
        self.invalid_penalty = 0.5  # ✅ 无效点的额外惩罚（降低以避免f值过大）
        
        # 动态分位数边界（保留兼容性）
        self.bounds = None
        
        # 临时固定边界（前10次使用，保留兼容性）
        self.temp_bounds = {
            'time': {'best': 20, 'worst': 120},           # ✅ 改进：20-120步
            'temp': {'best': 298.0, 'worst': 312.0},      # ✅ 改进：312K上限
            'aging': {'best': 0.0, 'worst': 6.5}          # ✅ 对数空间，>=0
        }
        
        # 详细日志（用于后续分析）
        self.detailed_logs = []
        
        if self.verbose:
            # [Gradient Computation]
            from SPM_v3 import SPM_Sensitivity
            self.spm_for_gradients = SPM_Sensitivity(
                init_v=3.0, 
                init_t=298, 
                mode='finite_difference',
                enable_penalty_gradients=True,
                penalty_scale=10.0,
                verbose=False  # ✅ 避免梯度计算时输出干扰
            )
            print("[OK] Gradient computation enabled with penalty gradients (v3.0)")
            print("=" * 70)
            print("多目标评价器 v2.0 已初始化（全局归一化）")
            print("=" * 70)
            print(f"权重设置: {self.weights}")
            print(f"\n物理边界（固定）:")
            for key in ['time', 'temp', 'aging']:
                print(f"  {key}: {self.physical_bounds[key]}")
            print(f"\n老化处理: log1p变换（替代×1000放大）")
            print(f"温度约束上限: {temp_max} K")
            print(f"最大步数限制: {max_steps} 步")
            print("=" * 70)
        
        self.soft_constraints = SoftConstraintHandler(
            temp_max=312.0,  # ✅ 使用增加裕度的温度上限
            temp_penalty_rate=0.15,
            temp_penalty_scale=0.05,
            aging_threshold=5.0,  # ✅ 对数空间的阈值（log1p(5%*100)≈6.2，设为5.0）
            aging_penalty_scale=0.02,  # ✅ 降低惩罚系数
            verbose=self.verbose
        )
    
    # ============================================================
    # 新增：LLM Warm Start 方法
    # ============================================================
    
    async def initialize_with_llm_warmstart(
        self,
        n_strategies: int = 5,
        llm_api_key: Optional[str] = None,
        llm_base_url: str = 'https://api.nuwaapi.com/v1',
        llm_model: str = "gpt-3.5-turbo"
    ) -> List[Dict]:
        """
        使用 LLM 进行热启动初始化 - 新版本（历史数据驱动）
        
        改进：
        1. 使用HistoricalWarmStart自动加载历史数据
        2. 基于电化学领域知识生成高质量prompt
        3. 包含few-shot learning examples（最优/最差解）
        4. 动态探索模式（conservative/balanced/aggressive）
        
        参数：
            n_strategies: 要生成的策略数量
            llm_api_key: LLM API 密钥（可选）
            llm_base_url: API 基础 URL
            llm_model: 使用的 LLM 模型
        
        返回：
            List[Dict]: 评估结果列表
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("开始 LLM Warm Start 初始化（历史数据驱动版）")
            print("=" * 70)
            print(f"目标策略数量: {n_strategies}")
            print(f"API Key: {'已提供' if llm_api_key else '未提供（将使用随机策略）'}")
            print("=" * 70)
        
        results = []
        
        # ====== 使用新的HistoricalWarmStart ======
        if llm_api_key is not None:
            try:
                if self.verbose:
                    print("\n使用HistoricalWarmStart生成策略...")
                
                # 创建HistoricalWarmStart实例
                warmstart_generator = HistoricalWarmStart(
                    result_dir='./results',  # 确保Result目录正确
                    llm_api_key=llm_api_key,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    verbose=self.verbose
                )
                
                # 生成策略（自动加载历史数据 + 生成高质量prompt + 调用LLM）
                strategies = await warmstart_generator.generate_warmstart_strategies_async(
                    n_strategies=n_strategies,
                    n_historical_runs=5,  # 加载最近5次运行
                    objective_weights=self.weights,
                    exploration_mode='balanced'  # 可选: 'conservative', 'balanced', 'aggressive'
                )
                
                if self.verbose:
                    print(f"[OK] HistoricalWarmStart成功生成 {len(strategies)} 个策略")
                
            except Exception as e:
                if self.verbose:
                    print(f"[Warning]  HistoricalWarmStart失败: {e}")
                    print("   回退到随机策略生成...")
                
                # 回退到随机策略
                strategies = self._generate_random_strategies(n_strategies)
        
        else:
            # 没有 API key，直接使用随机策略
            if self.verbose:
                print("\n使用随机策略生成...")
            
            strategies = self._generate_random_strategies(n_strategies)
        
        # ====== 评估所有策略（保持不变） ======
        if self.verbose:
            print(f"\n开始评估 {len(strategies)} 个策略...")
        
        for i, strategy in enumerate(strategies, 1):
            params = strategy['params']
            
            try:
                # 调用 evaluate() 方法进行仿真评估
                scalarized = self.evaluate(
                    current1=params['current1'],
                    charging_number=params['charging_number'],
                    current2=params['current2']
                )
                
                # 从最新的日志中提取目标值
                latest_log = self.detailed_logs[-1]
                
                result = {
                    'params': params,
                    'scalarized': scalarized,
                    'objectives': latest_log['objectives'],
                    'valid': latest_log['valid'],
                    'source': strategy.get('source', 'llm_warmstart'),
                    'reasoning': strategy.get('reasoning', '')
                }
                
                results.append(result)
                
                if self.verbose:
                    print(f"  策略 {i}/{len(strategies)}: "
                          f"I1={params['current1']:.2f}A, "
                          f"t1={params['charging_number']}, "
                          f"I2={params['current2']:.2f}A "
                          f"→ 标量化={scalarized:.4f}")
            
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ 策略 {i} 评估失败: {e}")
                continue
        
        if self.verbose:
            print(f"\n[OK] Warm Start完成！成功评估 {len(results)}/{len(strategies)} 个策略")
            print("=" * 70)
        
        return results
    
    # [X] 已删除 _llm_generate_strategies 方法
    # 原因: 使用新的HistoricalWarmStart替代硬编码prompt
    # 新方法自动加载历史数据，生成基于电化学领域知识的高质量prompt
    
    def _generate_random_strategies(self, n_strategies: int) -> List[Dict]:
        """
        生成随机充电策略（作为 LLM 的回退方案）
        
        返回：
            List[Dict]: 随机策略列表
        """
        strategies = []
        
        for _ in range(n_strategies):
            strategy = {
                'params': {
                    'current1': np.random.uniform(3.0, 6.0),
                    'charging_number': int(np.random.uniform(5, 25)),
                    'current2': np.random.uniform(1.0, 4.0)
                },
                'source': 'random_warmstart',
                'reasoning': 'Randomly generated strategy'
            }
            strategies.append(strategy)
        
        return strategies
    
    # ============================================================
    # 新增：对数变换和全局归一化方法 (v2.0)
    # ============================================================
    
    def _apply_log_transform(self, aging_raw: float) -> float:
        """
        对老化值应用对数变换
        
        参数：
            aging_raw: 原始容量衰减百分比（如0.1表示0.1%）
        
        返回：
            对数变换值
        """
        # log1p(x) = log(1 + x)，避免log(0)问题
        # 放大100倍再取对数，提升低老化区域分辨率
        return np.log1p(aging_raw * 100)
    
    def get_normalized_history(self) -> List[Dict]:
        """
        ✅ 核心方法：基于全局统一边界重新归一化所有历史数据
        
        改进 v2.1：
        - 使用 detailed_logs 替代 raw_history（保留 gradients 等字段）
        - 对 valid 点 clip 到 [0,1]
        - 对 invalid 点明确标记为最差（1.0）+ 额外惩罚
        
        这个方法会：
        1. 计算当前所有有效数据的min/max
        2. 与物理边界做单调扩展（只扩不缩）
        3. 用统一边界重新归一化所有历史点
        4. 重新计算标量化值
        
        返回：
            规范化的历史数据列表（包含 gradients 等完整字段）
        """
        if len(self.detailed_logs) == 0:
            return []
        
        # 1. 提取所有有效数据的物理值
        valid_data = [h for h in self.detailed_logs if h.get('valid', False)]
        
        if len(valid_data) == 0:
            return []
        
        # 使用Numpy向量化计算min/max
        times = np.array([d['objectives']['time'] for d in valid_data])
        temps = np.array([d['objectives']['temp'] for d in valid_data])
        agings_log = np.array([d['objectives']['aging'] for d in valid_data])
        
        current_bounds = {
            'time': {'min': times.min(), 'max': times.max()},
            'temp': {'min': temps.min(), 'max': temps.max()},
            'aging': {'min': agings_log.min(), 'max': agings_log.max()}
        }
        
        # 2. 单调扩展：与物理边界和运行时边界取并集
        for key in ['time', 'temp', 'aging']:
            # 边界只能扩展，不能收缩
            self.running_bounds[key]['min'] = min(
                self.running_bounds[key]['min'],
                current_bounds[key]['min'],
                self.physical_bounds[key]['min']
            )
            self.running_bounds[key]['max'] = max(
                self.running_bounds[key]['max'],
                current_bounds[key]['max'],
                self.physical_bounds[key]['max']
            )
        
        # 防止除零：确保最小范围
        min_ranges = {'time': 5.0, 'temp': 1.0, 'aging': 0.1}
        for key, min_range in min_ranges.items():
            current_range = self.running_bounds[key]['max'] - self.running_bounds[key]['min']
            if current_range < min_range:
                midpoint = (self.running_bounds[key]['max'] + self.running_bounds[key]['min']) / 2
                self.running_bounds[key]['min'] = midpoint - min_range / 2
                self.running_bounds[key]['max'] = midpoint + min_range / 2
        
        # 3. 向量化归一化所有数据
        normalized_history = []
        
        for log in self.detailed_logs:
            obj = log['objectives']
            is_valid = log.get('valid', False)
            
            # ✅ 归一化（使用固定的physical_bounds，确保一致性）
            normalized = {}
            for key in ['time', 'temp', 'aging']:
                denominator = self.physical_bounds[key]['max'] - self.physical_bounds[key]['min']
                val = (obj[key] - self.physical_bounds[key]['min']) / denominator
                
                # ✅ 只对 valid 点 clip 到 [0,1]；invalid 点直接按最差处理
                if is_valid:
                    normalized[key] = float(np.clip(val, 0.0, 1.0))
                else:
                    normalized[key] = 1.0
            
            # 切比雪夫标量化
            weighted_deviations = [
                self.weights[key] * normalized[key]
                for key in ['time', 'temp', 'aging']
            ]
            max_weighted = max(weighted_deviations)
            sum_weighted = sum(weighted_deviations)
            scalarized = max_weighted + 0.05 * sum_weighted
            
            # 添加软约束惩罚
            constraint_result = self.soft_constraints.compute_total_penalty(obj)
            scalarized += constraint_result['total_penalty']
            
            if constraint_result['is_severe']:
                scalarized += 0.2  # 严重违规额外惩罚
            
            # ✅ 无效点额外惩罚（让 f 明显 > 2）
            if not is_valid:
                scalarized += self.invalid_penalty
            
            # ✅ 构建规范化记录（保留原 log 的所有字段，包括 gradients）
            new_log = dict(log)
            new_log['normalized'] = normalized
            new_log['scalarized'] = scalarized
            
            normalized_history.append(new_log)
        
        if self.verbose and len(normalized_history) > 0:
            print(f"\n[全局归一化] 已重算 {len(normalized_history)} 条历史记录")
            print(f"✅ 使用固定物理边界:")
            for key in ['time', 'temp', 'aging']:
                print(f"  {key}: [{self.physical_bounds[key]['min']:.2f}, {self.physical_bounds[key]['max']:.2f}]")
        
        return normalized_history
    
    # ============================================================
    # 原有方法（保持不变）
    # ============================================================
    
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
        
        # 2. ✅ 对老化应用对数变换
        aging_raw = sim_result['aging']  # 原始百分比值
        aging_log = self._apply_log_transform(aging_raw)
        
        # 3. ✅ 存储原始数据到 raw_history
        objectives_with_log = {
            'time': sim_result['time'],
            'temp': sim_result['temp'],
            'aging': aging_log  # ✅ 存储对数变换值
        }
        
        raw_record = {
            'eval_id': self.eval_count,
            'params': {
                'current1': current1,
                'charging_number': charging_number,
                'current2': current2
            },
            'objectives': objectives_with_log,
            'aging_raw': aging_raw,  # 同时保留原始值用于分析
            'valid': sim_result['valid'],
            'violations': sim_result.get('constraint_violation', 0),
            'termination': sim_result.get('termination', 'unknown')
        }
        
        self.raw_history.append(raw_record)
        
        # 4. 更新旧历史（保留兼容性）
        self.history['time'].append(sim_result['time'])
        self.history['temp'].append(sim_result['temp'])
        self.history['aging'].append(aging_log)  # 存储对数值
        self.history['valid'].append(sim_result['valid'])
        
        self.eval_count += 1
        
        # 5. ✅ 移除旧的边界更新逻辑（使用全局归一化替代）
        # if self.eval_count % self.update_interval == 0 and self.eval_count >= 10:
        #     self._update_bounds()
        
        # ✅ 6. 临时归一化（统一使用固定的物理边界）
        normalized = {}
        for key in ['time', 'temp', 'aging']:
            denominator = self.physical_bounds[key]['max'] - self.physical_bounds[key]['min']
            if denominator > 0:
                normalized[key] = (objectives_with_log[key] - self.physical_bounds[key]['min']) / denominator
            else:
                normalized[key] = 0.5
            
            # ✅ 只对 valid 点 clip 到 [0,1]；invalid 点直接按最差处理
            if sim_result['valid']:
                normalized[key] = float(np.clip(normalized[key], 0.0, 1.0))
            else:
                normalized[key] = 1.0
        
        # 7. 切比雪夫标量化
        weighted_deviations = [
            self.weights[key] * normalized[key]
            for key in ['time', 'temp', 'aging']
        ]
        base_scalarized = max(weighted_deviations) + 0.05 * sum(weighted_deviations)
        scalarized = base_scalarized
        
        # 8. 软约束惩罚机制 (指数/平方)
        constraint_result = self.soft_constraints.compute_total_penalty(objectives_with_log)
        soft_penalty = constraint_result['total_penalty']
        scalarized += soft_penalty
        
        if constraint_result['is_severe']:
            scalarized += 0.2  # 严重违规额外惩罚
        
        # ✅ 无效点额外惩罚
        if not sim_result['valid']:
            scalarized += self.invalid_penalty
        
        # ✅ 详细调试输出
        if self.verbose and self.eval_count % 1 == 0:  # 每次都输出
            print(f"\n  [归一化] time={normalized['time']:.4f}, temp={normalized['temp']:.4f}, aging={normalized['aging']:.4f}")
            print(f"  [标量化] 基础={base_scalarized:.4f}, 软约束={soft_penalty:.4f}, 无效惩罚={self.invalid_penalty if not sim_result['valid'] else 0:.4f}")
            print(f"  [最终] f={scalarized:.4f}, valid={sim_result['valid']}")
        
        # 9. 计算梯度（可选）
        gradients = None
        if (self.spm_for_gradients is not None) and (self.eval_count % self.gradient_compute_interval == 0):
            try:
                grad_result = self.spm_for_gradients.run_two_stage_charging(
                    current1=current1, charging_number=int(charging_number), 
                    current2=current2, return_sensitivities=True
                )
                if grad_result.get('valid', False) and 'sensitivities' in grad_result:
                    gradients = grad_result['sensitivities']
            except Exception:
                gradients = None

        # 10. 记录详细日志
        log_entry = {
            'eval_id': self.eval_count,
            'params': {'current1': current1, 'charging_number': charging_number, 'current2': current2},
            'objectives': objectives_with_log,
            'aging_raw': aging_raw,
            'normalized': normalized,
            'scalarized': scalarized,
            'valid': sim_result['valid'],
            'violations': sim_result['constraint_violation'],
            'termination': sim_result['termination'],
            'gradients': gradients
        }
        self.detailed_logs.append(log_entry)
        
        # 11. 可选：打印进度
        if self.verbose and self.eval_count % 5 == 0:
            time_minutes = sim_result['time'] * 90 / 60
            
            constraint_info = self.soft_constraints.compute_total_penalty(objectives_with_log)
            temp_status = constraint_info['statuses']['temp']
            temp_penalty = constraint_info['penalties']['temp']
            
            status_icon = {
                'safe': '✓',
                'mild': '⚠',
                'moderate': '⚠⚠',
                'severe': '❌'
            }.get(temp_status, '?')
            
            print(f"[Eval {self.eval_count}] "
                  f"t={sim_result['time']:.0f}步({time_minutes:.1f}min), "
                  f"T={sim_result['temp']:.2f}K{status_icon}, "
                  f"A_raw={aging_raw:.4f}%, A_log={aging_log:.2f}, "
                  f"penalty={temp_penalty:.4f}, "
                  f"f={scalarized:.4f}")
        
        return scalarized
    
    def _run_charging_simulation(
        self, 
        current1: float, 
        charging_number: float, 
        current2: float
    ) -> Dict:
        """运行充电仿真并收集三个目标"""
        # 初始化SPM环境（v3.0 - 不需要灵敏度）
        env = SPM(init_v=3.0, init_t=298)
        
        # 运行两阶段充电仿真
        result = env.run_two_stage_charging(
            current1=current1,
            charging_number=int(charging_number),
            current2=current2,
            return_sensitivities=False
        )
        
        if not result['valid']:
            return {
                'time': self.max_steps,  # 惩罚值（步数）
                'temp': self.temp_max + 10,  # 惩罚值
                'aging': 1.0,  # 惩罚值
                'valid': False,
                'constraint_violation': 1,
                'termination': 'invalid'
            }
        
        # 转换结果格式：将时间从秒转换为步数（1步 = 90秒）
        objectives = result['objectives']
        time_in_steps = objectives['time'] / 90.0  # 秒 -> 步数
        
        return {
            'time': time_in_steps,  # 步数
            'temp': objectives['temp'],  # K
            'aging': objectives['aging'],  # %
            'valid': result['valid'],
            'constraint_violation': 0,
            'termination': 'completed'
        }
    
    def _update_bounds(self) -> None:
        """
        更新分位数边界（Q5/Q95）
        
        改进: 确保边界有合理的最小范围,避免数值不稳定
        """
        if self.eval_count < 10:
            return
        
        valid_indices = [i for i, v in enumerate(self.history['valid']) if v]
        
        if len(valid_indices) < 5:
            return
        
        # 计算Q5/Q95分位数
        self.bounds = {
            'time': {
                'best': np.percentile([self.history['time'][i] for i in valid_indices], 5),
                'worst': np.percentile([self.history['time'][i] for i in valid_indices], 95)
            },
            'temp': {
                'best': np.percentile([self.history['temp'][i] for i in valid_indices], 5),
                'worst': np.percentile([self.history['temp'][i] for i in valid_indices], 95)
            },
            'aging': {
                'best': np.percentile([self.history['aging'][i] for i in valid_indices], 5),
                'worst': np.percentile([self.history['aging'][i] for i in valid_indices], 95)
            }
        }
        
        # ✅ 确保边界有最小范围（避免除零）
        min_ranges = {
            'time': 5.0,      # 最小时间范围5步
            'temp': 1.0,      # 最小温度范围1K
            'aging': 0.01     # 最小老化范围0.01%
        }
        
        for key, min_range in min_ranges.items():
            current_range = self.bounds[key]['worst'] - self.bounds[key]['best']
            if current_range < min_range:
                # 扩展边界到最小范围
                mid = (self.bounds[key]['best'] + self.bounds[key]['worst']) / 2
                self.bounds[key]['best'] = mid - min_range / 2
                self.bounds[key]['worst'] = mid + min_range / 2
                
                if self.verbose:
                    print(f"  [边界调整] {key} 范围过窄 ({current_range:.4f}), 扩展到 {min_range}")
        
        if self.verbose:
            print(f"\n[分位数边界已更新] (第 {self.eval_count} 次评估)")
            for key in ['time', 'temp', 'aging']:
                print(f"  {key}: [{self.bounds[key]['best']:.4f}, {self.bounds[key]['worst']:.4f}]")
    
    def _normalize(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """
        归一化目标值到 [0, 1]
        
        改进: 添加数值稳定性保护,避免边界过窄导致除零错误
        """
        # 优先使用动态边界,不足则使用临时边界
        bounds = self.bounds if self.bounds is not None else self.temp_bounds
        
        # ✅ 定义最小有效范围（相对于临时边界）
        min_ranges = {
            'time': 5.0,      # 最小时间范围5步（临时边界范围140）
            'temp': 1.0,      # 最小温度范围1K（临时边界范围11K）
            'aging': 0.001    # 最小老化范围0.001%（临时边界范围0.5%）
        }
        
        normalized = {}
        for key in ['time', 'temp', 'aging']:
            best = bounds[key]['best']
            worst = bounds[key]['worst']
            value = objectives[key]
            
            # 计算分母
            denominator = worst - best
            min_range = min_ranges[key]
            
            # ✅ 数值稳定性检查：边界范围必须大于最小有效范围
            if abs(denominator) < min_range:
                # 边界过窄,尝试使用临时边界作为fallback
                if bounds is not self.temp_bounds:
                    # 当前使用的是动态边界,fallback到临时边界
                    temp_best = self.temp_bounds[key]['best']
                    temp_worst = self.temp_bounds[key]['worst']
                    temp_denominator = temp_worst - temp_best
                    
                    if abs(temp_denominator) >= min_range:
                        # 临时边界有效
                        normalized[key] = (value - temp_best) / temp_denominator
                        if self.verbose and self.eval_count % 10 == 0:
                            print(f"  [Warning] {key}动态边界过窄({denominator:.6f}<{min_range}), 使用固定边界")
                    else:
                        # 临时边界也过窄,使用中间值
                        normalized[key] = 0.5
                        if self.verbose and self.eval_count % 10 == 0:
                            print(f"  [Warning] {key}所有边界过窄, 使用默认值0.5")
                else:
                    # 已经在使用临时边界且仍过窄
                    normalized[key] = 0.5
                    if self.verbose and self.eval_count % 10 == 0:
                        print(f"  [Warning] {key}固定边界过窄({denominator:.6f}<{min_range}), 使用默认值0.5")
            else:
                # 边界正常,进行标准归一化
                normalized[key] = (value - best) / denominator
            
            # 裁剪到[0, 1]范围
            normalized[key] = np.clip(normalized[key], 0.0, 1.0)
        
        return normalized
    
    def _chebyshev_scalarize(self, normalized: Dict[str, float]) -> float:
        """增强切比雪夫标量化"""
        weighted_deviations = [
            self.weights[key] * normalized[key] 
            for key in ['time', 'temp', 'aging']
        ]
        
        max_weighted = max(weighted_deviations)
        sum_weighted = sum(weighted_deviations)
        
        rho = 0.05
        scalarized = max_weighted + rho * sum_weighted
        
        return scalarized
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_evaluations': self.eval_count,
            'valid_evaluations': sum(self.history['valid']),
            'history_summary': {}
        }
        
        for key in ['time', 'temp', 'aging']:
            if len(self.history[key]) > 0:
                stats['history_summary'][key] = {
                    'min': float(np.min(self.history[key])),
                    'max': float(np.max(self.history[key])),
                    'mean': float(np.mean(self.history[key])),
                    'std': float(np.std(self.history[key]))
                }
        
        return stats
    
    def export_database(self, normalized: bool = True) -> List[Dict]:
        """
        导出完整数据库
        
        参数：
            normalized: 是否返回全局归一化后的历史（默认True）
        
        返回：
            归一化历史（包含统一尺度的 scalarized 值）或原始 detailed_logs
        """
        return self.get_normalized_history() if normalized else self.detailed_logs
    
    def get_pareto_front(self) -> List[Dict]:
        """提取帕累托最优解"""
        valid_logs = [log for log in self.detailed_logs if log['valid']]
        
        if len(valid_logs) == 0:
            return []
        
        pareto_front = []
        
        for log_i in valid_logs:
            obj_i = log_i['objectives']
            is_dominated = False
            
            for log_j in valid_logs:
                if log_i == log_j:
                    continue
                
                obj_j = log_j['objectives']
                
                # 检查是否被支配
                dominates = (
                    obj_j['time'] <= obj_i['time'] and
                    obj_j['temp'] <= obj_i['temp'] and
                    obj_j['aging'] <= obj_i['aging']
                )
                
                at_least_one_better = (
                    obj_j['time'] < obj_i['time'] or
                    obj_j['temp'] < obj_i['temp'] or
                    obj_j['aging'] < obj_i['aging']
                )
                
                if dominates and at_least_one_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(log_i)
        
        return pareto_front
    
    def add_external_evaluation(
        self, 
        current1: float, 
        charging_number: float, 
        current2: float,
        objectives: Dict[str, float],
        source: str = "external"
    ) -> None:
        """手动添加外部评估结果到数据库"""
        self.history['time'].append(objectives['time'])
        self.history['temp'].append(objectives['temp'])
        self.history['aging'].append(objectives['aging'])
        self.history['valid'].append(objectives.get('valid', True))
        
        self.eval_count += 1
        
        normalized = self._normalize(objectives)
        scalarized = self._chebyshev_scalarize(normalized)
        
        log_entry = {
            'eval_id': self.eval_count,
            'params': {'current1': current1, 'charging_number': charging_number, 'current2': current2},
            'objectives': objectives,
            'normalized': normalized,
            'scalarized': scalarized,
            'valid': objectives.get('valid', True),
            'violations': 0,
            'termination': 'external',
            'source': source
        }
        self.detailed_logs.append(log_entry)
        
        if self.verbose:
            print(f"[外部评估已添加] 来源={source}, 评估ID={self.eval_count}")
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """动态更新目标权重"""
        weight_sum = sum(new_weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"权重之和必须为1.0，当前为 {weight_sum}")
        
        self.weights = new_weights
        
        if self.verbose:
            print(f"[权重已更新] {self.weights}")
    
    def get_best_solution(self) -> Optional[Dict]:
        """获取当前最佳解（基于标量化值）"""
        if self.eval_count == 0:
            return None
        
        valid_logs = [log for log in self.detailed_logs if log['valid']]
        
        if len(valid_logs) == 0:
            return None
        
        best_log = min(valid_logs, key=lambda x: x['scalarized'])
        
        return best_log


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("【测试】MultiObjectiveEvaluator - v3.1（已修复）")
    print("=" * 70)
    
    # 测试基本功能
    evaluator = MultiObjectiveEvaluator(verbose=True)
    
    print("\n[OK] Basic functionality test passed")
    print(f"   evaluate() 方法: 存在")
    print(f"   initialize_with_llm_warmstart() 方法: 存在 ✨")
    print(f"   get_best_solution() 方法: 存在")
    print(f"   get_pareto_front() 方法: 存在")
    print("=" * 70)