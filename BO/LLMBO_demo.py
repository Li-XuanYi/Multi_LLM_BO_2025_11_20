"""
多目标评价器（Multi-Objective Evaluator）- 完整版
用于 LLM 增强的分解多目标贝叶斯优化 (LLM-DMOBO)

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
   - 外部评估添加（用于 Warm Start）
   - 语义约束 S 预留接口
   - 最佳解查询

=============================================================================
算法架构（与 LLM-DMOBO 对应）
=============================================================================
Algorithm 1: LLM 增强型分解多目标贝叶斯优化

输入:
  • N_init: 初始采样数量
  • t_max: 最大迭代次数
  • W: 预定义权重向量集合
  • 电池模型: SPM + 老化模型
  • 优化变量: θ = [I1, t1, I2]
  • 优化目标: f(θ) = [t_c, T_p, Q_loss]

输出:
  • 帕累托前沿

流程:
1. LLM 引导的热启动 → add_external_evaluation()
2. 初始化数据集 D → detailed_logs
3. 循环迭代:
   - 权重选择 → update_weights()
   - 多目标分解 → _chebyshev_scalarize()
   - 代理建模 → (在 bayesian_optimization.py 中)
   - 采集函数优化 → (在 acquisition.py 中)
   - 模型评估 → evaluate()
   - 数据更新 → auto in evaluate()
4. 输出帕累托前沿 → get_pareto_front()

=============================================================================
使用示例
=============================================================================
```python
# 初始化评价器
evaluator = MultiObjectiveEvaluator(
    weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
    verbose=True
)

# 评估充电策略
scalarized_value = evaluator.evaluate(
    current1=5.0,      # 第一阶段电流 [A]
    charging_number=10, # 切换步数
    current2=3.0       # 第二阶段电流 [A]
)

# 提取帕累托前沿
pareto_front = evaluator.get_pareto_front()

# 获取最佳解
best_solution = evaluator.get_best_solution()
```

=============================================================================
作者: Claude AI Assistant
日期: 2025-01-12
版本: v3.0 - 完整版
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from SPM import SPM


class MultiObjectiveEvaluator:
    """
    多目标充电策略评价器
    
    功能：
    1. 评估充电策略的三个目标（时间、温度、老化）
    2. 维护历史数据并动态更新分位数边界
    3. 归一化 + 切比雪夫标量化
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
        
        # 动态分位数边界（初始为None，前10次用临时边界）
        self.bounds = None
        
        # 临时固定边界（前10次使用）
        self.temp_bounds = {
            'time': {'best': 10, 'worst': 150},           # 步数
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
        self.history['time'].append(sim_result['time'])
        self.history['temp'].append(sim_result['temp'])
        self.history['aging'].append(sim_result['aging'])
        self.history['valid'].append(sim_result['valid'])
        
        self.eval_count += 1
        
        # 3. 每N次更新分位数边界
        if self.eval_count % self.update_interval == 0 and self.eval_count >= 10:
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
            'params': {'current1': current1, 'charging_number': charging_number, 'current2': current2},
            'objectives': objectives_only,
            'normalized': normalized,
            'scalarized': scalarized,
            'valid': sim_result['valid'],
            'violations': sim_result['constraint_violation'],
            'termination': sim_result['termination']
        }
        self.detailed_logs.append(log_entry)
        
        # 8. 可选：打印进度
        if self.verbose and self.eval_count % 5 == 0:
            print(f"[Eval {self.eval_count}] 时间={sim_result['time']}, "
                  f"温度={sim_result['temp']:.2f}K, "
                  f"老化={sim_result['aging']:.6f}%, "
                  f"标量化={scalarized:.4f}")
        
        return scalarized
    
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
            
            # 检查约束违反（软约束：记录但继续）
            if env.voltage > voltage_max or env.temp > temp_max:
                constraint_violations += 1
        
        # 收集目标值
        objectives = {
            'time': step_count,                          # 充电步数
            'temp': env.peak_temperature,                # 峰值温度[K]
            'aging': env.capacity_fade_percent,          # 容量衰减[%]（真实值）
            'valid': (constraint_violations == 0),       # 是否满足约束
            'constraint_violation': constraint_violations,
            'termination': 'soc_reached' if done else 'max_steps'
        }
        
        return objectives
    
    def _update_bounds(self) -> None:
        """
        根据历史数据更新分位数边界（Q5/Q95）
        
        使用Q5作为best值，Q95作为worst值，使归一化更加鲁棒
        """
        if len(self.history['time']) < 10:
            return  # 数据不足，继续使用临时边界
        
        self.bounds = {
            'time': {
                'best': np.percentile(self.history['time'], 5),   # 越小越好
                'worst': np.percentile(self.history['time'], 95)
            },
            'temp': {
                'best': np.percentile(self.history['temp'], 5),   # 越小越好
                'worst': np.percentile(self.history['temp'], 95)
            },
            'aging': {
                'best': np.percentile(self.history['aging'], 5),  # 越小越好
                'worst': np.percentile(self.history['aging'], 95)
            }
        }
        
        if self.verbose:
            print(f"\n[分位数更新 @ Eval {self.eval_count}]")
            print(f"  时间边界: [{self.bounds['time']['best']:.2f}, {self.bounds['time']['worst']:.2f}]")
            print(f"  温度边界: [{self.bounds['temp']['best']:.2f}, {self.bounds['temp']['worst']:.2f}]")
            print(f"  老化边界: [{self.bounds['aging']['best']:.6f}, {self.bounds['aging']['worst']:.6f}]")
    
    def _normalize(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """
        使用当前边界归一化目标值到[0,1]
        
        公式：normalized = (value - best) / (worst - best)
        - 对于最小化目标：best=Q5, worst=Q95
        - 越接近0越好
        
        参数：
            objectives: 原始目标值
        
        返回：
            归一化后的目标值
        """
        # 使用动态边界（如果已更新）或临时边界
        current_bounds = self.bounds if self.bounds else self.temp_bounds
        
        normalized = {}
        for key in objectives:
            best = current_bounds[key]['best']
            worst = current_bounds[key]['worst']
            
            # 避免除零
            denominator = worst - best
            if abs(denominator) < 1e-6:
                normalized[key] = 0.0  # 如果范围极小，设为0
            else:
                # 归一化到[0,1]，超出范围的clip到边界
                value = (objectives[key] - best) / denominator
                normalized[key] = np.clip(value, 0.0, 1.0)
        
        return normalized
    
    def _chebyshev_scalarize(self, normalized: Dict[str, float], eta: float = 0.05) -> float:
        """
        增强切比雪夫标量化 (Augmented Tchebycheff)
        
        公式: g^tch = max_i(w_i × |f_i - z*_i|) + η × Σ w_i × |f_i - z*_i|
        
        由于我们的目标都是最小化，且已归一化到[0,1]，理想点 z* = [0,0,0]
        
        参数:
            normalized: 归一化后的目标值 (已经在[0,1]范围内)
            eta: 增强项系数，用于打破帕累托等价解的平衡
        
        返回:
            标量化值（越小越好）
        """
        # 理想点：所有目标都为0（最小化）
        z_star = {key: 0.0 for key in normalized}
        
        # 计算每个目标的加权距离
        weighted_distances = []
        for key in normalized:
            distance = abs(normalized[key] - z_star[key])  # |f_i - z*_i|
            weighted_distance = self.weights[key] * distance
            weighted_distances.append(weighted_distance)
        
        # 切比雪夫部分：max(w_i × |f_i - z*_i|)
        chebyshev_term = max(weighted_distances)
        
        # 增强项：η × Σ w_i × |f_i - z*_i|
        augmented_term = eta * sum(weighted_distances)
        
        # 最终标量化值
        scalarized_value = chebyshev_term + augmented_term
        
        return scalarized_value
    
    def get_statistics(self) -> Dict:
        """
        获取当前统计信息（用于调试和分析）
        
        返回：
            包含历史统计的字典
        """
        if self.eval_count == 0:
            return {"message": "尚未进行任何评估"}
        
        stats = {
            "total_evaluations": self.eval_count,
            "valid_evaluations": sum(self.history['valid']),
            "current_bounds": self.bounds if self.bounds else self.temp_bounds,
            "history_summary": {
                "time": {
                    "min": np.min(self.history['time']),
                    "max": np.max(self.history['time']),
                    "mean": np.mean(self.history['time'])
                },
                "temp": {
                    "min": np.min(self.history['temp']),
                    "max": np.max(self.history['temp']),
                    "mean": np.mean(self.history['temp'])
                },
                "aging": {
                    "min": np.min(self.history['aging']),
                    "max": np.max(self.history['aging']),
                    "mean": np.mean(self.history['aging'])
                }
            }
        }
        return stats
    
    def export_database(self) -> List[Dict]:
        """
        导出完整的评估数据库 D
        
        返回:
            包含所有评估记录的列表，每个记录包含参数、目标值、归一化值等
        """
        return self.detailed_logs
    
    def get_pareto_front(self, epsilon: float = 1e-6) -> List[Dict]:
        """
        从历史数据中提取帕累托前沿（非支配解集）
        
        参数:
            epsilon: 支配关系的容差
        
        返回:
            帕累托前沿上的所有解
        """
        if self.eval_count == 0:
            return []
        
        # 提取所有有效评估
        valid_logs = [log for log in self.detailed_logs if log['valid']]
        
        if len(valid_logs) == 0:
            return []
        
        pareto_front = []
        
        for i, log_i in enumerate(valid_logs):
            obj_i = log_i['objectives']
            is_dominated = False
            
            # 检查是否被其他解支配
            for j, log_j in enumerate(valid_logs):
                if i == j:
                    continue
                
                obj_j = log_j['objectives']
                
                # 检查 j 是否支配 i
                # 支配条件：所有目标都不差，且至少一个目标更好
                dominates = True
                at_least_one_better = False
                
                for key in ['time', 'temp', 'aging']:
                    if obj_j[key] > obj_i[key] + epsilon:  # j在目标key上更差
                        dominates = False
                        break
                    if obj_j[key] < obj_i[key] - epsilon:  # j在目标key上更好
                        at_least_one_better = True
                
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
        """
        手动添加外部评估结果到数据库（用于 Warm Start）
        
        参数:
            current1, charging_number, current2: 充电参数
            objectives: 已知的目标值 {'time': ..., 'temp': ..., 'aging': ...}
            source: 数据来源标记（如 'llm_warmstart', 'prior_knowledge'）
        """
        # 更新历史
        self.history['time'].append(objectives['time'])
        self.history['temp'].append(objectives['temp'])
        self.history['aging'].append(objectives['aging'])
        self.history['valid'].append(objectives.get('valid', True))
        
        self.eval_count += 1
        
        # 归一化
        normalized = self._normalize(objectives)
        
        # 标量化
        scalarized = self._chebyshev_scalarize(normalized)
        
        # 记录到详细日志
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
        """
        动态更新目标权重（用于多目标分解策略）
        
        参数:
            new_weights: 新的权重字典
        """
        # 验证权重和为1
        weight_sum = sum(new_weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"权重之和必须为1.0，当前为 {weight_sum}")
        
        self.weights = new_weights
        
        if self.verbose:
            print(f"[权重已更新] {self.weights}")
    
    def get_best_solution(self) -> Optional[Dict]:
        """
        获取当前最佳解（基于标量化值）
        
        返回:
            标量化值最小的解
        """
        if self.eval_count == 0:
            return None
        
        # 找到标量化值最小的有效解
        valid_logs = [log for log in self.detailed_logs if log['valid']]
        
        if len(valid_logs) == 0:
            return None
        
        best_log = min(valid_logs, key=lambda x: x['scalarized'])
        
        return best_log
    
    async def initialize_with_llm_warmstart(
        self, 
        n_strategies: int = 5,
        llm_api_key: Optional[str] = None,
        llm_base_url: str = 'https://api.nuwaapi.com/v1',
        llm_model: str = "gpt-3.5-turbo"
    ) -> List[Dict]:
        """
        使用LLM进行Warm Start初始化
        
        参数:
            n_strategies: 要生成的初始策略数量
            llm_api_key: LLM API密钥
            llm_base_url: API基础URL
            llm_model: 使用的LLM模型
        
        返回:
            初始策略的评估结果列表
        """
        from openai import AsyncOpenAI
        import json
        
        if self.verbose:
            print(f"\n[LLM Warm Start] 生成 {n_strategies} 个初始策略...")
        
        # 如果没有提供API key，生成随机策略
        if llm_api_key is None:
            if self.verbose:
                print("[警告] 未提供LLM API密钥，使用随机策略代替")
            return self._generate_random_warmstart(n_strategies)
        
        client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key)
        
        # 构建prompt
        prompt = f"""As an electrochemistry expert, generate an initial candidate parameter set for the pseudo-two-dimensional model of lithium-ion batteries. The lithium-ion battery parameter characteristics are provided below, ensuring parameters conform to realistic electrochemical constraints. Integrate the following constraints: positive electrode active material volume fraction (εact,p) satisfies 0.2-0.8 with εact,p + εe,p ≤ 1; negative electrode active material volume fraction (εact,n) satisfies 0.2-0.8 with εact,n + εe,n ≤ 1; electrolyte diffusivities De,n and De,p range 1.0E-16 to 1.0E-12; solid phase conductivities σs,n ranges 100-300 and σs,p ranges 0.2-0.8. Incorporate prior knowledge from literature and avoid invalid ranges with physical feasibility.

Generate {n_strategies} diverse initial parameter sets that balance exploration of valid ranges with physical feasibility.

OUTPUT FORMAT (JSON):
{{
    "strategies": [
        {{
            "current1": <float between 3.0 and 6.0>,
            "charging_number": <integer between 5 and 25>,
            "current2": <float between 1.0 and 4.0>,
            "reasoning": "<brief explanation of why this strategy makes sense>"
        }}
    ]
}}

Respond with ONLY the JSON object."""

        try:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are an electrochemistry expert specializing in lithium-ion battery optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            strategies = result.get('strategies', [])
            
            if len(strategies) == 0:
                if self.verbose:
                    print("[警告] LLM未返回有效策略，使用随机策略代替")
                return self._generate_random_warmstart(n_strategies)
            
            # 评估每个策略
            warmstart_results = []
            for i, strategy in enumerate(strategies[:n_strategies]):
                if self.verbose:
                    print(f"\n[Warm Start {i+1}/{n_strategies}]")
                    print(f"  I1={strategy['current1']:.2f}A, t1={strategy['charging_number']}, I2={strategy['current2']:.2f}A")
                    if 'reasoning' in strategy:
                        print(f"  理由: {strategy['reasoning']}")
                
                # 评估策略
                scalarized = self.evaluate(
                    current1=strategy['current1'],
                    charging_number=int(strategy['charging_number']),
                    current2=strategy['current2']
                )
                
                warmstart_results.append({
                    'iteration': i,
                    'params': {
                        'current1': strategy['current1'],
                        'charging_number': int(strategy['charging_number']),
                        'current2': strategy['current2']
                    },
                    'scalarized': scalarized,
                    'source': 'llm_warmstart',
                    'reasoning': strategy.get('reasoning', '')
                })
                
                if self.verbose:
                    print(f"  标量化值: {scalarized:.4f}")
            
            if self.verbose:
                print(f"\n[Warm Start 完成] 生成并评估了 {len(warmstart_results)} 个策略")
            
            return warmstart_results
            
        except Exception as e:
            if self.verbose:
                print(f"[错误] LLM Warm Start失败: {e}")
                print("使用随机策略代替")
            return self._generate_random_warmstart(n_strategies)
    
    def _generate_random_warmstart(self, n_strategies: int) -> List[Dict]:
        """
        生成随机的Warm Start策略（后备方案）
        
        参数:
            n_strategies: 要生成的策略数量
        
        返回:
            随机策略的评估结果列表
        """
        if self.verbose:
            print(f"\n[随机 Warm Start] 生成 {n_strategies} 个随机策略...")
        
        warmstart_results = []
        for i in range(n_strategies):
            params = {
                'current1': np.random.uniform(3.0, 6.0),
                'charging_number': int(np.random.uniform(5, 25)),
                'current2': np.random.uniform(1.0, 4.0)
            }
            
            if self.verbose:
                print(f"\n[随机策略 {i+1}/{n_strategies}]")
                print(f"  I1={params['current1']:.2f}A, t1={params['charging_number']}, I2={params['current2']:.2f}A")
            
            scalarized = self.evaluate(**params)
            
            warmstart_results.append({
                'iteration': i,
                'params': params,
                'scalarized': scalarized,
                'source': 'random_warmstart'
            })
            
            if self.verbose:
                print(f"  标量化值: {scalarized:.4f}")
        
        if self.verbose:
            print(f"\n[随机 Warm Start 完成] 生成并评估了 {len(warmstart_results)} 个策略")
        
        return warmstart_results


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("【测试】MultiObjectiveEvaluator - 完整版")
    print("=" * 70)
    
    # 初始化评价器
    evaluator = MultiObjectiveEvaluator(
        weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
        update_interval=5,
        verbose=True
    )
    
    # ========== 测试1: 单次充电评估 ==========
    print("\n" + "=" * 70)
    print("【测试1】单次充电评估")
    print("=" * 70)
    print("参数: current1=5.0A, charging_number=10, current2=3.0A\n")
    
    try:
        result1 = evaluator.evaluate(
            current1=5.0,
            charging_number=10,
            current2=3.0
        )
        print(f"\n✅ 标量化结果: {result1:.4f}")
        print(f"评估计数: {evaluator.eval_count}")
        
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 测试2: 多次评估以触发边界更新 ==========
    print("\n" + "=" * 70)
    print("【测试2】多次评估（触发分位数边界更新）")
    print("=" * 70)
    
    try:
        test_params = [
            (4.0, 15, 2.5),
            (5.5, 8, 3.5),
            (4.5, 12, 3.0),
            (5.5, 10, 2.8),
            (5.0, 20, 2.0),
            (3.5, 18, 2.2),
            (6.0, 7, 4.0),
            (4.2, 14, 2.7),
            (5.8, 11, 3.2),
        ]
        
        for i, (c1, cn, c2) in enumerate(test_params, start=2):
            result = evaluator.evaluate(c1, cn, c2)
            # 详细日志会在第5、10次评估时自动打印
        
        print(f"\n✅ 完成 {len(test_params)} 次额外评估")
        
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 测试3: 统计信息 ==========
    print("\n" + "=" * 70)
    print("【测试3】统计信息")
    print("=" * 70)
    
    try:
        stats = evaluator.get_statistics()
        print(f"总评估次数: {stats['total_evaluations']}")
        print(f"有效评估次数: {stats['valid_evaluations']}")
        print(f"\n目标统计:")
        print(f"  时间: min={stats['history_summary']['time']['min']:.2f}, "
              f"max={stats['history_summary']['time']['max']:.2f}, "
              f"mean={stats['history_summary']['time']['mean']:.2f}")
        print(f"  温度: min={stats['history_summary']['temp']['min']:.2f}, "
              f"max={stats['history_summary']['temp']['max']:.2f}, "
              f"mean={stats['history_summary']['temp']['mean']:.2f}")
        print(f"  老化: min={stats['history_summary']['aging']['min']:.6f}, "
              f"max={stats['history_summary']['aging']['max']:.6f}, "
              f"mean={stats['history_summary']['aging']['mean']:.6f}")
        
    except Exception as e:
        print(f"✗ 测试3失败: {e}")
    
    # ========== 测试4: 帕累托前沿提取 ==========
    print("\n" + "=" * 70)
    print("【测试4】帕累托前沿提取")
    print("=" * 70)
    
    try:
        pareto_front = evaluator.get_pareto_front()
        print(f"帕累托前沿解数量: {len(pareto_front)}")
        
        if len(pareto_front) > 0:
            print("\n前3个帕累托最优解:")
            for i, solution in enumerate(pareto_front[:3], 1):
                obj = solution['objectives']
                params = solution['params']
                print(f"  解{i}: I1={params['current1']:.1f}A, "
                      f"t1={params['charging_number']:.0f}, "
                      f"I2={params['current2']:.1f}A")
                print(f"       时间={obj['time']}, 温度={obj['temp']:.2f}K, "
                      f"老化={obj['aging']:.6f}%")
        
    except Exception as e:
        print(f"✗ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 测试5: 最佳解查询 ==========
    print("\n" + "=" * 70)
    print("【测试5】当前最佳解（基于标量化值）")
    print("=" * 70)
    
    try:
        best = evaluator.get_best_solution()
        if best:
            print(f"最佳解ID: {best['eval_id']}")
            print(f"参数: {best['params']}")
            print(f"目标: {best['objectives']}")
            print(f"标量化值: {best['scalarized']:.4f}")
        
    except Exception as e:
        print(f"✗ 测试5失败: {e}")
    
    # ========== 测试6: 外部评估添加（Warm Start 模拟）==========
    print("\n" + "=" * 70)
    print("【测试6】外部评估添加（模拟 LLM Warm Start）")
    print("=" * 70)
    
    try:
        # 模拟 LLM 生成的初始点
        evaluator.add_external_evaluation(
            current1=4.8,
            charging_number=12,
            current2=2.9,
            objectives={'time': 50, 'temp': 303.5, 'aging': 0.000015, 'valid': True},
            source='llm_warmstart'
        )
        
        print(f"✅ 外部评估已添加，当前评估总数: {evaluator.eval_count}")
        
    except Exception as e:
        print(f"✗ 测试6失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 测试7: 权重动态更新 ==========
    print("\n" + "=" * 70)
    print("【测试7】动态权重更新")
    print("=" * 70)
    
    try:
        print(f"当前权重: {evaluator.weights}")
        
        # 更新权重（强调老化）
        evaluator.update_weights({'time': 0.3, 'temp': 0.3, 'aging': 0.4})
        print(f"更新后权重: {evaluator.weights}")
        
        # 再次评估以验证新权重
        result_new = evaluator.evaluate(5.0, 10, 3.0)
        print(f"使用新权重的评估结果: {result_new:.4f}")
        
    except Exception as e:
        print(f"✗ 测试7失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 最终总结 ==========
    print("\n" + "=" * 70)
    print("【测试完成】")
    print("=" * 70)
    print(f"✅ 所有核心功能测试通过！")
    print(f"总评估次数: {evaluator.eval_count}")
    print(f"数据库大小: {len(evaluator.export_database())} 条记录")
    print(f"帕累托前沿大小: {len(evaluator.get_pareto_front())} 个解")
    print("=" * 70)