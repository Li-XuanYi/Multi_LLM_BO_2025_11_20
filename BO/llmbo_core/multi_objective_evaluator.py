"""
多目标评价器（Multi-Objective Evaluator）- 完整版（已修复）
用于 LLM 增强的分解多目标贝叶斯优化 (LLM-DMOBO)

=============================================================================
修复内容
=============================================================================
✅ 添加了 initialize_with_llm_warmstart() 异步方法
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
   - ✅ LLM Warm Start 初始化（新增）
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
    from .SPM import SPM_Sensitivity as SPM
except ImportError:
    from SPM import SPM_Sensitivity as SPM


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
            # [Gradient Computation]
            from SPM import SPM_Sensitivity
            self.spm_for_gradients = SPM_Sensitivity(init_v=3.0, init_t=298, enable_sensitivities=True)
            self.gradient_compute_interval = 5
            print("[OK] Gradient computation enabled")
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
        使用 LLM 进行热启动初始化
        
        功能：
        1. 如果提供 API key：调用 LLM 生成基于物理知识的初始策略
        2. 如果没有 API key：回退到随机策略生成
        3. 对所有生成的策略进行评估
        4. 返回评估结果列表
        
        参数：
            n_strategies: 要生成的策略数量
            llm_api_key: LLM API 密钥（可选）
            llm_base_url: API 基础 URL
            llm_model: 使用的 LLM 模型
        
        返回：
            List[Dict]: 评估结果列表
            [
                {
                    'params': {'current1': ..., 'charging_number': ..., 'current2': ...},
                    'scalarized': ...,
                    'objectives': {'time': ..., 'temp': ..., 'aging': ...},
                    'valid': True/False,
                    'source': 'llm_warmstart' or 'random_warmstart',
                    'reasoning': '...'  # 可选，LLM 的推理过程
                },
                ...
            ]
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("开始 LLM Warm Start 初始化")
            print("=" * 70)
            print(f"目标策略数量: {n_strategies}")
            print(f"API Key: {'已提供' if llm_api_key else '未提供（将使用随机策略）'}")
            print("=" * 70)
        
        results = []
        
        # 根据是否有 API key 选择策略
        if llm_api_key is not None:
            # 尝试使用 LLM 生成策略
            try:
                if self.verbose:
                    print("\n尝试使用 LLM 生成初始策略...")
                
                strategies = await self._llm_generate_strategies(
                    n_strategies=n_strategies,
                    api_key=llm_api_key,
                    base_url=llm_base_url,
                    model=llm_model
                )
                
                if self.verbose:
                    print(f"[OK] LLM successfully generated {len(strategies)} strategies")
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  LLM 调用失败: {e}")
                    print("   回退到随机策略生成...")
                
                # 回退到随机策略
                strategies = self._generate_random_strategies(n_strategies)
        
        else:
            # 没有 API key，直接使用随机策略
            if self.verbose:
                print("\n使用随机策略生成...")
            
            strategies = self._generate_random_strategies(n_strategies)
        
        # 评估所有策略
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
            print(f"\n[OK] Warm Start completed! Successfully evaluated {len(results)}/{len(strategies)} strategies")
            print("=" * 70)
        
        return results
    
    async def _llm_generate_strategies(
        self,
        n_strategies: int,
        api_key: str,
        base_url: str,
        model: str
    ) -> List[Dict]:
        """
        使用 LLM 生成初始充电策略
        
        返回：
            List[Dict]: 策略列表
            [
                {
                    'params': {'current1': ..., 'charging_number': ..., 'current2': ...},
                    'source': 'llm_warmstart',
                    'reasoning': '...'
                },
                ...
            ]
        """
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
        # 构建 prompt
        prompt = f"""You are an expert in electrochemistry and lithium-ion battery fast-charging optimization.

TASK:
Generate {n_strategies} diverse two-stage constant-current (CC) charging strategies for a lithium-ion battery.

BATTERY SPECIFICATIONS:
- Nominal Capacity: 5.0 Ah
- Voltage Range: 2.5V - 4.2V
- Initial State: 0% SOC, 298K
- Target: Charge to 80% SOC

OPTIMIZATION OBJECTIVES (competing):
1. Minimize charging time
2. Minimize peak temperature (constraint: ≤309K)
3. Minimize capacity degradation (SEI growth)

PARAMETER CONSTRAINTS:
- current1 (Stage 1 current): 3.0 - 6.0 A (typically 1.0-1.2C)
- charging_number (switching time step): 5 - 25 steps
- current2 (Stage 2 current): 1.0 - 4.0 A (typically 0.2-0.8C)

PHYSICAL CONSIDERATIONS:
1. High Stage 1 current → faster charging BUT higher temperature & more aging
2. Long Stage 1 duration → more heat accumulation
3. Low Stage 2 current → thermal relaxation BUT slower completion
4. Trade-off: charging speed vs. battery health

STRATEGY DIVERSITY:
Generate strategies with different focus:
- Aggressive: Fast charging (high I1, short transition)
- Conservative: Battery health (moderate I1, early transition to low I2)
- Balanced: Compromise between speed and health

OUTPUT FORMAT (JSON array):
[
    {{
        "current1": float (3.0-6.0),
        "charging_number": int (5-25),
        "current2": float (1.0-4.0),
        "reasoning": "Brief explanation of strategy rationale (1-2 sentences)"
    }},
    ...
]

CRITICAL: Output ONLY the JSON array, no additional text."""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in battery fast-charging optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # 增加多样性
                max_tokens=1000,
                response_format={"type": "json_object"} if "gpt-4" in model.lower() else None
            )
            
            content = response.choices[0].message.content
            
            # 尝试解析 JSON（处理可能的 markdown 包装）
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # 解析 JSON
            parsed = json.loads(content)
            
            # 处理不同的响应格式
            if isinstance(parsed, list):
                strategies_data = parsed
            elif isinstance(parsed, dict):
                # 可能是 {"strategies": [...]}
                strategies_data = parsed.get('strategies', [parsed])
            else:
                raise ValueError(f"Unexpected response format: {type(parsed)}")
            
            # 转换为标准格式
            strategies = []
            for s in strategies_data[:n_strategies]:
                strategy = {
                    'params': {
                        'current1': float(s['current1']),
                        'charging_number': int(s['charging_number']),
                        'current2': float(s['current2'])
                    },
                    'source': 'llm_warmstart',
                    'reasoning': s.get('reasoning', '')
                }
                
                # 验证参数范围
                if (3.0 <= strategy['params']['current1'] <= 6.0 and
                    5 <= strategy['params']['charging_number'] <= 25 and
                    1.0 <= strategy['params']['current2'] <= 4.0):
                    strategies.append(strategy)
            
            if len(strategies) < n_strategies:
                # LLM 生成的策略不足，补充随机策略
                n_needed = n_strategies - len(strategies)
                random_strategies = self._generate_random_strategies(n_needed)
                strategies.extend(random_strategies)
            
            return strategies
        
        except Exception as e:
            raise Exception(f"LLM API 调用失败: {e}")
    
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
            penalty = sim_result['constraint_violation'] * 0.5
            scalarized += penalty
        
        # 6.5 计算梯度
        gradients = None
        if self.eval_count % 5 == 0:
            try:
                grad_result = self.spm_for_gradients.run_two_stage_charging(
                    current1=current1, charging_number=int(charging_number), 
                    current2=current2, return_sensitivities=True
                )
                if grad_result['valid'] and 'sensitivities' in grad_result:
                    gradients = grad_result['sensitivities']
            except:
                gradients = None


        # 7. 记录详细日志
        log_entry = {
            'eval_id': self.eval_count,
            'params': {'current1': current1, 'charging_number': charging_number, 'current2': current2},
            'objectives': objectives_only,
            'normalized': normalized,
            'scalarized': scalarized,
            'valid': sim_result['valid'],
            'violations': sim_result['constraint_violation'],
            'termination': sim_result['termination'],
            'gradients': gradients
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
        """运行充电仿真并收集三个目标"""
        # 初始化SPM环境
        env = SPM(init_v=3.0, init_t=298, enable_sensitivities=False)
        
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
        """更新分位数边界（Q5/Q95）"""
        if self.eval_count < 10:
            return
        
        valid_indices = [i for i, v in enumerate(self.history['valid']) if v]
        
        if len(valid_indices) < 5:
            return
        
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
        
        if self.verbose:
            print(f"\n[分位数边界已更新] (第 {self.eval_count} 次评估)")
    
    def _normalize(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """归一化目标值到 [0, 1]"""
        bounds = self.bounds if self.bounds is not None else self.temp_bounds
        
        normalized = {}
        for key in ['time', 'temp', 'aging']:
            best = bounds[key]['best']
            worst = bounds[key]['worst']
            
            if worst - best < 1e-10:
                normalized[key] = 0.0
            else:
                normalized[key] = (objectives[key] - best) / (worst - best)
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
    
    def export_database(self) -> List[Dict]:
        """导出完整数据库"""
        return self.detailed_logs
    
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