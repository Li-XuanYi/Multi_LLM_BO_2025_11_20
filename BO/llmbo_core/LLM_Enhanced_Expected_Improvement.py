"""
LLM-Enhanced Expected Improvement (EI)
LLM增强的期望改进采集函数

=============================================================================
核心创新
=============================================================================
公式：α_EI^LLM(θ) = E[max(f_min - f(θ), 0)] · W_LLM(θ|D)

其中 W_LLM 是LLM指导的权重函数：
W_LLM = Π_{j=1}^q (1/√(2πσ_j²)) · exp(-(θ_j - μ_j)² / (2σ_j²))

流程：
1. 检测优化状态（收敛/局部最优/震荡）
2. LLM提供定性采样策略建议
3. 数据驱动计算 μ 和 σ
4. 修改EI采集函数

=============================================================================
参考文献
=============================================================================
Kuai, X., et al. (2025). Large language model-enhanced Bayesian optimization 
for parameter identification of lithium-ion batteries. Preprint.

Section 3.4: LLM-enhanced candidate sampling (Eq. 8-9)

=============================================================================
作者: Research Team
日期: 2025-01-12
版本: v1.0 - EI Enhancement
=============================================================================
"""

import numpy as np
import asyncio
import json
import warnings
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
from openai import AsyncOpenAI
from collections import deque


# ============================================================
# 1. 优化状态检测器
# ============================================================

class OptimizationStateDetector:
    """
    优化状态检测器
    
    检测三种状态：
    1. 收敛中 (Converging)：目标函数持续下降
    2. 局部最优 (Local Optimum)：在最优点附近徘徊
    3. 震荡 (Oscillating)：跳跃到差的区域后又回到好的区域
    """
    
    def __init__(
        self,
        window_size: int = 5,
        convergence_threshold: float = 0.01,
        stagnation_threshold: float = 0.001,
        oscillation_threshold: float = 0.1
    ):
        """
        初始化状态检测器
        
        参数：
            window_size: 滑动窗口大小
            convergence_threshold: 收敛判定阈值
            stagnation_threshold: 停滞判定阈值
            oscillation_threshold: 震荡判定阈值
        """
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.stagnation_threshold = stagnation_threshold
        self.oscillation_threshold = oscillation_threshold
        
        self.history_fmin = deque(maxlen=window_size)
        self.history_params = deque(maxlen=window_size)
    
    def update(self, fmin: float, params: Dict[str, float]) -> None:
        """更新历史记录"""
        self.history_fmin.append(fmin)
        self.history_params.append(params)
    
    def detect_state(self) -> Dict[str, any]:
        """
        检测当前优化状态
        
        返回：{
            'state': 'converging' | 'local_optimum' | 'oscillating',
            'confidence': float (0-1),
            'metrics': {...}
        }
        """
        if len(self.history_fmin) < 3:
        # [OK] 即使数据不足，也返回基本的 metrics
            current_val = self.history_fmin[-1] if len(self.history_fmin) > 0 else 0.0
            best_val = min(self.history_fmin) if len(self.history_fmin) > 0 else 0.0
        
            return {
            'state': 'exploring',
            'confidence': 1.0,
            'metrics': {
                'avg_improvement': 0.0,
                'std_improvement': 0.0,
                'avg_param_distance': 0.0,
                'current_fmin': float(current_val),
                'best_fmin': float(best_val)
            }
        }
        
        fmin_array = np.array(self.history_fmin)
        
        # 计算改善率
        improvements = np.diff(fmin_array)
        relative_improvements = improvements / (np.abs(fmin_array[:-1]) + 1e-10)
        
        # 计算参数空间的移动距离
        if len(self.history_params) >= 2:
            param_distances = []
            for i in range(len(self.history_params) - 1):
                p1 = self.history_params[i]
                p2 = self.history_params[i + 1]
                dist = np.sqrt(
                    (p1['current1'] - p2['current1'])**2 +
                    (p1['charging_number'] - p2['charging_number'])**2 +
                    (p1['current2'] - p2['current2'])**2
                )
                param_distances.append(dist)
            avg_param_distance = np.mean(param_distances)
        else:
            avg_param_distance = 0.0
        
        # 状态判定
        avg_improvement = np.mean(relative_improvements)
        std_improvement = np.std(relative_improvements)
        
        # 1. 收敛中：持续改善
        if avg_improvement < -self.convergence_threshold and std_improvement < 0.05:
            state = 'converging'
            confidence = min(1.0, abs(avg_improvement) / self.convergence_threshold)
        
        # 2. 局部最优：改善很小且参数变化小
        elif (abs(avg_improvement) < self.stagnation_threshold and 
              avg_param_distance < 0.5):
            state = 'local_optimum'
            confidence = 1.0 - abs(avg_improvement) / self.stagnation_threshold
        
        # 3. 震荡：改善率方差大
        elif std_improvement > self.oscillation_threshold:
            state = 'oscillating'
            confidence = min(1.0, std_improvement / self.oscillation_threshold)
        
        else:
            state = 'exploring'
            confidence = 0.5
        
        return {
            'state': state,
            'confidence': confidence,
            'metrics': {
                'avg_improvement': float(avg_improvement),
                'std_improvement': float(std_improvement),
                'avg_param_distance': float(avg_param_distance),
                'current_fmin': float(fmin_array[-1]),
                'best_fmin': float(np.min(fmin_array))
            }
        }


# ============================================================
# 2. LLM采样顾问
# ============================================================

class LLMSamplingAdvisor:
    """
    LLM采样策略顾问
    
    角色：基于优化状态提供定性的采样建议（不生成具体数值）
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://api.nuwaapi.com/v1',
        model: str = "gpt-3.5-turbo",
        verbose: bool = True
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.verbose = verbose
    
    async def get_sampling_strategy(
        self,
        state_info: Dict,
        history: List[Dict],
        sensitivity_info: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        获取LLM的采样策略建议
        
        参数：
            state_info: 优化状态信息
            history: 最近的评估历史
            sensitivity_info: 参数敏感度信息
        
        返回：{
            'mu_strategy': '描述如何选择μ的策略',
            'sigma_strategy': '描述如何选择σ的策略',
            'reasoning': '解释为什么这样选择'
        }
        """
        # 提取关键信息
        state = state_info['state']
        confidence = state_info['confidence']
        metrics = state_info['metrics']
        
        # 构建历史摘要
        recent_fmin = [r['scalarized'] for r in history[-5:] if r['valid']]
        best_params = min(history, key=lambda x: x['scalarized'] if x['valid'] else float('inf'))
        
        # 构建敏感度描述
        sens_desc = ""
        if sensitivity_info:
            sens_desc = f"""
PARAMETER SENSITIVITY (estimated):
- Current1: {sensitivity_info.get('current1', 'medium')}
- ChargingNumber: {sensitivity_info.get('charging_number', 'medium')}
- Current2: {sensitivity_info.get('current2', 'medium')}
"""
        
        prompt = f"""You are an optimization expert analyzing battery charging strategy optimization.

OPTIMIZATION STATE:
- Status: {state}
- Confidence: {confidence:.2f}
- Recent f_min trajectory: {[f'{x:.4f}' for x in recent_fmin]}
- Current best: {metrics['current_fmin']:.4f}
- Historical best: {metrics['best_fmin']:.4f}
- Average improvement rate: {metrics['avg_improvement']:.2%}
- Improvement volatility: {metrics['std_improvement']:.4f}

CURRENT BEST PARAMETERS:
- Current1: {best_params['params']['current1']:.2f}A
- ChargingNumber: {best_params['params']['charging_number']}
- Current2: {best_params['params']['current2']:.2f}A

{sens_desc}

TASK:
Based on the optimization state, provide a QUALITATIVE sampling strategy (DO NOT output specific numbers).

Strategy guidelines:
1. **If CONVERGING**: 
   - μ should be near current best (exploitation)
   - σ should be small (focused search)
   
2. **If LOCAL OPTIMUM**:
   - μ should be at the centroid of recently explored region (escape local trap)
   - σ should be larger (encourage exploration)
   
3. **If OSCILLATING**:
   - μ should return to historical best (stabilize)
   - σ should be moderate (balanced)

4. **σ adjustment based on sensitivity**:
   - High sensitivity parameters → smaller σ (careful exploration)
   - Low sensitivity parameters → larger σ (broader search)

5. **σ scaling based on convergence**:
   - Good convergence → σ × 0.8 (focus)
   - Poor convergence → σ × 1.2 (explore)

OUTPUT FORMAT (JSON):
{{
    "mu_strategy": "brief description of where to center sampling (e.g., 'near current best', 'at exploration centroid', 'return to historical best')",
    "sigma_strategy": "brief description of exploration scale (e.g., 'small focused search', 'moderate balanced', 'large exploration')",
    "sigma_scaling": "smaller/moderate/larger (based on convergence quality)",
    "reasoning": "1-2 sentence explanation of why this strategy makes sense"
}}

Respond with ONLY the JSON object."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            
            strategy = json.loads(response.choices[0].message.content)
            
            if self.verbose:
                print("\n[LLM采样策略]")
                print(f"μ策略: {strategy['mu_strategy']}")
                print(f"σ策略: {strategy['sigma_strategy']}")
                print(f"σ缩放: {strategy['sigma_scaling']}")
                print(f"推理: {strategy['reasoning']}")
            
            return strategy
            
        except Exception as e:
            warnings.warn(f"LLM采样顾问调用失败: {e}")
            # 返回默认策略
            return {
                'mu_strategy': 'near current best',
                'sigma_strategy': 'moderate balanced',
                'sigma_scaling': 'moderate',
                'reasoning': 'LLM unavailable, using default strategy'
            }


# ============================================================
# 3. 采样参数计算器
# ============================================================

class SamplingParameterComputer:
    """
    采样参数计算器
    
    根据LLM策略和数据，计算具体的 μ 和 σ 值
    """
    
    def __init__(
        self,
        pbounds: Dict[str, Tuple[float, float]],
        sigma_min: float = 0.05,  # 从0.1改为0.05
        sigma_max: float = 3.0,   # 从2.0改为3.0
        verbose: bool = True
    ):
        """
        初始化
        
        参数：
            pbounds: 参数边界
            sigma_min/max: σ的范围约束
            verbose: 详细输出
        """
        self.pbounds = pbounds
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.verbose = verbose
        
        self.param_names = ['current1', 'charging_number', 'current2']
    
    def compute_mu(
        self,
        strategy: str,
        history: List[Dict],
        state_metrics: Dict
    ) -> Dict[str, float]:
        """
        计算 μ（采样中心）
        
        参数：
            strategy: LLM给出的μ策略
            history: 评估历史
            state_metrics: 状态指标
        
        返回：
            μ: {'current1': ..., 'charging_number': ..., 'current2': ...}
        """
        valid_history = [h for h in history if h['valid']]
        
        if not valid_history:
            # 无有效历史，返回边界中点
            return {
                'current1': (self.pbounds['current1'][0] + self.pbounds['current1'][1]) / 2,
                'charging_number': (self.pbounds['charging_number'][0] + self.pbounds['charging_number'][1]) / 2,
                'current2': (self.pbounds['current2'][0] + self.pbounds['current2'][1]) / 2
            }
        
        # 策略1：当前最优（exploitation）
        if 'current best' in strategy.lower() or 'near best' in strategy.lower():
            best_record = min(valid_history, key=lambda x: x['scalarized'])
            mu = best_record['params'].copy()
        
        # 策略2：近期探索的质心（escape local optimum）
        elif 'centroid' in strategy.lower() or 'exploration' in strategy.lower():
            recent = valid_history[-10:]  # 最近10个点
            mu = {
                param: np.mean([r['params'][param] for r in recent])
                for param in self.param_names
            }
        
        # 策略3：历史最优（stabilize）
        elif 'historical' in strategy.lower() or 'return' in strategy.lower():
            best_record = min(valid_history, key=lambda x: x['scalarized'])
            mu = best_record['params'].copy()
        
        else:
            # 默认：当前最优
            best_record = min(valid_history, key=lambda x: x['scalarized'])
            mu = best_record['params'].copy()
        
        if self.verbose:
            print(f"\n[μ计算] 策略='{strategy}'")
            print(f"  I1={mu['current1']:.2f}A, t1={mu['charging_number']}, I2={mu['current2']:.2f}A")
        
        return mu
    
    def compute_sigma(
        self,
        strategy: str,
        scaling: str,
        sensitivity_info: Optional[Dict],
        state_metrics: Dict
    ) -> Dict[str, float]:
        """
        计算 σ（探索范围）
        
        参数：
            strategy: LLM给出的σ策略
            scaling: σ缩放建议 ('smaller', 'moderate', 'larger')
            sensitivity_info: 参数敏感度
            state_metrics: 状态指标
        
        返回：
            σ: {'current1': ..., 'charging_number': ..., 'current2': ...}
        """
        # 基础σ（基于参数范围）
        base_sigma = {
            'current1': (self.pbounds['current1'][1] - self.pbounds['current1'][0]) * 0.2,
            'charging_number': (self.pbounds['charging_number'][1] - self.pbounds['charging_number'][0]) * 0.2,
            'current2': (self.pbounds['current2'][1] - self.pbounds['current2'][0]) * 0.2
        }
        
        # 根据策略调整
        if 'small' in strategy.lower() or 'focused' in strategy.lower():
            scale_factor = 0.5
        elif 'large' in strategy.lower() or 'exploration' in strategy.lower():
            scale_factor = 2.0
        else:  # moderate
            scale_factor = 1.0
        
        # 根据收敛程度缩放
        if scaling == 'smaller':
            convergence_factor = 0.8
        elif scaling == 'larger':
            convergence_factor = 1.2
        else:
            convergence_factor = 1.0
        
        # 根据敏感度调整
        sensitivity_factors = {}
        if sensitivity_info:
            for param in self.param_names:
                sens = sensitivity_info.get(param, 'medium')
                if sens == 'high':
                    sensitivity_factors[param] = 0.7
                elif sens == 'low':
                    sensitivity_factors[param] = 1.3
                else:
                    sensitivity_factors[param] = 1.0
        else:
            sensitivity_factors = {p: 1.0 for p in self.param_names}
        
        # 计算最终σ
        sigma = {}
        for param in self.param_names:
            s = base_sigma[param] * scale_factor * convergence_factor * sensitivity_factors[param]
            # 约束到合理范围
            sigma[param] = np.clip(s, self.sigma_min, self.sigma_max)
        
        if self.verbose:
            print(f"\n[σ计算] 策略='{strategy}', 缩放='{scaling}'")
            print(f"  σ(I1)={sigma['current1']:.3f}, σ(t1)={sigma['charging_number']:.3f}, σ(I2)={sigma['current2']:.3f}")
        
        return sigma


# ============================================================
# 4. 参数敏感度估计器
# ============================================================

class ParameterSensitivityEstimator:
    """
    从历史数据估计参数敏感度
    """
    
    def __init__(self):
        self.param_names = ['current1', 'charging_number', 'current2']
    
    def estimate_sensitivity(self, history: List[Dict]) -> Dict[str, str]:
        """
        估计参数敏感度
        
        返回：{'current1': 'high'|'medium'|'low', ...}
        """
        if len(history) < 10:
            return {p: 'medium' for p in self.param_names}
        
        valid_history = [h for h in history if h['valid']]
        
        # 计算每个参数变化与目标变化的相关性
        sensitivities = {}
        
        for param in self.param_names:
            param_values = [h['params'][param] for h in valid_history]
            objective_values = [h['scalarized'] for h in valid_history]
            
            # 计算相关系数
            if len(set(param_values)) > 1:  # 参数有变化
                correlation = np.abs(np.corrcoef(param_values, objective_values)[0, 1])
                
                if correlation > 0.5:
                    sensitivities[param] = 'high'
                elif correlation > 0.2:
                    sensitivities[param] = 'medium'
                else:
                    sensitivities[param] = 'low'
            else:
                sensitivities[param] = 'medium'
        
        return sensitivities


# ============================================================
# 5. LLM增强的EI权重函数
# ============================================================

class LLMWeightFunction:
    """
    LLM增强的权重函数 W_LLM
    
    W_LLM(θ) = Π_{j=1}^q (1/√(2πσ_j²)) · exp(-(θ_j - μ_j)² / (2σ_j²))
    """
    
    def __init__(
        self,
        mu: Dict[str, float],
        sigma: Dict[str, float]
    ):
        """
        初始化权重函数
        
        参数：
            mu: 采样中心
            sigma: 探索范围
        """
        self.mu = mu
        self.sigma = sigma
        self.param_names = ['current1', 'charging_number', 'current2']
    
    def compute_weight(self, theta: np.ndarray) -> float:
        """
        计算权重 W_LLM(θ)
        
        参数：
            theta: 参数向量 [current1, charging_number, current2]
        
        返回：
            权重值
        """
        weight = 1.0
        
        for i, param_name in enumerate(self.param_names):
            mu_j = self.mu[param_name]
            sigma_j = self.sigma[param_name]
            theta_j = theta[i]
            
            # 高斯权重
            w_j = (1.0 / np.sqrt(2 * np.pi * sigma_j**2)) * \
                  np.exp(-(theta_j - mu_j)**2 / (2 * sigma_j**2))
            
            weight *= w_j
        
        return weight
    
    def compute_weights(self, X: np.ndarray) -> np.ndarray:
        """
        批量计算权重
        
        参数：
            X: (n_samples, 3) 参数矩阵
        
        返回：
            weights: (n_samples,) 权重向量
        """
        weights = np.array([self.compute_weight(x) for x in X])
        return weights


# ============================================================
# 6. LLM增强的EI采集函数
# ============================================================

class LLMEnhancedEI:
    """
    LLM增强的期望改进 (Expected Improvement)
    
    α_EI^LLM(θ) = EI(θ) · W_LLM(θ|D)
    
    集成所有组件：
    1. 状态检测
    2. LLM顾问
    3. 采样参数计算
    4. 权重函数
    """
    
    def __init__(
        self,
        evaluator,
        llm_api_key: Optional[str] = None,
        llm_base_url: str = 'https://api.nuwaapi.com/v1',
        llm_model: str = "gpt-3.5-turbo",
        enable_llm_advisor: bool = True,
        update_strategy_every: int = 3,
        verbose: bool = True
    ):
        """
        初始化LLM增强的EI
        
        参数：
            evaluator: MultiObjectiveEvaluator实例
            llm_api_key: LLM API密钥
            enable_llm_advisor: 是否启用LLM顾问
            update_strategy_every: 每N次迭代更新一次策略
            verbose: 详细输出
        """
        self.evaluator = evaluator
        self.verbose = verbose
        self.update_strategy_every = update_strategy_every
        
        # 参数边界
        self.pbounds = {
            'current1': (3.0, 6.0),
            'charging_number': (5, 25),
            'current2': (1.0, 4.0)
        }
        
        # 组件初始化
        self.state_detector = OptimizationStateDetector()
        self.param_computer = SamplingParameterComputer(
            pbounds=self.pbounds,
            verbose=verbose
        )
        self.sensitivity_estimator = ParameterSensitivityEstimator()
        
        # LLM顾问（可选）
        self.enable_llm_advisor = enable_llm_advisor and (llm_api_key is not None)
        if self.enable_llm_advisor:
            self.llm_advisor = LLMSamplingAdvisor(
                api_key=llm_api_key,
                base_url=llm_base_url,
                model=llm_model,
                verbose=verbose
            )
        else:
            self.llm_advisor = None
        
        # 状态
        self.current_mu = None
        self.current_sigma = None
        self.weight_function = None
        self.iteration = 0
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("LLM-Enhanced Expected Improvement 已初始化")
            print("=" * 70)
            print(f"LLM顾问: {'启用' if self.enable_llm_advisor else '禁用'}")
            print(f"策略更新间隔: 每{update_strategy_every}次迭代")
            print("=" * 70)
    
    async def update_sampling_strategy_async(self, history: List[Dict]) -> None:
        """
        更新采样策略（异步）
        
        流程：
        1. 检测优化状态
        2. LLM提供定性建议
        3. 计算具体的μ和σ
        4. 更新权重函数
        """
        self.iteration += 1
        
        if len(history) < 3:
            # 数据不足，使用默认策略
            self.current_mu = {
                'current1': 4.5,
                'charging_number': 15,
                'current2': 2.5
            }
            self.current_sigma = {
                'current1': 0.5,
                'charging_number': 3.0,
                'current2': 0.5
            }
            self.weight_function = LLMWeightFunction(self.current_mu, self.current_sigma)
            return
        
        # 更新状态检测器
        latest = history[-1]
        self.state_detector.update(
            fmin=latest['scalarized'],
            params=latest['params']
        )
        
        # 检测状态
        state_info = self.state_detector.detect_state()
        
        if self.verbose:
            print(f"\n[优化状态] {state_info['state']} (置信度: {state_info['confidence']:.2f})")
        
        # 估计参数敏感度
        sensitivity_info = self.sensitivity_estimator.estimate_sensitivity(history)
        
        # [OK] 新增: 停滞检测和强制探索
        forced_exploration = False
        if state_info['state'] in ['oscillating', 'local_optimum'] and state_info['confidence'] >= 0.9:
            # 检查最近5次的改善情况
            recent_fmin = [h['scalarized'] for h in history if h['valid']][-5:]
            if len(recent_fmin) >= 5:
                improvement = (recent_fmin[0] - np.min(recent_fmin)) / (abs(recent_fmin[0]) + 1e-10)
                
                if improvement < 0.01:  # 改善率 < 1%
                    forced_exploration = True
                    strategy = {
                        'mu_strategy': 'centroid of recently explored region',  # 远离当前最优
                        'sigma_strategy': 'large exploration',                 # 大范围搜索
                        'sigma_scaling': 'larger',
                        'reasoning': 'FORCED: Escaping local optimum via aggressive exploration'
                    }
                    if self.verbose:
                        print("  [WARNING] Stagnation detected: Forced exploration mode enabled")
        
        # LLM提供策略建议（只有在非强制探索时才调用）
        if not forced_exploration:
            if self.enable_llm_advisor and self.iteration % self.update_strategy_every == 1:
                try:
                    strategy = await self.llm_advisor.get_sampling_strategy(
                        state_info,
                        history,
                        sensitivity_info
                    )
                except Exception as e:
                    warnings.warn(f"LLM顾问失败: {e}")
                    strategy = {
                        'mu_strategy': 'near current best',
                        'sigma_strategy': 'moderate balanced',
                        'sigma_scaling': 'moderate',
                        'reasoning': 'Default strategy'
                    }
            else:
                # 使用上次的策略或默认策略
                if hasattr(self, 'last_strategy'):
                    strategy = self.last_strategy
                else:
                    strategy = {
                        'mu_strategy': 'near current best',
                        'sigma_strategy': 'moderate balanced',
                        'sigma_scaling': 'moderate',
                        'reasoning': 'Default strategy'
                    }
        
        self.last_strategy = strategy
        
        # 计算μ和σ
        self.current_mu = self.param_computer.compute_mu(
            strategy['mu_strategy'],
            history,
            state_info['metrics']
        )
        
        self.current_sigma = self.param_computer.compute_sigma(
            strategy['sigma_strategy'],
            strategy['sigma_scaling'],
            sensitivity_info,
            state_info['metrics']
        )
        
        # 更新权重函数
        self.weight_function = LLMWeightFunction(self.current_mu, self.current_sigma)
    
    def update_sampling_strategy(self, history: List[Dict]) -> None:
        """更新采样策略（同步包装）"""
        asyncio.run(self.update_sampling_strategy_async(history))
    
    def compute_llm_weights(self, X: np.ndarray) -> np.ndarray:
        """
        计算LLM权重 W_LLM
        
        参数：
            X: (n_samples, 3) 候选点
        
        返回：
            weights: (n_samples,) 权重
        """
        if self.weight_function is None:
            return np.ones(len(X))
        
        return self.weight_function.compute_weights(X)
    
    def get_current_strategy(self) -> Dict:
        """获取当前策略"""
        return {
            'mu': self.current_mu,
            'sigma': self.current_sigma,
            'iteration': self.iteration
        }


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("测试 LLM-Enhanced Expected Improvement")
    print("=" * 70)
    
    print("\n[OK] 模块加载成功")
    print("[OK] OptimizationStateDetector")
    print("[OK] LLMSamplingAdvisor")
    print("[OK] SamplingParameterComputer")
    print("[OK] ParameterSensitivityEstimator")
    print("[OK] LLMWeightFunction")
    print("[OK] LLMEnhancedEI")
    
    print("\n" + "=" * 70)
    print("准备就绪！可以与bayes_opt集成。")
    print("=" * 70)