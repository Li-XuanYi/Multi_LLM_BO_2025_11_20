"""
高质量LLM Prompt生成器 - 基于电化学领域知识
Dynamic Prompt Generator for Battery Fast-Charging Optimization

基于论文和最新研究的物理约束和领域知识
参考来源:
- LLMBO论文 Figure 2 (Warm Start Prompt)
- 多阶段恒流充电优化研究
- SEI生长与锂析出机制

Author: Research Team  
Date: 2025-12-06
Version: 2.0 - 领域知识驱动
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class BatteryKnowledgePromptGenerator:
    """
    电池充电优化的领域知识Prompt生成器
    
    功能:
    1. 生成基于电化学原理的WarmStart prompts
    2. 结合历史数据提供few-shot learning examples
    3. 动态调整prompt以探索不同设计空间
    """
    
    def __init__(self):
        """初始化领域知识库"""
        
        # 核心物理约束和机制
        self.physics_knowledge = {
            'thermal': {
                'mechanism': 'Joule heating Q̇ = I²R causes temperature rise quadratically with current',
                'constraint': 'Peak temperature must stay below 309K to prevent thermal runaway and SEI decomposition',
                'coupling': 'Temperature affects reaction kinetics, diffusivity, and SEI growth rate exponentially (Arrhenius)'
            },
            'sei_growth': {
                'mechanism': 'Solid Electrolyte Interphase (SEI) forms from electrolyte reduction at anode surface',
                'impact': 'SEI growth consumes lithium, increases internal resistance, causes irreversible capacity fade',
                'rate_dependency': 'Continuous high-rate charging accelerates SEI layer thickness growth',
                'formula': 'SEI thickness ∝ ∫I(t)·exp(E_a/RT) dt'
            },
            'lithium_plating': {
                'mechanism': 'Li+ ions plate as metallic lithium when anode potential drops below 0V vs Li/Li+',
                'triggers': 'High current density, low temperature, high SOC state',
                'danger': 'Dendrite formation → separator puncture → internal short circuit → fire/explosion',
                'prevention': 'Keep anode potential > 0V, limit current especially at high SOC'
            },
            'two_stage_strategy': {
                'principle': 'High-Low current pattern superior to Low-High for thermal management',
                'stage1': 'High current (1.0-1.2C) delivers bulk energy fast but generates significant heat',
                'stage2': 'Low current (0.2-0.8C) completes charging with minimal additional heating',
                'transition': 'Switching time critical - balances first-stage energy input vs thermal relaxation'
            },
            'voltage_limits': {
                'lower': '2.5V cutoff prevents over-discharge and anode degradation',
                'upper': '4.2V cutoff prevents cathode material decomposition and electrolyte oxidation',
                'anode_potential': 'Graphite operates at 65-200mV vs Li/Li+, very sensitive to overpotential'
            }
        }
        
        # 参数的物理意义和典型范围
        self.parameter_knowledge = {
            'current1': {
                'physical_meaning': 'First-stage charging current delivering bulk energy',
                'typical_range': '1.0C to 1.2C (5.0A to 6.0A for 5Ah cell)',
                'tradeoff': 'Higher current → faster charging BUT higher temperature & SEI growth',
                'constraints': 'Must prevent lithium plating: I1·R_total < V_anode_safe'
            },
            'charging_number': {
                'physical_meaning': 'Time steps before switching to stage 2 (controls stage 1 energy input)',
                'typical_range': '5-25 steps (varies with current magnitude)',
                'tradeoff': 'Longer stage 1 → more total energy but more heat accumulation',
                'coupling': 'Strongly coupled with current1: high I1 requires shorter duration'
            },
            'current2': {
                'physical_meaning': 'Second-stage current completing charge with minimal heating',
                'typical_range': '0.2C to 0.8C (1.0A to 4.0A for 5Ah cell)',
                'tradeoff': 'Higher current → faster completion BUT less thermal recovery',
                'optimization': 'Often optimized to balance time vs temperature/aging objectives'
            }
        }
    
    def generate_warmstart_prompt(
        self,
        n_strategies: int = 5,
        objective_weights: Optional[Dict[str, float]] = None,
        historical_best: Optional[List[Dict]] = None,
        historical_worst: Optional[List[Dict]] = None,
        exploration_emphasis: str = 'balanced'  # 'conservative', 'balanced', 'aggressive'
    ) -> str:
        """
        生成高质量的WarmStart prompt
        
        参数:
            n_strategies: 需要生成的策略数量
            objective_weights: 优化目标权重 {'time': 0.4, 'temp': 0.35, 'aging': 0.25}
            historical_best: 历史最优解列表(few-shot examples)
            historical_worst: 历史最差解列表(用于避免区域)
            exploration_emphasis: 探索倾向
        
        返回:
            完整的LLM prompt字符串
        """
        
        # 设置默认权重
        if objective_weights is None:
            objective_weights = {'time': 0.4, 'temp': 0.35, 'aging': 0.25}
        
        # 构建Prompt
        prompt_sections = []
        
        # ============ 1. 角色定义 ============
        prompt_sections.append(
            "You are an expert electrochemist specializing in lithium-ion battery fast-charging optimization. "
            "Your expertise includes thermal management, SEI formation kinetics, lithium plating prevention, "
            "and multi-objective optimization under strict safety constraints."
        )
        
        # ============ 2. 任务描述 ============
        prompt_sections.append(f"\n\nTASK:")
        prompt_sections.append(
            f"Generate {n_strategies} diverse and physically plausible two-stage constant-current (CC) "
            f"charging strategies for a lithium-ion battery. Each strategy must balance competing objectives "
            f"while respecting electrochemical constraints and safety limits."
        )
        
        # ============ 3. 电池规格 ============
        prompt_sections.append("\n\nBATTERY SPECIFICATIONS:")
        prompt_sections.append("- Chemistry: Graphite anode / NMC or LFP cathode (typical Li-ion)")
        prompt_sections.append("- Nominal Capacity: 5.0 Ah")
        prompt_sections.append("- Voltage Range: 2.5V (discharge cutoff) to 4.2V (charge cutoff)")
        prompt_sections.append("- Initial Conditions: 0% SOC, 298K ambient temperature")
        prompt_sections.append("- Target: Charge to 80% SOC (avoiding stress from extreme SOC levels)")
        prompt_sections.append("- Anode Potential Operating Range: 65-200mV vs Li/Li+ (graphite)")
        
        # ============ 4. 优化目标 ============
        prompt_sections.append("\n\nOPTIMIZATION OBJECTIVES (weighted multi-objective):")
        prompt_sections.append(f"1. Minimize Charging Time (weight: {objective_weights['time']:.2f})")
        prompt_sections.append(f"   - Target: 25-42 minutes vs current 80+ minute baseline")
        prompt_sections.append(f"   - Measured in time steps (each step ≈ 2-3 seconds)")
        
        prompt_sections.append(f"\n2. Minimize Peak Temperature (weight: {objective_weights['temp']:.2f})")
        prompt_sections.append(f"   - HARD CONSTRAINT: T_max ≤ 309K (36°C)")
        prompt_sections.append(f"   - Exceeding 309K risks thermal runaway and accelerated degradation")
        prompt_sections.append(f"   - Heat generation: Q̇ = I²R (quadratic with current)")
        
        prompt_sections.append(f"\n3. Minimize Capacity Fade (weight: {objective_weights['aging']:.2f})")
        prompt_sections.append(f"   - Primary mechanism: SEI layer growth at anode")
        prompt_sections.append(f"   - SEI rate ∝ current density × exp(E_a/RT)")
        prompt_sections.append(f"   - Irreversible lithium loss reduces usable capacity")
        
        # ============ 5. 参数约束和物理意义 ============
        prompt_sections.append("\n\nPARAMETER CONSTRAINTS AND PHYSICAL MEANING:")
        
        prompt_sections.append("\n1. current1 (First-Stage Current): 3.0 - 6.0 A")
        prompt_sections.append(f"   - Physical meaning: {self.parameter_knowledge['current1']['physical_meaning']}")
        prompt_sections.append(f"   - Typical best range: {self.parameter_knowledge['current1']['typical_range']}")
        prompt_sections.append(f"   - Tradeoff: {self.parameter_knowledge['current1']['tradeoff']}")
        prompt_sections.append(f"   - Safety: {self.parameter_knowledge['current1']['constraints']}")
        
        prompt_sections.append("\n2. charging_number (Stage 1 Duration): 5 - 25 time steps")
        prompt_sections.append(f"   - Physical meaning: {self.parameter_knowledge['charging_number']['physical_meaning']}")
        prompt_sections.append(f"   - Typical range: {self.parameter_knowledge['charging_number']['typical_range']}")
        prompt_sections.append(f"   - Tradeoff: {self.parameter_knowledge['charging_number']['tradeoff']}")
        prompt_sections.append(f"   - Coupling: {self.parameter_knowledge['charging_number']['coupling']}")
        
        prompt_sections.append("\n3. current2 (Second-Stage Current): 1.0 - 4.0 A")
        prompt_sections.append(f"   - Physical meaning: {self.parameter_knowledge['current2']['physical_meaning']}")
        prompt_sections.append(f"   - Typical best range: {self.parameter_knowledge['current2']['typical_range']}")
        prompt_sections.append(f"   - Tradeoff: {self.parameter_knowledge['current2']['tradeoff']}")
        prompt_sections.append(f"   - Optimization: {self.parameter_knowledge['current2']['optimization']}")
        
        # ============ 6. 关键物理考虑 ============
        prompt_sections.append("\n\nCRITICAL PHYSICAL CONSIDERATIONS:")
        
        prompt_sections.append("\n【Thermal Management】")
        prompt_sections.append(f"- {self.physics_knowledge['thermal']['mechanism']}")
        prompt_sections.append(f"- {self.physics_knowledge['thermal']['constraint']}")
        prompt_sections.append(f"- {self.physics_knowledge['thermal']['coupling']}")
        prompt_sections.append("- High current1 + long charging_number = excessive heat accumulation")
        
        prompt_sections.append("\n【SEI Growth and Capacity Fade】")
        prompt_sections.append(f"- {self.physics_knowledge['sei_growth']['mechanism']}")
        prompt_sections.append(f"- {self.physics_knowledge['sei_growth']['impact']}")
        prompt_sections.append(f"- {self.physics_knowledge['sei_growth']['rate_dependency']}")
        prompt_sections.append("- Minimize ∫I²(t) dt to slow SEI growth")
        
        prompt_sections.append("\n【Lithium Plating Prevention】")
        prompt_sections.append(f"- {self.physics_knowledge['lithium_plating']['mechanism']}")
        prompt_sections.append(f"- {self.physics_knowledge['lithium_plating']['triggers']}")
        prompt_sections.append(f"- {self.physics_knowledge['lithium_plating']['danger']}")
        prompt_sections.append(f"- {self.physics_knowledge['lithium_plating']['prevention']}")
        prompt_sections.append("- Particularly critical when current1 > 5.5A or at end of stage 1")
        
        prompt_sections.append("\n【Two-Stage Strategy Rationale】")
        prompt_sections.append(f"- {self.physics_knowledge['two_stage_strategy']['principle']}")
        prompt_sections.append(f"- Stage 1: {self.physics_knowledge['two_stage_strategy']['stage1']}")
        prompt_sections.append(f"- Stage 2: {self.physics_knowledge['two_stage_strategy']['stage2']}")
        prompt_sections.append(f"- Transition: {self.physics_knowledge['two_stage_strategy']['transition']}")
        
        # ============ 7. Few-Shot Examples (如果有历史数据) ============
        if historical_best and len(historical_best) > 0:
            prompt_sections.append("\n\nHISTORICAL BEST SOLUTIONS (learn from these):")
            for i, sol in enumerate(historical_best[:3], 1):  # 最多3个最优解
                prompt_sections.append(f"\nExample {i}:")
                prompt_sections.append(f"  current1 = {sol['params']['current1']:.2f}A")
                prompt_sections.append(f"  charging_number = {sol['params']['charging_number']}")
                prompt_sections.append(f"  current2 = {sol['params']['current2']:.2f}A")
                prompt_sections.append(f"  → time = {sol['objectives']['time']:.0f} steps, "
                                     f"temp = {sol['objectives']['temp']:.1f}K, "
                                     f"aging = {sol['objectives']['aging']:.6f}%")
                prompt_sections.append(f"  → scalarized score = {sol['scalarized']:.4f}")
        
        if historical_worst and len(historical_worst) > 0:
            prompt_sections.append("\n\nHISTORICAL WORST SOLUTIONS (avoid these regions):")
            for i, sol in enumerate(historical_worst[:2], 1):  # 最多2个最差解
                prompt_sections.append(f"\nBad Example {i}:")
                prompt_sections.append(f"  current1 = {sol['params']['current1']:.2f}A, "
                                     f"charging_number = {sol['params']['charging_number']}, "
                                     f"current2 = {sol['params']['current2']:.2f}A")
                prompt_sections.append(f"  → POOR: score = {sol['scalarized']:.4f}")
                if not sol.get('valid', True):
                    prompt_sections.append(f"  → VIOLATED CONSTRAINTS!")
        
        # ============ 8. 探索策略指导 ============
        prompt_sections.append("\n\nSTRATEGY GENERATION GUIDANCE:")
        
        if exploration_emphasis == 'conservative':
            prompt_sections.append("- CONSERVATIVE MODE: Generate safe, proven strategies close to known good solutions")
            prompt_sections.append("- Prioritize feasibility and constraint satisfaction")
            prompt_sections.append("- Vary parameters by ±10-20% from historical best")
            
        elif exploration_emphasis == 'aggressive':
            prompt_sections.append("- AGGRESSIVE MODE: Explore boundary regions and unconventional combinations")
            prompt_sections.append("- Test higher currents and extreme transition points")
            prompt_sections.append("- Accept some risk of constraint violations to discover novel strategies")
            
        else:  # balanced
            prompt_sections.append("- BALANCED MODE: Mix proven strategies with exploratory variations")
            prompt_sections.append("- 60% strategies near known good regions, 40% exploring new areas")
            prompt_sections.append("- Systematically vary one parameter at a time while keeping others stable")
        
        prompt_sections.append("\n- Ensure diversity: avoid generating similar strategies")
        prompt_sections.append("- Consider parameter coupling: high current1 should pair with shorter charging_number")
        prompt_sections.append("- Respect physical causality: stage 1 sets thermal state for stage 2")
        
        # ============ 9. 输出格式要求 ============
        prompt_sections.append("\n\nOUTPUT FORMAT:")
        prompt_sections.append("For each strategy, provide:")
        prompt_sections.append("1. Parameter values (current1, charging_number, current2)")
        prompt_sections.append("2. Brief physical reasoning (1-2 sentences)")
        prompt_sections.append("3. Expected behavior prediction")
        
        prompt_sections.append("\n\nRESPOND IN JSON FORMAT:")
        prompt_sections.append("""
[
  {
    "current1": <float between 3.0-6.0>,
    "charging_number": <integer between 5-25>,
    "current2": <float between 1.0-4.0>,
    "reasoning": "<brief explanation of the strategy's physical rationale>"
  },
  ...
]
""")
        
        prompt_sections.append("\n\nREMEMBER:")
        prompt_sections.append("- All parameter values MUST be within specified bounds")
        prompt_sections.append("- Strategies should reflect deep electrochemical understanding, not random guessing")
        prompt_sections.append("- Diversity is key: generate {n_strategies} DIFFERENT strategies")
        prompt_sections.append("- Prioritize physical feasibility over extreme optimization")
        
        return "\n".join(prompt_sections)
    
    def generate_exploration_guidance(
        self,
        current_iteration: int,
        stagnation_detected: bool = False
    ) -> str:
        """
        根据优化进度生成探索指导
        
        参数:
            current_iteration: 当前迭代次数
            stagnation_detected: 是否检测到停滞
        
        返回:
            探索指导建议
        """
        
        if current_iteration < 10:
            return "balanced"  # 早期: 平衡探索
        elif stagnation_detected:
            return "aggressive"  # 停滞: 激进探索
        else:
            return "conservative"  # 后期: 保守利用


# ============ 测试代码 ============
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("测试 BatteryKnowledgePromptGenerator")
    print("=" * 80)
    
    generator = BatteryKnowledgePromptGenerator()
    
    # 模拟历史数据
    historical_best = [
        {
            'params': {'current1': 4.8, 'charging_number': 8, 'current2': 2.9},
            'objectives': {'time': 35, 'temp': 305.2, 'aging': 0.0012},
            'scalarized': 0.15
        }
    ]
    
    historical_worst = [
        {
            'params': {'current1': 5.9, 'charging_number': 20, 'current2': 3.8},
            'objectives': {'time': 42, 'temp': 312.5, 'aging': 0.0045},
            'scalarized': 0.89,
            'valid': False
        }
    ]
    
    # 生成prompt
    prompt = generator.generate_warmstart_prompt(
        n_strategies=5,
        objective_weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
        historical_best=historical_best,
        historical_worst=historical_worst,
        exploration_emphasis='balanced'
    )
    
    print("\n生成的Prompt (前1000字符):")
    print("=" * 80)
    print(prompt[:1000])
    print("...")
    print(f"\n总长度: {len(prompt)} 字符")
    print("\n[OK] Prompt生成器测试完成")
