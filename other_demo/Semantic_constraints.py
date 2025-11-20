"""
Semantic Constraints Module (语义约束 S)
用于 LLM 增强的分解多目标贝叶斯优化 (LLM-DMOBO)

功能：
1. 定义基于物理知识的语义约束
2. 使用 LLM 生成额外的专家级约束
3. 验证充电策略是否满足约束
4. 提供约束违反的详细报告

基于 manuscript 中的约束方法设计

作者: Claude AI Assistant
日期: 2025-01-12
版本: v1.0
"""

import json
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple, Any
from openai import OpenAI


class SemanticConstraints:
    """
    语义约束管理器
    
    结合专家知识和 LLM 生成的约束，确保充电策略的物理合理性和安全性
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = 'https://api.nuwaapi.com/v1',
        model: str = "gpt-3.5-turbo",
        verbose: bool = True
    ):
        """
        初始化语义约束管理器
        
        参数:
            api_key: OpenAI API 密钥（可选，不使用 LLM 生成时可为 None）
            base_url: API 基础URL
            model: 使用的模型名称
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        self.constraints = []
        
        # LLM 客户端（可选）
        if api_key:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model
            self.llm_available = True
        else:
            self.client = None
            self.llm_available = False
        
        # 添加预定义的专家约束
        self._add_expert_constraints()
        
        if self.verbose:
            print("=" * 70)
            print("语义约束管理器已初始化")
            print("=" * 70)
            print(f"预定义约束数量: {len(self.constraints)}")
            print(f"LLM 可用: {self.llm_available}")
            print("=" * 70)
    
    def _add_expert_constraints(self) -> None:
        """
        添加预定义的专家约束
        
        基于锂电池充电领域的专家知识
        """
        # 约束1: I1 必须大于 I2（先快后慢策略）
        self.add_constraint(
            name="current_order",
            check_func=lambda I1, t1, I2: I1 > I2,
            description="第一阶段电流必须大于第二阶段电流（先快后慢策略）",
            priority=1,
            category="physical"
        )
        
        # 约束2: 高电流不宜持续太久（防止过热）
        self.add_constraint(
            name="high_current_duration",
            check_func=lambda I1, t1, I2: not (I1 > 6.0 and t1 > 20),
            description="高电流（>6A）不应持续超过20步（防止热累积）",
            priority=2,
            category="thermal"
        )
        
        # 约束3: I2 不能太低（避免充电过慢）
        self.add_constraint(
            name="min_second_current",
            check_func=lambda I1, t1, I2: I2 >= 2.0,
            description="第二阶段电流不应低于2.0A（保证充电效率）",
            priority=3,
            category="efficiency"
        )
        
        # 约束4: 电流变化不应过于剧烈（平稳过渡）
        self.add_constraint(
            name="smooth_transition",
            check_func=lambda I1, t1, I2: abs(I1 - I2) <= 5.0,
            description="两阶段电流差不应超过5.0A（避免电压突变）",
            priority=2,
            category="voltage"
        )
        
        # 约束5: 切换时机不应太早（确保初期快充效果）
        self.add_constraint(
            name="min_first_stage_duration",
            check_func=lambda I1, t1, I2: t1 >= 5,
            description="第一阶段至少持续5步（确保快充效果）",
            priority=3,
            category="efficiency"
        )
        
        # 约束6: 切换时机不应太晚（避免过度快充）
        self.add_constraint(
            name="max_first_stage_duration",
            check_func=lambda I1, t1, I2: t1 <= 25,
            description="第一阶段不应超过25步（避免过度快充导致老化）",
            priority=2,
            category="aging"
        )
        
        # 约束7: 极高电流需要更早切换（安全保护）
        self.add_constraint(
            name="extreme_current_safety",
            check_func=lambda I1, t1, I2: not (I1 > 7.0 and t1 > 15),
            description="极高电流（>7A）必须在15步内切换（安全保护）",
            priority=1,
            category="safety"
        )
        
        # 约束8: 电流比例合理性（经验规则）
        self.add_constraint(
            name="current_ratio",
            check_func=lambda I1, t1, I2: I2 / I1 >= 0.3,
            description="I2/I1 应≥0.3（避免过度不平衡）",
            priority=3,
            category="physical"
        )
    
    def add_constraint(
        self,
        name: str,
        check_func: Callable[[float, float, float], bool],
        description: str,
        priority: int = 2,
        category: str = "general"
    ) -> None:
        """
        添加单个约束
        
        参数:
            name: 约束名称
            check_func: 检查函数 (I1, t1, I2) -> bool
            description: 约束描述
            priority: 优先级 (1=最高, 3=最低)
            category: 约束类别
        """
        constraint = {
            'name': name,
            'check': check_func,
            'description': description,
            'priority': priority,
            'category': category,
            'source': 'expert'
        }
        self.constraints.append(constraint)
    
    def check_all(
        self,
        I1: float,
        t1: float,
        I2: float,
        return_details: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        检查所有约束
        
        参数:
            I1: 第一阶段电流 [A]
            t1: 切换步数
            I2: 第二阶段电流 [A]
            return_details: 是否返回详细的违反信息
        
        返回:
            (is_valid, violations)
            - is_valid: 是否通过所有约束
            - violations: 违反的约束列表
        """
        violations = []
        
        for constraint in self.constraints:
            try:
                if not constraint['check'](I1, t1, I2):
                    violation_info = {
                        'name': constraint['name'],
                        'description': constraint['description'],
                        'priority': constraint['priority'],
                        'category': constraint['category']
                    }
                    violations.append(violation_info if return_details else constraint['description'])
            except Exception as e:
                if self.verbose:
                    print(f"  约束 '{constraint['name']}' 检查失败: {e}")
        
        is_valid = len(violations) == 0
        
        return is_valid, violations
    
    def generate_constraints_from_llm(
        self,
        num_constraints: int = 5,
        focus_area: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 生成额外的语义约束
        
        参数:
            num_constraints: 生成的约束数量
            focus_area: 关注领域（如 'thermal', 'aging', 'safety'）
        
        返回:
            生成的约束列表
        """
        if not self.llm_available:
            print(" LLM 不可用，无法生成约束")
            return []
        
        if self.verbose:
            print(f"\n 使用 LLM 生成 {num_constraints} 个语义约束...")
        
        # 构建 Prompt
        prompt = self._construct_llm_constraint_prompt(num_constraints, focus_area)
        
        # 调用 LLM
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in lithium-ion battery charging physics and safety. "
                                   "You provide precise, executable Python constraints."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=2000
            )
            
            response = completion.choices[0].message.content
            
            # 解析 LLM 响应
            new_constraints = self._parse_llm_constraints(response)
            
            if self.verbose:
                print(f" 成功生成 {len(new_constraints)} 个LLM约束")
            
            return new_constraints
            
        except Exception as e:
            print(f" LLM 约束生成失败: {e}")
            return []
    
    def _construct_llm_constraint_prompt(
        self,
        num_constraints: int,
        focus_area: Optional[str]
    ) -> str:
        """
        构建 LLM 约束生成的 Prompt
        """
        prompt = f"""You are an expert in lithium-ion battery two-stage constant-current fast charging.

**Task**: Generate {num_constraints} additional semantic constraints for charging strategy validation.

**Parameters**:
- I1: First-stage current [A], range [3.0, 6.0]
- t1: Switching time [steps], range [5, 30]
- I2: Second-stage current [A], range [1.0, 4.0]

**Existing Constraints** (for reference, don't duplicate):
1. I1 > I2 (fast-then-slow)
2. High current (>5A) should not last >20 steps
3. I2 ≥ 2.0A (efficiency)
4. |I1 - I2| ≤ 4.5A (smooth transition)

"""
        
        if focus_area:
            prompt += f"""**Focus Area**: {focus_area}
Please emphasize constraints related to {focus_area}.

"""
        
        prompt += """**Requirements**:
1. Each constraint should be based on physical principles or expert knowledge
2. Constraints should be practical and verifiable
3. Provide diverse constraints covering different aspects (thermal, aging, voltage, safety)

**Output Format** (CRITICAL):
Return ONLY a valid JSON array. Each constraint must have:
- name: Short identifier (e.g., "thermal_limit_advanced")
- description: Clear explanation
- priority: 1 (highest) to 3 (lowest)
- category: One of [thermal, aging, voltage, safety, efficiency, physical]
- condition: Python lambda expression string, e.g., "lambda I1, t1, I2: I1 * t1 < 120"

Example format:
[
  {
    "name": "thermal_energy_limit",
    "description": "Total thermal energy (I1 * t1) should not exceed 120 to prevent overheating",
    "priority": 1,
    "category": "thermal",
    "condition": "lambda I1, t1, I2: I1 * t1 < 120"
  },
  ...
]

**Important**:
- The condition must be a valid Python lambda expression
- It must take exactly 3 parameters: I1, t1, I2
- It must return a boolean
- Use simple mathematical operations only

Generate {num_constraints} constraints now:"""
        
        return prompt
    
    def _parse_llm_constraints(self, response: str) -> List[Dict[str, Any]]:
        """
        解析 LLM 生成的约束
        """
        # 清理响应
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            constraints_data = json.loads(response)
            
            parsed_constraints = []
            for constraint_data in constraints_data:
                try:
                    # 安全地评估 lambda 表达式
                    # 注意：这里有安全风险，实际生产环境需要更严格的验证
                    condition_str = constraint_data['condition']
                    check_func = eval(condition_str)
                    
                    # 添加约束
                    self.add_constraint(
                        name=constraint_data['name'],
                        check_func=check_func,
                        description=constraint_data['description'],
                        priority=constraint_data.get('priority', 2),
                        category=constraint_data.get('category', 'general')
                    )
                    
                    # 更新 source
                    self.constraints[-1]['source'] = 'llm'
                    
                    parsed_constraints.append(constraint_data)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  约束解析失败: {e}")
            
            return parsed_constraints
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
            return []
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """
        获取约束摘要
        """
        summary = {
            'total': len(self.constraints),
            'by_source': {
                'expert': sum(1 for c in self.constraints if c['source'] == 'expert'),
                'llm': sum(1 for c in self.constraints if c['source'] == 'llm')
            },
            'by_category': {},
            'by_priority': {1: 0, 2: 0, 3: 0}
        }
        
        for constraint in self.constraints:
            # 按类别统计
            category = constraint['category']
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # 按优先级统计
            priority = constraint['priority']
            summary['by_priority'][priority] += 1
        
        return summary
    
    def print_all_constraints(self) -> None:
        """
        打印所有约束
        """
        print("\n" + "=" * 70)
        print("所有语义约束:")
        print("=" * 70)
        
        for i, constraint in enumerate(self.constraints, 1):
            print(f"\n{i}. {constraint['name']} (优先级{constraint['priority']}, {constraint['category']})")
            print(f"   描述: {constraint['description']}")
            print(f"   来源: {constraint['source']}")
        
        print("\n" + "=" * 70)
        
        # 打印摘要
        summary = self.get_constraint_summary()
        print("约束摘要:")
        print(f"  总数: {summary['total']}")
        print(f"  专家约束: {summary['by_source']['expert']}")
        print(f"  LLM约束: {summary['by_source']['llm']}")
        print(f"  按类别: {summary['by_category']}")
        print(f"  按优先级: {summary['by_priority']}")
        print("=" * 70)


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("【测试】Semantic Constraints")
    print("=" * 70)
    
    # 测试1: 不使用 LLM（仅专家约束）
    print("\n【测试1】专家约束")
    constraints = SemanticConstraints(api_key=None, verbose=True)
    constraints.print_all_constraints()
    
    # 测试一些策略
    test_cases = [
        (5.0, 10, 3.0, "正常策略"),
        (2.0, 25, 3.5, "违反 I1>I2"),
        (6.0, 25, 3.0, "违反高电流持续时间"),
        (5.0, 10, 0.5, "违反 I2 最小值"),
        (6.0, 10, 1.0, "违反电流变化"),
    ]
    
    print("\n" + "=" * 70)
    print("测试策略验证:")
    print("=" * 70)
    
    for I1, t1, I2, label in test_cases:
        is_valid, violations = constraints.check_all(I1, t1, I2, return_details=True)
        status = "✅ 通过" if is_valid else "❌ 违反"
        print(f"\n{label}: I1={I1}, t1={t1}, I2={I2}")
        print(f"  {status}")
        if not is_valid:
            print(f"  违反的约束:")
            for v in violations:
                print(f"    - [{v['category']}] {v['description']}")
    
    # 测试2: 使用 LLM 生成约束
    print("\n\n【测试2】LLM 生成约束")
    try:
        constraints_with_llm = SemanticConstraints(
            api_key='sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK',
            base_url='https://api.nuwaapi.com/v1',
            model='gpt-3.5-turbo',
            verbose=True
        )
        
        # 生成额外约束
        new_constraints = constraints_with_llm.generate_constraints_from_llm(
            num_constraints=3,
            focus_area="thermal management and battery aging"
        )
        
        # 打印所有约束
        constraints_with_llm.print_all_constraints()
        
        print("\n✅ 测试2通过")
        
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)