"""
调试归一化fallback逻辑
"""
import sys
sys.path.insert(0, './BO/llmbo_core')

import numpy as np
from multi_objective_evaluator import MultiObjectiveEvaluator

print("调试归一化fallback逻辑")
print("=" * 80)

evaluator = MultiObjectiveEvaluator(
    weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
    verbose=True
)

# 设置过窄的动态边界
evaluator.bounds = {
    'time': {'best': 50.0, 'worst': 50.0001},  
    'temp': {'best': 305.0, 'worst': 305.0001},
    'aging': {'best': 0.005, 'worst': 0.005001}
}

evaluator.eval_count = 15  # 触发verbose输出

print("\n当前边界设置:")
print(f"  bounds: {evaluator.bounds}")
print(f"  temp_bounds: {evaluator.temp_bounds}")

objectives = {'time': 50, 'temp': 305.0, 'aging': 0.005}
print(f"\n测试归一化: {objectives}")

# 逐步调试
for key in ['time', 'temp', 'aging']:
    best = evaluator.bounds[key]['best']
    worst = evaluator.bounds[key]['worst']
    denominator = worst - best
    print(f"\n{key}:")
    print(f"  best={best}, worst={worst}")
    print(f"  denominator={denominator}")
    print(f"  is narrow? {abs(denominator) < 1e-6}")
    
    if abs(denominator) < 1e-6:
        temp_best = evaluator.temp_bounds[key]['best']
        temp_worst = evaluator.temp_bounds[key]['worst']
        temp_denom = temp_worst - temp_best
        print(f"  → Fallback: temp_best={temp_best}, temp_worst={temp_worst}")
        print(f"  → temp_denominator={temp_denom}")

normalized = evaluator._normalize(objectives)
print(f"\n最终归一化结果: {normalized}")

scalarized = evaluator._chebyshev_scalarize(normalized)
print(f"标量化: {scalarized:.4f}")
