"""
测试归一化修复 - 验证边界过窄时的处理
"""
import sys
sys.path.insert(0, './BO/llmbo_core')

import numpy as np
from multi_objective_evaluator import MultiObjectiveEvaluator

print("=" * 80)
print("测试归一化数值稳定性修复")
print("=" * 80)

# 初始化评估器
evaluator = MultiObjectiveEvaluator(
    weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
    verbose=True
)

print("\n[测试1] 正常情况 - 边界合理")
print("-" * 80)
objectives1 = {'time': 50, 'temp': 305.0, 'aging': 0.005}
normalized1 = evaluator._normalize(objectives1)
scalarized1 = evaluator._chebyshev_scalarize(normalized1)
print(f"目标值: {objectives1}")
print(f"归一化: {normalized1}")
print(f"标量化: {scalarized1:.4f}")

print("\n[测试2] 边界过窄情况 - 模拟worst≈best")
print("-" * 80)
# 手动设置过窄的边界
evaluator.bounds = {
    'time': {'best': 50.0, 'worst': 50.001},  # 极窄范围
    'temp': {'best': 305.0, 'worst': 305.001},
    'aging': {'best': 0.005, 'worst': 0.005001}
}
objectives2 = {'time': 50, 'temp': 305.0, 'aging': 0.005}
normalized2 = evaluator._normalize(objectives2)
scalarized2 = evaluator._chebyshev_scalarize(normalized2)
print(f"目标值: {objectives2}")
print(f"归一化: {normalized2}")
print(f"标量化: {scalarized2:.4f}")

print("\n[测试3] 边界更新后 - 确保最小范围")
print("-" * 80)
# 添加历史数据（所有值接近）
evaluator.eval_count = 15
for _ in range(10):
    evaluator.history['time'].append(50.0 + np.random.uniform(-0.5, 0.5))
    evaluator.history['temp'].append(305.0 + np.random.uniform(-0.1, 0.1))
    evaluator.history['aging'].append(0.005 + np.random.uniform(-0.0001, 0.0001))
    evaluator.history['valid'].append(True)

evaluator._update_bounds()
print(f"\n更新后的边界:")
for key in ['time', 'temp', 'aging']:
    best = evaluator.bounds[key]['best']
    worst = evaluator.bounds[key]['worst']
    range_val = worst - best
    print(f"  {key}: [{best:.6f}, {worst:.6f}], 范围={range_val:.6f}")

objectives3 = {'time': 50, 'temp': 305.0, 'aging': 0.005}
normalized3 = evaluator._normalize(objectives3)
scalarized3 = evaluator._chebyshev_scalarize(normalized3)
print(f"\n目标值: {objectives3}")
print(f"归一化: {normalized3}")
print(f"标量化: {scalarized3:.4f}")

print("\n" + "=" * 80)
print("测试结果:")
print("=" * 80)
print(f"✅ 测试1 (正常): 标量化 = {scalarized1:.4f} (应该 > 0)")
print(f"✅ 测试2 (边界过窄): 标量化 = {scalarized2:.4f} (修复后 > 0)")
print(f"✅ 测试3 (最小范围): 标量化 = {scalarized3:.4f} (修复后 > 0)")

if scalarized1 > 0 and scalarized2 > 0 and scalarized3 > 0:
    print("\n[OK] 所有测试通过，归一化数值稳定性修复成功!")
else:
    print("\n[X] 仍有问题，需要进一步调试")
print("=" * 80)
