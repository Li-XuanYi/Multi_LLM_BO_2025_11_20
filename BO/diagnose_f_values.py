#!/usr/bin/env python3
"""
诊断 f 值大于 1 的原因
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llmbo_core.multi_objective_evaluator import MultiObjectiveEvaluator

print("=" * 80)
print("诊断 f 值 > 1 的原因")
print("=" * 80)

# 创建评估器
evaluator = MultiObjectiveEvaluator(verbose=False)

# 模拟用户看到的数据
test_cases = [
    {"I1": 5.50, "t1": 10, "I2": 2.50, "f": 4.4363, "case": "策略1"},
    {"I1": 4.00, "t1": 15, "I2": 3.00, "f": 2.5293, "case": "策略2"},
    {"I1": 3.70, "t1": 20, "I2": 3.80, "f": 2.5150, "case": "策略3"},
    {"I1": 6.00, "t1": 8, "I2": 1.00, "f": 4.4363, "case": "策略4"},
    {"I1": 3.50, "t1": 25, "I2": 4.00, "f": 2.5912, "case": "策略5", 
     "time": 64, "temp": 306.69, "aging_raw": 1.6815},
]

print("\n物理边界:")
for key in ['time', 'temp', 'aging']:
    print(f"  {key}: {evaluator.physical_bounds[key]}")

print("\n运行时边界（初始）:")
for key in ['time', 'temp', 'aging']:
    print(f"  {key}: {evaluator.running_bounds[key]}")

print("\n临时边界:")
for key in ['time', 'temp', 'aging']:
    print(f"  {key}: {evaluator.temp_bounds[key]}")

print("\n权重:")
print(f"  {evaluator.weights}")

# 分析策略5（有完整数据）
print("\n" + "=" * 80)
print("详细分析策略5")
print("=" * 80)

case5 = test_cases[4]
time = case5.get("time", 64)
temp = case5.get("temp", 306.69)
aging_raw = case5.get("aging_raw", 1.6815)

# 对数变换
aging_log = evaluator._apply_log_transform(aging_raw)

print(f"\n原始值:")
print(f"  时间: {time} 步")
print(f"  温度: {temp} K")
print(f"  老化（原始）: {aging_raw} %")
print(f"  老化（对数）: {aging_log:.4f}")

# 使用临时边界归一化（前几次评估）
temp_bounds = evaluator.temp_bounds
objectives_with_log = {
    'time': time,
    'temp': temp,
    'aging': aging_log
}

print(f"\n使用临时边界归一化:")
normalized = {}
for key in ['time', 'temp', 'aging']:
    denom = temp_bounds[key]['worst'] - temp_bounds[key]['best']
    if key == 'time':
        # 时间越小越好
        normalized[key] = (objectives_with_log[key] - temp_bounds[key]['best']) / denom
    elif key == 'temp':
        # 温度越低越好
        normalized[key] = (objectives_with_log[key] - temp_bounds[key]['best']) / denom
    else:  # aging
        # 老化越低越好
        normalized[key] = (objectives_with_log[key] - temp_bounds[key]['best']) / denom
    
    print(f"  {key}: ({objectives_with_log[key]:.2f} - {temp_bounds[key]['best']}) / {denom:.2f} = {normalized[key]:.4f}")

# 切比雪夫标量化
weighted = {k: evaluator.weights[k] * normalized[k] for k in ['time', 'temp', 'aging']}
max_weighted = max(weighted.values())
sum_weighted = sum(weighted.values())
scalarized_base = max_weighted + 0.05 * sum_weighted

print(f"\n加权归一化:")
for k in ['time', 'temp', 'aging']:
    print(f"  {k}: {evaluator.weights[k]} × {normalized[k]:.4f} = {weighted[k]:.4f}")

print(f"\n切比雪夫标量化:")
print(f"  max(weighted) = {max_weighted:.4f}")
print(f"  sum(weighted) = {sum_weighted:.4f}")
print(f"  基础 f = {max_weighted:.4f} + 0.05 × {sum_weighted:.4f} = {scalarized_base:.4f}")

# 软约束
constraint_result = evaluator.soft_constraints.compute_total_penalty(objectives_with_log)
total_penalty = constraint_result['total_penalty']

print(f"\n软约束惩罚:")
print(f"  温度惩罚: {constraint_result['penalties']['temp']:.4f} (状态: {constraint_result['statuses']['temp']})")
print(f"  老化惩罚: {constraint_result['penalties']['aging']:.4f} (状态: {constraint_result['statuses']['aging']})")
print(f"  总惩罚: {total_penalty:.4f}")

scalarized_with_penalty = scalarized_base + total_penalty

if constraint_result['is_severe']:
    scalarized_with_penalty += 0.2
    print(f"  严重违规惩罚: +0.2")

print(f"\n最终 f = {scalarized_with_penalty:.4f}")
print(f"用户看到的 f = {case5['f']:.4f}")
print(f"差异 = {abs(scalarized_with_penalty - case5['f']):.4f}")

# 分析问题
print("\n" + "=" * 80)
print("问题分析")
print("=" * 80)

print("\n⚠️ 发现的问题:")

# 问题1: 归一化值 > 1
issues = []
for key, val in normalized.items():
    if val > 1.0:
        issues.append(f"  ❌ {key} 归一化值 = {val:.4f} > 1.0")
        issues.append(f"     原因: {objectives_with_log[key]:.2f} 超出边界 [{temp_bounds[key]['best']}, {temp_bounds[key]['worst']}]")

if issues:
    print("\n".join(issues))
else:
    print("  ✅ 归一化值都在 [0, 1] 范围内")

# 问题2: 边界设置
print("\n边界问题:")
print(f"  临时边界 aging: [{temp_bounds['aging']['best']}, {temp_bounds['aging']['worst']}]")
print(f"  实际 aging_log: {aging_log:.4f}")

if aging_log > temp_bounds['aging']['worst']:
    print(f"  ❌ aging_log ({aging_log:.4f}) > worst ({temp_bounds['aging']['worst']})")
    print(f"     这会导致归一化值 > 1")
else:
    print(f"  ✅ aging_log 在边界内")

# 问题3: 权重加和
print("\n切比雪夫标量化分析:")
print(f"  max(weighted) = {max_weighted:.4f}")
if max_weighted > 0.5:
    print(f"  ⚠️ max(weighted) > 0.5，这是主要原因")
    print(f"     这意味着至少有一个目标的归一化值很高")
    
    for k, w in weighted.items():
        if w == max_weighted:
            print(f"     → {k} 是瓶颈: {normalized[k]:.4f} (归一化) × {evaluator.weights[k]} = {w:.4f}")

# 根本原因总结
print("\n" + "=" * 80)
print("根本原因总结")
print("=" * 80)

print("""
f > 1 的主要原因：

1. ❗ 老化边界不合理
   - temp_bounds['aging']['worst'] = 6.5
   - 但实际 aging_log 可能接近或超过这个值
   - 导致归一化值 > 1

2. ❗ 时间/温度也可能超边界
   - 如果仿真时间很长（接近120步）或温度很高（接近312K）
   - 归一化值会接近 1.0

3. 切比雪夫标量化的特性
   - f = max(weighted) + 0.05 * sum(weighted)
   - 如果任意一个归一化值 > 0.5，对应的加权值就可能 > 0.4
   - max(weighted) 就会 > 0.4，导致 f > 0.4

4. 软约束惩罚
   - 温度接近312K会有轻微惩罚
   - 老化高也会有惩罚
   - 累加后可能让 f 从 0.4 增加到 2.5

解决方案：

✅ 方案1（推荐）: 扩大物理边界
   - aging: 0.0 → 8.0（而不是6.5）
   - time: 20 → 150（而不是120）
   - 这样能容纳更极端的情况

✅ 方案2: 归一化后强制 clip
   - 无论边界如何，都 clip(normalized[key], 0.0, 1.0)
   - 已在代码中实现，但只对 valid=True 的点

✅ 方案3: 调整权重
   - 降低某些目标的权重，让 max(weighted) < 0.5
""")
