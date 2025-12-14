#!/usr/bin/env python3
"""
快速测试comparison_runner的输出格式
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llmbo_core import MultiObjectiveEvaluator
from Comparison.base_optimizer import OptimizerFactory

print("\n" + "=" * 80)
print("测试comparison_runner的输出格式")
print("=" * 80)

# 创建评估器
evaluator = MultiObjectiveEvaluator(verbose=False)

# 运行几次评估
test_params = [
    {'current1': 4.0, 'charging_number': 15, 'current2': 2.5},
    {'current1': 4.5, 'charging_number': 12, 'current2': 3.0},
    {'current1': 3.5, 'charging_number': 18, 'current2': 2.0}
]

for i, params in enumerate(test_params, 1):
    print(f"\n评估 {i}: {params}")
    try:
        scalarized = evaluator.evaluate(**params)
        print(f"  标量化值: {scalarized:.4f}")
    except Exception as e:
        print(f"  评估失败: {e}")

# 获取最优解并检查结构
best = evaluator.get_best_solution()

if best:
    print("\n" + "=" * 80)
    print("最优解结构:")
    print("=" * 80)
    print(f"Keys: {list(best.keys())}")
    
    print("\n模拟输出格式:")
    print("-" * 80)
    print(f"  [OK] Trial 1 completed")
    print(f"    标量化值: {best['scalarized']:.4f}")
    print(f"    运行时间: 73.4s")
    
    # 显示最优充电参数
    params = best['params']
    print(f"    充电参数:")
    print(f"      电流1 (I1):      {params['current1']:.4f} A")
    print(f"      充电次数 (t1):   {params['charging_number']:.0f} 次")
    print(f"      电流2 (I2):      {params['current2']:.4f} A")
    
    # 显示性能指标
    obj = best['objectives']
    print(f"    性能指标:")
    print(f"      充电时间:   {obj['time']:.2f} 步")
    print(f"      峰值温度:   {obj['temp']:.2f} K")
    print(f"      容量衰减:   {obj['aging']:.6f} %")
    
    print("\n✅ 输出格式正确！")
else:
    print("\n❌ 没有有效解")
