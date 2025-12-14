#!/usr/bin/env python3
"""
快速测试 - 只运行1个trial验证输出格式
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llmbo_core import MultiObjectiveEvaluator
from Comparison.base_optimizer import OptimizerFactory

# 导入优化器以触发注册
try:
    from Comparison.traditional_bo import TraditionalBO
except ImportError:
    pass

print("\n" + "=" * 80)
print("快速测试 - 验证输出格式")
print("=" * 80)

# 只测试BO算法，1个trial，5次迭代
algorithm = 'BO'
n_trials = 1
n_iterations = 5
n_random_init = 3

pbounds = {
    'current1': (3.0, 6.0),
    'charging_number': (5, 25),
    'current2': (1.0, 4.0)
}

print(f"\n测试配置:")
print(f"  算法: {algorithm}")
print(f"  Trials: {n_trials}")
print(f"  Iterations: {n_iterations}")
print(f"  Random init: {n_random_init}")

# 创建评估器和优化器
evaluator = MultiObjectiveEvaluator(verbose=False)

optimizer = OptimizerFactory.create(
    name=algorithm,
    evaluator=evaluator,
    pbounds=pbounds,
    random_state=42,
    verbose=False
)

print(f"\n开始优化...")
import time
start_time = time.time()

result = optimizer.optimize(
    n_iterations=n_iterations,
    n_random_init=n_random_init
)

elapsed_time = time.time() - start_time

# 模拟comparison_runner的输出格式
print(f"\n{'=' * 80}")
print(f"Running algorithm: {algorithm}")
print(f"{'=' * 80}")

print(f"\n--- Trial 1/{n_trials} ---")
print(f"  [OK] Trial 1 completed")
print(f"    标量化值: {result['best_solution']['scalarized']:.4f}")
print(f"    运行时间: {elapsed_time:.1f}s")

# 显示最优充电参数
params = result['best_solution']['params']
print(f"    充电参数:")
print(f"      电流1 (I1):      {params['current1']:.4f} A")
print(f"      充电次数 (t1):   {params['charging_number']:.0f} 次")
print(f"      电流2 (I2):      {params['current2']:.4f} A")

# 显示性能指标
obj = result['best_solution']['objectives']
print(f"    性能指标:")
print(f"      充电时间:   {obj['time']:.2f} 步")
print(f"      峰值温度:   {obj['temp']:.2f} K")
print(f"      容量衰减:   {obj['aging']:.6f} %")

print(f"\n{'=' * 80}")
print("✅ 测试完成！输出格式符合预期")
print(f"{'=' * 80}")
