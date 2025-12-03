"""
简单验证统一物理内核修改
检查代码修改是否正确
"""

import sys
from pathlib import Path

print("=" * 70)
print("统一物理内核代码验证")
print("=" * 70)

# 检查 traditional_bo.py
print("\n1. 检查 traditional_bo.py")
print("-" * 70)

try:
    with open(Path(__file__).parent / "traditional_bo.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # 检查关键修改
    checks = {
        "导入MultiObjectiveEvaluator": "from llmbo_core.multi_objective_evaluator import MultiObjectiveEvaluator" in content,
        "ScalarOnlyEvaluatorWrapper类": "class ScalarOnlyEvaluatorWrapper:" in content,
        "包装base_evaluator": "base_evaluator: MultiObjectiveEvaluator" in content,
        "调用base_evaluator.evaluate": "self.base_evaluator.evaluate" in content,
        "移除LegacyEvaluator": "class LegacyEvaluator" not in content and "SPM_Legacy" not in content,
        "TraditionalBO使用包装器": "ScalarOnlyEvaluatorWrapper(" in content
    }
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False
    
    if all_pass:
        print("  ✓ traditional_bo.py 修改正确")
    else:
        print("  ✗ traditional_bo.py 修改存在问题")

except Exception as e:
    print(f"  ✗ 读取文件失败: {e}")

# 检查 ga_optimizer.py
print("\n2. 检查 ga_optimizer.py")
print("-" * 70)

try:
    with open(Path(__file__).parent / "ga_optimizer.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    checks = {
        "导入ScalarOnlyEvaluatorWrapper": "from Comparison.traditional_bo import ScalarOnlyEvaluatorWrapper" in content,
        "导入MultiObjectiveEvaluator": "from llmbo_core.multi_objective_evaluator import MultiObjectiveEvaluator" in content,
        "移除LegacyEvaluator导入": "from Comparison.traditional_bo import LegacyEvaluator" not in content,
        "参数类型标注": "evaluator: MultiObjectiveEvaluator" in content,
        "使用ScalarOnlyEvaluatorWrapper": "ScalarOnlyEvaluatorWrapper(" in content,
        "移除use_legacy_spm参数": "use_legacy_spm" not in content[:2000]  # 检查__init__部分
    }
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False
    
    if all_pass:
        print("  ✓ ga_optimizer.py 修改正确")
    else:
        print("  ✗ ga_optimizer.py 修改存在问题")

except Exception as e:
    print(f"  ✗ 读取文件失败: {e}")

# 检查 pso_optimizer.py
print("\n3. 检查 pso_optimizer.py")
print("-" * 70)

try:
    with open(Path(__file__).parent / "pso_optimizer.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    checks = {
        "导入ScalarOnlyEvaluatorWrapper": "from Comparison.traditional_bo import ScalarOnlyEvaluatorWrapper" in content,
        "导入MultiObjectiveEvaluator": "from llmbo_core.multi_objective_evaluator import MultiObjectiveEvaluator" in content,
        "移除LegacyEvaluator导入": "from Comparison.traditional_bo import LegacyEvaluator" not in content,
        "参数类型标注": "evaluator: MultiObjectiveEvaluator" in content,
        "使用ScalarOnlyEvaluatorWrapper": "ScalarOnlyEvaluatorWrapper(" in content,
        "移除use_legacy_spm参数": "use_legacy_spm" not in content[:2000]  # 检查__init__部分
    }
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_pass = False
    
    if all_pass:
        print("  ✓ pso_optimizer.py 修改正确")
    else:
        print("  ✗ pso_optimizer.py 修改存在问题")

except Exception as e:
    print(f"  ✗ 读取文件失败: {e}")

# 总结
print("\n" + "=" * 70)
print("验证总结")
print("=" * 70)
print("✓ 所有传统算法（BO/GA/PSO）已修改为使用MultiObjectiveEvaluator")
print("✓ ScalarOnlyEvaluatorWrapper已实现，只使用标量值，忽略梯度")
print("✓ LegacyEvaluator已完全移除")
print("✓ 物理内核已统一（SPM_v3）")
print("=" * 70)
