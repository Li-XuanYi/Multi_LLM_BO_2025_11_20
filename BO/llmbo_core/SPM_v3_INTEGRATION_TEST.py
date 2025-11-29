"""
SPM v3.0 集成测试
验证 SPM_v3 + PybammSensitivity_v3 + MultiObjectiveEvaluator 整合
"""

import numpy as np
import time

def test_v3_integration():
    """测试 v3.0 完整集成"""
    print("\n" + "=" * 70)
    print("SPM v3.0 Integration Test")
    print("测试 SPM v3 + 多目标评价器 + 惩罚梯度")
    print("=" * 70)
    
    # 导入模块
    from multi_objective_evaluator import MultiObjectiveEvaluator
    
    print("\n[1] 初始化评价器（应使用 SPM_v3）")
    evaluator = MultiObjectiveEvaluator(verbose=True)
    
    print("\n[2] 测试可行点")
    params1 = {'current1': 4.5, 'charging_number': 18, 'current2': 3.5}
    start = time.time()
    result1 = evaluator._run_charging_simulation(**params1)
    t1 = time.time() - start
    
    print(f"\n  参数: I1={params1['current1']}, t1={params1['charging_number']}, I2={params1['current2']}")
    print(f"  Valid: {result1['valid']}")
    print(f"  目标值:")
    print(f"    Time: {result1['time']:.2f} steps")
    print(f"    Temp: {result1['temp']:.2f} K")
    print(f"    Aging: {result1['aging']:.4f}%")
    print(f"  评估时间: {t1:.2f}s")
    
    print("\n[3] 测试边界点（应违反约束）")
    params2 = {'current1': 5.5, 'charging_number': 20, 'current2': 5.0}
    start = time.time()
    result2 = evaluator._run_charging_simulation(**params2)
    t2 = time.time() - start
    
    print(f"\n  参数: I1={params2['current1']}, t1={params2['charging_number']}, I2={params2['current2']}")
    print(f"  Valid: {result2['valid']}")
    print(f"  目标值:")
    print(f"    Time: {result2['time']:.2f} steps")
    print(f"    Temp: {result2['temp']:.2f} K")
    print(f"  评估时间: {t2:.2f}s")
    
    if not result2['valid']:
        print(f"  ✓ 检测到约束违反（符合预期）")
    
    print("\n[4] 测试梯度计算（evaluator的间隔触发）")
    # 连续评估5次以触发梯度计算
    for i in range(5):
        _ = evaluator._run_charging_simulation(
            current1=4.0 + i * 0.2,
            charging_number=16 + i,
            current2=3.0
        )
    
    print(f"  已完成 {evaluator.eval_count} 次评估")
    if evaluator.eval_count >= evaluator.gradient_compute_interval:
        print(f"  ✓ 梯度计算间隔达到，SPM_v3 的灵敏度功能应已触发")
    
    print("\n[5] 验证 v3.0 特性")
    checks = []
    
    # 检查1: evaluator使用的是SPM_v3
    spm_type = type(evaluator.spm_for_gradients).__name__
    checks.append(("SPM v3 loaded", spm_type == "SPM_Sensitivity"))
    
    # 检查2: 惩罚梯度是否启用
    penalty_enabled = getattr(evaluator.spm_for_gradients, 'enable_penalty_gradients', False)
    checks.append(("Penalty gradients enabled", penalty_enabled))
    
    # 检查3: 模式是否正确
    mode = getattr(evaluator.spm_for_gradients, 'mode', 'unknown')
    checks.append(("Finite difference mode", mode == 'finite_difference'))
    
    print()
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 集成测试通过！SPM v3.0 已成功整合")
    else:
        print("✗ 部分检查未通过")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = test_v3_integration()
    exit(0 if success else 1)
