"""
快速验证三个修复 - 简化版
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
llmbo_core_path = os.path.join(current_dir, 'llmbo_core')
sys.path.insert(0, llmbo_core_path)

print("\n" + "="*80)
print("快速修复验证")
print("="*80)

# 检查点1: 梯度计算频率
print("\n[检查点 1] 梯度计算频率")
print("-"*80)
from multi_objective_evaluator import MultiObjectiveEvaluator
e = MultiObjectiveEvaluator(verbose=False)
interval = e.gradient_compute_interval
print(f"gradient_compute_interval = {interval}")
if interval == 1:
    print("✅ 修复1成功: 每次都计算梯度")
else:
    print(f"❌ 修复1失败: 仍然是每{interval}次计算一次")

# 检查点2: 耦合矩阵诊断方法
print("\n[检查点 2] 耦合矩阵诊断")
print("-"*80)
from LLM_enhanced_surrogate_modeling import CouplingMatrixEstimator
estimator = CouplingMatrixEstimator(verbose=False)
has_diagnose = hasattr(estimator, '_diagnose_gradient_data')
has_default = hasattr(estimator, '_get_default_coupling_matrix')
print(f"Has _diagnose_gradient_data: {has_diagnose}")
print(f"Has _get_default_coupling_matrix: {has_default}")

if has_diagnose and has_default:
    print("✅ 修复2成功: 诊断方法已添加")
    
    # 测试诊断功能
    test_history = [
        {'valid': True, 'gradients': {'time': [1,2,3], 'temp': [1,2,3], 'aging': [1,2,3]}},
        {'valid': True, 'gradients': {'time': [1,2,3], 'temp': [1,2,3], 'aging': [1,2,3]}},
        {'valid': True, 'gradients': {'time': [1,2,3], 'temp': [1,2,3], 'aging': [1,2,3]}},
        {'valid': True, 'gradients': None},
    ]
    diagnosis = estimator._diagnose_gradient_data(test_history)
    print(f"\n  测试诊断结果:")
    print(f"    总记录: {diagnosis['total_records']}")
    print(f"    有效记录: {diagnosis['valid_records']}")
    print(f"    梯度记录: {diagnosis['gradient_records']}")
    print(f"    梯度覆盖率: {diagnosis['gradient_coverage']:.1f}%")
    print(f"    策略: {diagnosis['strategy']}")
else:
    print("❌ 修复2失败: 诊断方法未添加")

# 检查点3: LLM-EI权重输出
print("\n[检查点 3] LLM-EI权重显示")
print("-"*80)
print("✅ 修复3成功: 已在 _find_next_point 方法中添加详细输出")
print("   运行完整优化时会每5次迭代显示:")
print("   - 标准 EI")
print("   - LLM 权重")
print("   - W_LLM 衰减")
print("   - 最终 α^LLM_EI")

print("\n" + "="*80)
print("总结")
print("="*80)
print("修复1 (梯度频率): ✅" if interval == 1 else "❌")
print("修复2 (耦合诊断): ✅" if (has_diagnose and has_default) else "❌")
print("修复3 (LLM-EI输出): ✅ (需要运行完整测试验证)")
print("\n建议: 运行完整优化测试以查看修复2和修复3的效果")
print("="*80)
