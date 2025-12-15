"""
测试归一化修复 - 快速验证

测试内容：
1. 创建MultiObjectiveEvaluator
2. 评估几个策略
3. 检查f值范围
4. 确认归一化是否正确
"""

import sys
sys.path.append('d:/Users/aa133/Desktop/BO_Multi_11_12/BO/llmbo_core')

from multi_objective_evaluator import MultiObjectiveEvaluator

def test_normalization_fix():
    """测试归一化修复"""
    
    print("\n" + "=" * 80)
    print("测试归一化修复")
    print("=" * 80)
    
    # 1. 创建评价器
    evaluator = MultiObjectiveEvaluator(
        weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("测试策略 (与你的输出相同)")
    print("=" * 80)
    
    # 2. 测试策略（你的5个策略）
    test_strategies = [
        {'I1': 5.50, 't1': 10, 'I2': 2.50, 'name': '策略1'},
        {'I1': 4.00, 't1': 15, 'I2': 3.00, 'name': '策略2'},
        {'I1': 3.50, 't1': 20, 'I2': 3.50, 'name': '策略3'},
        {'I1': 6.00, 't1': 8, 'I2': 1.50, 'name': '策略4'},
        {'I1': 4.50, 't1': 7, 'I2': 4.00, 'name': '策略5'},
    ]
    
    results = []
    
    for i, strategy in enumerate(test_strategies, 1):
        print(f"\n{'='*80}")
        print(f"评估 {strategy['name']}: I1={strategy['I1']:.2f}A, t1={strategy['t1']}, I2={strategy['I2']:.2f}A")
        print(f"{'='*80}")
        
        try:
            f = evaluator.evaluate(
                current1=strategy['I1'],
                charging_number=strategy['t1'],
                current2=strategy['I2']
            )
            
            results.append({
                'strategy': strategy['name'],
                'params': strategy,
                'f': f,
                'success': True
            })
            
            print(f"\n✓ 评估成功: f = {f:.4f}")
            
        except Exception as e:
            print(f"\n✗ 评估失败: {e}")
            results.append({
                'strategy': strategy['name'],
                'params': strategy,
                'f': None,
                'success': False,
                'error': str(e)
            })
    
    # 3. 分析结果
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        f_values = [r['f'] for r in successful_results]
        
        print(f"\n成功评估: {len(successful_results)} / {len(results)}")
        print(f"\nf 值范围:")
        print(f"  最小: {min(f_values):.4f}")
        print(f"  最大: {max(f_values):.4f}")
        print(f"  平均: {sum(f_values)/len(f_values):.4f}")
        
        print(f"\n详细结果:")
        for r in results:
            if r['success']:
                p = r['params']
                print(f"  {r['strategy']}: I1={p['I1']:.2f}, t1={p['t1']}, I2={p['I2']:.2f} → f={r['f']:.4f}")
            else:
                print(f"  {r['strategy']}: 失败 - {r['error']}")
        
        # 检查是否修复
        print("\n" + "=" * 80)
        print("修复验证")
        print("=" * 80)
        
        if max(f_values) <= 1.5:
            print("\n✓ 修复成功！")
            print("  f 值范围在预期内 [0, 1.5]")
        elif max(f_values) <= 2.0:
            print("\n⚠️ 部分修复")
            print("  f 值范围在 [0, 2.0]，比之前好但仍可能有软约束惩罚")
        else:
            print("\n✗ 仍需调整")
            print(f"  最大 f 值 = {max(f_values):.4f} > 2.0")
            print("  可能原因:")
            print("  1. 仍有无效点")
            print("  2. 软约束惩罚过大")
            print("  3. 归一化边界仍有问题")
        
        # 理论分析
        print("\n理论预期:")
        print("  - 基础标量化: [0, 1.05]（当所有norm_i ∈ [0,1]）")
        print("  - 软约束惩罚: [0, 0.5+]（取决于温度/老化超限程度）")
        print("  - 无效点惩罚: +0.5（修复后）")
        print("  → 总计: f ∈ [0, 2.0+]")
        print("  → 大部分有效点应在 [0.2, 0.8] 范围内")
        
    else:
        print("\n✗ 所有评估都失败了")
        print("  请检查 SPM 仿真是否正常")
    
    # 4. 检查归一化历史
    print("\n" + "=" * 80)
    print("归一化历史检查")
    print("=" * 80)
    
    normalized_history = evaluator.get_normalized_history()
    
    if normalized_history:
        print(f"\n归一化历史记录数: {len(normalized_history)}")
        
        # 检查归一化值范围
        for key in ['time', 'temp', 'aging']:
            norm_values = [r['normalized'][key] for r in normalized_history if r['valid']]
            if norm_values:
                min_norm = min(norm_values)
                max_norm = max(norm_values)
                print(f"\n{key} 的归一化值范围: [{min_norm:.4f}, {max_norm:.4f}]", end="")
                if min_norm >= 0 and max_norm <= 1:
                    print(" ✓")
                else:
                    print(" ✗ 超出 [0, 1] 范围!")


if __name__ == "__main__":
    test_normalization_fix()
