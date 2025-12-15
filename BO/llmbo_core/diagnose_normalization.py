"""
归一化诊断工具 - 分析为什么 f 值不为 1

检查内容：
1. 原始目标值范围
2. 归一化后的值范围
3. 标量化公式的每一步
4. 惩罚项的贡献
5. 有效性标记
"""

import json
import numpy as np
from pathlib import Path

def diagnose_normalization():
    """诊断归一化问题"""
    
    # 查找最新的结果文件
    result_dir = Path('d:/Users/aa133/Desktop/BO_Multi_11_12/comparison_results')
    if not result_dir.exists():
        result_dir = Path('./comparison_results')
    
    if not result_dir.exists():
        print("❌ 找不到结果目录")
        return
    
    # 找到最新的详细结果文件
    json_files = sorted(result_dir.glob('detailed_results_*.json'))
    if not json_files:
        print("❌ 找不到结果文件")
        return
    
    latest_file = json_files[-1]
    print(f"\n分析文件: {latest_file.name}")
    print("=" * 80)
    
    # 加载数据
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    history = data.get('history', [])
    if not history:
        print("❌ 历史记录为空")
        return
    
    print(f"\n总评估次数: {len(history)}")
    
    # 分析每个点
    print("\n" + "=" * 80)
    print("详细分析前10个点:")
    print("=" * 80)
    
    for i, record in enumerate(history[:10]):
        print(f"\n【点 {i+1}】")
        
        # 参数
        params = record.get('params', {})
        print(f"  参数: I1={params.get('current1', 0):.2f}A, "
              f"t1={params.get('charging_number', 0)}, "
              f"I2={params.get('current2', 0):.2f}A")
        
        # 原始目标值
        obj = record.get('objectives', {})
        print(f"  原始目标:")
        print(f"    时间: {obj.get('time', 0):.2f} 步")
        print(f"    温度: {obj.get('temp', 0):.2f} K")
        print(f"    老化: {obj.get('aging', 0):.4f} (对数值)")
        
        # 归一化值
        norm = record.get('normalized', {})
        print(f"  归一化值:")
        print(f"    时间: {norm.get('time', 0):.4f}")
        print(f"    温度: {norm.get('temp', 0):.4f}")
        print(f"    老化: {norm.get('aging', 0):.4f}")
        
        # 计算标量化（假设权重）
        weights = {'time': 0.4, 'temp': 0.35, 'aging': 0.25}
        weighted = {k: weights[k] * norm.get(k, 0) for k in weights}
        max_weighted = max(weighted.values())
        sum_weighted = sum(weighted.values())
        base_scalarized = max_weighted + 0.05 * sum_weighted
        
        print(f"  加权偏差:")
        for k, v in weighted.items():
            print(f"    w_{k} * norm_{k} = {weights[k]:.2f} * {norm.get(k, 0):.4f} = {v:.4f}")
        print(f"  切比雪夫: max={max_weighted:.4f}, sum={sum_weighted:.4f}")
        print(f"  基础标量化: {base_scalarized:.4f}")
        
        # 实际标量化值
        actual_scalarized = record.get('scalarized', 0)
        print(f"  实际标量化: {actual_scalarized:.4f}")
        
        # 差异（惩罚项）
        penalty = actual_scalarized - base_scalarized
        print(f"  惩罚项: {penalty:.4f}")
        
        # 有效性
        is_valid = record.get('valid', True)
        print(f"  有效性: {'✓ 有效' if is_valid else '❌ 无效 (+2.0 惩罚)'}")
        
        if penalty > 0.1:
            print(f"  ⚠️ 警告: 存在较大惩罚 ({penalty:.4f})")
            if penalty >= 2.0:
                print(f"     → 可能是无效点 (invalid_penalty = 2.0)")
            else:
                print(f"     → 可能是软约束惩罚 (温度/老化超限)")
    
    # 统计分析
    print("\n" + "=" * 80)
    print("统计分析:")
    print("=" * 80)
    
    # 有效点数量
    valid_count = sum(1 for r in history if r.get('valid', True))
    invalid_count = len(history) - valid_count
    print(f"\n有效点: {valid_count} / {len(history)} ({valid_count/len(history)*100:.1f}%)")
    print(f"无效点: {invalid_count} / {len(history)} ({invalid_count/len(history)*100:.1f}%)")
    
    # 标量化值分布
    scalarized_values = [r.get('scalarized', 0) for r in history]
    print(f"\n标量化值分布:")
    print(f"  最小: {min(scalarized_values):.4f}")
    print(f"  最大: {max(scalarized_values):.4f}")
    print(f"  平均: {np.mean(scalarized_values):.4f}")
    print(f"  中位数: {np.median(scalarized_values):.4f}")
    
    # 分析 f > 2 的点
    high_f_points = [r for r in history if r.get('scalarized', 0) > 2.0]
    if high_f_points:
        print(f"\n⚠️ 发现 {len(high_f_points)} 个 f > 2.0 的点:")
        for r in high_f_points[:5]:
            params = r.get('params', {})
            is_valid = r.get('valid', True)
            f_val = r.get('scalarized', 0)
            print(f"  I1={params.get('current1', 0):.2f}, "
                  f"t1={params.get('charging_number', 0)}, "
                  f"I2={params.get('current2', 0):.2f} "
                  f"→ f={f_val:.4f}, valid={is_valid}")
    
    # 归一化边界
    print("\n" + "=" * 80)
    print("归一化边界检查:")
    print("=" * 80)
    
    # 从有效点计算实际范围
    valid_records = [r for r in history if r.get('valid', True)]
    if valid_records:
        objectives_arrays = {
            'time': [r['objectives']['time'] for r in valid_records],
            'temp': [r['objectives']['temp'] for r in valid_records],
            'aging': [r['objectives']['aging'] for r in valid_records]
        }
        
        print("\n有效点的目标值范围:")
        for key, values in objectives_arrays.items():
            print(f"  {key}: [{min(values):.2f}, {max(values):.2f}]")
        
        # 检查归一化后是否在[0,1]
        print("\n有效点的归一化值范围:")
        for key in ['time', 'temp', 'aging']:
            norm_values = [r['normalized'][key] for r in valid_records]
            min_norm = min(norm_values)
            max_norm = max(norm_values)
            print(f"  {key}: [{min_norm:.4f}, {max_norm:.4f}]", end="")
            if min_norm < 0 or max_norm > 1:
                print(f" ❌ 超出 [0, 1] 范围!")
            else:
                print(f" ✓")
    
    # 建议
    print("\n" + "=" * 80)
    print("诊断结论:")
    print("=" * 80)
    
    if invalid_count > 0:
        print(f"\n1. ❌ 发现 {invalid_count} 个无效点 (valid=False)")
        print(f"   → 这些点自动添加了 +2.0 惩罚")
        print(f"   → 导致 f > 2.0")
        print(f"   → 建议: 检查充电仿真为什么失败")
    
    if high_f_points:
        high_f_valid = [r for r in high_f_points if r.get('valid', True)]
        if high_f_valid:
            print(f"\n2. ⚠️ 发现 {len(high_f_valid)} 个有效但 f > 2.0 的点")
            print(f"   → 可能是软约束惩罚过大")
            print(f"   → 建议: 检查温度和老化是否严重超限")
    
    # 检查归一化是否合理
    norm_issues = []
    for r in valid_records:
        for key in ['time', 'temp', 'aging']:
            if r['normalized'][key] < -0.01 or r['normalized'][key] > 1.01:
                norm_issues.append(r)
                break
    
    if norm_issues:
        print(f"\n3. ❌ 发现 {len(norm_issues)} 个归一化异常的点")
        print(f"   → 归一化值超出 [0, 1] 范围")
        print(f"   → 建议: 检查 running_bounds 的更新逻辑")
    
    # 最终建议
    print("\n" + "=" * 80)
    print("修复建议:")
    print("=" * 80)
    
    if invalid_count == 0 and len(norm_issues) == 0:
        print("\n✓ 归一化工作正常！")
        print("  f 值范围取决于:")
        print("  - 基础标量化: [0, 1.05] (当所有norm_i ∈ [0,1])")
        print("  - 软约束惩罚: [0, 0.5+] (取决于违规程度)")
        print("  - 无效点惩罚: +2.0")
        print("\n  因此 f ∈ [0, 3.5+] 是正常的")
    else:
        print("\n需要修复:")
        print("1. 减少无效点数量")
        print("   → 调整参数搜索范围")
        print("   → 检查SPM仿真的稳定性")
        print("\n2. 优化软约束惩罚")
        print("   → 如果大部分点都有惩罚，考虑放宽约束")
        print("   → 或减小惩罚系数")
        print("\n3. 确保归一化正确")
        print("   → 使用固定的物理边界")
        print("   → 或确保 running_bounds 正确更新")


if __name__ == "__main__":
    diagnose_normalization()
