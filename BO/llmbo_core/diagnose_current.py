"""
归一化诊断工具 v2 - 从当前运行的数据中分析

基于你提供的输出：
  策略 1/5: I1=5.50A, t1=10, I2=2.50A → 标量化=4.4363
  策略 2/5: I1=4.00A, t1=15, I2=3.00A → 标量化=2.5190
  策略 3/5: I1=3.50A, t1=20, I2=3.50A → 标量化=2.5244
  策略 4/5: I1=6.00A, t1=8, I2=1.50A → 标量化=4.4363
  策略 5/5: I1=4.50A, t1=7, I2=4.00A → 标量化=2.1941
"""

def analyze_current_results():
    """分析当前的运行结果"""
    
    print("\n" + "=" * 80)
    print("归一化诊断 - 基于当前输出")
    print("=" * 80)
    
    # 你的数据
    strategies = [
        {'I1': 5.50, 't1': 10, 'I2': 2.50, 'f': 4.4363},
        {'I1': 4.00, 't1': 15, 'I2': 3.00, 'f': 2.5190},
        {'I1': 3.50, 't1': 20, 'I2': 3.50, 'f': 2.5244},
        {'I1': 6.00, 't1': 8, 'I2': 1.50, 'f': 4.4363},
        {'I1': 4.50, 't1': 7, 'I2': 4.00, 'f': 2.1941},
    ]
    
    print("\n【问题分析】")
    print("=" * 80)
    
    print("\n1. 标量化值范围:")
    f_values = [s['f'] for s in strategies]
    print(f"   最小: {min(f_values):.4f}")
    print(f"   最大: {max(f_values):.4f}")
    print(f"   平均: {sum(f_values)/len(f_values):.4f}")
    
    print("\n2. 问题诊断:")
    print(f"   ✗ f 值全部 > 2.0")
    print(f"   ✗ f 值范围 [2.19, 4.44]，远超预期的 [0, 1.5]")
    
    print("\n3. 可能原因:")
    
    # 分析是否是无效点
    invalid_penalty = 2.0
    base_max = 1.05  # 理论最大基础标量化
    
    high_f = [s for s in strategies if s['f'] > invalid_penalty]
    medium_f = [s for s in strategies if invalid_penalty < s['f'] <= invalid_penalty + 0.5]
    
    print(f"\n   原因1: 无效点惩罚 (invalid_penalty = 2.0)")
    if high_f:
        print(f"   → 发现 {len(high_f)} 个点的 f > 2.0")
        print(f"   → 这些点可能是仿真失败的无效点")
        print(f"   → 代码自动添加了 +2.0 惩罚")
        for s in high_f:
            print(f"      • I1={s['I1']:.2f}, t1={s['t1']}, I2={s['I2']:.2f} → f={s['f']:.4f}")
    
    print(f"\n   原因2: 软约束惩罚 (温度/老化超限)")
    print(f"   → 即使仿真成功，如果温度超过 312K 或老化过大")
    print(f"   → 会添加指数/平方惩罚")
    print(f"   → 惩罚范围: [0, 0.5+]")
    
    print(f"\n   原因3: 归一化边界问题")
    print(f"   → 如果归一化边界不合理")
    print(f"   → 会导致归一化值 > 1")
    print(f"   → 进而导致标量化值偏大")
    
    # 分析策略特征
    print("\n" + "=" * 80)
    print("【策略特征分析】")
    print("=" * 80)
    
    print("\n策略 1 & 4 (f ≈ 4.44): 高电流 + 短时间")
    print("  • 策略1: I1=5.50A, t1=10, I2=2.50A")
    print("  • 策略4: I1=6.00A, t1=8, I2=1.50A")
    print("  → 特点: 第一阶段高电流 (5.5-6A)")
    print("  → 问题: 可能导致温度过高或仿真失败")
    print("  → 推测: 这些点是无效点 (valid=False)")
    
    print("\n策略 2, 3, 5 (f ≈ 2.2-2.5): 中等电流")
    print("  • 策略2: I1=4.00A, t1=15, I2=3.00A")
    print("  • 策略3: I1=3.50A, t1=20, I2=3.50A")
    print("  • 策略5: I1=4.50A, t1=7, I2=4.00A")
    print("  → 特点: 更均衡的电流分配")
    print("  → 问题: f 仍然 > 2.0")
    print("  → 推测: 可能有软约束惩罚或归一化问题")
    
    # 解决方案
    print("\n" + "=" * 80)
    print("【解决方案】")
    print("=" * 80)
    
    print("\n方案1: 诊断为什么这些点是无效的")
    print("  → 运行充电仿真，检查返回的 valid 标记")
    print("  → 查看是否有温度爆炸、求解器失败等问题")
    print("  → 如果大部分点都无效，需要调整搜索空间")
    
    print("\n方案2: 检查归一化逻辑")
    print("  → 检查 running_bounds 是否正确更新")
    print("  → 确认归一化后的值是否在 [0, 1] 范围内")
    print("  → 如果不在，说明边界有问题")
    
    print("\n方案3: 调整惩罚系数")
    print("  → 如果所有点都有较大惩罚")
    print("  → 考虑减小 invalid_penalty (从 2.0 降到 1.0)")
    print("  → 或调整软约束的惩罚系数")
    
    print("\n方案4: 使用固定物理边界")
    print("  → 替代动态的 running_bounds")
    print("  → 使用固定的 physical_bounds")
    print("  → 确保归一化的一致性")
    
    # 代码修复建议
    print("\n" + "=" * 80)
    print("【代码修复建议】")
    print("=" * 80)
    
    print("\n修复1: 在 multi_objective_evaluator.py 中")
    print("  找到 get_normalized_history() 方法")
    print("  将归一化改为使用固定的 physical_bounds:")
    print("""
    # 修改前:
    for key in ['time', 'temp', 'aging']:
        denominator = self.running_bounds[key]['max'] - self.running_bounds[key]['min']
        val = (obj[key] - self.running_bounds[key]['min']) / denominator
    
    # 修改后:
    for key in ['time', 'temp', 'aging']:
        denominator = self.physical_bounds[key]['max'] - self.physical_bounds[key]['min']
        val = (obj[key] - self.physical_bounds[key]['min']) / denominator
    """)
    
    print("\n修复2: 减小无效点惩罚")
    print("  找到 __init__ 方法中的:")
    print("    self.invalid_penalty = 2.0")
    print("  改为:")
    print("    self.invalid_penalty = 1.0  # 或 0.5")
    
    print("\n修复3: 添加调试输出")
    print("  在 evaluate() 方法中添加:")
    print("""
    if self.verbose:
        print(f"  归一化: time={normalized['time']:.4f}, "
              f"temp={normalized['temp']:.4f}, "
              f"aging={normalized['aging']:.4f}")
        print(f"  基础标量化: {base_scalarized:.4f}")
        print(f"  惩罚: {penalty:.4f}")
        print(f"  最终标量化: {scalarized:.4f}")
    """)
    
    print("\n" + "=" * 80)
    print("【建议执行顺序】")
    print("=" * 80)
    print("\n1. 先运行诊断：添加调试输出，查看归一化值")
    print("2. 修复归一化：使用 physical_bounds 替代 running_bounds")
    print("3. 调整惩罚：如果需要，减小 invalid_penalty")
    print("4. 重新测试：运行优化，检查 f 值范围")
    print("\n预期结果：f ∈ [0, 1.5]，大部分有效点在 [0.2, 0.8]")
    

if __name__ == "__main__":
    analyze_current_results()
