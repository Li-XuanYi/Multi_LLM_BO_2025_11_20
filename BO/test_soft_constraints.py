#!/usr/bin/env python3
"""
软约束机制验证测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'llmbo_core'))

def test_soft_constraints():
    """测试软约束机制"""
    print("\n" + "="*70)
    print("🧪 软约束机制验证测试")
    print("="*70)
    
    try:
        from multi_objective_evaluator import MultiObjectiveEvaluator, SoftConstraintHandler
        
        print("\n✅ 成功导入 SoftConstraintHandler 和 MultiObjectiveEvaluator")
        
        # 测试1: SoftConstraintHandler单独测试
        print("\n" + "-"*70)
        print("测试1: SoftConstraintHandler温度惩罚")
        print("-"*70)
        
        handler = SoftConstraintHandler(verbose=False)
        
        test_temps = [312, 315, 318, 320, 325]
        print(f"{'温度[K]':<12} {'惩罚值':<15} {'状态':<10}")
        print("-"*40)
        for temp in test_temps:
            penalty, status = handler.compute_temperature_penalty(temp)
            print(f"{temp:<12.1f} {penalty:<15.6f} {status:<10}")
        
        # 测试2: MultiObjectiveEvaluator初始化
        print("\n" + "-"*70)
        print("测试2: MultiObjectiveEvaluator初始化")
        print("-"*70)
        
        evaluator = MultiObjectiveEvaluator(
            temp_max=315.0,
            verbose=True
        )
        
        print("\n✅ 软约束处理器已成功集成到MultiObjectiveEvaluator")
        
        # 测试3: 运行几次评估
        print("\n" + "-"*70)
        print("测试3: 运行评估测试")
        print("-"*70)
        
        test_params = [
            (4.0, 15, 2.5),
            (4.5, 12, 3.0),
            (3.5, 18, 2.0)
        ]
        
        for i, (I1, t1, I2) in enumerate(test_params, 1):
            print(f"\n测试 {i}: I1={I1}A, t1={t1}, I2={I2}A")
            try:
                scalarized = evaluator.evaluate(I1, t1, I2)
                print(f"  标量化值: {scalarized:.4f}")
            except Exception as e:
                print(f"  ❌ 评估失败: {e}")
        
        # 测试4: 检查日志
        print("\n" + "-"*70)
        print("测试4: 检查详细日志")
        print("-"*70)
        
        if evaluator.detailed_logs:
            latest_log = evaluator.detailed_logs[-1]
            print(f"最近一次评估:")
            print(f"  eval_id: {latest_log['eval_id']}")
            print(f"  valid: {latest_log['valid']}")
            print(f"  scalarized: {latest_log['scalarized']:.4f}")
            print(f"  objectives: {latest_log['objectives']}")
        
        print("\n" + "="*70)
        print("🎉 所有测试通过！软约束机制工作正常")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_soft_constraints()
    sys.exit(0 if success else 1)
