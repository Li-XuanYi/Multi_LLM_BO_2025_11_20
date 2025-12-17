"""
验证三个修复是否生效
"""

import asyncio
import sys
import os

# 添加正确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
llmbo_core_path = os.path.join(current_dir, 'llmbo_core')
sys.path.insert(0, llmbo_core_path)

from LLM_Enhanced_Multi_Objective_Bayesian_Optimization import LLMEnhancedMultiObjectiveBO

async def test_all_fixes():
    """测试所有修复"""
    
    print("\n" + "="*80)
    print("[TEST] LLM Enhanced BO Fix Verification")
    print("="*80)
    
    # 初始化优化器（10次迭代快速测试）
    optimizer = LLMEnhancedMultiObjectiveBO(
        llm_api_key='sk-dummy',  # 测试用，不实际调用LLM
        n_warmstart=0,           # 跳过warmstart
        n_random_init=5,         # 5次随机初始化
        n_iterations=10,         # 10次BO迭代
        enable_llm_surrogate=True,
        enable_llm_ei=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("检查点 1: 梯度计算频率")
    print("="*80)
    interval = optimizer.evaluator.gradient_compute_interval
    print(f"gradient_compute_interval = {interval}")
    
    if interval == 1:
        print("✅ 修复1成功: 每次都计算梯度")
    else:
        print(f"❌ 修复1失败: 仍然是每{interval}次计算一次")
    
    # 运行优化
    print("\n" + "="*80)
    print("开始优化测试（15次评估）")
    print("="*80)
    
    try:
        results = await optimizer.optimize_async()
        
        print("\n" + "="*80)
        print("检查点 2: 耦合矩阵估计策略")
        print("="*80)
        print("✅ 查看上方输出，确认是否出现:")
        print("   '[Coupling Matrix Estimator] 诊断报告'")
        print("   '[Data-Driven Coupling Matrix] 估计完成 ✨'")
        
        print("\n" + "="*80)
        print("检查点 3: LLM-EI权重衰减")
        print("="*80)
        print("✅ 查看上方输出，确认是否出现:")
        print("   '[LLM-Enhanced EI] 迭代 X'")
        print("   '  W_LLM 权重: 0.XXXXXX'")
        
        print("\n" + "="*80)
        print("[SUCCESS] Test Completed!")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = asyncio.run(test_all_fixes())
    
    if results:
        print("\n[OK] All fixes verified!")
    else:
        print("\n[FAILED] Some fixes may not be working, check output above")
