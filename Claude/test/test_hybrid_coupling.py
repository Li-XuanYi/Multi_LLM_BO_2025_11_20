"""
测试混合耦合估计器在同步环境中的工作
"""
import sys
sys.path.insert(0, './BO/llmbo_core')

from coupling_estimator_hybrid import HybridCouplingEstimator
import numpy as np

print("=" * 70)
print("测试混合耦合估计器 - 同步环境")
print("=" * 70)

# 初始化估计器
estimator = HybridCouplingEstimator(
    result_dir='./results',
    llm_api_key='sk-BxuZpZM1iQTbA5Jl10A32e47C1C74057Bc9fF13aFf7c5fD7',  # 使用真实密钥测试LLM调用
    llm_base_url='https://api.nuwaapi.com/v1',
    llm_model='gpt-4o',
    verbose=True
)

print("\n[测试1] 数据驱动耦合估计")
print("-" * 70)
historical_data = estimator.load_historical_data(n_recent=3)
data_coupling = estimator.compute_data_driven_coupling(historical_data)
print("✅ 数据驱动耦合矩阵:")
print(data_coupling)

print("\n[测试2] LLM知识获取 (同步)")
print("-" * 70)
try:
    llm_coupling = estimator.get_llm_coupling_knowledge()
    print("✅ LLM耦合矩阵:")
    print(llm_coupling)
except Exception as e:
    print(f"❌ LLM调用失败: {e}")

print("\n[测试3] 混合耦合矩阵估计 (完整流程)")
print("-" * 70)
try:
    hybrid_coupling = estimator.estimate_coupling_matrix(
        n_historical_runs=3,
        data_weight=0.6,
        llm_weight=0.4
    )
    print("✅ 混合估计成功!")
except Exception as e:
    print(f"❌ 混合估计失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
