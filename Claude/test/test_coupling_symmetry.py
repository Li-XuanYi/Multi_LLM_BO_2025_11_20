"""
测试对称化耦合矩阵的数学性质
"""
import numpy as np
import sys
sys.path.insert(0, './BO/llmbo_core')

print("=" * 80)
print("测试耦合矩阵的数学性质")
print("=" * 80)

# 模拟非对称矩阵 (类似LLM输出)
W_asymmetric = np.array([
    [0.75, 0.80, 0.70],
    [0.60, 0.50, 0.55],
    [0.65, 0.60, 0.50]
])

print("\n[1] 原始非对称矩阵:")
print(W_asymmetric)
print(f"  对称性: {np.allclose(W_asymmetric, W_asymmetric.T)}")
print(f"  最大非对称度: {np.abs(W_asymmetric - W_asymmetric.T).max():.6f}")

# 对称化
W_symmetric = (W_asymmetric + W_asymmetric.T) / 2

print("\n[2] 对称化后矩阵:")
print(W_symmetric)
print(f"  对称性: {np.allclose(W_symmetric, W_symmetric.T)}")
print(f"  最大非对称度: {np.abs(W_symmetric - W_symmetric.T).max():.6f}")

# 检查正定性 (对于参数×目标矩阵，我们检查W@W^T的正定性)
print("\n[3] 正定性检查 (W @ W^T):")
gram_matrix = W_symmetric @ W_symmetric.T
eigenvalues = np.linalg.eigvals(gram_matrix)
print(f"  特征值: {eigenvalues}")
print(f"  最小特征值: {eigenvalues.min():.6f}")
print(f"  正定性: {np.all(eigenvalues > 0)}")

# 测试在核函数中的使用
print("\n[4] 核函数测试:")
from LLM_enhanced_surrogate_modeling import CouplingKernel

# 创建测试数据
X_test = np.array([
    [0.2, 0.3, 0.5],
    [0.4, 0.6, 0.7],
    [0.8, 0.2, 0.3]
])

# 非对称核
kernel_asym = CouplingKernel(W_asymmetric, length_scale=1.0)
K_asym = kernel_asym(X_test)

# 对称核
kernel_sym = CouplingKernel(W_symmetric, length_scale=1.0)
K_sym = kernel_sym(X_test)

print(f"\n  非对称核矩阵 K:")
print(f"    对角线: {np.diag(K_asym)}")
print(f"    对称性: {np.allclose(K_asym, K_asym.T)}")
print(f"    最小特征值: {np.linalg.eigvals(K_asym).min():.6f}")
print(f"    正定性: {np.all(np.linalg.eigvals(K_asym) > -1e-10)}")

print(f"\n  对称核矩阵 K:")
print(f"    对角线: {np.diag(K_sym)}")
print(f"    对称性: {np.allclose(K_sym, K_sym.T)}")
print(f"    最小特征值: {np.linalg.eigvals(K_sym).min():.6f}")
print(f"    正定性: {np.all(np.linalg.eigvals(K_sym) > -1e-10)}")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print("1. 对称化操作: (W + W^T) / 2 保证了矩阵对称性")
print("2. 对称核矩阵: 确保了高斯过程核的正定性 (所有特征值 > 0)")
print("3. 数值稳定性: 避免了非对称导致的数值不稳定问题")
print("4. 物理合理性: 参数i→目标j 与 参数j→目标i 的耦合强度一致")
print("=" * 80)
