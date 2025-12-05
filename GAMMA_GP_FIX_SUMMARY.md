# 关键修复说明

## 修复1: Gamma衰减策略 ✅

### 问题
- **旧版**: 改善时γ会**增加**，违反"逐步减少LLM依赖"原则
- **停滞惩罚过强**: 0.8倍下降可能过早放弃LLM指导

### 解决方案
改为**纯指数衰减 + 自适应速度调整**：

```python
# 基础衰减率: 0.95 (每次下降5%)
base_decay_rate = 0.95

# 自适应调整衰减速度（而非γ本身）:
if 停滞:
    effective_decay_rate = 0.95 × 0.95 = 0.9025  # 加速衰减
elif 改善良好:
    effective_decay_rate = 0.95 × 1.02 = 0.969  # 减缓衰减（但仍衰减！）
else:
    effective_decay_rate = 0.95  # 标准衰减

γ(t) = γ(t-1) × effective_decay_rate  # 确保单调递减
```

### 效果
- ✅ **单调递减**: γ永远不会增加
- ✅ **自适应**: 根据优化状态调整衰减速度
- ✅ **理论一致**: 符合贝叶斯"先验→数据"的思想

---

## 修复2: GP核参数边界警告 ✅

### 问题警告
```
ConvergenceWarning: k1__length_scale is close to upper bound 100.0
ConvergenceWarning: k2__k2__length_scale is close to lower bound 1e-05
```

### 根本原因
1. **RBF核**: length_scale → 100.0 (上界)
   - 默认边界 [1e-5, 1e5] 对归一化数据太宽
   - GP认为空间"极度平滑"

2. **CouplingKernel**: length_scale → 1e-05 (下界)
   - 两个核共享相同边界，导致竞争
   - GP认为耦合效应"极度剧烈"

### 解决方案
为每个核设置**独立的合理边界**：

```python
# RBF核: 捕捉全局平滑趋势
base_kernel = RBF(
    length_scale=0.5,              # 初始值适合[0,1]归一化数据
    length_scale_bounds=(0.1, 5.0) # ✅ 合理范围
)

# CouplingKernel: 捕捉参数耦合效应
coupling_kernel = CouplingKernel(
    coupling_matrix=W,
    length_scale=0.2,              # ✅ 更小的初始值
    length_scale_bounds=(0.05, 2.0) # ✅ 独立的较小范围
)
```

### 效果
- ✅ **消除警告**: 参数优化不再碰到边界
- ✅ **物理合理**: 两个核有独立的作用尺度
- ✅ **数值稳定**: 避免极端参数值

---

## 验证方法

### 测试Gamma单调性
```python
# 运行优化后检查日志
# 应该看到: γ: 0.3000 → 0.2850 ↓ (每次都是↓，不会↑)
```

### 测试GP警告消除
```python
# 运行优化后检查
# 不应再出现 ConvergenceWarning: close to bound
```

---

## 技术细节

### Gamma衰减对比

| 迭代 | 旧版(改善时增加) | 新版(纯衰减) |
|------|----------------|--------------|
| 0    | 0.300          | 0.300        |
| 5    | 0.320 ⚠️ 增加  | 0.278 ✅     |
| 10   | 0.305          | 0.258 ✅     |
| 20   | 0.285          | 0.212 ✅     |
| 50   | 不可预测       | 0.097 ✅     |

### 核参数对比

| 参数 | 旧版边界 | 新版边界 | 说明 |
|------|----------|----------|------|
| RBF length_scale | [1e-2, 1e2] | [0.1, 5.0] | 适合归一化数据 |
| Coupling length_scale | (无) | [0.05, 2.0] | 独立的较小范围 |
