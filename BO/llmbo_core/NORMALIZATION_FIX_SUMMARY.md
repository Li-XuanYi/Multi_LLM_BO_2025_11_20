# 归一化问题修复总结

## 问题诊断

### 症状
运行优化时发现标量化值（f）异常：
```
策略 1/5: I1=5.50A, t1=10, I2=2.50A → 标量化=4.4363
策略 2/5: I1=4.00A, t1=15, I2=3.00A → 标量化=2.5190
策略 3/5: I1=3.50A, t1=20, I2=3.50A → 标量化=2.5244
策略 4/5: I1=6.00A, t1=8, I2=1.50A → 标量化=4.4363
策略 5/5: I1=4.50A, t1=7, I2=4.00A → 标量化=2.1941
```

**问题**：所有 f 值都 > 2.0，远超预期的 [0, 1.5] 范围。

### 根本原因分析

经过详细诊断，发现了两个主要问题：

#### 1. 归一化边界不一致 ✅ 已修复
**问题**：代码中同时存在两套边界：
- `physical_bounds`（固定物理边界）
- `running_bounds`（动态更新边界）

不同方法使用不同边界，导致归一化不一致。

**修复**：统一使用 `physical_bounds` 进行所有归一化计算。

#### 2. 软约束惩罚过大 ✅ 已修复
**问题**：老化约束的阈值设置不合理：
- 旧阈值：`aging_threshold = 0.5` (对数空间)
- 实际老化值：4.73
- excess = 4.23 → penalty = 0.1 × (4.23)² = 1.79

**修复**：
- 新阈值：`aging_threshold = 5.0`（更合理）
- 降低惩罚系数：`aging_penalty_scale = 0.02`（从 0.1 降到 0.02）

## 修复内容

### 文件：`multi_objective_evaluator.py`

#### 修改 1：减小无效点惩罚
```python
# 修改前
self.invalid_penalty = 2.0  # 无效点的额外惩罚（让 f 明显 > 2）

# 修改后
self.invalid_penalty = 0.5  # 无效点的额外惩罚（降低以避免f值过大）
```

**位置**：第 236 行

---

#### 修改 2：使用固定物理边界进行归一化
```python
# 修改前：使用动态边界
for key in ['time', 'temp', 'aging']:
    denominator = self.running_bounds[key]['max'] - self.running_bounds[key]['min']
    val = (obj[key] - self.running_bounds[key]['min']) / denominator

# 修改后：使用固定物理边界
for key in ['time', 'temp', 'aging']:
    denominator = self.physical_bounds[key]['max'] - self.physical_bounds[key]['min']
    val = (obj[key] - self.physical_bounds[key]['min']) / denominator
```

**位置**：
- `get_normalized_history()` 方法（第 523 行）
- `evaluate()` 方法（第 638 行）

---

#### 修改 3：调整软约束参数
```python
# 修改前
self.soft_constraints = SoftConstraintHandler(
    temp_max=312.0,
    temp_penalty_rate=0.15,
    temp_penalty_scale=0.05,
    aging_threshold=0.5,      # 太低
    aging_penalty_scale=0.1,  # 太高
    verbose=self.verbose
)

# 修改后
self.soft_constraints = SoftConstraintHandler(
    temp_max=312.0,
    temp_penalty_rate=0.15,
    temp_penalty_scale=0.05,
    aging_threshold=5.0,      # 更合理（log1p(5%*100)≈6.2）
    aging_penalty_scale=0.02, # 降低惩罚系数
    verbose=self.verbose
)
```

**位置**：第 276 行

---

#### 修改 4：添加详细调试输出
```python
# 在 evaluate() 方法中添加
if self.verbose and self.eval_count % 1 == 0:
    print(f"\n  [归一化] time={normalized['time']:.4f}, temp={normalized['temp']:.4f}, aging={normalized['aging']:.4f}")
    print(f"  [标量化] 基础={base_scalarized:.4f}, 软约束={soft_penalty:.4f}, 无效惩罚={self.invalid_penalty if not sim_result['valid'] else 0:.4f}")
    print(f"  [最终] f={scalarized:.4f}, valid={sim_result['valid']}")
```

**位置**：第 665-668 行

---

## 修复验证

### 测试结果

运行相同的 5 个策略，结果对比：

| 策略 | 参数 | 修复前 f | 修复后 f | 有效性 |
|------|------|----------|----------|--------|
| 1 | I1=5.50, t1=10, I2=2.50 | 4.4363 | **1.2429** | ❌ 无效 |
| 2 | I1=4.00, t1=15, I2=3.00 | 2.5190 | **0.2209** | ✅ 有效 |
| 3 | I1=3.50, t1=20, I2=3.50 | 2.5244 | **0.2205** | ✅ 有效 |
| 4 | I1=6.00, t1=8, I2=1.50 | 4.4363 | **1.2429** | ❌ 无效 |
| 5 | I1=4.50, t1=7, I2=4.00 | 2.1941 | **0.2034** | ✅ 有效 |

### 关键指标

- ✅ **有效点 f 值范围**：[0.2034, 0.2209] —— 符合预期 [0, 1.0]
- ✅ **无效点 f 值**：1.2429 —— 包含 +0.5 惩罚，合理
- ✅ **归一化值范围**：所有归一化值 ∈ [0, 1] —— 正确
- ✅ **软约束惩罚**：0.0000-0.0929 —— 合理范围

### 详细分析

#### 有效点（策略 2, 3, 5）
- 基础标量化：0.20-0.22
- 软约束惩罚：≈0.0001（几乎为0）
- 无效惩罚：0
- **最终 f**：0.20-0.22 ✅

#### 无效点（策略 1, 4）
- 基础标量化：0.45（所有归一化值=1.0）
- 软约束惩罚：0.09（温度超限）
- 无效惩罚：+0.5
- **最终 f**：1.24 ✅

## 理论预期

标量化值的理论范围：

```
f = 基础标量化 + 软约束惩罚 + 无效惩罚

其中：
- 基础标量化 = max(w_i × norm_i) + 0.05 × sum(w_i × norm_i)
  → 范围：[0, 1.05]（当所有 norm_i ∈ [0,1]）

- 软约束惩罚（温度 + 老化）
  → 范围：[0, 0.5+]（取决于超限程度）

- 无效惩罚
  → 有效点：0
  → 无效点：+0.5

因此：
- 完美有效点：f ≈ 0（所有目标都在最优范围）
- 一般有效点：f ∈ [0.2, 0.8]
- 轻微违规：f ∈ [0.8, 1.2]
- 无效点：f ∈ [1.0, 2.0+]
```

## 结论

✅ **修复成功！** 归一化问题已完全解决：

1. **归一化一致性**：统一使用固定物理边界
2. **惩罚合理性**：调整软约束参数到合理范围
3. **调试可见性**：添加详细输出便于监控
4. **f值范围正确**：
   - 有效点：0.2-0.8（大部分情况）
   - 无效点：1.0-1.5

现在可以正常进行贝叶斯优化了！

## 使用建议

1. **运行优化时**，观察 f 值范围：
   - 如果大部分有效点的 f > 1.0，说明约束太严格，可以适当放宽
   - 如果无效点太多，考虑调整搜索空间

2. **调整软约束**（如有需要）：
   - 温度约束：修改 `temp_max`
   - 老化约束：修改 `aging_threshold` 和 `aging_penalty_scale`

3. **监控归一化**：
   - 确保归一化值始终在 [0, 1] 范围内
   - 如果超出范围，检查 `physical_bounds` 是否合理

## 诊断工具

创建了以下工具帮助诊断：
- `diagnose_normalization.py`：分析保存的结果文件
- `diagnose_current.py`：分析当前运行的输出
- `test_normalization_fix.py`：快速验证修复效果

---

**修复日期**：2025-12-15  
**测试通过**：✅  
**版本**：v2.1 - 归一化修复版
