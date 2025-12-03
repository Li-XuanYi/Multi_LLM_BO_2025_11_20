# 统一物理内核修改完成

## 修改日期
2025-12-02

## 修改内容

### 1. 核心修改：统一物理内核

所有传统算法（Traditional BO、GA、PSO）现在都使用 **MultiObjectiveEvaluator（SPM_v3）** 作为统一的物理内核，确保公平对比。

### 2. 主要文件修改

#### 2.1 traditional_bo.py
- ✅ 移除 `LegacyEvaluator` 类（旧版SPM v2.1）
- ✅ 新增 `ScalarOnlyEvaluatorWrapper` 类
  - 包装 `MultiObjectiveEvaluator`
  - 只使用标量值，忽略梯度信息
  - 代理所有必要的方法（`get_best_solution`、`export_database`等）
- ✅ 更新 `TraditionalBO` 类
  - 移除 `use_legacy_spm` 参数
  - 自动包装 `MultiObjectiveEvaluator` 为 `ScalarOnlyEvaluatorWrapper`
  - 更新初始化日志显示"SPM v3.0 (统一物理内核)"

#### 2.2 ga_optimizer.py
- ✅ 移除 `LegacyEvaluator` 导入
- ✅ 导入 `ScalarOnlyEvaluatorWrapper` 和 `MultiObjectiveEvaluator`
- ✅ 更新 `GeneticAlgorithm` 类
  - 移除 `use_legacy_spm` 参数
  - 参数类型标注：`evaluator: MultiObjectiveEvaluator`
  - 自动包装为 `ScalarOnlyEvaluatorWrapper`
  - 更新日志显示"使用SPM_v3（统一物理内核）"

#### 2.3 pso_optimizer.py
- ✅ 移除 `LegacyEvaluator` 导入
- ✅ 导入 `ScalarOnlyEvaluatorWrapper` 和 `MultiObjectiveEvaluator`
- ✅ 更新 `ParticleSwarmOptimization` 类
  - 移除 `use_legacy_spm` 参数
  - 参数类型标注：`evaluator: MultiObjectiveEvaluator`
  - 自动包装为 `ScalarOnlyEvaluatorWrapper`
  - 更新日志显示"使用SPM_v3（统一物理内核）"

#### 2.4 comparison_runner.py
- ✅ 更新文件头说明为"统一物理内核版"
- ✅ 已经正确使用 `MultiObjectiveEvaluator`（无需修改）

### 3. 设计模式

使用 **Adapter/Wrapper 模式** 实现统一物理内核：

```
MultiObjectiveEvaluator (SPM_v3)
        ↓
ScalarOnlyEvaluatorWrapper (适配层)
        ↓ (只传递标量值)
Traditional Algorithms (BO/GA/PSO)
```

### 4. 关键特性

#### 4.1 ScalarOnlyEvaluatorWrapper
- **包装对象**：`MultiObjectiveEvaluator` 实例
- **核心方法**：
  - `evaluate()`: 调用 `base_evaluator.evaluate()`，直接返回标量值
  - `get_best_solution()`: 代理到基础评估器
  - `export_database()`: 代理到基础评估器
  - `get_statistics()`: 代理到基础评估器
  - `weights` 属性: 代理到基础评估器
- **特点**：
  - 梯度信息被忽略（传统算法不使用梯度）
  - 保持与旧接口的兼容性
  - 所有历史数据由底层 `MultiObjectiveEvaluator` 管理

#### 4.2 物理内核统一
- **所有算法使用**：SPM_v3（支持自动微分）
- **传统算法**：只使用标量评估值（`scalarized`）
- **LLM-Enhanced BO**：使用标量值 + 梯度信息（如果可用）
- **公平性**：确保所有算法在相同的物理模型上评估

### 5. 验证测试

#### 5.1 代码验证（verify_unified_kernel.py）
```bash
python verify_unified_kernel.py
```
验证结果：
- ✓ traditional_bo.py 修改正确
- ✓ ga_optimizer.py 修改正确
- ✓ pso_optimizer.py 修改正确
- ✓ 所有 `LegacyEvaluator` 引用已移除

#### 5.2 功能测试（test_unified_kernel.py）
```bash
python test_unified_kernel.py
```
测试结果：
- ✓ `ScalarOnlyEvaluatorWrapper` 工作正常
- ✓ Traditional BO 初始化成功
- ✓ GA 初始化成功
- ✓ PSO 初始化成功
- ✓ 所有算法都使用 SPM_v3

### 6. 使用方法

#### 6.1 创建评估器
```python
from llmbo_core.multi_objective_evaluator import MultiObjectiveEvaluator

# 创建统一的评估器（SPM_v3）
evaluator = MultiObjectiveEvaluator(
    weights={'time': 0.4, 'temp': 0.35, 'aging': 0.25},
    temp_max=309.0,
    max_steps=300,
    verbose=False
)
```

#### 6.2 使用Traditional BO
```python
from Comparison.traditional_bo import TraditionalBO

# 自动包装为标量值接口
bo = TraditionalBO(
    evaluator=evaluator,  # MultiObjectiveEvaluator实例
    pbounds=pbounds,
    random_state=42,
    verbose=True
)

# 运行优化
result = bo.optimize(n_iterations=50, n_random_init=10)
```

#### 6.3 使用GA/PSO
```python
from Comparison.ga_optimizer import GeneticAlgorithm
from Comparison.pso_optimizer import ParticleSwarmOptimization

# GA（自动包装）
ga = GeneticAlgorithm(
    evaluator=evaluator,
    pbounds=pbounds,
    random_state=42,
    population_size=10
)

# PSO（自动包装）
pso = ParticleSwarmOptimization(
    evaluator=evaluator,
    pbounds=pbounds,
    random_state=42,
    n_particles=10
)
```

### 7. 迁移指南

如果您的代码中使用了旧的接口，请按以下方式更新：

#### 7.1 移除 use_legacy_spm 参数

**旧代码**：
```python
bo = TraditionalBO(
    evaluator=evaluator,
    pbounds=pbounds,
    use_legacy_spm=True  # ❌ 已废弃
)
```

**新代码**：
```python
bo = TraditionalBO(
    evaluator=evaluator,
    pbounds=pbounds
    # ✅ 自动使用SPM_v3
)
```

#### 7.2 确保传递正确的评估器类型

**旧代码**：
```python
# 可能传递任何评估器
ga = GeneticAlgorithm(evaluator=some_evaluator, ...)
```

**新代码**：
```python
from llmbo_core.multi_objective_evaluator import MultiObjectiveEvaluator

# 必须传递MultiObjectiveEvaluator实例
evaluator = MultiObjectiveEvaluator(...)
ga = GeneticAlgorithm(evaluator=evaluator, ...)
```

### 8. 运行对比实验

使用统一物理内核运行算法对比：

```bash
cd D:\Users\aa133\Desktop\BO_Multi_11_12\BO\Comparison
python comparison_runner.py
```

所有算法（BO、GA、PSO、LLM-BO）现在都使用相同的 SPM_v3 物理内核，确保公平对比。

### 9. 已知问题和解决方案

#### 问题 1: MultiObjectiveEvaluator 参数错误
**错误信息**：
```
TypeError: MultiObjectiveEvaluator.__init__() got an unexpected keyword argument 'enable_sensitivities'
```

**解决方案**：
`MultiObjectiveEvaluator` 不接受 `enable_sensitivities` 参数。创建时只需提供：
```python
evaluator = MultiObjectiveEvaluator(
    weights=weights,
    temp_max=309.0,
    max_steps=300,
    verbose=False
)
```

#### 问题 2: evaluate() 返回值类型错误
**错误信息**：
```
IndexError: invalid index to scalar variable
```

**解决方案**：
`MultiObjectiveEvaluator.evaluate()` 直接返回标量浮点数，不是字典。正确用法：
```python
# ✅ 正确
scalarized = evaluator.evaluate(current1, charging_number, current2)

# ❌ 错误
scalarized = evaluator.evaluate(...)['scalarized']
```

### 10. 总结

✅ **完成事项**：
- 所有传统算法统一使用 SPM_v3 物理内核
- 移除旧版 `LegacyEvaluator`（SPM v2.1）
- 实现 `ScalarOnlyEvaluatorWrapper` 适配层
- 更新所有相关代码和文档
- 通过验证测试

✅ **收益**：
- 确保算法对比的公平性
- 代码结构更清晰
- 易于维护和扩展
- 统一的物理模拟精度

✅ **兼容性**：
- 对外接口保持不变
- 现有调用代码无需修改（除非使用了 `use_legacy_spm` 参数）
- 所有功能正常工作
