"""
Multi_objective_evaluator 修改示例
展示如何集成新的 HistoricalWarmStart 替换硬编码prompt

修改要点:
1. 移除硬编码的_llm_generate_strategies方法
2. 使用HistoricalWarmStart生成策略
3. 保持原有接口兼容性

Author: Research Team
Date: 2025-12-06
"""

# ============================================================================
# 修改前 (原有硬编码版本) - 需要删除或注释
# ============================================================================

"""
原有的硬编码WarmStart (multi_objective_evaluator.py 中):

async def _llm_generate_strategies(
    self,
    n_strategies: int,
    api_key: str,
    base_url: str,
    model: str
) -> List[Dict]:
    # 构建 prompt - 硬编码!!!
    prompt = f\"\"\"You are an expert in electrochemistry...
    
TASK:
Generate {n_strategies} diverse two-stage constant-current (CC) charging strategies...

BATTERY SPECIFICATIONS:
- Nominal Capacity: 5.0 Ah
...

PHYSICAL CONSIDERATIONS:
1. High Stage 1 current → faster charging BUT higher temperature & more aging
2. Long Stage 1 duration → more heat accumulation
3. Stage 2 current affects completion time and thermal recovery
...
\"\"\"

    # 调用LLM
    ...
"""

# ============================================================================
# 修改后 (使用新的HistoricalWarmStart) - 推荐方式
# ============================================================================

"""
在 multi_objective_evaluator.py 中的修改:

1. 在文件开头添加导入:
"""

# 新增导入
from historical_warmstart import HistoricalWarmStart

"""
2. 修改 initialize_with_llm_warmstart 方法:
"""

async def initialize_with_llm_warmstart(
    self,
    n_strategies: int = 5,
    llm_api_key: Optional[str] = None,
    llm_base_url: str = 'https://api.nuwaapi.com/v1',
    llm_model: str = "gpt-3.5-turbo"
) -> List[Dict]:
    """
    使用 LLM 进行热启动初始化 - 新版本
    
    修改:
    - 不再使用硬编码prompt
    - 使用HistoricalWarmStart从历史数据学习
    - 自动生成高质量prompt
    """
    
    if self.verbose:
        print("\n" + "=" * 70)
        print("开始 LLM Warm Start 初始化 (历史数据驱动版)")
        print("=" * 70)
        print(f"目标策略数量: {n_strategies}")
        print(f"API Key: {'已提供' if llm_api_key else '未提供（将使用随机策略）'}")
        print("=" * 70)
    
    results = []
    
    # ====== 使用新的HistoricalWarmStart ======
    if llm_api_key is not None:
        try:
            if self.verbose:
                print("\n使用HistoricalWarmStart生成策略...")
            
            # 创建HistoricalWarmStart实例
            warmstart_generator = HistoricalWarmStart(
                result_dir='./results',  # 确保Result目录正确
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                verbose=self.verbose
            )
            
            # 生成策略 (自动加载历史数据 + 生成高质量prompt + 调用LLM)
            strategies = await warmstart_generator.generate_warmstart_strategies_async(
                n_strategies=n_strategies,
                n_historical_runs=5,  # 加载最近5次运行
                objective_weights=self.weights,
                exploration_mode='balanced'  # 可选: 'conservative', 'balanced', 'aggressive'
            )
            
            if self.verbose:
                print(f"[OK] HistoricalWarmStart成功生成 {len(strategies)} 个策略")
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  HistoricalWarmStart失败: {e}")
                print("   回退到随机策略生成...")
            
            # 回退到随机策略
            strategies = self._generate_random_strategies(n_strategies)
    
    else:
        # 没有 API key，直接使用随机策略
        if self.verbose:
            print("\n使用随机策略生成...")
        
        strategies = self._generate_random_strategies(n_strategies)
    
    # ====== 评估所有策略 (保持不变) ======
    if self.verbose:
        print(f"\n开始评估 {len(strategies)} 个策略...")
    
    for i, strategy in enumerate(strategies, 1):
        params = strategy['params']
        
        try:
            # 调用 evaluate() 方法进行仿真评估
            scalarized = self.evaluate(
                current1=params['current1'],
                charging_number=params['charging_number'],
                current2=params['current2']
            )
            
            # 从最新的日志中提取目标值
            latest_log = self.detailed_logs[-1]
            
            result = {
                'params': params,
                'scalarized': scalarized,
                'objectives': latest_log['objectives'],
                'valid': latest_log['valid'],
                'source': strategy.get('source', 'llm_warmstart'),
                'reasoning': strategy.get('reasoning', '')
            }
            
            results.append(result)
            
            if self.verbose:
                print(f"  策略 {i}/{len(strategies)}: "
                      f"I1={params['current1']:.2f}A, "
                      f"t1={params['charging_number']}, "
                      f"I2={params['current2']:.2f}A "
                      f"→ 标量化={scalarized:.4f}")
        
        except Exception as e:
            if self.verbose:
                print(f"  ✗ 策略 {i} 评估失败: {e}")
            continue
    
    if self.verbose:
        print(f"\n[OK] Warm Start completed! Successfully evaluated {len(results)}/{len(strategies)} strategies")
        print("=" * 70)
    
    return results


"""
3. 删除或注释掉原有的 _llm_generate_strategies 方法:
"""

# ❌ 删除这个方法 (硬编码prompt)
# async def _llm_generate_strategies(...):
#     prompt = f\"\"\"硬编码的prompt...\"\"\"
#     ...


"""
4. _generate_random_strategies 保持不变 (作为fallback)
"""

def _generate_random_strategies(self, n_strategies: int) -> List[Dict]:
    """生成随机策略 (fallback方案)"""
    import numpy as np
    
    strategies = []
    for i in range(n_strategies):
        strategies.append({
            'params': {
                'current1': np.random.uniform(3.0, 6.0),
                'charging_number': int(np.random.uniform(5, 25)),
                'current2': np.random.uniform(1.0, 4.0)
            },
            'source': 'random_warmstart',
            'reasoning': 'Random fallback strategy'
        })
    
    return strategies


# ============================================================================
# 主优化器中的修改 (LLM_Enhanced_Multi-Objective_Bayesian_Optimization.py)
# ============================================================================

"""
在主优化器中集成ResultManager:

1. 在文件开头添加导入:
"""

from result_manager import ResultManager

"""
2. 在__init__中初始化ResultManager:
"""

def __init__(self): # ...
    # ... 原有代码 ...
    
    # 初始化ResultManager
    self.result_manager = ResultManager(save_dir=save_dir)


"""
3. 修改export_results方法,使用ResultManager保存完整数据:
"""

def export_results(self, filename: str = None) -> str:
    """
    导出结果到JSON文件 - 使用ResultManager
    
    修改:
    - 使用ResultManager.save_optimization_run保存完整数据
    - 不仅保存最优解,还保存所有评估点
    """
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"llm_mobo_{timestamp}"
    else:
        run_id = filename.replace('.json', '')
    
    # 获取数据
    database = self.evaluator.export_database()
    best_solution = self.evaluator.get_best_solution()
    pareto_front = self.evaluator.get_pareto_front()
    statistics = self.evaluator.get_statistics()
    
    # 配置信息
    config = {
        'llm_model': self.llm_model,
        'n_warmstart': self.n_warmstart,
        'n_random_init': self.n_random_init,
        'n_iterations': self.n_iterations,
        'objective_weights': self.evaluator.weights,
        'enable_llm_warmstart': self.enable_llm_warmstart,
        'enable_llm_surrogate': self.enable_llm_surrogate,
        'enable_llm_ei': self.enable_llm_ei
    }
    
    # 使用ResultManager保存完整数据
    filepath = self.result_manager.save_optimization_run(
        run_id=run_id,
        database=database,
        best_solution=best_solution,
        pareto_front=pareto_front,
        config=config,
        statistics=statistics,
        metadata={
            'elapsed_time': self.optimization_history[-1]['elapsed_time'] if self.optimization_history else 0
        }
    )
    
    print(f"\n✓ 完整结果已保存至: {filepath}")
    return filepath


# ============================================================================
# 总结
# ============================================================================

"""
主要修改:

1. **multi_objective_evaluator.py**:
   - 添加: from historical_warmstart import HistoricalWarmStart
   - 修改: initialize_with_llm_warmstart() 使用HistoricalWarmStart
   - 删除: _llm_generate_strategies() 硬编码方法

2. **LLM_Enhanced_Multi-Objective_Bayesian_Optimization.py**:
   - 添加: from result_manager import ResultManager
   - 添加: self.result_manager = ResultManager(...)
   - 修改: export_results() 使用ResultManager保存完整数据

优势:
✅ 移除硬编码prompt
✅ 自动从历史数据学习
✅ 高质量的领域知识prompt
✅ 完整的数据保存(所有评估点)
✅ 支持最优/最差/随机解查询
✅ Few-shot learning from历史

使用方法见 USAGE_GUIDE.md
"""
