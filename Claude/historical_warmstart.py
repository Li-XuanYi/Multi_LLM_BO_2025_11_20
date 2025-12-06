"""
基于历史数据的智能WarmStart模块
Historical Data-Driven Intelligent Warm Start

功能:
1. 从历史Result中学习最优/最差/随机解
2. 结合领域知识生成高质量prompt
3. 调用LLM生成physically plausible初始策略
4. 替换原有的硬编码WarmStart

Author: Research Team
Date: 2025-12-06
Version: 2.0 - 历史数据驱动
"""

import json
import asyncio
from typing import Dict, List, Optional
from openai import AsyncOpenAI

# 导入我们创建的模块
try:
    from prompt_generator import BatteryKnowledgePromptGenerator
    from result_manager import ResultManager
except ImportError:
    # 如果在其他目录运行,尝试相对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from prompt_generator import BatteryKnowledgePromptGenerator
    from result_manager import ResultManager


class HistoricalWarmStart:
    """
    历史数据驱动的智能WarmStart
    
    集成:
    - ResultManager: 加载和查询历史数据
    - PromptGenerator: 生成高质量LLM prompts
    - LLM API: 生成physically plausible策略
    """
    
    def __init__(
        self,
        result_dir: str = './results',
        llm_api_key: Optional[str] = None,
        llm_base_url: str = 'https://api.nuwaapi.com/v1',
        llm_model: str = "gpt-3.5-turbo",
        verbose: bool = True
    ):
        """
        初始化HistoricalWarmStart
        
        参数:
            result_dir: 历史结果目录
            llm_api_key: LLM API密钥
            llm_base_url: API基础URL
            llm_model: LLM模型名称
            verbose: 详细输出
        """
        
        self.result_manager = ResultManager(save_dir=result_dir)
        self.prompt_generator = BatteryKnowledgePromptGenerator()
        
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.verbose = verbose
        
        # 初始化LLM客户端(如果有API key)
        if llm_api_key:
            self.llm_client = AsyncOpenAI(
                base_url=llm_base_url,
                api_key=llm_api_key
            )
        else:
            self.llm_client = None
        
        if self.verbose:
            print("[HistoricalWarmStart] 初始化完成")
            print(f"  结果目录: {result_dir}")
            print(f"  LLM模型: {llm_model if llm_api_key else '未配置(将使用随机策略)'}")
    
    async def generate_warmstart_strategies_async(
        self,
        n_strategies: int = 5,
        n_historical_runs: int = 5,
        objective_weights: Optional[Dict[str, float]] = None,
        exploration_mode: str = 'balanced'
    ) -> List[Dict]:
        """
        生成WarmStart策略(异步)
        
        流程:
        1. 加载历史数据(最近n_historical_runs次运行)
        2. 筛选最优10个、最差10个、随机5个解
        3. 生成包含历史样例的高质量prompt
        4. 调用LLM生成策略
        5. 返回策略列表
        
        参数:
            n_strategies: 需要生成的策略数量
            n_historical_runs: 加载最近几次运行
            objective_weights: 目标权重
            exploration_mode: 探索模式 ('conservative', 'balanced', 'aggressive')
        
        返回:
            策略列表 [{'params': {...}, 'reasoning': '...', 'source': 'llm_warmstart'}, ...]
        """
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("【HistoricalWarmStart】开始生成策略")
            print("=" * 80)
        
        # ====== 1. 加载历史数据 ======
        if self.verbose:
            print("\n[1/4] 加载历史数据...")
        
        historical_data = self.result_manager.load_historical_data(
            n_recent=n_historical_runs
        )
        
        # ====== 2. 筛选示例解 ======
        if self.verbose:
            print("\n[2/4] 筛选示例解...")
        
        historical_best = []
        historical_worst = []
        
        if historical_data:
            # 最优10个
            historical_best = self.result_manager.get_top_k_solutions(
                historical_data,
                k=10,
                metric='scalarized'
            )
            
            # 最差10个
            historical_worst = self.result_manager.get_worst_k_solutions(
                historical_data,
                k=10,
                metric='scalarized'
            )
            
            if self.verbose:
                print(f"  ✓ 最优解: {len(historical_best)} 个")
                print(f"  ✓ 最差解: {len(historical_worst)} 个")
                if historical_best:
                    print(f"  全局最优标量值: {historical_best[0].get('scalarized', 'N/A'):.4f}")
        else:
            if self.verbose:
                print("  ! 未找到历史数据,将使用纯领域知识生成")
        
        # ====== 3. 生成高质量Prompt ======
        if self.verbose:
            print("\n[3/4] 生成高质量Prompt...")
        
        prompt = self.prompt_generator.generate_warmstart_prompt(
            n_strategies=n_strategies,
            objective_weights=objective_weights or {'time': 0.4, 'temp': 0.35, 'aging': 0.25},
            historical_best=historical_best[:3] if historical_best else None,  # 只传递top-3
            historical_worst=historical_worst[:2] if historical_worst else None,  # 只传递worst-2
            exploration_emphasis=exploration_mode
        )
        
        if self.verbose:
            print(f"  ✓ Prompt长度: {len(prompt)} 字符")
            print(f"  包含历史样例: {len(historical_best[:3]) if historical_best else 0} 个最优 + {len(historical_worst[:2]) if historical_worst else 0} 个最差")
        
        # ====== 4. 调用LLM生成策略 ======
        if self.verbose:
            print(f"\n[4/4] 调用LLM生成策略 ({self.llm_model})...")
        
        if self.llm_client is None:
            if self.verbose:
                print("  ! 未配置LLM API,回退到随机策略生成")
            return self._generate_random_fallback_strategies(n_strategies)
        
        try:
            strategies = await self._call_llm_async(prompt, n_strategies)
            
            if self.verbose:
                print(f"  ✓ LLM成功生成 {len(strategies)} 个策略")
            
            return strategies
            
        except Exception as e:
            if self.verbose:
                print(f"  ✗ LLM调用失败: {e}")
                print("  回退到随机策略生成")
            
            return self._generate_random_fallback_strategies(n_strategies)
    
    async def _call_llm_async(
        self,
        prompt: str,
        n_strategies: int
    ) -> List[Dict]:
        """
        异步调用LLM API生成策略
        
        参数:
            prompt: 完整的prompt字符串
            n_strategies: 期望的策略数量
        
        返回:
            策略列表
        """
        
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in electrochemistry and battery optimization. "
                               "Always respond with valid JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,  # 适度的创造性
            max_tokens=2000,
            response_format={"type": "json_object"}  # 强制JSON格式
        )
        
        # 解析响应
        response_text = response.choices[0].message.content
        
        try:
            # 尝试直接解析JSON
            strategies_data = json.loads(response_text)
            
            # 检查格式
            if isinstance(strategies_data, dict):
                # 可能是 {"strategies": [...]} 格式
                if 'strategies' in strategies_data:
                    strategies_list = strategies_data['strategies']
                else:
                    # 或者是单个策略对象,包装成列表
                    strategies_list = [strategies_data]
            elif isinstance(strategies_data, list):
                strategies_list = strategies_data
            else:
                raise ValueError("LLM返回的不是字典或列表")
            
            # 验证和格式化策略
            validated_strategies = []
            for strategy in strategies_list:
                if self._validate_strategy(strategy):
                    validated_strategies.append({
                        'params': {
                            'current1': float(strategy['current1']),
                            'charging_number': int(strategy['charging_number']),
                            'current2': float(strategy['current2'])
                        },
                        'reasoning': strategy.get('reasoning', ''),
                        'source': 'llm_warmstart'
                    })
            
            if len(validated_strategies) < n_strategies:
                if self.verbose:
                    print(f"  警告: LLM只生成了 {len(validated_strategies)}/{n_strategies} 个有效策略")
            
            return validated_strategies
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  JSON解析失败: {e}")
                print(f"  LLM响应: {response_text[:200]}...")
            raise
    
    def _validate_strategy(self, strategy: Dict) -> bool:
        """
        验证策略是否满足约束
        
        参数:
            strategy: 策略字典
        
        返回:
            是否有效
        """
        
        try:
            # 检查必需字段
            if 'current1' not in strategy or 'charging_number' not in strategy or 'current2' not in strategy:
                return False
            
            current1 = float(strategy['current1'])
            charging_number = int(strategy['charging_number'])
            current2 = float(strategy['current2'])
            
            # 检查范围
            if not (3.0 <= current1 <= 6.0):
                return False
            if not (5 <= charging_number <= 25):
                return False
            if not (1.0 <= current2 <= 4.0):
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _generate_random_fallback_strategies(
        self,
        n_strategies: int
    ) -> List[Dict]:
        """
        回退方案:生成随机策略
        
        参数:
            n_strategies: 策略数量
        
        返回:
            随机策略列表
        """
        
        import numpy as np
        
        strategies = []
        for i in range(n_strategies):
            strategies.append({
                'params': {
                    'current1': np.random.uniform(3.0, 6.0),
                    'charging_number': int(np.random.uniform(5, 25)),
                    'current2': np.random.uniform(1.0, 4.0)
                },
                'reasoning': 'Random fallback strategy (LLM unavailable)',
                'source': 'random_warmstart'
            })
        
        return strategies
    
    # 同步版本(用于兼容)
    def generate_warmstart_strategies(self, **kwargs) -> List[Dict]:
        """同步版本的WarmStart策略生成"""
        return asyncio.run(self.generate_warmstart_strategies_async(**kwargs))


# ============ 测试代码 ============
if __name__ == "__main__":
    import asyncio
    
    print("\n" + "=" * 80)
    print("测试 HistoricalWarmStart")
    print("=" * 80)
    
    # 创建HistoricalWarmStart(不提供API key,测试随机回退)
    warmstart = HistoricalWarmStart(
        result_dir='./test_results',
        llm_api_key=None,  # 测试随机回退
        verbose=True
    )
    
    # 测试生成策略
    async def test():
        strategies = await warmstart.generate_warmstart_strategies_async(
            n_strategies=3,
            n_historical_runs=1,
            exploration_mode='balanced'
        )
        
        print("\n" + "=" * 80)
        print(f"生成的策略 ({len(strategies)} 个):")
        print("=" * 80)
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n策略 {i}:")
            print(f"  current1 = {strategy['params']['current1']:.2f} A")
            print(f"  charging_number = {strategy['params']['charging_number']}")
            print(f"  current2 = {strategy['params']['current2']:.2f} A")
            print(f"  推理: {strategy['reasoning']}")
            print(f"  来源: {strategy['source']}")
    
    asyncio.run(test())
    
    print("\n" + "=" * 80)
    print("HistoricalWarmStart 测试完成")
    print("=" * 80)
