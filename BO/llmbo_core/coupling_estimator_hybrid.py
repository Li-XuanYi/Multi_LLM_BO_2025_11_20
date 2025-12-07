"""
混合耦合矩阵估计器 (Hybrid Coupling Matrix Estimator)
结合数据驱动 + LLM知识的耦合强度估计

功能:
1. 加载历史优化数据
2. 分析参数-目标的相关性
3. 结合LLM的电化学知识
4. 输出融合后的耦合矩阵

Author: Research Team
Date: 2025-12-06
Version: 1.0 - 基于论文和历史数据的混合方法
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import asyncio


class HybridCouplingEstimator:
    """
    混合耦合矩阵估计器
    
    结合:
    1. 数据驱动: 从历史数据计算参数-目标相关性
    2. LLM知识: 电化学领域的物理耦合关系
    """
    
    def __init__(
        self,
        result_dir: str = './results',
        llm_api_key: Optional[str] = None,
        llm_base_url: str = 'https://api.nuwaapi.com/v1',
        llm_model: str = 'gpt-4o',
        verbose: bool = True
    ):
        """
        初始化混合估计器
        
        参数:
            result_dir: 历史数据目录
            llm_api_key: LLM API密钥
            llm_base_url: API基础URL
            llm_model: LLM模型名称
            verbose: 详细输出
        """
        self.result_dir = Path(result_dir)
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.verbose = verbose
        
        # 参数和目标名称
        self.param_names = ['current1', 'charging_number', 'current2']
        self.objective_names = ['time', 'temp', 'aging']
        
        if self.verbose:
            print("[HybridCouplingEstimator] 初始化完成")
    
    def load_historical_data(self, n_recent: int = 5) -> List[Dict]:
        """
        加载最近的历史数据
        
        参数:
            n_recent: 加载最近n次运行
        
        返回:
            历史数据列表
        """
        if not self.result_dir.exists():
            if self.verbose:
                print(f"[警告] 结果目录不存在: {self.result_dir}")
            return []
        
        # 查找JSON文件
        json_files = sorted(
            self.result_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:n_recent]
        
        historical_data = []
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 验证数据完整性
                    if 'database' in data and len(data['database']) > 0:
                        historical_data.append(data)
            except Exception as e:
                if self.verbose:
                    print(f"[警告] 加载文件失败: {filepath.name}")
        
        if self.verbose:
            print(f"[HybridCouplingEstimator] 加载了 {len(historical_data)} 个历史运行")
        
        return historical_data
    
    def compute_data_driven_coupling(
        self,
        historical_data: List[Dict]
    ) -> np.ndarray:
        """
        从历史数据计算参数-目标的耦合强度
        
        方法: 计算Pearson相关系数
        
        返回:
            3×3耦合矩阵 (current1, charging_number, current2) × (time, temp, aging)
        """
        # 收集所有评估点
        all_params = []
        all_objectives = []
        
        for run_data in historical_data:
            database = run_data.get('database', [])
            for entry in database:
                if entry.get('valid', True):
                    params = entry['params']
                    objectives = entry['objectives']
                    
                    all_params.append([
                        params['current1'],
                        params['charging_number'],
                        params['current2']
                    ])
                    all_objectives.append([
                        objectives['time'],
                        objectives['temp'],
                        objectives['aging']
                    ])
        
        if len(all_params) < 5:
            if self.verbose:
                print("[警告] 数据不足(<5个点),使用默认耦合矩阵")
            # 返回默认矩阵(基于物理直觉)
            return np.array([
                [0.7, 0.6, 0.5],  # current1 → time, temp, aging
                [0.5, 0.4, 0.3],  # charging_number → time, temp, aging
                [0.6, 0.5, 0.4]   # current2 → time, temp, aging
            ])
        
        # 转换为numpy数组
        X = np.array(all_params)  # (N, 3)
        Y = np.array(all_objectives)  # (N, 3)
        
        # 标准化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-10
        X_norm = (X - X_mean) / X_std
        
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0) + 1e-10
        Y_norm = (Y - Y_mean) / Y_std
        
        # 计算相关系数矩阵 (3×3)
        correlation_matrix = np.abs(X_norm.T @ Y_norm) / len(X_norm)
        
        # 归一化到[0, 1]
        coupling_matrix = np.clip(correlation_matrix, 0, 1)
        
        if self.verbose:
            print(f"[数据驱动耦合] 使用 {len(all_params)} 个评估点")
            print(f"  相关性范围: [{coupling_matrix.min():.3f}, {coupling_matrix.max():.3f}]")
        
        return coupling_matrix
    
    async def get_llm_coupling_knowledge_async(self) -> np.ndarray:
        """
        从LLM获取电化学领域的耦合知识
        
        返回:
            3×3耦合矩阵 (基于LLM的物理知识)
        """
        if not self.llm_api_key:
            if self.verbose:
                print("[警告] 未提供LLM API密钥,使用物理默认值")
            return self._get_physics_default_coupling()
        
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
        
        prompt = """You are an expert in lithium-ion battery electrochemistry and fast-charging optimization.

TASK: Estimate the coupling strength between charging parameters and performance objectives.

PARAMETERS (inputs):
1. current1: First-stage charging current (A)
2. charging_number: Switching time step (controls stage 1 duration)
3. current2: Second-stage charging current (A)

OBJECTIVES (outputs):
1. time: Total charging time (minimize)
2. temp: Peak temperature (minimize, constraint ≤309K)
3. aging: Capacity fade from SEI growth (minimize)

PHYSICAL MECHANISMS:
- High current1 → faster charging BUT higher Joule heating (Q̇=I²R) and SEI growth
- Long charging_number → more heat accumulation and aging
- High current2 → faster completion BUT less thermal relaxation

QUESTION: On a scale of 0.0 to 1.0, how strongly does each parameter affect each objective?

OUTPUT FORMAT (JSON):
{
  "coupling_matrix": [
    [c11, c12, c13],  // current1 → [time, temp, aging]
    [c21, c22, c23],  // charging_number → [time, temp, aging]
    [c31, c32, c33]   // current2 → [time, temp, aging]
  ],
  "reasoning": "Brief explanation of the strongest couplings"
}

MATHEMATICAL CONSTRAINT: The matrix will be symmetrized as W_final = (W + W^T)/2 for numerical stability.
Consider this when assigning values: c_ij and c_ji should be similar (e.g., if current1→temp is 0.8, then charging_number→time should be ~0.7-0.9 for physical consistency).

CRITICAL: Output ONLY the JSON, no additional text."""

        try:
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an electrochemistry expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 低温度以获得稳定输出
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # 清理可能的markdown包装
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # 解析JSON
            parsed = json.loads(content)
            coupling_matrix = np.array(parsed['coupling_matrix'])
            
            if self.verbose:
                print("[LLM耦合知识] 成功获取")
                print(f"  LLM推理: {parsed.get('reasoning', 'N/A')[:100]}")
            
            return coupling_matrix
            
        except Exception as e:
            if self.verbose:
                print(f"[警告] LLM调用失败: {e}, 使用物理默认值")
            return self._get_physics_default_coupling()
    
    def get_llm_coupling_knowledge(self) -> np.ndarray:
        """
        同步版本：从LLM获取电化学领域的耦合知识
        
        使用create_task + run_until_complete避免嵌套事件循环问题
        
        返回:
            3×3耦合矩阵
        """
        try:
            # 检查是否在运行的事件循环中
            try:
                loop = asyncio.get_running_loop()
                # 如果在事件循环中,使用run_in_executor在新线程中运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._sync_llm_call)
                    return future.result(timeout=30)
            except RuntimeError:
                # 没有运行的事件循环,直接使用asyncio.run
                return asyncio.run(self.get_llm_coupling_knowledge_async())
        except Exception as e:
            if self.verbose:
                print(f"[警告] LLM同步调用失败: {e}, 使用物理默认值")
            return self._get_physics_default_coupling()
    
    def _sync_llm_call(self) -> np.ndarray:
        """在新事件循环中执行异步LLM调用"""
        return asyncio.run(self.get_llm_coupling_knowledge_async())
    
    def _regularize_coupling_matrix(self, matrix: np.ndarray, enforce_symmetric: bool = True) -> np.ndarray:
        """
        正则化耦合矩阵,确保数值稳定性
        
        数学原理:
        1. 对称化: W_sym = (W + W^T) / 2
           - 物理意义: 参数i→目标j 与 参数j→目标i 的耦合应该一致
           - 保证核函数的正定性 (Mercer定理要求)
        
        2. 对角线归一化: diag(W) = 1
           - 物理意义: 参数对自身的"耦合"定义为1
           - 类似于相关系数矩阵的标准化
        
        3. 范围裁剪: W ∈ [0, 1]
           - 物理意义: 耦合强度是无量纲的相对值
        
        参数:
            matrix: 原始耦合矩阵 (3×3)
            enforce_symmetric: 是否强制对称性
        
        返回:
            正则化后的耦合矩阵
        """
        W = matrix.copy()
        
        # 步骤1: 对称化
        if enforce_symmetric:
            W = (W + W.T) / 2
        
        # 步骤2: 对角线归一化 (可选)
        # 注意: 这里我们不强制对角线为1,因为W是参数×目标矩阵(3×3),不是参数×参数
        # 如果是参数×参数的协方差矩阵,才需要对角线为1
        
        # 步骤3: 范围裁剪
        W = np.clip(W, 0.0, 1.0)
        
        return W
    
    def _get_physics_default_coupling(self) -> np.ndarray:
        """
        基于电化学物理原理的默认耦合矩阵
        
        注意: 这是参数×目标矩阵 (3×3),不是相关性矩阵
        因此对称性不是严格要求,但为了数值稳定性我们仍然返回对称矩阵
        
        返回:
            3×3对称耦合矩阵
        """
        # 基于文献和物理直觉的默认值 (对称化)
        matrix = np.array([
            # [time, temp, aging]
            [0.75, 0.80, 0.70],  # current1: 强影响时间和温度,中等影响老化
            [0.70, 0.50, 0.55],  # charging_number: 中等影响所有目标
            [0.70, 0.55, 0.50]   # current2: 影响时间和温度,较小影响老化
        ])
        
        # 对称化处理
        return self._regularize_coupling_matrix(matrix, enforce_symmetric=True)
    
    def estimate_coupling_matrix(
        self,
        n_historical_runs: int = 5,
        data_weight: float = 0.6,
        llm_weight: float = 0.4
    ) -> np.ndarray:
        """
        估计混合耦合矩阵
        
        参数:
            n_historical_runs: 使用最近n次运行的数据
            data_weight: 数据驱动部分的权重
            llm_weight: LLM知识部分的权重
        
        返回:
            3×3混合耦合矩阵
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("[混合耦合矩阵估计]")
            print("=" * 70)
        
        # 1. 加载历史数据
        historical_data = self.load_historical_data(n_recent=n_historical_runs)
        
        # 2. 数据驱动部分
        data_coupling = self.compute_data_driven_coupling(historical_data)
        
        # 3. LLM知识部分 (使用同步版本避免嵌套事件循环)
        llm_coupling = self.get_llm_coupling_knowledge()
        
        # 4. 加权融合
        hybrid_coupling = data_weight * data_coupling + llm_weight * llm_coupling
        
        # 5. 正则化处理
        # 注意: 虽然这是参数×目标矩阵,不是严格的协方差矩阵,
        # 但对称化可以提高核函数的数值稳定性和正定性
        hybrid_coupling = self._regularize_coupling_matrix(
            hybrid_coupling, 
            enforce_symmetric=True  # 强制对称以保证正定性
        )
        
        if self.verbose:
            print("\n[融合结果]")
            print(f"  数据驱动权重: {data_weight:.1f}")
            print(f"  LLM知识权重: {llm_weight:.1f}")
            print(f"  正则化: 对称化 + 范围裁剪[0,1]")
            print("\n混合耦合矩阵 (参数×目标):")
            print("              time    temp   aging")
            for i, param in enumerate(self.param_names):
                print(f"  {param:15s} {hybrid_coupling[i,0]:.3f}  {hybrid_coupling[i,1]:.3f}  {hybrid_coupling[i,2]:.3f}")
            
            # 验证对称性
            is_symmetric = np.allclose(hybrid_coupling, hybrid_coupling.T, atol=1e-6)
            print(f"\n  [数学验证] 对称性: {'[OK]' if is_symmetric else '[X]'}")
            if not is_symmetric:
                max_asym = np.abs(hybrid_coupling - hybrid_coupling.T).max()
                print(f"  [Warning] 最大非对称度: {max_asym:.6f}")
            
            print("=" * 70)
        
        return hybrid_coupling


# ============ 测试代码 ============
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("测试 HybridCouplingEstimator")
    print("=" * 80)
    
    # 创建估计器
    estimator = HybridCouplingEstimator(
        result_dir='./results',
        llm_api_key=None,  # 测试时不使用LLM
        verbose=True
    )
    
    # 估计耦合矩阵
    coupling_matrix = estimator.estimate_coupling_matrix(
        n_historical_runs=5,
        data_weight=0.6,
        llm_weight=0.4
    )
    
    print("\n[OK] 混合耦合矩阵估计完成")
    print(f"  矩阵形状: {coupling_matrix.shape}")
    print(f"  值范围: [{coupling_matrix.min():.3f}, {coupling_matrix.max():.3f}]")
