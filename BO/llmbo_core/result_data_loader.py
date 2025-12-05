"""
历史结果数据加载器 - 用于LLM历史学习
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


class ResultDataLoader:
    """加载和总结历史优化结果，供LLM历史学习使用"""
    
    def __init__(self, results_dirs: List[str] = None):
        """
        Args:
            results_dirs: 结果文件所在目录列表
                         默认: ['results/', 'comparison_results/']
        """
        if results_dirs is None:
            # 默认搜索两个目录
            base_path = Path(__file__).parent.parent.parent  # BO_Multi_11_12/
            self.results_dirs = [
                base_path / 'results',
                base_path / 'comparison_results'
            ]
        else:
            self.results_dirs = [Path(d) for d in results_dirs]
        
        self.loaded_results = []
    
    def load_all_results(
        self, 
        pattern: str = "*.json",
        limit: int = None
    ) -> List[Dict]:
        """
        加载所有历史结果JSON文件
        
        Args:
            pattern: 文件名匹配模式
            limit: 最多加载多少个文件（None=全部）
        
        Returns:
            结果列表，每个元素包含 {'file', 'data', 'timestamp'}
        """
        all_files = []
        
        # 收集所有匹配文件
        for results_dir in self.results_dirs:
            if results_dir.exists():
                all_files.extend(list(results_dir.glob(pattern)))
        
        # 按修改时间排序（最新优先）
        all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # 应用限制
        if limit is not None:
            all_files = all_files[:limit]
        
        # 加载文件
        self.loaded_results = []
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.loaded_results.append({
                    'file': str(file_path),
                    'data': data,
                    'timestamp': file_path.stem.split('_')[-1]  # 提取时间戳
                })
            except Exception as e:
                print(f"[警告] 加载文件失败: {file_path}, 错误: {e}")
        
        print(f"[历史结果加载] 成功加载 {len(self.loaded_results)} 个结果文件")
        return self.loaded_results
    
    def get_statistics_summary(self) -> Dict:
        """
        生成统计摘要（供LLM参考）
        
        Returns:
            统计信息字典，包含:
            - total_runs: 总运行次数
            - best_overall: 所有运行中的最优解
            - objective_ranges: 各目标的范围
            - param_ranges: 各参数的范围
        """
        if not self.loaded_results:
            return {}
        
        all_best_solutions = []
        all_eval_counts = []
        
        # 收集所有最优解
        for result in self.loaded_results:
            data = result['data']
            
            # 处理不同格式的结果文件
            if 'best_solution' in data:
                # llm_mobo格式
                all_best_solutions.append(data['best_solution'])
                all_eval_counts.append(data.get('total_evaluations', 0))
            elif 'results' in data:
                # comparison格式
                for algo, algo_result in data['results'].items():
                    if 'best_solution' in algo_result:
                        all_best_solutions.append(algo_result['best_solution'])
                        all_eval_counts.append(algo_result.get('total_evaluations', 0))
        
        if not all_best_solutions:
            return {}
        
        # 找到全局最优（基于scalarized）
        valid_solutions = [s for s in all_best_solutions if s.get('valid', True)]
        if not valid_solutions:
            return {}
        
        best_overall = min(valid_solutions, key=lambda x: x.get('scalarized', float('inf')))
        
        # 统计目标值范围
        objective_ranges = {
            'time': self._get_range([s['objectives']['time'] for s in valid_solutions]),
            'temp': self._get_range([s['objectives']['temp'] for s in valid_solutions]),
            'aging': self._get_range([s['objectives']['aging'] for s in valid_solutions])
        }
        
        # 统计参数范围
        param_ranges = {
            'current1': self._get_range([s['params']['current1'] for s in valid_solutions]),
            'charging_number': self._get_range([s['params']['charging_number'] for s in valid_solutions]),
            'current2': self._get_range([s['params']['current2'] for s in valid_solutions])
        }
        
        return {
            'total_runs': len(self.loaded_results),
            'total_evaluations': sum(all_eval_counts) if all_eval_counts else 0,
            'best_overall': best_overall,
            'objective_ranges': objective_ranges,
            'param_ranges': param_ranges,
            'valid_solution_count': len(valid_solutions)
        }
    
    def get_top_k_solutions(self, k: int = 10) -> List[Dict]:
        """
        获取历史中top-k最优解
        
        Args:
            k: 返回前k个最优解
        
        Returns:
            最优解列表
        """
        all_solutions = []
        
        for result in self.loaded_results:
            data = result['data']
            
            if 'best_solution' in data:
                all_solutions.append(data['best_solution'])
            elif 'results' in data:
                for algo_result in data['results'].values():
                    if 'best_solution' in algo_result:
                        all_solutions.append(algo_result['best_solution'])
        
        # 过滤有效解并排序
        valid_solutions = [s for s in all_solutions if s.get('valid', True)]
        valid_solutions.sort(key=lambda x: x.get('scalarized', float('inf')))
        
        return valid_solutions[:k]
    
    def format_for_llm_prompt(self, top_k: int = 5) -> str:
        """
        格式化为LLM提示词的历史知识部分
        
        Args:
            top_k: 包含前k个历史最优解
        
        Returns:
            格式化的字符串
        """
        stats = self.get_statistics_summary()
        
        if not stats:
            return "无历史结果数据。"
        
        top_solutions = self.get_top_k_solutions(k=top_k)
        
        prompt = f"""
## 历史优化结果知识库

**总体统计:**
- 历史运行次数: {stats['total_runs']}
- 总评估次数: {stats['total_evaluations']}
- 有效解数量: {stats['valid_solution_count']}

**目标值范围 (历史观测):**
- 充电时间: {stats['objective_ranges']['time']['min']:.2f} ~ {stats['objective_ranges']['time']['max']:.2f} min (均值: {stats['objective_ranges']['time']['mean']:.2f})
- 最高温度: {stats['objective_ranges']['temp']['min']:.2f} ~ {stats['objective_ranges']['temp']['max']:.2f} K (均值: {stats['objective_ranges']['temp']['mean']:.2f})
- 老化损失: {stats['objective_ranges']['aging']['min']:.4f} ~ {stats['objective_ranges']['aging']['max']:.4f} (均值: {stats['objective_ranges']['aging']['mean']:.4f})

**参数空间探索范围 (历史):**
- current1 (I1): {stats['param_ranges']['current1']['min']:.2f} ~ {stats['param_ranges']['current1']['max']:.2f} A
- charging_number (t1): {int(stats['param_ranges']['charging_number']['min'])} ~ {int(stats['param_ranges']['charging_number']['max'])} 步
- current2 (I2): {stats['param_ranges']['current2']['min']:.2f} ~ {stats['param_ranges']['current2']['max']:.2f} A

**历史Top-{top_k}最优解:**
"""
        
        for i, sol in enumerate(top_solutions, 1):
            prompt += f"""
{i}. 参数: I1={sol['params']['current1']:.2f}A, t1={sol['params']['charging_number']}, I2={sol['params']['current2']:.2f}A
   目标: 时间={sol['objectives']['time']:.2f}min, 温度={sol['objectives']['temp']:.2f}K, 老化={sol['objectives']['aging']:.4f}
   标量化得分: {sol['scalarized']:.6f}"""
        
        prompt += "\n\n**知识启发:** 基于历史数据，优化算法应关注上述成功参数区域，避免已知的失败配置。"
        
        return prompt
    
    def _get_range(self, values: List[float]) -> Dict:
        """计算数值列表的统计范围"""
        values = [v for v in values if v is not None and not np.isnan(v)]
        
        if not values:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
        
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }


# 单例实例（可选）
_global_loader = None


def get_global_loader() -> ResultDataLoader:
    """获取全局单例ResultDataLoader"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ResultDataLoader()
        _global_loader.load_all_results(limit=20)  # 默认加载最新20个
    return _global_loader
