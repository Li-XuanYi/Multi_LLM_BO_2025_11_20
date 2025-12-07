"""
Result数据管理系统 - 完善的优化结果存储
Comprehensive Result Data Manager for Optimization Records

功能:
1. 完整保存每次运行的所有评估数据(不仅是最优解)
2. 支持多种查询:最优/最差/随机解
3. 统计分析和可视化支持
4. 为LLM WarmStart提供历史数据

Author: Research Team
Date: 2025-12-06  
Version: 2.0 - 完整数据管理
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


def convert_to_json_serializable(obj):
    """
    递归转换numpy类型为Python原生类型,解决JSON序列化问题
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


class ResultManager:
    """
    优化结果管理器
    
    负责:
    - 保存完整的运行数据(所有评估点,不仅是最优解)
    - 提供多种查询接口
    - 支持历史数据加载和分析
    """
    
    def __init__(self, save_dir: str = './results'):
        """
        初始化Result管理器
        
        参数:
            save_dir: 结果保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[ResultManager] 初始化完成,保存目录: {self.save_dir}")
    
    def save_optimization_run(
        self,
        run_id: str,
        database: List[Dict],
        best_solution: Dict,
        pareto_front: List[Dict],
        config: Dict,
        statistics: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        保存一次完整的优化运行结果
        
        参数:
            run_id: 运行标识(例如: "llm_mobo_20251206_143022")
            database: 完整的评估数据库(所有评估点)
            best_solution: 最优解
            pareto_front: 帕累托前沿
            config: 运行配置
            statistics: 统计信息
            metadata: 额外的元数据
        
        返回:
            保存的文件路径
        """
        
        # 准备完整的数据结构
        complete_data = {
            # ====== 元信息 ======
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            
            # ====== 配置信息 ======
            'config': config,
            
            # ====== 统计摘要 ======
            'statistics': statistics,
            
            # ====== 最优解 ======
            'best_solution': {
                'eval_id': best_solution.get('eval_id'),
                'params': best_solution.get('params'),
                'objectives': best_solution.get('objectives'),
                'normalized': best_solution.get('normalized'),
                'scalarized': best_solution.get('scalarized'),
                'valid': best_solution.get('valid'),
                'source': best_solution.get('source', 'unknown')
            },
            
            # ====== 帕累托前沿 ======
            'pareto_front': [
                {
                    'eval_id': sol.get('eval_id'),
                    'params': sol.get('params'),
                    'objectives': sol.get('objectives'),
                    'scalarized': sol.get('scalarized'),
                    'valid': sol.get('valid')
                }
                for sol in pareto_front
            ],
            
            # ====== 完整评估数据库 ======
            'database': database,  # 这是关键!保存所有评估点
            
            # ====== 数据分析 ======
            'analysis': self._analyze_database(database),
            
            # ====== 额外元数据 ======
            'metadata': metadata or {}
        }
        
        # 生成文件名
        filename = f"{run_id}.json"
        filepath = self.save_dir / filename
        
        # 转换为JSON可序列化格式
        complete_data_serializable = convert_to_json_serializable(complete_data)
        
        # 保存JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(complete_data_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n[ResultManager] [OK] 完整运行数据已保存:")
        print(f"  文件: {filepath}")
        print(f"  总评估数: {len(database)}")
        print(f"  有效评估: {statistics.get('valid_evaluations', 'N/A')}")
        print(f"  最优标量值: {best_solution.get('scalarized', 'N/A'):.4f}")
        
        return str(filepath)
    
    def _analyze_database(self, database: List[Dict]) -> Dict:
        """
        分析评估数据库,生成统计信息
        
        参数:
            database: 完整的评估数据
        
        返回:
            统计分析结果
        """
        
        if not database:
            return {}
        
        # 分离有效和无效评估
        valid_evals = [e for e in database if e.get('valid', True)]
        invalid_evals = [e for e in database if not e.get('valid', True)]
        
        analysis = {
            'total_evaluations': len(database),
            'valid_count': len(valid_evals),
            'invalid_count': len(invalid_evals),
            'validity_rate': len(valid_evals) / len(database) if database else 0
        }
        
        if valid_evals:
            # 提取目标值
            times = [e['objectives']['time'] for e in valid_evals]
            temps = [e['objectives']['temp'] for e in valid_evals]
            agings = [e['objectives']['aging'] for e in valid_evals]
            scalarizeds = [e['scalarized'] for e in valid_evals]
            
            # 参数分布
            current1s = [e['params']['current1'] for e in valid_evals]
            charging_numbers = [e['params']['charging_number'] for e in valid_evals]
            current2s = [e['params']['current2'] for e in valid_evals]
            
            analysis.update({
                'objectives': {
                    'time': {
                        'min': float(np.min(times)),
                        'max': float(np.max(times)),
                        'mean': float(np.mean(times)),
                        'std': float(np.std(times))
                    },
                    'temp': {
                        'min': float(np.min(temps)),
                        'max': float(np.max(temps)),
                        'mean': float(np.mean(temps)),
                        'std': float(np.std(temps))
                    },
                    'aging': {
                        'min': float(np.min(agings)),
                        'max': float(np.max(agings)),
                        'mean': float(np.mean(agings)),
                        'std': float(np.std(agings))
                    }
                },
                'scalarized': {
                    'best': float(np.min(scalarizeds)),
                    'worst': float(np.max(scalarizeds)),
                    'mean': float(np.mean(scalarizeds)),
                    'std': float(np.std(scalarizeds)),
                    'median': float(np.median(scalarizeds))
                },
                'parameters': {
                    'current1': {
                        'min': float(np.min(current1s)),
                        'max': float(np.max(current1s)),
                        'mean': float(np.mean(current1s))
                    },
                    'charging_number': {
                        'min': int(np.min(charging_numbers)),
                        'max': int(np.max(charging_numbers)),
                        'mean': float(np.mean(charging_numbers))
                    },
                    'current2': {
                        'min': float(np.min(current2s)),
                        'max': float(np.max(current2s)),
                        'mean': float(np.mean(current2s))
                    }
                }
            })
        
        return analysis
    
    def load_historical_data(
        self,
        n_recent: Optional[int] = None,
        pattern: str = "*.json"
    ) -> List[Dict]:
        """
        加载历史优化结果
        
        参数:
            n_recent: 加载最近n次运行(None=全部)
            pattern: 文件名匹配模式
        
        返回:
            历史数据列表
        """
        
        # 查找所有结果文件
        result_files = sorted(
            self.save_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # 最新的在前
        )
        
        # 限制数量
        if n_recent is not None:
            result_files = result_files[:n_recent]
        
        # 加载数据
        historical_data = []
        failed_files = []
        for filepath in result_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 验证数据完整性
                    if 'database' in data and 'best_solution' in data:
                        historical_data.append(data)
                    else:
                        failed_files.append((filepath.name, "格式不完整(缺少database或best_solution)"))
                        
            except json.JSONDecodeError as e:
                failed_files.append((filepath.name, f"JSON解析错误: {str(e)[:50]}"))
            except Exception as e:
                failed_files.append((filepath.name, f"其他错误: {str(e)[:50]}"))
        
        if failed_files:
            print(f"[警告] {len(failed_files)} 个文件加载失败(可能是旧格式或损坏):")
            for fname, error in failed_files[:3]:  # 只显示前3个
                print(f"  - {fname}: {error}")
            if len(failed_files) > 3:
                print(f"  - ... 还有 {len(failed_files)-3} 个文件")
        
        print(f"[ResultManager] 成功加载 {len(historical_data)} 个历史运行数据")
        return historical_data
    
    def get_top_k_solutions(
        self,
        historical_data: List[Dict],
        k: int = 10,
        metric: str = 'scalarized'
    ) -> List[Dict]:
        """
        获取历史top-k最优解
        
        参数:
            historical_data: 历史数据列表
            k: 返回前k个
            metric: 排序指标 ('scalarized', 'time', 'temp', 'aging')
        
        返回:
            top-k解列表
        """
        
        all_solutions = []
        
        # 从所有历史运行中收集有效解
        for run_data in historical_data:
            database = run_data.get('database', [])
            for eval_point in database:
                if eval_point.get('valid', True):
                    all_solutions.append(eval_point)
        
        if not all_solutions:
            return []
        
        # 排序
        if metric == 'scalarized':
            all_solutions.sort(key=lambda x: x.get('scalarized', float('inf')))
        elif metric in ['time', 'temp', 'aging']:
            all_solutions.sort(key=lambda x: x.get('objectives', {}).get(metric, float('inf')))
        else:
            raise ValueError(f"未知的metric: {metric}")
        
        return all_solutions[:k]
    
    def get_worst_k_solutions(
        self,
        historical_data: List[Dict],
        k: int = 10,
        metric: str = 'scalarized'
    ) -> List[Dict]:
        """
        获取历史worst-k最差解(用于避免)
        
        参数:
            historical_data: 历史数据列表
            k: 返回最差的k个
            metric: 排序指标
        
        返回:
            worst-k解列表
        """
        
        all_solutions = []
        
        for run_data in historical_data:
            database = run_data.get('database', [])
            for eval_point in database:
                # 包括无效解
                all_solutions.append(eval_point)
        
        if not all_solutions:
            return []
        
        # 反向排序
        if metric == 'scalarized':
            all_solutions.sort(key=lambda x: x.get('scalarized', -float('inf')), reverse=True)
        elif metric in ['time', 'temp', 'aging']:
            all_solutions.sort(key=lambda x: x.get('objectives', {}).get(metric, -float('inf')), reverse=True)
        else:
            raise ValueError(f"未知的metric: {metric}")
        
        return all_solutions[:k]
    
    def get_random_solutions(
        self,
        historical_data: List[Dict],
        k: int = 5,
        valid_only: bool = True
    ) -> List[Dict]:
        """
        随机采样k个历史解
        
        参数:
            historical_data: 历史数据
            k: 采样数量
            valid_only: 是否只采样有效解
        
        返回:
            随机解列表
        """
        
        all_solutions = []
        
        for run_data in historical_data:
            database = run_data.get('database', [])
            for eval_point in database:
                if valid_only:
                    if eval_point.get('valid', True):
                        all_solutions.append(eval_point)
                else:
                    all_solutions.append(eval_point)
        
        if not all_solutions:
            return []
        
        # 随机采样
        k = min(k, len(all_solutions))
        indices = np.random.choice(len(all_solutions), size=k, replace=False)
        
        return [all_solutions[i] for i in indices]
    
    def get_statistics_summary(
        self,
        historical_data: List[Dict]
    ) -> Dict:
        """
        生成历史数据的统计摘要
        
        参数:
            historical_data: 历史数据列表
        
        返回:
            统计摘要字典
        """
        
        if not historical_data:
            return {}
        
        total_runs = len(historical_data)
        total_evaluations = sum(
            len(run.get('database', []))
            for run in historical_data
        )
        
        # 收集所有最优解
        all_best_solutions = [
            run['best_solution']
            for run in historical_data
            if 'best_solution' in run and run['best_solution'].get('valid', True)
        ]
        
        if not all_best_solutions:
            return {
                'total_runs': total_runs,
                'total_evaluations': total_evaluations
            }
        
        # 全局最优
        global_best = min(all_best_solutions, key=lambda x: x.get('scalarized', float('inf')))
        
        # 目标值范围
        best_times = [s['objectives']['time'] for s in all_best_solutions]
        best_temps = [s['objectives']['temp'] for s in all_best_solutions]
        best_agings = [s['objectives']['aging'] for s in all_best_solutions]
        
        return {
            'total_runs': total_runs,
            'total_evaluations': total_evaluations,
            'global_best': global_best,
            'objective_ranges': {
                'time': {'min': min(best_times), 'max': max(best_times), 'mean': np.mean(best_times)},
                'temp': {'min': min(best_temps), 'max': max(best_temps), 'mean': np.mean(best_temps)},
                'aging': {'min': min(best_agings), 'max': max(best_agings), 'mean': np.mean(best_agings)}
            },
            'scalarized_range': {
                'best': global_best['scalarized'],
                'mean': np.mean([s['scalarized'] for s in all_best_solutions]),
                'worst': max(s['scalarized'] for s in all_best_solutions)
            }
        }


# ============ 测试代码 ============
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("测试 ResultManager")
    print("=" * 80)
    
    # 创建管理器
    manager = ResultManager(save_dir='./test_results')
    
    # 模拟优化数据
    mock_database = [
        {
            'eval_id': 1,
            'params': {'current1': 4.5, 'charging_number': 10, 'current2': 2.8},
            'objectives': {'time': 38, 'temp': 304.5, 'aging': 0.0015},
            'normalized': {'time': 0.2, 'temp': 0.1, 'aging': 0.15},
            'scalarized': 0.16,
            'valid': True,
            'source': 'warmstart'
        },
        {
            'eval_id': 2,
            'params': {'current1': 5.2, 'charging_number': 8, 'current2': 3.1},
            'objectives': {'time': 32, 'temp': 307.2, 'aging': 0.0028},
            'normalized': {'time': 0.1, 'temp': 0.35, 'aging': 0.40},
            'scalarized': 0.28,
            'valid': True,
            'source': 'bo'
        },
        {
            'eval_id': 3,
            'params': {'current1': 6.0, 'charging_number': 15, 'current2': 4.0},
            'objectives': {'time': 28, 'temp': 312.0, 'aging': 0.0055},
            'normalized': {'time': 0.05, 'temp': 0.95, 'aging': 0.95},
            'scalarized': 0.85,
            'valid': False,  # 违反温度约束
            'source': 'bo'
        }
    ]
    
    mock_best = mock_database[0]
    mock_pareto = [mock_database[0], mock_database[1]]
    
    mock_config = {
        'n_warmstart': 5,
        'n_iterations': 20,
        'llm_model': 'gpt-3.5-turbo'
    }
    
    mock_stats = {
        'total_evaluations': 3,
        'valid_evaluations': 2,
        'pareto_count': 2
    }
    
    # 保存数据
    filepath = manager.save_optimization_run(
        run_id='test_run_001',
        database=mock_database,
        best_solution=mock_best,
        pareto_front=mock_pareto,
        config=mock_config,
        statistics=mock_stats
    )
    
    print(f"\n[OK] 测试数据已保存到: {filepath}")
    
    # 测试加载
    historical = manager.load_historical_data(n_recent=1)
    print(f"\n[OK] 加载了 {len(historical)} 个历史文件")
    
    # 测试查询
    top_solutions = manager.get_top_k_solutions(historical, k=2)
    print(f"\n[OK] Top-2 solutions: {len(top_solutions)} 个")
    
    print("\n" + "=" * 80)
    print("ResultManager 测试完成")
    print("=" * 80)
