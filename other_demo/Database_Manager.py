"""
Database Manager Module
用于 LLM 增强的分解多目标贝叶斯优化 (LLM-DMOBO)

功能：
1. 持久化存储优化历史数据
2. 支持增量保存和加载
3. 数据导出（CSV, JSON, Pickle）
4. 迁移学习支持
5. 数据分析和可视化准备

作者: Claude AI Assistant
日期: 2025-01-12
版本: v1.0
"""

import os
import json
import pickle
import csv
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path


class DatabaseManager:
    """
    优化数据库管理器
    
    负责历史数据的持久化存储、加载和管理
    """
    
    def __init__(
        self,
        save_dir: str = "./optimization_data",
        auto_save: bool = True,
        save_interval: int = 10,
        verbose: bool = True
    ):
        """
        初始化数据库管理器
        
        参数:
            save_dir: 数据保存目录
            auto_save: 是否自动保存
            save_interval: 自动保存间隔（每N次评估）
            verbose: 是否打印详细日志
        """
        self.save_dir = Path(save_dir)
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.verbose = verbose
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.database = []  # 主数据库
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'total_evaluations': 0,
            'valid_evaluations': 0,
            'optimization_settings': {}
        }
        
        # 保存计数器
        self._save_counter = 0
        
        if self.verbose:
            print("=" * 70)
            print("数据库管理器已初始化")
            print("=" * 70)
            print(f"保存目录: {self.save_dir.absolute()}")
            print(f"自动保存: {auto_save} (间隔={save_interval})")
            print("=" * 70)
    
    def add_evaluation(
        self,
        eval_data: Dict[str, Any],
        trigger_save: bool = False
    ) -> None:
        """
        添加单次评估记录
        
        参数:
            eval_data: 评估数据字典
            trigger_save: 是否触发保存
        """
        # 添加时间戳
        eval_data['timestamp'] = datetime.now().isoformat()
        
        # 添加到数据库
        self.database.append(eval_data)
        
        # 更新元数据
        self.metadata['total_evaluations'] = len(self.database)
        self.metadata['last_updated'] = datetime.now().isoformat()
        if eval_data.get('valid', False):
            self.metadata['valid_evaluations'] += 1
        
        # 自动保存检查
        self._save_counter += 1
        if self.auto_save and self._save_counter >= self.save_interval:
            self.save_database()
            self._save_counter = 0
        elif trigger_save:
            self.save_database()
    
    def bulk_add_evaluations(
        self,
        eval_data_list: List[Dict[str, Any]],
        trigger_save: bool = True
    ) -> None:
        """
        批量添加评估记录
        
        参数:
            eval_data_list: 评估数据列表
            trigger_save: 是否触发保存
        """
        for eval_data in eval_data_list:
            self.add_evaluation(eval_data, trigger_save=False)
        
        if trigger_save:
            self.save_database()
    
    def save_database(
        self,
        format: str = 'all',
        custom_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        保存数据库
        
        参数:
            format: 保存格式 ('json', 'pickle', 'csv', 'all')
            custom_name: 自定义文件名前缀
        
        返回:
            保存的文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = custom_name or f"optimization_db_{timestamp}"
        
        saved_files = {}
        
        # JSON 格式
        if format in ['json', 'all']:
            json_path = self.save_dir / f"{base_name}.json"
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': self.metadata,
                        'database': self.database
                    }, f, indent=2, default=str)
                saved_files['json'] = str(json_path)
                if self.verbose:
                    print(f"✅ 已保存 JSON: {json_path}")
            except Exception as e:
                print(f"❌ JSON 保存失败: {e}")
        
        # Pickle 格式（用于快速加载）
        if format in ['pickle', 'all']:
            pickle_path = self.save_dir / f"{base_name}.pkl"
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump({
                        'metadata': self.metadata,
                        'database': self.database
                    }, f)
                saved_files['pickle'] = str(pickle_path)
                if self.verbose:
                    print(f"✅ 已保存 Pickle: {pickle_path}")
            except Exception as e:
                print(f"❌ Pickle 保存失败: {e}")
        
        # CSV 格式（用于分析）
        if format in ['csv', 'all']:
            csv_path = self.save_dir / f"{base_name}.csv"
            try:
                self._save_as_csv(csv_path)
                saved_files['csv'] = str(csv_path)
                if self.verbose:
                    print(f"✅ 已保存 CSV: {csv_path}")
            except Exception as e:
                print(f"❌ CSV 保存失败: {e}")
        
        return saved_files
    
    def _save_as_csv(self, csv_path: Path) -> None:
        """
        保存为 CSV 格式
        """
        if len(self.database) == 0:
            return
        
        # 提取列名
        fieldnames = set()
        for record in self.database:
            # 展平嵌套字典
            flat_record = self._flatten_dict(record)
            fieldnames.update(flat_record.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        # 写入 CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in self.database:
                flat_record = self._flatten_dict(record)
                writer.writerow(flat_record)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        展平嵌套字典
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def load_database(
        self,
        file_path: str,
        format: Optional[str] = None
    ) -> bool:
        """
        加载数据库
        
        参数:
            file_path: 文件路径
            format: 文件格式（可选，会自动推断）
        
        返回:
            是否加载成功
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return False
        
        # 自动推断格式
        if format is None:
            format = file_path.suffix[1:]  # 去掉 '.'
        
        try:
            if format == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif format in ['pkl', 'pickle']:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f"❌ 不支持的格式: {format}")
                return False
            
            # 加载数据
            self.database = data['database']
            self.metadata = data['metadata']
            
            if self.verbose:
                print(f"✅ 成功加载数据库: {file_path}")
                print(f"   评估记录数: {len(self.database)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def merge_database(
        self,
        other_db_path: str,
        remove_duplicates: bool = True
    ) -> int:
        """
        合并另一个数据库
        
        参数:
            other_db_path: 另一个数据库文件路径
            remove_duplicates: 是否移除重复记录
        
        返回:
            合并的记录数
        """
        other_db_path = Path(other_db_path)
        
        # 加载另一个数据库
        try:
            with open(other_db_path, 'rb') as f:
                other_data = pickle.load(f)
            
            other_database = other_data['database']
            
            # 合并
            original_count = len(self.database)
            
            if remove_duplicates:
                # 简单的去重：基于参数
                existing_params = set()
                for record in self.database:
                    params = record.get('params', {})
                    param_tuple = (params.get('current1'), params.get('charging_number'), params.get('current2'))
                    existing_params.add(param_tuple)
                
                for record in other_database:
                    params = record.get('params', {})
                    param_tuple = (params.get('current1'), params.get('charging_number'), params.get('current2'))
                    if param_tuple not in existing_params:
                        self.database.append(record)
                        existing_params.add(param_tuple)
            else:
                self.database.extend(other_database)
            
            merged_count = len(self.database) - original_count
            
            # 更新元数据
            self.metadata['total_evaluations'] = len(self.database)
            self.metadata['last_updated'] = datetime.now().isoformat()
            
            if self.verbose:
                print(f"✅ 成功合并 {merged_count} 条记录")
            
            return merged_count
            
        except Exception as e:
            print(f"❌ 合并失败: {e}")
            return 0
    
    def get_database(self) -> List[Dict[str, Any]]:
        """
        获取完整数据库
        """
        return self.database
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        """
        if len(self.database) == 0:
            return {"message": "数据库为空"}
        
        # 提取目标值
        times = [r['objectives']['time'] for r in self.database if 'objectives' in r]
        temps = [r['objectives']['temp'] for r in self.database if 'objectives' in r]
        agings = [r['objectives']['aging'] for r in self.database if 'objectives' in r]
        scalarizeds = [r['scalarized'] for r in self.database if 'scalarized' in r]
        
        stats = {
            'metadata': self.metadata,
            'objectives': {
                'time': {
                    'min': np.min(times) if times else None,
                    'max': np.max(times) if times else None,
                    'mean': np.mean(times) if times else None,
                    'std': np.std(times) if times else None
                },
                'temp': {
                    'min': np.min(temps) if temps else None,
                    'max': np.max(temps) if temps else None,
                    'mean': np.mean(temps) if temps else None,
                    'std': np.std(temps) if temps else None
                },
                'aging': {
                    'min': np.min(agings) if agings else None,
                    'max': np.max(agings) if agings else None,
                    'mean': np.mean(agings) if agings else None,
                    'std': np.std(agings) if agings else None
                },
                'scalarized': {
                    'min': np.min(scalarizeds) if scalarizeds else None,
                    'max': np.max(scalarizeds) if scalarizeds else None,
                    'mean': np.mean(scalarizeds) if scalarizeds else None,
                    'std': np.std(scalarizeds) if scalarizeds else None
                }
            }
        }
        
        return stats
    
    def export_for_transfer_learning(
        self,
        output_path: Optional[str] = None,
        filter_valid_only: bool = True,
        top_n: Optional[int] = None
    ) -> str:
        """
        导出用于迁移学习的数据
        
        参数:
            output_path: 输出路径（可选）
            filter_valid_only: 是否只导出有效评估
            top_n: 只导出最优的 N 个（基于标量化值）
        
        返回:
            导出文件路径
        """
        # 过滤数据
        export_data = self.database.copy()
        
        if filter_valid_only:
            export_data = [r for r in export_data if r.get('valid', False)]
        
        if top_n is not None:
            export_data = sorted(export_data, key=lambda x: x.get('scalarized', float('inf')))
            export_data = export_data[:top_n]
        
        # 保存
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.save_dir / f"transfer_learning_{timestamp}.pkl"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'original_total': len(self.database),
                    'exported_count': len(export_data),
                    'filter_valid_only': filter_valid_only,
                    'top_n': top_n
                },
                'database': export_data
            }, f)
        
        if self.verbose:
            print(f"✅ 已导出 {len(export_data)} 条记录用于迁移学习: {output_path}")
        
        return str(output_path)
    
    def clear_database(self, confirm: bool = False) -> None:
        """
        清空数据库
        
        参数:
            confirm: 确认清空（防止误操作）
        """
        if not confirm:
            print("⚠️  请设置 confirm=True 以确认清空数据库")
            return
        
        self.database = []
        self.metadata['total_evaluations'] = 0
        self.metadata['valid_evaluations'] = 0
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        if self.verbose:
            print("✅ 数据库已清空")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("【测试】Database Manager")
    print("=" * 70)
    
    # 测试1: 创建数据库管理器
    print("\n【测试1】初始化")
    db_manager = DatabaseManager(
        save_dir="./test_optimization_data",
        auto_save=True,
        save_interval=5,
        verbose=True
    )
    
    # 测试2: 添加模拟数据
    print("\n【测试2】添加评估记录")
    for i in range(10):
        eval_data = {
            'eval_id': i + 1,
            'params': {
                'current1': 5.0 + np.random.rand(),
                'charging_number': 10 + np.random.randint(-2, 3),
                'current2': 3.0 + np.random.rand()
            },
            'objectives': {
                'time': 50 + np.random.randint(-5, 6),
                'temp': 305.0 + np.random.rand() * 5,
                'aging': 0.001 + np.random.rand() * 0.0005
            },
            'scalarized': 0.3 + np.random.rand() * 0.2,
            'valid': True,
            'source': 'simulation'
        }
        db_manager.add_evaluation(eval_data)
    
    print(f"✅ 已添加 10 条记录")
    
    # 测试3: 保存数据库
    print("\n【测试3】保存数据库")
    saved_files = db_manager.save_database(format='all', custom_name='test_db')
    print(f"保存的文件: {saved_files}")
    
    # 测试4: 统计信息
    print("\n【测试4】统计信息")
    stats = db_manager.get_statistics()
    print(f"总评估数: {stats['metadata']['total_evaluations']}")
    print(f"时间统计: {stats['objectives']['time']}")
    print(f"标量化值统计: {stats['objectives']['scalarized']}")
    
    # 测试5: 加载数据库
    print("\n【测试5】加载数据库")
    db_manager2 = DatabaseManager(save_dir="./test_optimization_data", verbose=True)
    success = db_manager2.load_database(saved_files['pickle'])
    if success:
        print(f"✅ 加载成功，记录数: {len(db_manager2.get_database())}")
    
    # 测试6: 导出迁移学习数据
    print("\n【测试6】导出迁移学习数据")
    transfer_file = db_manager.export_for_transfer_learning(
        filter_valid_only=True,
        top_n=5
    )
    print(f"迁移学习文件: {transfer_file}")
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)