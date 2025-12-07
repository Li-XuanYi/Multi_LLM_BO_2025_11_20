"""
验证result_manager的错误处理改进

测试内容:
1. 尝试加载所有历史文件(包括损坏的)
2. 验证容错机制工作正常
3. 显示哪些文件可用,哪些不可用
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BO', 'llmbo_core'))

from result_manager import ResultManager

print("\n" + "=" * 80)
print("测试ResultManager容错能力")
print("=" * 80)

# 创建ResultManager
manager = ResultManager(save_dir='./results')

# 尝试加载历史数据
print("\n[测试] 加载历史数据(包括损坏文件)...")
historical = manager.load_historical_data(n_recent=10)

print(f"\n[结果] 成功加载 {len(historical)} 个有效文件")

if historical:
    print("\n[详细信息] 已加载文件:")
    for i, data in enumerate(historical, 1):
        run_id = data.get('run_id', 'Unknown')
        n_evals = len(data.get('database', []))
        best_score = data.get('best_solution', {}).get('scalarized', 'N/A')
        print(f"  {i}. {run_id}: {n_evals} 个评估点, 最优={best_score}")
else:
    print("\n[提示] 没有有效的历史文件")
    print("  原因可能是:")
    print("  1. results/ 目录为空")
    print("  2. 所有文件都是旧格式(已备份到results_backup/)")
    print("  3. JSON文件损坏")
    print("\n  解决方法:")
    print("  - 运行一次新的优化,生成新格式的结果文件")
    print("  - 或使用 test_historical_warmstart.py 生成测试数据")

print("\n" + "=" * 80)
print("✓ 容错测试完成")
print("=" * 80)
