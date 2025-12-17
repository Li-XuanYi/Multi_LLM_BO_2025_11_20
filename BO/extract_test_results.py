"""
提取测试关键信息
"""
import subprocess
import sys

# 运行测试并提取关键行
result = subprocess.run(
    [sys.executable, 'd:/Users/aa133/Desktop/BO_Multi_11_12/BO/test_fixes.py'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

output = result.stdout + result.stderr

# 查找关键标记
key_markers = [
    "检查点 1",
    "gradient_compute_interval",
    "修复1",
    "检查点 2", 
    "Coupling Matrix Estimator",
    "诊断报告",
    "DATA-DRIVEN",
    "检查点 3",
    "LLM-Enhanced EI",
    "W_LLM",
    "SUCCESS",
    "Test Completed"
]

print("\n" + "="*80)
print("关键信息提取")
print("="*80)

lines = output.split('\n')
for i, line in enumerate(lines):
    # 检查是否包含关键标记
    if any(marker in line for marker in key_markers):
        # 打印该行及上下文
        start = max(0, i-1)
        end = min(len(lines), i+2)
        for j in range(start, end):
            print(lines[j])
        print()
