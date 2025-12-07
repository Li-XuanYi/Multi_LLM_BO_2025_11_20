"""
批量修复Python文件中的Unicode编码问题
将特殊符号替换为ASCII字符
"""
import os
import re

# 定义替换规则
REPLACEMENTS = {
    '✓': '[OK]',
    '✅': '[OK]',
    '❌': '[X]',
    '⚠️': '[Warning]',
    '⚠': '[Warning]',
    '🔥': '[!]',
    '📊': '[Data]',
    '⚡': '[Fast]',
    '∂': 'd',  # 偏导数符号
}

def fix_encoding_in_file(filepath):
    """修复单个文件的编码问题"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 替换特殊字符
        for old, new in REPLACEMENTS.items():
            content = content.replace(old, new)
        
        # 如果有修改,写回文件
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 修复: {filepath}")
            return True
        else:
            return False
    except Exception as e:
        print(f"✗ 失败: {filepath} - {e}")
        return False

def main():
    """主函数"""
    target_dir = './BO/llmbo_core'
    
    print("=" * 70)
    print("批量修复Python文件编码问题")
    print("=" * 70)
    
    if not os.path.exists(target_dir):
        print(f"错误: 目录 {target_dir} 不存在")
        return
    
    fixed_count = 0
    total_count = 0
    
    # 遍历所有Python文件
    for filename in os.listdir(target_dir):
        if filename.endswith('.py'):
            filepath = os.path.join(target_dir, filename)
            total_count += 1
            if fix_encoding_in_file(filepath):
                fixed_count += 1
    
    print("=" * 70)
    print(f"完成: 检查 {total_count} 个文件, 修复 {fixed_count} 个文件")
    print("=" * 70)

if __name__ == '__main__':
    main()
