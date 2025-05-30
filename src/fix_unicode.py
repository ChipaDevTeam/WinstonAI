#!/usr/bin/env python3
"""
Script to fix Unicode characters in Python files for Windows console compatibility
"""

import re

def fix_unicode_in_file(filepath):
    """Replace Unicode emojis with ASCII equivalents"""
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacements
    replacements = {
        '🚀': '[GPU]',
        '💾': '[MEMORY]',
        '⚠️': '[WARNING]',
        '📊': '[STATS]',
        '🎯': '[TARGET]',
        '⏰': '[TIME]',
        '🔧': '[CONFIG]',
        '📈': '[GROWTH]',
        '🧠': '[AI]',
        '🔢': '[DATA]',
        '🔄': '[LOADING]',
        '✅': '[OK]',
        '📂': '[FILE]',
        '🆕': '[NEW]',
        '🧪': '[TEST]',
        '💰': '[PROFIT]',
        '💹': '[ROI]',
        '📋': '[INFO]',
        '🤔': '[QUESTION]',
        '❌': '[ERROR]',
        '🔍': '[SEARCH]'
    }
    
    # Apply replacements
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed Unicode characters in {filepath}")

if __name__ == "__main__":
    files_to_fix = [
        'train_gpu_optimized.py',
        'gpu_benchmark.py',
        'gpu_monitor.py',
        'quick_start_gpu.py',
        'ultra_live_trading_bot.py'
    ]
    
    for file in files_to_fix:
        try:
            fix_unicode_in_file(file)
        except FileNotFoundError:
            print(f"File not found: {file}")
        except Exception as e:
            print(f"Error fixing {file}: {e}")
    
    print("Unicode fix complete!")
