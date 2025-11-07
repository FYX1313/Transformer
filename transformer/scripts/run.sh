#!/bin/bash
# 激活虚拟环境（如果使用了虚拟环境，替换为你的环境路径）
# source /path/to/your/venv/bin/activate  # Linux/Mac
# . /path/to/your/venv/Scripts/activate   # Windows（PowerShell）

# 运行主程序（src/main.py）
python src/main.py

# 若需要指定设备（如强制使用CPU），可添加参数：
# python src/main.py --device cpu