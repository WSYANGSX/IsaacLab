#!/bin/bash
 
# 要运行的 Python 脚本的路径
PYTHON_SCRIPT="/home/yangxf/my_projects/IsaacLab/local_projects/projects/skrl_train/cabinet/ddpg-dense.py"
 
# 运行 5 次
for i in {1..5}
do
    $ISAACSIM_PYTHON_EXE "$PYTHON_SCRIPT"
done
