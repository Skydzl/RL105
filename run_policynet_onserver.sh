#!/bin/bash
# Author :   Skydzl
# Description:  仅在服务器跑的时候用该文件，避免输出结果找不到，输出结果重定向在logs/output.log
nohup python -u train_worker_policynet.py > logs/output.log 2>&1