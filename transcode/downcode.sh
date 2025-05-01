#!/bin/bash
# 该脚本用于一键克隆 GitHub 项目到服务器本地
# 设置 GitHub 仓库地址，请根据实际情况修改
git_repo="git@github.com:inferno-cat/New_Net.git"
# 设置克隆的目标目录，请根据实际情况修改
clone_dir="/home/share3/zc/file/New_Net"
# 克隆仓库
git clone $git_repo $clone_dir
echo "项目克隆完成！"