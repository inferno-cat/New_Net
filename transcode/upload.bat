@echo off
REM 该脚本用于一键将本地项目上传到 GitHub
REM 设置本地项目路径，请根据实际情况修改
set project_path="D:\rawcode\NewCode\Sub_Net"
REM 切换到项目路径
cd /d %project_path%
REM 添加所有文件到暂存区
git add .
REM 提交修改，这里设置了一个默认的提交信息，你可以根据需要修改
git commit -m "自动提交：项目更新"
REM 推送代码到 GitHub
git push origin master
echo 代码上传完成！
pause