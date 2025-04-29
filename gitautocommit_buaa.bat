@echo off
:: 自动 Git 提交脚本 (Windows 版本)

:: 检查是否在 Git 仓库中
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo 错误：当前目录不是 Git 仓库
    pause
    exit /b 1
)

:: 获取当前分支
for /f "delims=" %%b in ('git symbolic-ref --short HEAD') do set "current_branch=%%b"

:: 添加所有更改
git add .

:: 检查是否有更改
git status --porcelain >nul 2>&1
if errorlevel 1 (
    echo 没有检测到任何更改，无需提交。
    pause
    exit /b 0
)

:: 提交信息
for /f "tokens=1-6 delims=/: " %%a in ('time /t') do set "time=%%a:%%b"
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "date=20%%c-%%a-%%b"
set "commit_message=update on buaa %date% %time%"

:: 执行提交
git commit -m "%commit_message%"

:: 推送到远程仓库
git push origin %current_branch%

echo 已成功提交并推送到分支 %current_branch%
pause