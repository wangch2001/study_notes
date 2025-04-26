#!/usr/bin/env zsh

# ==============================================
# macOS Git 自动提交脚本 (Zsh 版本)
# 功能：自动添加、提交并推送当前 Git 仓库的更改
# ==============================================

# 启用错误检查
set -e

# 颜色定义
autoload -U colors && colors
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
RESET="\033[0m"

# 检查是否在 Git 仓库中
check_git_repo() {
  if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "${RED}错误：当前目录不是 Git 仓库${RESET}"
    exit 1
  fi
}

# 获取当前分支
get_current_branch() {
  git branch --show-current
}

# 检查是否有未提交的更改
check_for_changes() {
  if [[ -z $(git status --porcelain) ]]; then
    echo "${YELLOW}没有检测到任何更改，无需提交。${RESET}"
    exit 0
  fi
}

# 生成智能提交信息
generate_commit_message() {
  local last_commit_msg=$(git log -1 --pretty=%B 2> /dev/null | head -n 1)
  local changed_files=$(git diff --name-only --cached)
  
  # 如果有特定文件类型更改，生成更有意义的提交信息
  if [[ $changed_files =~ "\.swift$" ]]; then
    echo "iOS: $(date '+%Y-%m-%d %H:%M:%S') 更新"
  elif [[ $changed_files =~ "\.js$" ]]; then
    echo "前端: $(date '+%Y-%m-%d %H:%M:%S') 更新"
  elif [[ $last_commit_msg ]]; then
    # 如果与上次提交信息相似，则增加序号
    if [[ $last_commit_msg =~ "^(.*) \((\d+)\)$" ]]; then
      local base_msg=${match[1]}
      local count=$((match[2] + 1))
      echo "${base_msg} (${count})"
    elif [[ $last_commit_msg =~ "^(.*)$" ]]; then
      echo "${match[1]} (2)"
    fi
  else
    echo "自动提交于 $(date '+%Y-%m-%d %H:%M:%S')"
  fi
}

# 主函数
main() {
  echo "${BLUE}=== macOS Git 自动提交脚本 ===${RESET}"
  
  check_git_repo
  
  local current_branch=$(get_current_branch)
  echo "当前分支: ${GREEN}${current_branch}${RESET}"
  
  check_for_changes
  
  # 添加所有更改
  echo -n "添加所有更改..."
  git add -A
  echo " ${GREEN}完成${RESET}"
  
  # 生成提交信息
  local commit_message=$(generate_commit_message)
  echo "提交信息: ${YELLOW}${commit_message}${RESET}"
  
  # 执行提交
  echo -n "正在提交..."
  git commit -m "${commit_message}" > /dev/null
  echo " ${GREEN}完成${RESET}"
  
  # 推送到远程仓库
  echo -n "正在推送到远程仓库..."
  git push origin ${current_branch} > /dev/null
  echo " ${GREEN}完成${RESET}"
  
  echo "${GREEN}✓ 已成功提交并推送到分支 ${current_branch}${RESET}"
  
  # macOS 通知
  if [[ $(command -v osascript) ]]; then
    osascript -e 'display notification "Git 提交成功" with title "自动 Git 提交"'
  fi
}

# 执行主函数
main "$@"
