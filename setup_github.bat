@echo off
echo Setting up Git repository for Elderly Care Multi-Agent System...

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo Git is not installed. Please install Git first: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Git is installed.

REM Get GitHub details
set /p github_username=Enter your GitHub username: 
set /p repo_name=Enter repository name (press Enter for default 'elderly-care-multiagent-system'): 

if "%repo_name%"=="" set repo_name=elderly-care-multiagent-system

set repo_url=https://github.com/%github_username%/%repo_name%.git

echo Repository URL: %repo_url%

REM Initialize Git if not already done
if not exist ".git" (
    echo Initializing Git repository...
    git init
) else (
    echo Git repository already exists.
)

REM Add files
echo Adding files to Git...
git add .

REM Commit
echo Creating initial commit...
git commit -m "Initial commit: Elderly Care Multi-Agent AI System with ML-powered health monitoring"

REM Add remote
echo Adding remote origin...
git remote add origin %repo_url% 2>nul
if errorlevel 1 (
    git remote set-url origin %repo_url%
)

REM Rename branch to main
git branch -M main

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo Failed to push. You may need to authenticate or check repository permissions.
    echo Try running: git push -u origin main
) else (
    echo Successfully pushed to GitHub!
    echo Your repository is available at: https://github.com/%github_username%/%repo_name%
)

pause
