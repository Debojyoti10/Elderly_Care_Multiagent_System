# GitHub Setup Script for Elderly Care Project
# Run this script in PowerShell to initialize Git and push to GitHub

Write-Host "🚀 Setting up Git repository for Elderly Care Multi-Agent System..." -ForegroundColor Green

# Check if Git is installed
try {
    git --version | Out-Null
    Write-Host "✅ Git is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed. Please install Git first: https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Prompt for GitHub repository URL
Write-Host "`n📝 Please provide your GitHub repository details:" -ForegroundColor Yellow
$githubUsername = Read-Host "Enter your GitHub username"
$repoName = Read-Host "Enter repository name (default: elderly-care-multiagent-system)"

if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "elderly-care-multiagent-system"
}

$repoUrl = "https://github.com/$githubUsername/$repoName.git"

Write-Host "`n🔧 Repository URL: $repoUrl" -ForegroundColor Cyan

# Initialize Git repository
Write-Host "`n📁 Initializing Git repository..." -ForegroundColor Yellow

if (Test-Path ".git") {
    Write-Host "⚠️  Git repository already exists. Skipping initialization." -ForegroundColor Yellow
} else {
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
}

# Add all files
Write-Host "`n📦 Adding files to Git..." -ForegroundColor Yellow
git add .

# Check status
Write-Host "`n📊 Git status:" -ForegroundColor Cyan
git status --short

# Commit changes
Write-Host "`n💾 Creating initial commit..." -ForegroundColor Yellow
$commitMessage = "Initial commit: Elderly Care Multi-Agent AI System

Features:
- ML-powered health monitoring agent
- Anomaly detection using Isolation Forest
- Real-time health prediction and risk assessment
- Event-driven multi-agent architecture
- Comprehensive testing suite
- Health insights and recommendations engine

Technologies: Python, Scikit-learn, Pandas, NumPy"

git commit -m $commitMessage

# Add remote origin
Write-Host "`n🔗 Adding remote origin..." -ForegroundColor Yellow
try {
    git remote add origin $repoUrl
    Write-Host "✅ Remote origin added" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Remote origin already exists or failed to add" -ForegroundColor Yellow
    git remote set-url origin $repoUrl
    Write-Host "✅ Remote origin URL updated" -ForegroundColor Green
}

# Check if main or master branch
$currentBranch = git rev-parse --abbrev-ref HEAD
if ($currentBranch -ne "main") {
    Write-Host "`n🌿 Renaming branch to 'main'..." -ForegroundColor Yellow
    git branch -M main
}

# Push to GitHub
Write-Host "`n🚀 Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "Note: You may be prompted for your GitHub credentials" -ForegroundColor Cyan

try {
    git push -u origin main
    Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
    
    Write-Host "`n🎉 Repository setup complete!" -ForegroundColor Green
    Write-Host "🔗 Your repository is available at: https://github.com/$githubUsername/$repoName" -ForegroundColor Cyan
    
    Write-Host "`n📋 Next steps:" -ForegroundColor Yellow
    Write-Host "1. Visit your GitHub repository and add a description" -ForegroundColor White
    Write-Host "2. Consider adding topics/tags for better discoverability" -ForegroundColor White
    Write-Host "3. Set up GitHub Actions for CI/CD (optional)" -ForegroundColor White
    Write-Host "4. Add collaborators if working in a team" -ForegroundColor White
    
} catch {
    Write-Host "❌ Failed to push to GitHub. Please check:" -ForegroundColor Red
    Write-Host "1. Repository exists on GitHub" -ForegroundColor White
    Write-Host "2. You have push permissions" -ForegroundColor White
    Write-Host "3. Your Git credentials are correct" -ForegroundColor White
    Write-Host "4. Try: git push -u origin main --force (if needed)" -ForegroundColor White
}

Write-Host "`n📁 Project structure:" -ForegroundColor Cyan
Get-ChildItem -Recurse -Name | Where-Object { $_ -notlike "*__pycache__*" -and $_ -notlike "*.git*" } | Sort-Object

Write-Host "`n✨ Happy coding!" -ForegroundColor Green
