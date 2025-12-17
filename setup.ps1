# Quick Setup Script for AI Service
# Run this from the server/python-ai-service directory

Write-Host "╔═══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Hippiekit AI Service - Quick Setup             ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Error: requirements.txt not found!" -ForegroundColor Red
    Write-Host "Please run this script from the server/python-ai-service directory" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
Write-Host "Dependencies installed" -ForegroundColor Green
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host ".env file not found" -ForegroundColor Yellow
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host ".env file created" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: Please edit .env and add your Pinecone API key!" -ForegroundColor Red
    Write-Host "   1. Sign up at https://app.pinecone.io" -ForegroundColor Yellow
    Write-Host "   2. Create an API key" -ForegroundColor Yellow
    Write-Host "   3. Add it to the .env file" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Press Enter when you've added your API key (or 'skip' to continue)"
    if ($response -eq "skip") {
        Write-Host "Skipping API key setup - service may not work!" -ForegroundColor Yellow
    }
} else {
    Write-Host ".env file exists" -ForegroundColor Green
}
Write-Host ""

Write-Host "╔═══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Setup Complete!                                 ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Make sure your Pinecone API key is in .env" -ForegroundColor White
Write-Host "2. Run: python main.py" -ForegroundColor White
Write-Host "3. Index products: curl -X POST 'http://localhost:8001/index/products?max_products=10'" -ForegroundColor White
Write-Host ""
Write-Host "Would you like to start the service now? (y/n)" -ForegroundColor Yellow
$start = Read-Host

if ($start -eq "y" -or $start -eq "Y") {
    Write-Host "Starting AI service..." -ForegroundColor Green
    python main.py
} else {
    Write-Host "You can start the service later with: python main.py" -ForegroundColor Yellow
}
