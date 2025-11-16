#!/bin/bash

# Project Cleanup Script
# This script removes unnecessary files and folders to reduce project size

echo "🧹 Starting project cleanup..."
echo ""

# Function to get folder size
get_size() {
    du -sh "$1" 2>/dev/null | cut -f1
}

# Function to remove directory safely
remove_dir() {
    if [ -d "$1" ]; then
        size=$(get_size "$1")
        echo "  Removing $1 ($size)..."
        rm -rf "$1"
        echo "  ✓ Removed"
    fi
}

# Function to remove files matching pattern
remove_files() {
    pattern="$1"
    desc="$2"
    count=$(find . -name "$pattern" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "  Removing $count $desc files..."
        find . -name "$pattern" -type f -delete 2>/dev/null
        echo "  ✓ Removed"
    fi
}

echo "📦 Cleaning Python cache files..."
remove_files "*.pyc" "Python bytecode"
remove_files "*.pyo" "Python optimized"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo ""

echo "📦 Cleaning build artifacts..."
remove_dir ".next"
remove_dir "out"
remove_dir "build"
remove_dir "dist"
echo ""

echo "📦 Cleaning test files..."
remove_dir "test-results"
remove_dir "tests/uploads"
remove_files "test_*.py" "test Python"
remove_files "test_*.js" "test JavaScript"
remove_files "test_*.sh" "test shell"
echo ""

echo "📦 Cleaning temporary files..."
remove_files "*.swp" "Vim swap"
remove_files "*.swo" "Vim swap"
remove_files ".DS_Store" "macOS"
remove_files "*.log" "log"
echo ""

echo "📦 Cleaning uploaded detection images..."
if [ -d "uploads" ]; then
    count=$(find uploads -type f 2>/dev/null | wc -l | tr -d ' ')
    size=$(get_size "uploads")
    echo "  Found $count files in uploads/ ($size)"
    read -p "  Remove all uploaded images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf uploads/*
        echo "  ✓ Cleared uploads/"
    else
        echo "  ⊘ Skipped"
    fi
fi

if [ -d "Backend/uploads" ]; then
    count=$(find Backend/uploads -type f 2>/dev/null | wc -l | tr -d ' ')
    size=$(get_size "Backend/uploads")
    echo "  Found $count files in Backend/uploads/ ($size)"
    read -p "  Remove all uploaded images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf Backend/uploads/*
        echo "  ✓ Cleared Backend/uploads/"
    else
        echo "  ⊘ Skipped"
    fi
fi
echo ""

echo "📦 Cleaning documentation files..."
remove_files "*_FIXED.md" "fix documentation"
remove_files "*_APPLIED.md" "applied documentation"
remove_files "*_GUIDE.md" "guide documentation"
remove_files "*_SUMMARY.md" "summary documentation"
echo ""

echo "✨ Cleanup complete!"
echo ""
echo "📊 Current project size:"
echo "  Total: $(get_size .)"
echo "  node_modules: $(get_size node_modules)"
echo "  Backend/venv: $(get_size Backend/venv)"
echo ""
echo "💡 Note: node_modules and Backend/venv are necessary for the project to run."
echo "   They are excluded from git by .gitignore"
echo ""
echo "🚀 To reinstall dependencies if needed:"
echo "   Frontend: npm install"
echo "   Backend: cd Backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
