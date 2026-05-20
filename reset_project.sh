#!/bin/zsh

# reset_project.sh - Clean up all generated artifacts for a fresh start.

echo "Resetting project state..."

# 1. Clear Optuna databases
if ls results/logs/*.db 1> /dev/null 2>&1; then
    echo "Deleting Optuna databases..."
    rm results/logs/*.db
fi

# 2. Clear saved models
echo "Clearing models directory..."
rm -f models/*

# 3. Clear results subdirectories (keeping the structure)
echo "Clearing results subdirectories..."
rm -f results/figures/*
rm -f results/tables/*
rm -f results/metrics/*
rm -f results/shap/*
rm -f results/logs/*.log
rm -f results/logs/*.txt

echo "Project reset complete (processed data preserved). Ready for a fresh run."
