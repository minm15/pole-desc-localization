#!/usr/bin/env bash
set -e

TARGET_DIRS=("eval1" "eval2")

echo "Starting ALL Experiments in: ${TARGET_DIRS[*]}..."

for dir in "${TARGET_DIRS[@]}"; do
    SCRIPT_PATH="./scripts/$dir"
    
    echo "========================================"
    echo "Processing Directory: $dir"
    echo "========================================"

    if [ -d "$SCRIPT_PATH" ]; then
        for script in "$SCRIPT_PATH"/*.sh; do
            if [ -f "$script" ]; then
                echo "----------------------------------------"
                echo "Running script: $script"
                
                chmod +x "$script"
                
                "$script"
            fi
        done
    else
        echo "Error: Directory $SCRIPT_PATH not found!"
        exit 1
    fi
done

echo "========================================"
echo "All experiments in eval1 and eval2 completed successfully."