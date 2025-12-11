#!/bin/bash

# Jaithon Test Runner
# Runs all .jai test files in organized subdirectories

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$SCRIPT_DIR/Jaithon"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

run_tests_in_dir() {
    local dir=$1
    local category=$(basename "$dir")
    
    echo -e "\n${YELLOW}=== $category ===${NC}"
    
    for file in "$dir"/*.jai; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .jai)
            printf "  %-20s " "$filename"
            
            output=$("$BASE_DIR/jaithon" "$file" 2>&1)
            exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                echo -e "${GREEN}PASS${NC}"
                ((passed++))
            else
                echo -e "${RED}FAIL${NC}"
                echo "$output" | head -3 | sed 's/^/    /'
                ((failed++))
            fi
        fi
    done
}

echo "Jaithon Test Suite"
echo "=================="

for subdir in "$TEST_DIR"/*/; do
    if [ -d "$subdir" ]; then
        run_tests_in_dir "$subdir"
    fi
done

echo ""
echo "=================="
echo -e "Results: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"

if [ $failed -gt 0 ]; then
    exit 1
fi
