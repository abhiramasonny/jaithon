import subprocess
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

@dataclass
class TestResult:
    name: str
    passed: bool
    output: str
    expected: List[str]
    actual: List[str]
    error: Optional[str] = None
    time_ms: float = 0.0

def get_expected_output(filepath: Path) -> List[str]:
    expected = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Expected output"):
                match = re.search(r"Expected output.*?:\s*(.+)", line)
                if match:
                    values = match.group(1).split(", ")
                    expected = [v.strip() for v in values]
                break
    return expected

def run_test(jaithon_path: str, test_file: Path) -> TestResult:
    import time
    
    expected = get_expected_output(test_file)
    
    start = time.time()
    try:
        result = subprocess.run(
            [jaithon_path, str(test_file)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=test_file.parent.parent.parent
        )
        output = result.stdout + result.stderr
        elapsed = (time.time() - start) * 1000
        
        actual_lines = []
        for line in output.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('[line') and not line.startswith('VM Error') and not line.startswith('Compile Error'):
                actual_lines.append(line)
        
        has_error = "Error in __main__" in output or "segmentation fault" in output.lower()
        
        if has_error:
            error_match = re.search(r"Error in __main__.*?:\s*(.+)", output)
            error_msg = error_match.group(1) if error_match else "Unknown error"
            return TestResult(
                name=test_file.stem,
                passed=False,
                output=output,
                expected=expected,
                actual=actual_lines,
                error=error_msg,
                time_ms=elapsed
            )
        
        passed = True
        if expected:
            if len(actual_lines) < len(expected):
                passed = False
            else:
                for i, exp in enumerate(expected):
                    if i >= len(actual_lines):
                        passed = False
                        break
                    try:
                        exp_float = float(exp)
                        act_float = float(actual_lines[i])
                        if abs(exp_float - act_float) > 0.01:
                            passed = False
                            break
                    except ValueError:
                        if actual_lines[i] != exp:
                            passed = False
                            break
        
        return TestResult(
            name=test_file.stem,
            passed=passed,
            output=output,
            expected=expected,
            actual=actual_lines,
            time_ms=elapsed
        )
        
    except subprocess.TimeoutExpired:
        return TestResult(
            name=test_file.stem,
            passed=False,
            output="",
            expected=expected,
            actual=[],
            error="Timeout (>10s)"
        )
    except Exception as e:
        return TestResult(
            name=test_file.stem,
            passed=False,
            output="",
            expected=expected,
            actual=[],
            error=str(e)
        )

def print_result(result: TestResult, verbose: bool = False):
    status = f"{GREEN}PASS{RESET}" if result.passed else f"{RED}FAIL{RESET}"
    time_str = f"({result.time_ms:.1f}ms)" if result.time_ms > 0 else ""
    
    print(f"  {status} {result.name} {time_str}")
    
    if not result.passed:
        if result.error:
            print(f"       {RED}Error: {result.error}{RESET}")
        if verbose or not result.passed:
            if result.expected and result.actual:
                print(f"       Expected: {', '.join(result.expected[:5])}{'...' if len(result.expected) > 5 else ''}")
                print(f"       Actual:   {', '.join(result.actual[:5])}{'...' if len(result.actual) > 5 else ''}")

def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    
    script_dir = Path(__file__).parent
    workspace = script_dir
    
    jaithon = workspace / "jaithon"
    if not jaithon.exists():
        print(f"{RED}Error: jaithon binary not found at {jaithon}{RESET}")
        print("Run 'make' first to build jaithon")
        sys.exit(1)
    
    checks_dir = workspace / "test" / "checks"
    if not checks_dir.exists():
        print(f"{RED}Error: test/checks directory not found{RESET}")
        sys.exit(1)
    
    test_files = sorted(checks_dir.glob("*.jai"))
    
    if not test_files:
        print(f"{YELLOW}No test files found in {checks_dir}{RESET}")
        sys.exit(1)
    
    print(f"\n{BOLD}{CYAN}=== Jaithon Test Suite ==={RESET}\n")
    print(f"Running {len(test_files)} tests from {checks_dir}\n")
    
    results: List[TestResult] = []
    for test_file in test_files:
        result = run_test(str(jaithon), test_file)
        results.append(result)
        print_result(result, verbose)
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print(f"\n{BOLD}{'='*40}{RESET}")
    print(f"{BOLD}Results:{RESET} {GREEN}{passed} passed{RESET}, {RED if failed else ''}{failed} failed{RESET}")
    
    if failed > 0:
        print(f"\n{YELLOW}Failed tests:{RESET}")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error or 'Output mismatch'}")
    
    total_time = sum(r.time_ms for r in results)
    print(f"\n{CYAN}Total time: {total_time:.0f}ms{RESET}\n")
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
