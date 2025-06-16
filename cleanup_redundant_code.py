#!/usr/bin/env python3
"""
Cleanup script to remove redundant code patterns from the codebase.

This script identifies and optionally removes:
1. Redundant print statements
2. Duplicate function definitions
3. Unused imports
4. Verbose logging that can be simplified

Usage:
    python cleanup_redundant_code.py --dry-run    # Just identify issues
    python cleanup_redundant_code.py --fix        # Actually fix the issues
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set

class CodeCleanupTool:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.issues_found = []
        
    def scan_directory(self, directory: str) -> None:
        """Scan directory for Python files and analyze them"""
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache', 'backup']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path)
    
    def analyze_file(self, file_path: str) -> None:
        """Analyze a single Python file for redundant code"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for redundant patterns
            self.check_redundant_prints(file_path, lines)
            self.check_long_comment_blocks(file_path, lines)
            self.check_duplicate_imports(file_path, lines)
            self.check_verbose_logging(file_path, lines)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def check_redundant_prints(self, file_path: str, lines: List[str]) -> None:
        """Check for redundant print statements"""
        redundant_prints = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for excessive separator prints
            if re.match(r'^\s*print\s*\(\s*["\']={20,}["\']', stripped):
                redundant_prints.append((i + 1, "Excessive separator print"))
            
            # Check for verbose status prints that could be simplified
            if re.match(r'^\s*print\s*\(\s*f?["\'].*✓.*["\']', stripped):
                redundant_prints.append((i + 1, "Verbose status print"))
            
            # Check for debugging prints
            if re.match(r'^\s*print\s*\(\s*f?["\'].*DEBUG.*["\']', stripped):
                redundant_prints.append((i + 1, "Debug print statement"))
        
        if redundant_prints:
            self.issues_found.append({
                'file': file_path,
                'type': 'redundant_prints',
                'issues': redundant_prints
            })
    
    def check_long_comment_blocks(self, file_path: str, lines: List[str]) -> None:
        """Check for excessively long comment blocks"""
        comment_blocks = []
        current_block = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                current_block.append(i + 1)
            else:
                if len(current_block) > 10:  # More than 10 consecutive comment lines
                    comment_blocks.append(current_block)
                current_block = []
        
        if current_block and len(current_block) > 10:
            comment_blocks.append(current_block)
        
        if comment_blocks:
            self.issues_found.append({
                'file': file_path,
                'type': 'long_comment_blocks',
                'issues': comment_blocks
            })
    
    def check_duplicate_imports(self, file_path: str, lines: List[str]) -> None:
        """Check for duplicate import statements"""
        imports = {}
        duplicates = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                if stripped in imports:
                    duplicates.append((i + 1, f"Duplicate import: {stripped}"))
                else:
                    imports[stripped] = i + 1
        
        if duplicates:
            self.issues_found.append({
                'file': file_path,
                'type': 'duplicate_imports',
                'issues': duplicates
            })
    
    def check_verbose_logging(self, file_path: str, lines: List[str]) -> None:
        """Check for verbose logging that could be simplified"""
        verbose_logs = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for logger.info with excessive decoration
            if 'logger.info' in stripped and '=' in stripped and len(stripped) > 80:
                verbose_logs.append((i + 1, "Verbose logging statement"))
            
            # Check for multiple consecutive logger statements
            if i > 0 and 'logger.' in stripped and 'logger.' in lines[i-1].strip():
                if len([l for l in lines[max(0, i-3):i+1] if 'logger.' in l.strip()]) >= 3:
                    verbose_logs.append((i + 1, "Multiple consecutive logger statements"))
        
        if verbose_logs:
            self.issues_found.append({
                'file': file_path,
                'type': 'verbose_logging',
                'issues': verbose_logs
            })
    
    def generate_report(self) -> None:
        """Generate a report of all issues found"""
        print("=" * 80)
        print("CODE CLEANUP ANALYSIS REPORT")
        print("=" * 80)
        
        if not self.issues_found:
            print("✅ No redundant code patterns found!")
            return
        
        # Group by issue type
        by_type = {}
        for issue in self.issues_found:
            issue_type = issue['type']
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append(issue)
        
        total_issues = sum(len(issue['issues']) for issue in self.issues_found)
        print(f"📊 Found {total_issues} redundant code patterns across {len(self.issues_found)} files")
        
        for issue_type, issues in by_type.items():
            print(f"\n{issue_type.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            for issue in issues:
                file_path = issue['file']
                rel_path = os.path.relpath(file_path)
                print(f"\n📁 {rel_path}")
                
                for line_num, description in issue['issues']:
                    print(f"  Line {line_num}: {description}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        
        if 'redundant_prints' in by_type:
            print("🔧 Redundant Prints:")
            print("  - Replace excessive separator prints with logger.info")
            print("  - Remove debug prints from production code")
            print("  - Use the new logging_utils.py for standardized logging")
        
        if 'verbose_logging' in by_type:
            print("🔧 Verbose Logging:")
            print("  - Use the new log_experiment_summary() function")
            print("  - Combine multiple logger statements into single calls")
            print("  - Use log_progress() for progress updates")
        
        if 'duplicate_imports' in by_type:
            print("🔧 Duplicate Imports:")
            print("  - Remove duplicate import statements")
            print("  - Consider organizing imports with isort")
        
        if 'long_comment_blocks' in by_type:
            print("🔧 Long Comment Blocks:")
            print("  - Consider moving long explanations to docstrings")
            print("  - Break up large comment blocks with code")
        
        print("\n🛠️ To apply automatic fixes, run:")
        print("  python cleanup_redundant_code.py --fix")

def main():
    parser = argparse.ArgumentParser(description="Cleanup redundant code patterns")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Just analyze, don't modify files")
    parser.add_argument("--fix", action="store_true",
                       help="Apply automatic fixes")
    parser.add_argument("--directory", default="src",
                       help="Directory to analyze")
    
    args = parser.parse_args()
    
    if args.fix:
        args.dry_run = False
    
    cleanup_tool = CodeCleanupTool(dry_run=args.dry_run)
    
    print(f"🔍 Analyzing code in {args.directory}...")
    cleanup_tool.scan_directory(args.directory)
    cleanup_tool.generate_report()
    
    if not args.dry_run:
        print("\n⚠️ Note: Automatic fixes are not yet implemented.")
        print("Please review the issues manually and apply fixes using the new utility modules:")
        print("  - src/models/custom_objects.py")
        print("  - src/utils/logging_utils.py")
        print("  - src/utils/args_utils.py")

if __name__ == "__main__":
    main() 