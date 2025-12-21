"""Test that all 6 required PII entity types are detected in output log files.

This test reads all *_log.txt files from the output directory and verifies that
the last log section contains all 6 required entity types:
- 名前 (PERSON / JP_PERSON)
- Email (EMAIL_ADDRESS)
- 住所 (JP_ADDRESS)
- 生年月日 (DATE_OF_BIRTH_JP)
- 郵便番号 (JP_ZIP_CODE)
- 電話番号 (PHONE_NUMBER_JP)
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest


# 6 required PII entity types (display name: [accepted entity type labels])
REQUIRED_ENTITY_TYPES: dict[str, list[str]] = {
    "名前": ["PERSON", "JP_PERSON"],
    "Email": ["EMAIL_ADDRESS"],
    "住所": ["JP_ADDRESS", "ADDRESS", "LOCATION"],
    "生年月日": ["DATE_OF_BIRTH_JP", "DATE_OF_BIRTH", "DATE"],
    "郵便番号": ["JP_ZIP_CODE", "ZIP_CODE"],
    "電話番号": ["PHONE_NUMBER_JP", "PHONE_NUMBER"],
}


class LogParseResult(NamedTuple):
    """Result of parsing a log file."""
    file_path: Path
    detected_types: set[str]
    missing_types: list[str]
    is_success: bool
    raw_entities: list[str]


def get_output_log_files(output_dir: Path) -> list[Path]:
    """Get all *_log.txt files from the output directory."""
    return sorted(output_dir.glob("*_log.txt"))


def parse_last_log_section(log_content: str) -> set[str]:
    """
    Parse the last log section and extract detected entity types.
    
    Log sections are separated by lines like:
    ============================================================
    Masking Log - 2025-12-21 05:03:39
    ============================================================
    
    Entity lines look like:
    [EMAIL_ADDRESS] "jessica_martinez384@gmail.com" (score: 1.00, pos: 25-54)
    """
    # Split by log section headers
    sections = re.split(r"={60,}", log_content)
    
    if len(sections) < 3:
        # No complete log section found
        return set()
    
    # Get the last complete section (after the last header pair)
    # Sections: ['', header, content, header, content, ...]
    last_section = sections[-1] if sections else ""
    
    # Extract entity types from lines like [ENTITY_TYPE] "value"
    entity_pattern = r"\[([A-Z_]+)\]"
    entity_types = set(re.findall(entity_pattern, last_section))
    
    return entity_types


def check_required_entities(detected_types: set[str]) -> list[str]:
    """
    Check if all required entity types are detected.
    
    Returns a list of missing entity type names (display names).
    """
    missing = []
    
    for display_name, accepted_labels in REQUIRED_ENTITY_TYPES.items():
        # Check if any of the accepted labels are in detected types
        found = any(label in detected_types for label in accepted_labels)
        if not found:
            missing.append(display_name)
    
    return missing


def parse_log_file(log_path: Path) -> LogParseResult:
    """Parse a single log file and check for required entity types."""
    try:
        content = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return LogParseResult(
            file_path=log_path,
            detected_types=set(),
            missing_types=list(REQUIRED_ENTITY_TYPES.keys()),
            is_success=False,
            raw_entities=[f"Error reading file: {e}"],
        )
    
    detected_types = parse_last_log_section(content)
    missing_types = check_required_entities(detected_types)
    
    return LogParseResult(
        file_path=log_path,
        detected_types=detected_types,
        missing_types=missing_types,
        is_success=len(missing_types) == 0,
        raw_entities=list(detected_types),
    )


def validate_all_logs(output_dir: Path) -> tuple[list[LogParseResult], list[LogParseResult]]:
    """
    Validate all log files in output directory.
    
    Returns (success_results, failure_results).
    """
    log_files = get_output_log_files(output_dir)
    
    successes = []
    failures = []
    
    for log_path in log_files:
        result = parse_log_file(log_path)
        if result.is_success:
            successes.append(result)
        else:
            failures.append(result)
    
    return successes, failures


class TestEntityDetectionCompleteness:
    """Test that all 6 required PII entity types are detected in log files."""
    
    @pytest.fixture
    def output_dir(self) -> Path:
        """Get the output directory path."""
        return Path(__file__).parent.parent.parent / "output"
    
    def test_all_log_files_have_required_entities(self, output_dir: Path):
        """
        Test that each *_log.txt file in output directory contains all 6 required
        entity types in its last log section.
        
        Success: All 6 entity types detected.
        Failure: Reports which entity types are missing.
        """
        if not output_dir.exists():
            pytest.skip(f"Output directory not found: {output_dir}")
        
        log_files = get_output_log_files(output_dir)
        
        if not log_files:
            pytest.skip("No log files found in output directory")
        
        successes, failures = validate_all_logs(output_dir)
        
        if failures:
            error_messages = []
            for result in failures:
                error_messages.append(
                    f"\n{result.file_path.name}:\n"
                    f"  検出されてないentity: {result.missing_types}\n"
                    f"  検出されたentity: {list(result.detected_types)}"
                )
            
            pytest.fail(
                f"テスト失敗: {len(failures)}/{len(log_files)} 件のログファイルで"
                f"すべてのエンティティが検出されていません\n"
                + "\n".join(error_messages)
            )
        
        # All tests passed
        print(f"\n✓ {len(successes)} 件のログファイルですべてのエンティティが検出されました")
    
    @pytest.mark.parametrize("log_filename", [
        "JP_001_フロントエンドエンジニア_log.txt",
        "JP_002_バックエンドエンジニア_log.txt",
        "JP_003_データエンジニア_log.txt",
        "JP_004_インフラエンジニア_log.txt",
        "JP_005_セキュリティエンジニア_log.txt",
    ])
    def test_japanese_resume_logs(self, output_dir: Path, log_filename: str):
        """
        Test individual Japanese resume log files for completeness.
        
        Each test reports which entity types are missing if any.
        """
        log_path = output_dir / log_filename
        
        if not log_path.exists():
            pytest.skip(f"Log file not found: {log_path}")
        
        result = parse_log_file(log_path)
        
        if not result.is_success:
            pytest.fail(
                f"検出されてないentity: {result.missing_types}\n"
                f"検出されたentity: {list(result.detected_types)}"
            )
    
    @pytest.mark.parametrize("log_filename", [
        "EN_001_Frontend_Engineer_log.txt",
        "EN_002_Backend_Engineer_log.txt",
        "EN_003_Data_Engineer_log.txt",
        "EN_004_Infrastructure_Engineer_log.txt",
        "EN_005_Security_Engineer_log.txt",
    ])
    def test_english_resume_logs(self, output_dir: Path, log_filename: str):
        """
        Test individual English resume log files for completeness.
        
        Each test reports which entity types are missing if any.
        """
        log_path = output_dir / log_filename
        
        if not log_path.exists():
            pytest.skip(f"Log file not found: {log_path}")
        
        result = parse_log_file(log_path)
        
        if not result.is_success:
            pytest.fail(
                f"検出されてないentity: {result.missing_types}\n"
                f"検出されたentity: {list(result.detected_types)}"
            )


def main():
    """Run validation and print results (for standalone usage)."""
    output_dir = Path(__file__).parent.parent.parent / "output"
    
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        return 1
    
    log_files = get_output_log_files(output_dir)
    
    if not log_files:
        print("No log files found")
        return 1
    
    print(f"Checking {len(log_files)} log files...\n")
    
    successes, failures = validate_all_logs(output_dir)
    
    # Print results
    for result in successes:
        print(f"✓ {result.file_path.name}")
    
    for result in failures:
        print(f"✗ {result.file_path.name}")
        print(f"  検出されてないentity: {result.missing_types}")
        print(f"  検出されたentity: {result.raw_entities}")
    
    print(f"\n結果: {len(successes)}/{len(log_files)} 成功")
    
    return 0 if not failures else 1


if __name__ == "__main__":
    exit(main())
