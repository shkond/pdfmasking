"""Unit tests for allow list functionality."""

import tempfile
from pathlib import Path

import pytest

from core.allow_list import get_allow_list, parse_dictionary


class TestParseDictionary:
    """Tests for parse_dictionary function."""

    def test_parse_simple_terms(self):
        """Test parsing simple terms without aliases."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dic", delete=False, encoding="utf-8") as f:
            f.write("Python\n")
            f.write("Docker\n")
            f.write("Git\n")
            f.name

        try:
            terms = parse_dictionary(f.name)
            assert "Python" in terms
            assert "Docker" in terms
            assert "Git" in terms
            assert len(terms) == 3
        finally:
            Path(f.name).unlink()

    def test_parse_terms_with_alias(self):
        """Test parsing terms with /alias[...] format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dic", delete=False, encoding="utf-8") as f:
            f.write("AI/alias[AI|Artificial Intelligence]\n")
            f.write("Machine Learning/alias[Machine Learning|ML]\n")
            f.name

        try:
            terms = parse_dictionary(f.name)
            assert "AI" in terms
            assert "Artificial Intelligence" in terms
            assert "Machine Learning" in terms
            assert "ML" in terms
        finally:
            Path(f.name).unlink()

    def test_parse_terms_with_js(self):
        """Test parsing terms with /js[...] format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dic", delete=False, encoding="utf-8") as f:
            f.write("node.js/js[node]\n")
            f.write("React/js[react]\n")
            f.name

        try:
            terms = parse_dictionary(f.name)
            assert "node.js" in terms
            assert "node" in terms
            assert "React" in terms
            assert "react" in terms
        finally:
            Path(f.name).unlink()

    def test_parse_real_dictionary(self):
        """Test parsing the actual softwaretec.dic file."""
        dict_path = Path(__file__).parent.parent.parent / "doc" / "softwaretec.dic"
        
        if not dict_path.exists():
            pytest.skip("softwaretec.dic not found")
        
        terms = parse_dictionary(dict_path)
        
        # Should have many terms
        assert len(terms) > 250
        
        # Check some known terms
        assert "Python" in terms
        assert "Docker" in terms
        assert "Git" in terms
        assert "JavaScript" in terms
        
    def test_parse_empty_lines(self):
        """Test that empty lines are ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dic", delete=False, encoding="utf-8") as f:
            f.write("Python\n")
            f.write("\n")
            f.write("Docker\n")
            f.write("  \n")
            f.write("Git\n")
            f.name

        try:
            terms = parse_dictionary(f.name)
            assert len(terms) == 3
        finally:
            Path(f.name).unlink()

    def test_parse_nonexistent_file(self):
        """Test that nonexistent file returns empty list."""
        terms = parse_dictionary("/nonexistent/path.dic")
        assert terms == []


class TestGetAllowList:
    """Tests for get_allow_list function."""

    def test_disabled_allow_list(self):
        """Test that disabled allow list returns empty."""
        config = {"allow_list": {"enabled": False}}
        terms = get_allow_list(config)
        assert terms == []

    def test_missing_allow_list_config(self):
        """Test that missing config returns empty."""
        config = {}
        terms = get_allow_list(config)
        assert terms == []

    def test_additional_terms(self):
        """Test that additional terms are included."""
        config = {
            "allow_list": {
                "enabled": True,
                "additional_terms": ["CustomTerm1", "CustomTerm2"]
            }
        }
        terms = get_allow_list(config)
        assert "CustomTerm1" in terms
        assert "CustomTerm2" in terms
