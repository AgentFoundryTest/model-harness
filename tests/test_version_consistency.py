"""
Tests for version consistency across the codebase.

Ensures that version numbers are consistent between:
- pyproject.toml
- mlx/__init__.py
- CLI --version output
"""

import subprocess
import re
from pathlib import Path


def test_version_in_pyproject_toml():
    """Test that version is defined in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"
    
    content = pyproject_path.read_text()
    version_match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    assert version_match, "Version not found in pyproject.toml"
    
    version = version_match.group(1)
    assert version, "Version string is empty"
    
    # Validate semantic versioning format (basic check)
    assert re.match(r'^\d+\.\d+\.\d+', version), f"Version '{version}' does not follow semantic versioning"


def test_version_in_init_py():
    """Test that __version__ is defined in mlx/__init__.py."""
    import mlx
    
    assert hasattr(mlx, '__version__'), "mlx.__version__ not defined"
    assert mlx.__version__, "__version__ is empty"
    
    # Validate semantic versioning format (basic check)
    assert re.match(r'^\d+\.\d+\.\d+', mlx.__version__), \
        f"Version '{mlx.__version__}' does not follow semantic versioning"


def test_version_consistency_between_files():
    """Test that version is consistent between pyproject.toml and mlx/__init__.py."""
    import mlx
    
    # Get version from pyproject.toml
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    version_match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    pyproject_version = version_match.group(1)
    
    # Compare with mlx.__version__
    assert mlx.__version__ == pyproject_version, \
        f"Version mismatch: mlx.__version__='{mlx.__version__}' != pyproject.toml='{pyproject_version}'"


def test_cli_version_output():
    """Test that CLI --version outputs the correct version."""
    import mlx
    
    # Run mlx --version
    result = subprocess.run(
        ["mlx", "--version"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "mlx --version failed"
    
    # Extract version from output (format: "mlx X.Y.Z")
    output = result.stdout.strip()
    version_match = re.search(r'mlx\s+(\d+\.\d+\.\d+)', output)
    assert version_match, f"Could not parse version from CLI output: '{output}'"
    
    cli_version = version_match.group(1)
    
    # Compare with mlx.__version__
    assert cli_version == mlx.__version__, \
        f"CLI version mismatch: CLI='{cli_version}' != mlx.__version__='{mlx.__version__}'"


def test_version_format_is_semantic():
    """Test that version follows semantic versioning format."""
    import mlx
    
    version = mlx.__version__
    
    # Basic semantic versioning: MAJOR.MINOR.PATCH
    # Optional: -PRERELEASE+BUILD
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
    
    match = re.match(pattern, version)
    assert match, f"Version '{version}' does not follow semantic versioning format"
    
    major, minor, patch, prerelease, build = match.groups()
    
    # Verify parts are valid
    assert int(major) >= 0, "Major version must be non-negative"
    assert int(minor) >= 0, "Minor version must be non-negative"
    assert int(patch) >= 0, "Patch version must be non-negative"


def test_version_in_changelog():
    """Test that current version is documented in CHANGELOG.md."""
    import mlx
    
    changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
    assert changelog_path.exists(), "CHANGELOG.md not found"
    
    content = changelog_path.read_text()
    
    # Look for version heading in changelog (format: ## [X.Y.Z])
    version_pattern = re.escape(f"[{mlx.__version__}]")
    assert re.search(version_pattern, content), \
        f"Version '{mlx.__version__}' not found in CHANGELOG.md"


def test_changelog_has_release_date():
    """Test that current version in CHANGELOG has a release date."""
    import mlx
    
    changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
    content = changelog_path.read_text()
    
    # Look for version with date (format: ## [X.Y.Z] - YYYY-MM-DD)
    version = re.escape(mlx.__version__)
    date_pattern = rf'##\s*\[{version}\]\s*-\s*\d{{4}}-\d{{2}}-\d{{2}}'
    
    assert re.search(date_pattern, content), \
        f"Version '{mlx.__version__}' in CHANGELOG.md does not have a proper release date"


def test_readme_references_version():
    """Test that README.md mentions the current version."""
    import mlx
    
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md not found"
    
    content = readme_path.read_text()
    
    # Check if version appears in README
    assert mlx.__version__ in content, \
        f"Version '{mlx.__version__}' not referenced in README.md"


def test_readme_links_to_changelog():
    """Test that README.md links to CHANGELOG.md."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    
    # Check for link to CHANGELOG.md
    assert "CHANGELOG.md" in content or "Changelog" in content, \
        "README.md does not link to CHANGELOG.md"


def test_python_version_compatibility():
    """Test that Python version requirements are consistent."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    
    # Extract requires-python
    python_req_match = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', content)
    assert python_req_match, "requires-python not found in pyproject.toml"
    
    python_req = python_req_match.group(1)
    
    # Verify it's a reasonable requirement
    assert ">=" in python_req or "==" in python_req, \
        f"Invalid Python version requirement: '{python_req}'"
    
    # Extract classifiers for Python versions
    classifier_matches = re.findall(r'Programming Language :: Python :: (3\.\d+)', content)
    assert len(classifier_matches) > 0, "No Python version classifiers found"
    
    # Verify minimum version in requires-python matches classifiers
    min_version_match = re.search(r'>=\s*(\d+\.\d+)', python_req)
    if min_version_match:
        min_version = min_version_match.group(1)
        # Check that this version is in classifiers
        assert any(version >= min_version for version in classifier_matches), \
            f"Minimum Python version '{min_version}' not in classifiers"


def test_dependency_version_specifiers():
    """Test that dependencies have proper version specifiers."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    
    # Extract dependencies section
    deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
    assert deps_match, "dependencies not found in pyproject.toml"
    
    deps_text = deps_match.group(1)
    
    # Find all dependency specifications
    dep_specs = re.findall(r'["\']([^"\']+)["\']', deps_text)
    
    for dep_spec in dep_specs:
        # Should have package name and version specifier
        assert any(op in dep_spec for op in ['>=', '<=', '==', '!=', '~=', '<', '>']), \
            f"Dependency '{dep_spec}' should have a version specifier"
        
        # Prefer >= for flexibility unless there's a good reason
        if '>=' in dep_spec:
            # This is the preferred pattern
            pass
        elif '==' in dep_spec:
            # Exact versions should be rare (only for known compatibility issues)
            print(f"Warning: Exact version specified for '{dep_spec}'. Consider using '>=' for flexibility.")


def test_version_is_not_placeholder():
    """Test that version is not a placeholder like '0.0.0' or '1.0.0-dev'."""
    import mlx
    
    version = mlx.__version__
    
    # Check for common placeholder versions
    placeholders = ['0.0.0', '0.0.1', '1.0.0-dev', 'dev', 'unknown']
    
    # For 0.1.0 (initial release), this is acceptable
    # But warn if it looks like a placeholder
    if version in placeholders:
        raise AssertionError(f"Version '{version}' appears to be a placeholder. Update to actual release version.")
