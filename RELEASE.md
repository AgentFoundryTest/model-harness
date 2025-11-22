# Release Process

This document outlines the manual release process for MLX ML Experiment Harness.

## Version Strategy

MLX follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR version** (X.0.0): Incompatible API changes
- **MINOR version** (0.X.0): New functionality in a backward-compatible manner
- **PATCH version** (0.0.X): Backward-compatible bug fixes

### Version Number Format

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

Examples:
- `0.1.0` - Initial release
- `0.2.0` - New features added
- `0.2.1` - Bug fixes only
- `1.0.0` - First stable release with API guarantees
- `1.0.0-alpha.1` - Pre-release version
- `1.0.0+20130313144700` - Build metadata

### Pre-1.0.0 Releases

During the 0.x.x series:
- The API may change between minor versions
- Breaking changes are allowed but should be documented
- The project is considered "in development"
- Users should expect potential instability

### Post-1.0.0 Releases

After reaching 1.0.0:
- MAJOR version must increment for breaking changes
- MINOR version increments for new features
- PATCH version increments for bug fixes only
- API stability is guaranteed within a major version

## Release Checklist

Use this checklist for each release to ensure consistency and completeness.

### Pre-Release (Preparation)

- [ ] **Verify all tests pass**
  ```bash
  pytest -v
  # Ensure all 258+ tests pass
  ```

- [ ] **Verify code quality**
  ```bash
  # Run any linters if configured
  # Check for code smells and technical debt
  ```

- [ ] **Update version numbers**
  - [ ] Update `version` in `pyproject.toml`
  - [ ] Update `__version__` in `mlx/__init__.py`
  - [ ] Verify consistency:
    ```bash
    python -c "import mlx; print(mlx.__version__)"
    mlx --version
    grep 'version = ' pyproject.toml
    ```

- [ ] **Update CHANGELOG.md**
  - [ ] Add new version section with current date
  - [ ] Document all changes since last release:
    - Added features
    - Changed behavior
    - Deprecated features
    - Removed features
    - Fixed bugs
    - Security fixes
  - [ ] Add comparison link at bottom
  - [ ] Move items from "Unreleased" section if using one

- [ ] **Update README.md**
  - [ ] Update version badge/reference
  - [ ] Update installation instructions if needed
  - [ ] Verify all examples work with new version
  - [ ] Update any version-specific documentation

- [ ] **Update documentation**
  - [ ] Verify `docs/usage.md` is current
  - [ ] Update API documentation if changed
  - [ ] Check all code examples in docs
  - [ ] Update any version-specific guides

- [ ] **Verify dependencies**
  - [ ] Check `dependencies` in `pyproject.toml` for correct version specifiers
  - [ ] Ensure `numpy>=1.20.0` is appropriate
  - [ ] Ensure `pyyaml>=5.1` is appropriate
  - [ ] Test with minimum supported versions
  - [ ] Test with latest versions

- [ ] **Check edge cases**
  - [ ] Verify NICE features (if any optional features exist)
  - [ ] Test with different Python versions (3.8-3.12)
  - [ ] Test on different platforms (Linux, macOS, Windows if applicable)
  - [ ] Verify deterministic behavior across platforms

- [ ] **Review breaking changes**
  - [ ] Document any breaking changes prominently
  - [ ] Provide migration guide if needed
  - [ ] Consider deprecation warnings for removed features

### Release (Execution)

- [ ] **Create release commit**
  ```bash
  git add pyproject.toml mlx/__init__.py CHANGELOG.md README.md
  git commit -m "Release version X.Y.Z"
  ```

- [ ] **Create and push git tag**
  ```bash
  git tag -a vX.Y.Z -m "Release version X.Y.Z"
  git push origin main  # or master
  git push origin vX.Y.Z
  ```

- [ ] **Verify tag on GitHub**
  - Navigate to repository tags page
  - Confirm tag vX.Y.Z appears correctly

- [ ] **Create GitHub Release**
  - Go to GitHub repository releases page
  - Click "Draft a new release"
  - Select the tag vX.Y.Z
  - Title: "MLX v{X.Y.Z}"
  - Description: Copy relevant section from CHANGELOG.md
  - Mark as pre-release if version < 1.0.0 or contains pre-release suffix
  - Publish release

- [ ] **Build distribution packages** (for PyPI when ready)
  ```bash
  # Clean previous builds
  rm -rf dist/ build/ *.egg-info

  # Build source and wheel distributions
  python -m build

  # Verify contents
  tar tzf dist/mlx-X.Y.Z.tar.gz
  unzip -l dist/mlx-X.Y.Z-py3-none-any.whl
  ```

- [ ] **Test installation from built packages** (local test)
  ```bash
  # Create clean virtual environment
  python -m venv test-env
  source test-env/bin/activate  # or test-env\Scripts\activate on Windows

  # Install from wheel
  pip install dist/mlx-X.Y.Z-py3-none-any.whl

  # Verify installation
  mlx --version
  python -c "import mlx; print(mlx.__version__)"

  # Run smoke tests
  mlx run-experiment --dry-run --config experiments/example.json

  # Clean up
  deactivate
  rm -rf test-env
  ```

### Post-Release (Publication - Future)

When ready to publish to PyPI:

- [ ] **Upload to Test PyPI** (recommended first)
  ```bash
  python -m twine upload --repository testpypi dist/*
  # Test installation: pip install --index-url https://test.pypi.org/simple/ mlx==X.Y.Z
  ```

- [ ] **Upload to PyPI** (production)
  ```bash
  python -m twine upload dist/*
  ```

- [ ] **Verify PyPI page**
  - Check package metadata
  - Verify README renders correctly
  - Check supported Python versions
  - Verify classifiers

- [ ] **Update installation instructions**
  ```bash
  # Update README with PyPI installation
  pip install mlx==X.Y.Z
  ```

### Post-Release (Verification)

- [ ] **Verify installation from GitHub**
  ```bash
  # In a clean environment
  pip install git+https://github.com/AgentFoundryTest/model-harness.git@vX.Y.Z
  mlx --version
  ```

- [ ] **Test example workflows**
  ```bash
  # Run the examples from documentation
  mlx run-experiment --config experiments/example.json
  mlx eval --run-dir runs/example-comprehensive/20251122_143025
  ```

- [ ] **Monitor for issues**
  - Check GitHub issues for bug reports
  - Monitor discussion channels
  - Watch for installation problems

### Post-Release (Maintenance)

- [ ] **Prepare for next release**
  - [ ] Create "Unreleased" section in CHANGELOG.md
  - [ ] Update development roadmap if exists
  - [ ] Close milestone for this release (if using GitHub milestones)
  - [ ] Create milestone for next release

- [ ] **Announce release** (optional)
  - Post to relevant forums/channels
  - Update project website if exists
  - Notify key users/contributors

- [ ] **Archive release artifacts**
  - Keep local copy of distribution packages
  - Document any platform-specific issues found

## Emergency Patch Release

If a critical bug is found after release:

1. **Create hotfix branch** from the release tag
   ```bash
   git checkout -b hotfix-X.Y.Z+1 vX.Y.Z
   ```

2. **Fix the bug** with minimal changes
   - Focus only on the critical issue
   - Avoid refactoring or feature additions

3. **Update version** to next patch version (X.Y.Z+1)

4. **Update CHANGELOG.md** with the fix

5. **Follow release checklist** for the patch version

6. **Merge back** to main branch
   ```bash
   git checkout main
   git merge hotfix-X.Y.Z+1
   ```

## Version Bump Guidelines

### When to Bump MAJOR (X.0.0)

- Removing or renaming public API functions/classes
- Changing function signatures in incompatible ways
- Removing command-line options
- Changing configuration file format incompatibly
- Removing supported Python versions
- Changing output format in breaking ways

### When to Bump MINOR (0.X.0)

- Adding new features
- Adding new CLI commands
- Adding new configuration options
- Adding new models or datasets
- Adding new optional dependencies
- Deprecating features (but not removing)
- Improving performance significantly
- Adding new Python version support

### When to Bump PATCH (0.0.X)

- Fixing bugs without changing API
- Fixing documentation
- Improving error messages
- Fixing edge cases
- Performance improvements without API changes
- Security fixes (may also warrant MINOR or MAJOR)
- Dependency updates (if compatible)

## Dependency Version Management

### Philosophy

- Use **minimum version specifiers** (`>=`) for flexibility
- Only use **maximum version specifiers** (`<`) for known incompatibilities
- Test with both minimum and latest versions before release

### Current Dependencies

```toml
dependencies = [
    "numpy>=1.20.0",
    "pyyaml>=5.1",
]
```

### Before Each Release

1. **Check for numpy updates**
   ```bash
   pip index versions numpy
   # Test with latest version
   pip install numpy==latest-version
   pytest
   ```

2. **Check for pyyaml updates**
   ```bash
   pip index versions pyyaml
   # Test with latest version
   pip install pyyaml==latest-version
   pytest
   ```

3. **Test with minimum versions**
   ```bash
   pip install numpy==1.20.0 pyyaml==5.1
   pytest
   ```

4. **Update minimum versions if needed**
   - If new features require newer versions
   - If minimum version has critical bugs
   - Document reason in commit message

## Release History

### v0.1.0 (2025-11-22)
- Initial release
- Core framework with CLI
- Synthetic datasets (regression, classification)
- Models (linear regression, MLP)
- Training and evaluation pipelines
- Comprehensive testing infrastructure
- Full documentation

## Notes for Future Releases

### Environments Without Build Isolation

If users report issues installing in environments without build isolation:

```bash
# Provide alternative installation method
pip install --no-build-isolation .
```

Document this in README if it becomes a common issue.

### Version Tag Ordering

Ensure semantic version ordering is preserved:
- v0.1.0 < v0.2.0 < v0.10.0 (not v0.1.0 < v0.10.0 < v0.2.0)
- Git tags are lexicographically sorted by default
- Use `git tag --sort=-version:refname` for semantic sorting

### Platform-Specific Releases

If platform-specific issues arise:
- Consider platform-specific builds/wheels
- Document known platform limitations
- Test on all supported platforms before release

## Tools and Automation

### Recommended Tools

```bash
# Build tool
pip install build

# Upload tool (for PyPI)
pip install twine

# Version bumping (optional)
pip install bump2version

# Changelog generation (optional)
pip install towncrier
```

### Future Automation Opportunities

- GitHub Actions workflow for automated releases
- Automated version bumping via scripts
- Automated CHANGELOG generation from commits
- Automated testing across Python versions
- Automated PyPI upload on tag push

## Contact

For questions about the release process:
- Open an issue on GitHub
- Contact maintainers: Agent Foundry and John Brosnihan

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
