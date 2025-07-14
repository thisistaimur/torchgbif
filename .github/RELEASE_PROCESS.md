# Automated Releases with Semantic Versioning

This repository uses GitHub Actions to automatically create releases based on semantic versioning tags and commit messages.

## How it works

### 1. Automatic Tagging (auto-tag.yml)

The repository monitors commit messages for version bump indicators and automatically creates tags:

- **Major version bump**: Include `[major]`, `major:`, `breaking:`, or `feat!:` in your commit message
- **Minor version bump**: Include `[minor]`, `minor:`, `feat:`, or `feature:` in your commit message  
- **Patch version bump**: Include `[patch]`, `patch:`, `fix:`, or `bugfix:` in your commit message
- **Release**: Include `[release]` or `release:` to create a release tag

**Examples:**

```bash
git commit -m "feat: add new GBIF image dataset loader"  # Creates minor version bump
git commit -m "fix: handle missing audio files gracefully"  # Creates patch version bump
git commit -m "feat!: redesign API for better PyTorch integration"  # Creates major version bump
git commit -m "[release] prepare for stable release"  # Creates patch version bump for release
```

### 2. Release Workflow (release.yml)

When a tag matching the pattern `v*.*.*` is pushed, the release workflow:

1. **Validates** the tag follows semantic versioning
2. **Builds** the Python package
3. **Tests** across multiple Python versions (3.8-3.11)
4. **Publishes** to PyPI (requires PyPI trusted publishing setup)
5. **Creates** a GitHub release with changelog

### 3. Continuous Integration (ci.yml)

Runs on every push and pull request to ensure code quality:

- **Linting** with flake8
- **Code formatting** check with black
- **Type checking** with mypy
- **Testing** with pytest
- **Build verification**

## Manual Release Process

If you prefer manual control, you can create releases manually:

### Option 1: Create tag manually

```bash
# Create a new tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Option 2: Use GitHub interface

1. Go to "Releases" in your GitHub repository
2. Click "Create a new release"
3. Create a new tag (e.g., `v1.0.0`)
4. Fill in release notes
5. Publish release

## PyPI Setup

To publish to PyPI automatically, you need to set up trusted publishing:

1. Go to <https://pypi.org/manage/account/publishing/>
2. Add a new publisher with these settings:
   - PyPI project name: `torchgbif`
   - Owner: `thisistaimur`
   - Repository name: `TorchGBIF`
   - Workflow name: `release.yml`
   - Environment name: `pypi` (optional but recommended)

## Pre-releases

The system supports pre-releases for development versions:

- Tags like `v1.0.0-alpha.1`, `v1.0.0-beta.1`, `v1.0.0-rc.1` will be marked as pre-releases
- Development branch commits with version indicators create pre-release tags

## Version History

The release workflow automatically generates changelogs based on commit messages since the last tag.

## Testing Releases

Before creating a real release, you can test the build process:

1. The CI workflow runs on every push to validate builds
2. Build artifacts are uploaded and can be downloaded for testing
3. The package can be installed locally for testing: `pip install -e .`

## Troubleshooting

### Release Failed

- Check the GitHub Actions logs for detailed error messages
- Ensure all tests pass in the CI workflow
- Verify the tag follows semantic versioning (v1.0.0 format)

### PyPI Upload Failed

- Verify PyPI trusted publishing is configured correctly
- Check that the package name is available on PyPI
- Ensure version number doesn't already exist

### Tag Already Exists

- Delete the existing tag: `git tag -d v1.0.0 && git push origin :refs/tags/v1.0.0`
- Create a new tag with incremented version
