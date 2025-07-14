# Release Workflow Examples

## Quick Start

1. **Make your changes and commit with appropriate message:**

   ```bash
   git add .
   git commit -m "feat: add new GBIF species dataset loader"
   git push origin master
   ```

2. **The system will automatically:**
   - Detect the `feat:` prefix
   - Create a new minor version tag (e.g., v0.1.0 → v0.2.0)
   - Trigger the release workflow
   - Build and publish to PyPI
   - Create GitHub release with changelog

## Commit Message Conventions

| Type | Example | Version Bump |
|------|---------|--------------|
| `feat:` | `feat: add audio dataset support` | Minor (0.1.0 → 0.2.0) |
| `fix:` | `fix: handle missing metadata gracefully` | Patch (0.1.0 → 0.1.1) |
| `feat!:` | `feat!: redesign API for better performance` | Major (0.1.0 → 1.0.0) |
| `[release]` | `[release] prepare for stable release` | Patch (0.1.0 → 0.1.1) |

## Manual Release Steps

If you prefer manual control:

```bash
# Create and push a tag manually
git tag -a v1.0.0 -m "Release version 1.0.0

- Add comprehensive GBIF dataset loaders
- Implement PyTorch DataLoader compatibility
- Add extensive documentation and examples"

git push origin v1.0.0
```

## Pre-release Workflow

For beta/alpha releases:

```bash
git tag -a v1.0.0-beta.1 -m "Beta release v1.0.0-beta.1"
git push origin v1.0.0-beta.1
```

This will create a pre-release on GitHub and publish to PyPI with the beta tag.

## First Release

For the very first release of the project:

```bash
git commit -m "[release] initial release of TorchGBIF"
git push origin master
```

This will create v0.0.1 (or bump from current version) and trigger the full release process.
