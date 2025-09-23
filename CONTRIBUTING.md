# Contributing
Contributing to pykilosort

## Installation

The installation process is the same for everyone. See the README.

## Dependencies

Dependencies need to be added to the conda environment files (`pyks2.yml`) and the requirements.txt. Please pin any critical dependencies to at least a minor version. Ideally we should avoid adding dependencies that are not actively maintained or are not widely used (unless we are willing to support them ourselves).

## Testing

### Running the tests

- **Unit tests** Simply run `pytest` in the root dir.
- **Integration tests** Run the `integration/integration_100s.py` script. Refer to the [integration/README](integration/README) on how to access the public test dataset.

## Contributing a feature

- [X] make sure the tests pass locally (see above)
- [X] [CHANGELOG.md](CHANGELOG.md) documents the changes, the date and the new version number in `./iblsorter/__init__.py`

### Reviewer steps for a feature PR
- [X] the CI passes
- [X] squash-merge upon a successful review

## Release

- [X] create tag corresponding to the version number `X.Y.Z` on the `main` branch

```shell
tag=X.Y.Z
git tag -a $tag 
git push origin $tag
```
