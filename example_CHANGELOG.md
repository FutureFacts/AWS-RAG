# Version history

_Version history of the python template repository._

## 0.4.0 (2024-11-29)

Modified this repo to be more in line with usage requirements.
Transfered some learnings from this repo: <https://github.com/FutureFacts/hackathon-multimodal-rag>

- Removed `tox` usage
- Removed `sphinx` usage
- Removed multiple python version compatibility
  - Mostly at Future Facts we wouldn't require our projects to be compatibly with multiple python versions.
- Improved `CHANGELOG.md` and `README.md`
- Fixed `mypy`
- Replaced `check-test.yml` with `ci.yml`
- Updated to python 3.12

## 0.3.1 (2024-05-29)

- Moved template to the Future Facts organization
- Fixed github actions pipeline

## 0.3.0 (2024-04-24)

- Moved linters and formatters to ruff
- Added pre-commit-hook
- Added vscode settings and recommended extensions

## 0.2.0 (2023-09-15)

- Added github actions pipeline

## 0.1.0 (2023-09-11)

- first release
- documentations added: `README.md` and `explanation/*.md`
- .gitignore
- Makefile script
- Poetry framework
- tox framework
- mypy framework
