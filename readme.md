# 🛸 XQubit Project

This repository is optimized for **Jupyter Notebook** research with a focus on clean version control and metadata management. All collaborators **must** follow the setup instructions below to ensure repository integrity.

---

## 🛠 Prerequisites

Ensure you have the following installed on your system:

- **Python 3.14+**
- **Pipenv** (`pip install pipenv`)
- **Git**

---

## 🚀 Environment Setup

### 1. Initialize the Virtual Environment

Clone the repository and install the required dependencies (including development tools).

```bash
git clone https://github.com/Operator-Syn/xqubit-project.git && \
cd xqubit-project && \
pipenv install --dev
```

### 2. Configure Notebook Tools

We use `nbdime` for clean diffs/merges and `pre-commit` to automatically strip notebook metadata (execution counts, etc.) before every commit.

```bash
# Activate the virtual environment
pipenv shell

# Enable nbdime for this repository
nbdime config-git --enable

# Install the pre-commit hooks
pre-commit install
```

---

## 📋 Collaboration Workflow

### Dealing with Jupyter Notebooks

Because `.ipynb` files are JSON, standard Git tools often struggle with them. We have configured specialized utilities to handle them correctly:

- **Clean Diffs:** When you run `git diff`, you will see code changes only, without the JSON metadata noise.
- **Smart Merges:** If a notebook has a conflict, **do not** fix it manually in a text editor. Instead, use the visual merge tool:
  ```bash
  pipenv run nbdime mergetool
  ```
  This launches a web-based 3-pane merge editor in your browser.
- **Automatic Scrubbing:** You don't need to clear your outputs manually. The `pre-commit` hook will run `nbstripout` automatically when you `git commit`.

### Working on Features

1. **Create a branch:** `git checkout -b feature/your-feature-name`
2. **Commit your changes:** If a commit fails on the first try, it is likely because `pre-commit` cleaned your notebook metadata. Simply `git add` the file again and re-commit.

---

## 📂 Repository Structure

| File | Purpose |
|------|---------|
| `.pre-commit-config.yaml` | Configuration for notebook scrubbing |
| `.gitattributes` | Settings for `nbdime` diff/merge drivers |
| `Pipfile` | Dependency management |
| `.gitignore` | Configured to ignore `.orig` merge backups and local checkpoints |

---

## 💡 Tips

> **Adding new packages?** Always use `pipenv install <package>` instead of `pip install` so the `Pipfile` stays updated for everyone on the team.