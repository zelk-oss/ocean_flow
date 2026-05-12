Template Updates
================

This project was generated from a cookiecutter template and can be kept in sync with template updates.

Checking for template updates
-----------------------------

A GitHub Action automatically checks for template updates once per day. When updates are available, 
a pull request will be opened to the `dev` branch.

Manual template sync
--------------------

To manually check for and merge template updates:

```bash
# Fetch latest template changes
git fetch template

# Check what changed
git log --oneline main..template-sync

# Merge template updates into current branch
git merge template-sync

# Resolve any conflicts and commit
```

Understanding the template-sync branch
--------------------------------------

- **template-sync**: This branch always mirrors the template repository (automatically maintained)
- **main/dev**: Your project branches - merge from template-sync to get updates
