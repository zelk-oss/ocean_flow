#!/usr/bin/env python3
"""
Post-generation hook for cookiecutter template.

This script:
1. Stores cookiecutter template metadata for future reference
2. Initializes git repository
3. Fetches the template repository to resolve the current SHA
4. Sets up git configuration for template tracking
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, check=True, capture_output=False):
    """Execute a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            shell=isinstance(cmd, str)
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd if isinstance(cmd, list) else [cmd])}")
        print(f"Error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
        if check:
            sys.exit(1)
        return None


def store_template_metadata(template_sha: str) -> dict:
    """Store cookiecutter template metadata for future updates."""
    metadata = {
        "template_repo": "{{ cookiecutter.template_repo | default('') }}",
        "template_branch": "{{ cookiecutter.template_branch | default('main') }}",
        "template_sha": template_sha,
        "cookiecutter_context": {
            "project_name": "{{ cookiecutter.project_name }}",
            "project_slug": "{{ cookiecutter.project_slug }}",
            "project_year": "{{ cookiecutter.project_year }}",
            "start_version": "{{ cookiecutter.start_version }}",
            "author_name": "{{ cookiecutter.author_name }}",
            "author_email": "{{ cookiecutter.author_email }}",
            "description": "{{ cookiecutter.description }}",
            "template_repo": "{{ cookiecutter.template_repo | default('') }}",
            "template_branch": "{{ cookiecutter.template_branch | default('main') }}",
        },
        "generated_at": datetime.now().isoformat(),
        "generated_with": "cookiecutter",
    }

    metadata_file = Path(".cookiecutter-metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Template metadata stored in {metadata_file}")
    return metadata


def init_git_repo():
    """Initialize git repository with basic configuration."""
    print("\n📦 Initializing git repository...")
    run_command(["git", "init"])
    print("✅ Git repository initialized")


def create_gitignore():
    """Ensure .gitignore exists to avoid committing sensitive files."""
    gitignore_path = Path(".gitignore")

    if not gitignore_path.exists():
        default_ignores = [
            "# Cookiecutter",
            ".cookiecutter-metadata.json",
            "",
            "# Python",
            "__pycache__/",
            "*.py[cod]",
            "*$py.class",
            "*.so",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "",
            "# Conda",
            "conda-env/",
            "env/",
            "venv/",
            "",
            "# IDEs",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "*~",
            ".DS_Store",
        ]
        with open(gitignore_path, "w") as f:
            f.write("\n".join(default_ignores) + "\n")
        print("✅ .gitignore created with default entries")
    else:
        print("ℹ️  .gitignore already exists")


def add_template_remote(template_repo):
    """Add the template repository as a remote."""
    print(f"\n🔗 Adding template repository as remote...")

    result = run_command(
        ["git", "remote", "add", "template", template_repo],
        check=False,
        capture_output=True
    )

    if result.returncode == 0:
        print(f"✅ Template remote added: {template_repo}")
    else:
        if "already exists" in result.stderr:
            print(f"ℹ️  Template remote already exists")
        else:
            print(f"⚠️  Could not add template remote: {result.stderr}")


def fetch_template(template_branch="main"):
    """Fetch the template repository and return its HEAD SHA."""
    print(f"\n📥 Fetching template repository (branch: {template_branch})...")

    result = run_command(
        ["git", "fetch", "template", template_branch],
        check=False,
        capture_output=True
    )

    if result.returncode != 0:
        print(f"⚠️  Could not fetch template: {result.stderr}")
        print("   This may be expected if the remote repository is not accessible yet.")
        print("   You can manually sync by running: git fetch template")
        return None

    print(f"✅ Template fetched successfully")

    sha_result = run_command(
        ["git", "rev-parse", f"template/{template_branch}"],
        check=False,
        capture_output=True
    )
    sha = sha_result.stdout.strip() if sha_result.returncode == 0 else ""
    return sha


def commit_generated_files():
    """Commit the generated files on main branch."""
    print(f"\n💾 Committing generated project files...")

    run_command(["git", "add", "."])

    result = run_command(
        ["git", "commit", "-m", "Initial commit: Generated from cookiecutter template"],
        check=False,
        capture_output=True
    )

    if result.returncode == 0:
        print("✅ Initial commit created")
    else:
        print("⚠️  No files to commit or commit failed")


def print_summary(metadata):
    """Print a summary of what was configured."""
    print("\n" + "="*70)
    print("🎉 COOKIECUTTER TEMPLATE SETUP COMPLETE")
    print("="*70)
    print(f"\n📋 Project: {metadata['cookiecutter_context']['project_name']}")
    print(f"📦 Template: {metadata['template_repo']}")
    print(f"🌿 Template Branch: {metadata['template_branch']}")
    sha = metadata.get("template_sha", "")
    if sha:
        print(f"🔖 Template SHA: {sha[:12]}")
    print("\n🔄 Workflow for template updates:")
    print("  1. Run: scripts/sync-template.sh --check")
    print("  2. Run: scripts/sync-template.sh --merge")
    print("  3. Resolve any merge conflicts")
    print("  4. Review changes, run tests, and push")
    print("\n📖 For more info, see the 'Template Updates' section in the documentation")
    print("="*70 + "\n")


def _display_post_gen_message():
    """Display an inlined post-generation message"""
    project_name = "{{ cookiecutter.project_name }}"
    project_slug = "{{ cookiecutter.project_slug }}"
    author_name = "{{ cookiecutter.author_name }}"

    message = f"""
{'=' * 60}
{project_name}  —  project created successfully!
{'=' * 60}

Next steps
----------

1. Create and activate the conda environment and install requirements:

    conda env create -f environment.yml
    conda activate {project_slug}
    uv pip install -r requirements.txt

2. Install the package in editable mode:

    pip install -e .

3. Implement your changes as described in the TODOs in the code.

4. Start training with the default config:

    python scripts/train.py

5. Run the test suite to verify everything is wired up:

    pytest

Happy modelling,
{author_name}
{'=' * 60}
    """
    print(message)


def main():
    """Main execution flow."""
    try:
        print("🚀 Starting cookiecutter post-generation setup...\n")

        template_repo = "{{ cookiecutter.template_repo | default('') }}"
        template_branch = "{{ cookiecutter.template_branch | default('main') }}"

        # Initialize git first — everything else depends on a repo existing
        init_git_repo()
        create_gitignore()

        # Commit generated files
        commit_generated_files()

        # Fetch the template to resolve its current SHA (may fail if offline)
        template_sha = ""
        if template_repo:
            add_template_remote(template_repo)
            template_sha = fetch_template(template_branch) or ""
        else:
            print("\nℹ️  No template_repo configured; skipping remote setup")

        # Store full context + resolved SHA (file is gitignored, lives on disk)
        metadata = store_template_metadata(template_sha)

        # Print summary
        print_summary(metadata)

        # Display the post-generation user message
        _display_post_gen_message()

        print("✨ Setup complete! You can now start using your project.")
        print("   Run scripts/sync-template.sh --check to check for template updates.\n")

    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
