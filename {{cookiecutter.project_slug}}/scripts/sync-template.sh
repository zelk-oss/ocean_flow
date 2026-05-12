#!/usr/bin/env bash
#
# Template sync helper — re-renders the template at the latest commit and
# merges the delta into your current branch.
#
# Usage:
#   ./sync-template.sh                        # Check for updates (dry run)
#   ./sync-template.sh --merge                # Update template-sync and merge
#   ./sync-template.sh --set-remote <url>     # Change the template repo URL
#   ./sync-template.sh --set-branch <branch>  # Change the tracked branch
#   ./sync-template.sh --help                 # Show this help
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header()  { echo -e "${BLUE}===================================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}===================================================${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
METADATA_FILE=".cookiecutter-metadata.json"

check_metadata() {
    if [ ! -f "$METADATA_FILE" ]; then
        print_error "Metadata file not found: $METADATA_FILE"
        print_info "This project may not have been generated from the cookiecutter template."
        exit 1
    fi
}

# Read a top-level key from the metadata JSON.
# Falls back to grep+sed when jq is unavailable.
read_meta() {
    local key="$1"
    if command -v jq &>/dev/null; then
        jq -r ".$key // empty" "$METADATA_FILE"
    else
        grep "\"$key\"" "$METADATA_FILE" \
            | sed 's/.*: "\(.*\)".*/\1/' \
            | head -1
    fi
}

# Read a key nested under cookiecutter_context.
# Falls back to the flat top-level format produced by older hook versions.
read_context() {
    local key="$1"
    local val=""
    if command -v jq &>/dev/null; then
        val=$(jq -r ".cookiecutter_context.$key // empty" "$METADATA_FILE")
        # Old metadata: flat keys at top level
        if [ -z "$val" ]; then
            val=$(jq -r ".$key // empty" "$METADATA_FILE")
        fi
    else
        # Try nested block first
        val=$(awk '/"cookiecutter_context"/,/^\s*\}/' "$METADATA_FILE" \
            | grep "\"$key\"" \
            | sed 's/.*: "\(.*\)".*/\1/' \
            | head -1)
        # Fall back to flat format
        if [ -z "$val" ]; then
            val=$(grep "\"$key\"" "$METADATA_FILE" \
                | sed 's/.*: "\(.*\)".*/\1/' \
                | head -1)
        fi
    fi
    echo "$val"
}

# Write a new value for any top-level key in a metadata file in-place.
# $1 = key; $2 = new value; $3 = path to metadata file (defaults to $METADATA_FILE)
update_meta_value() {
    local key="$1"
    local value="$2"
    local meta_path="${3:-$METADATA_FILE}"
    if command -v jq &>/dev/null; then
        local tmp
        tmp=$(mktemp)
        jq --arg v "$value" ".$key = \$v" "$meta_path" > "$tmp"
        mv "$tmp" "$meta_path"
    else
        sed -i.bak "s|\"$key\": \"[^\"]*\"|\"$key\": \"$value\"|" \
            "$meta_path"
        rm -f "${meta_path}.bak"
    fi
}

# Write a new template_sha into a metadata file in-place.
# $1 = new SHA; $2 = path to metadata file (defaults to $METADATA_FILE)
update_meta_sha() {
    local new_sha="$1"
    local meta_path="${2:-$METADATA_FILE}"
    update_meta_value "template_sha" "$new_sha" "$meta_path"
}

load_metadata() {
    TEMPLATE_REPO=$(read_meta "template_repo")
    TEMPLATE_BRANCH=$(read_meta "template_branch")
    STORED_SHA=$(read_meta "template_sha")

    PROJECT_SLUG=$(read_context "project_slug")
    PROJECT_NAME=$(read_context "project_name")
    PROJECT_YEAR=$(read_context "project_year")
    START_VERSION=$(read_context "start_version")
    AUTHOR_NAME=$(read_context "author_name")
    AUTHOR_EMAIL=$(read_context "author_email")
    DESCRIPTION=$(read_context "description")

    if [ -z "$TEMPLATE_REPO" ]; then
        print_error "template_repo is not set in $METADATA_FILE"
        print_info "Set it manually: edit $METADATA_FILE and add" \
                   "\"template_repo\": \"<your-template-repo-url>\""
        exit 1
    fi
    if [ -z "$TEMPLATE_BRANCH" ]; then
        print_error "template_branch is not set in $METADATA_FILE"
        print_info "Set it manually: edit $METADATA_FILE and add" \
                   "\"template_branch\": \"<branch-name>\""
        exit 1
    fi
    if [ -z "$PROJECT_SLUG" ]; then
        print_error "cookiecutter_context.project_slug not set in $METADATA_FILE"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
check_git_clean() {
    if ! git diff --quiet || ! git diff --cached --quiet; then
        print_warning "Working directory has uncommitted changes."
        print_info "Stash or commit your changes before syncing."
        exit 1
    fi
    local untracked
    untracked=$(git ls-files --others --exclude-standard)
    if [ -n "$untracked" ]; then
        print_warning "Working directory has untracked files."
        print_info "Stash or commit your changes before syncing."
        exit 1
    fi
}

ensure_template_remote() {
    if ! git remote get-url template &>/dev/null; then
        print_info "Adding template remote: $TEMPLATE_REPO"
        git remote add template "$TEMPLATE_REPO"
    fi
}

fetch_template() {
    print_info "Fetching template remote (branch: $TEMPLATE_BRANCH)..."
    if ! git fetch template "$TEMPLATE_BRANCH" 2>&1; then
        print_error "Failed to fetch template remote."
        print_info "Check access to: $TEMPLATE_REPO"
        exit 1
    fi
    print_success "Template fetched."
}

get_new_sha() {
    git rev-parse "template/$TEMPLATE_BRANCH"
}

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
check_cookiecutter() {
    if ! command -v cookiecutter &>/dev/null; then
        print_error "cookiecutter is not installed / not on PATH."
        print_info "Install it with: pip install cookiecutter"
        exit 1
    fi
}

render_template() {
    local template_sha="$1"
    local render_dir="$2"   # output goes here: render_dir/<project_slug>/

    local archive_dir
    archive_dir=$(mktemp -d)
    trap 'rm -rf "$archive_dir"' RETURN

    print_info "Checking out template at $template_sha..."
    git archive "$template_sha" | tar -x -C "$archive_dir"

    print_info "Rendering template with cookiecutter..."
    cookiecutter "$archive_dir" \
        --no-input \
        --output-dir "$render_dir" \
        project_name="$PROJECT_NAME" \
        project_slug="$PROJECT_SLUG" \
        project_year="$PROJECT_YEAR" \
        start_version="$START_VERSION" \
        author_name="$AUTHOR_NAME" \
        author_email="$AUTHOR_EMAIL" \
        description="$DESCRIPTION" \
        template_repo="$TEMPLATE_REPO" \
        template_branch="$TEMPLATE_BRANCH"

    print_success "Template rendered."
    trap - RETURN
}

# ---------------------------------------------------------------------------
# template-sync branch management
# ---------------------------------------------------------------------------
ensure_template_sync_branch() {
    if ! git rev-parse --verify template-sync &>/dev/null; then
        print_warning "template-sync branch does not exist."
        print_info "Creating it from the initial commit of main..."
        local initial_commit
        initial_commit=$(git rev-list --max-parents=0 HEAD)
        git branch template-sync "$initial_commit"
        print_success "template-sync branch created."
    fi
}

remove_existing_template_sync_worktree() {
    local repo_root
    repo_root=$(git rev-parse --show-toplevel)

    local existing_worktree
    existing_worktree=$(git worktree list --porcelain \
        | awk '/^worktree / { wt=$2 } /^branch refs\/heads\/template-sync$/ { print wt; exit }')

    if [ -z "$existing_worktree" ]; then
        return 0
    fi

    if [ "$existing_worktree" = "$repo_root" ]; then
        print_error "template-sync is checked out in the main working tree."
        print_info "Switch to a different branch and rerun --merge."
        return 1
    fi

    print_warning "template-sync is already checked out at: $existing_worktree"
    print_info "Removing stale template-sync worktree..."
    if ! git worktree remove --force "$existing_worktree"; then
        print_error "Failed to remove existing template-sync worktree."
        print_info "Try running: git worktree list  and  git worktree prune"
        return 1
    fi
    print_success "Removed stale template-sync worktree."
    return 0
}

apply_rendered_to_sync_branch() {
    local rendered_root="$1"   # path to <render_dir>/<project_slug>/
    local new_sha="$2"

    # Remove stale worktree registrations left by previous interrupted runs.
    git worktree prune

    # If template-sync is still checked out in another worktree, remove it.
    if ! remove_existing_template_sync_worktree; then
        return 1
    fi

    # Work in a temporary worktree so the main working tree is never touched.
    local worktree_dir
    worktree_dir=$(mktemp -d)
    rmdir "$worktree_dir"   # git worktree add creates the directory itself
    if ! git worktree add "$worktree_dir" template-sync; then
        print_error "Failed to create worktree for template-sync."
        print_info "Try running: git worktree list  and  git worktree prune"
        return 1
    fi
    # shellcheck disable=SC2064
    trap "git worktree remove --force '$worktree_dir' 2>/dev/null || true" RETURN

    # Replace all tracked files with the rendered output.
    git -C "$worktree_dir" ls-files -z \
        | xargs -0 git -C "$worktree_dir" rm -f --quiet

    rsync -a --exclude='.git' "$rendered_root/" "$worktree_dir/"

    # Update metadata SHA inside the worktree.
    update_meta_sha "$new_sha" "$worktree_dir/$METADATA_FILE"

    git -C "$worktree_dir" add -A
    if git -C "$worktree_dir" diff --cached --quiet; then
        print_info "No changes to template-sync after rendering."
        return 1   # signal: nothing changed
    fi

    git -C "$worktree_dir" commit -m "chore: Sync template to ${new_sha:0:12}"
    print_success "template-sync updated."
    trap - RETURN
    return 0
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_set_remote() {
    local new_url="$1"
    print_header "Updating Template Remote"
    print_info "Old: $TEMPLATE_REPO"
    print_info "New: $new_url"

    update_meta_value "template_repo" "$new_url"
    print_success "template_repo updated in $METADATA_FILE."

    if git remote get-url template &>/dev/null; then
        git remote set-url template "$new_url"
        print_success "Git 'template' remote URL updated."
    else
        print_info "Git 'template' remote not yet added; it will be" \
                   "added automatically on the next --check or --merge."
    fi
}

cmd_set_branch() {
    local new_branch_name="$1"
    print_header "Updating Template Branch"
    print_info "Old: $TEMPLATE_BRANCH"
    print_info "New: $new_branch_name"

    update_meta_value "template_branch" "$new_branch_name"
    print_success "template_branch updated in $METADATA_FILE."
}

show_usage() {
    cat <<EOF
${BLUE}Cookiecutter Template Sync${NC}

${YELLOW}Usage:${NC}
  $0 [OPTIONS]

${YELLOW}Options:${NC}
  --check              Fetch and compare, do not modify anything (default)
  --merge              Re-render template, update template-sync, merge into
                       current branch
  --status             Show branch and metadata status
  --set-remote <url>   Update template_repo in $METADATA_FILE (and the
                       'template' git remote if it already exists)
  --set-branch <name>  Update template_branch in $METADATA_FILE
  --help               Show this help

${YELLOW}Workflow:${NC}
  1. $0 --check          (inspect what changed in the template)
  2. $0 --merge          (apply changes; resolve conflicts if any)
  3. Review, test, push

${YELLOW}How it works:${NC}
  template-sync is a local branch rooted at your project's initial commit.
  Each sync re-renders the template at the new SHA and commits the result to
  template-sync. Merging template-sync into your branch shows only the delta
  since the version you were initialised from.
EOF
}

cmd_check() {
    print_header "Checking for Template Updates"
    print_info "Template: $TEMPLATE_REPO  branch: $TEMPLATE_BRANCH"
    echo ""

    ensure_template_remote
    fetch_template

    local new_sha
    new_sha=$(get_new_sha)

    if [ "$new_sha" = "$STORED_SHA" ]; then
        print_success "Already up to date (SHA: ${new_sha:0:12})."
        exit 0
    fi

    print_warning "New template commits available."
    echo ""
    echo -e "${YELLOW}Changes since your version (${STORED_SHA:0:12}):${NC}"
    git log --oneline --graph \
        "${STORED_SHA}..template/${TEMPLATE_BRANCH}" 2>/dev/null \
        | head -20 \
        || git log --oneline "template/$TEMPLATE_BRANCH" | head -20
    echo ""
    print_info "Run:  $0 --merge  to apply these changes."
}

cmd_merge() {
    print_header "Syncing Template Updates"

    if [ -z "$STORED_SHA" ]; then
        print_error "template_sha is not set in $METADATA_FILE"
        print_info "Set it manually: edit $METADATA_FILE and add" \
                   "\"template_sha\": \"<last-synced-commit-sha>\""
        print_info "Use the full 40-character SHA of the template commit" \
                   "your project was last synced with."
        exit 1
    fi

    check_git_clean
    check_cookiecutter
    ensure_template_remote
    fetch_template

    local new_sha
    new_sha=$(get_new_sha)

    if [ "$new_sha" = "$STORED_SHA" ]; then
        print_success "Already up to date (SHA: ${new_sha:0:12}). Nothing to do."
        exit 0
    fi

    print_info "New SHA: ${new_sha:0:12}  (was: ${STORED_SHA:0:12})"

    echo ""
    local COMMIT_LOG
    COMMIT_LOG=$(git log --oneline --reverse \
        "${STORED_SHA}..template/${TEMPLATE_BRANCH}" 2>/dev/null) || true

    if [ -n "$COMMIT_LOG" ]; then
        echo -e "${YELLOW}Commits since last sync (${STORED_SHA:0:12}):${NC}"
        echo "$COMMIT_LOG"
    else
        print_info "No commit messages available (SHA range may be invalid)."
    fi

    echo ""
    local answer
    read -rp "Do you want to perform the update? (yes/no): " answer
    answer=$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')
    case "$answer" in
        yes|y) ;;
        *)
            print_info "Update aborted by user."
            exit 0
            ;;
    esac

    render_dir=$(mktemp -d)
    trap 'rm -rf "$render_dir"' EXIT

    render_template "$new_sha" "$render_dir"

    local rendered_root="$render_dir/$PROJECT_SLUG"
    if [ ! -d "$rendered_root" ]; then
        print_error "Rendered output not found at: $rendered_root"
        exit 1
    fi

    ensure_template_sync_branch

    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    if apply_rendered_to_sync_branch "$rendered_root" "$new_sha"; then
        echo ""
        print_info "Merging template-sync into $current_branch..."
        local MERGE_SUBJECT="Merging of template into repository (${STORED_SHA:0:12} -> ${new_sha:0:12})"
        local merge_args=(git merge --no-ff -m "$MERGE_SUBJECT")
        if [ -n "$COMMIT_LOG" ]; then
            merge_args+=(-m "$COMMIT_LOG")
        fi
        merge_args+=(template-sync)

        if "${merge_args[@]}"; then
            # Update metadata SHA after the merge so we do not
            # dirty the tree before the merge commit is created.
            update_meta_sha "$new_sha"
            if git ls-files --error-unmatch "$METADATA_FILE" >/dev/null 2>&1; then
                git add "$METADATA_FILE"
                if ! git diff --cached --quiet; then
                    git commit -m "chore: Update template_sha to ${new_sha:0:12}"
                fi
            else
                print_info "$METADATA_FILE is not tracked; skipping metadata add."
            fi

            print_success "Merge complete. Template SHA updated to ${new_sha:0:12}."
            echo ""
            print_info "Next steps:"
            echo "  1. Review the changes"
            echo "  2. Run tests"
            echo "  3. Push when ready"
        else
            # Merge had conflicts — update metadata so the user
            # does not have to do it manually after resolving.
            update_meta_sha "$new_sha"
            print_warning "Merge conflicts detected. Template SHA updated to ${new_sha:0:12} in metadata."
            echo ""
            print_info "Resolve conflicts, then:"
            echo "  1. git add <resolved-files>"
            echo "  2. git commit"
            echo ""
            print_info "After resolving, verify that template_sha in"
            print_info "$METADATA_FILE is set to: $new_sha"
            exit 1
        fi
    else
        print_info "template-sync is identical to current project — nothing merged."
    fi
}

cmd_status() {
    print_header "Template Sync Status"

    print_info "Template: $TEMPLATE_REPO  branch: $TEMPLATE_BRANCH"
    sha="${STORED_SHA:-<not set>}"
    print_info "Stored template SHA: ${sha:0:12}"
    echo ""
    echo -e "${YELLOW}Git branches:${NC}"
    git branch -vv | sed 's/^/  /'
    echo ""

    if git rev-parse --verify template-sync &>/dev/null; then
        local behind ahead
        behind=$(git rev-list --count HEAD..template-sync 2>/dev/null || echo "?")
        ahead=$(git rev-list --count template-sync..HEAD 2>/dev/null || echo "?")
        print_success "template-sync exists"
        echo "  Current branch is $behind commit(s) behind template-sync"
        echo "  Current branch is $ahead commit(s) ahead of template-sync"
    else
        print_warning "template-sync branch not found"
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main() {
    local cmd="--check"
    local new_repo=""
    local new_branch_name=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check|--merge|--status|--help)
                cmd="$1"
                shift
                ;;
            --set-remote)
                if [[ -z "${2:-}" ]]; then
                    print_error "--set-remote requires a URL argument."
                    exit 1
                fi
                cmd="--set-remote"
                new_repo="$2"
                shift 2
                ;;
            --set-branch)
                if [[ -z "${2:-}" ]]; then
                    print_error "--set-branch requires a branch name."
                    exit 1
                fi
                cmd="--set-branch"
                new_branch_name="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done

    if [[ "$cmd" != "--help" ]]; then
        check_metadata
        load_metadata
    fi

    case "$cmd" in
        --check)      cmd_check ;;
        --merge)      cmd_merge ;;
        --status)     cmd_status ;;
        --set-remote) cmd_set_remote "$new_repo" ;;
        --set-branch) cmd_set_branch "$new_branch_name" ;;
        --help)       show_usage ;;
    esac
}

main "$@"
