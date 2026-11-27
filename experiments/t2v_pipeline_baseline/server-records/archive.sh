set -euo pipefail
run="$HOME/experiments/t2v_pipeline_baseline.VN08gB"
repo="$HOME/lmp-e1-e2-followup-20261121/repo"
bundle="$run/publish"
path=experiments/t2v_pipeline_baseline
branch=refs/heads/codex/t2v-pipeline-baseline-20261122
test -f "$run/COMPLETE"
test -f "$run/RGB_COMPLETE"
mkdir "$bundle"
tar -xf - -C "$bundle"
test -s "$bundle/$path/BASELINE_REPORT.md"
test -s "$bundle/$path/results/dit_step_timing.csv"
test -s "$bundle/$path/figures/pipeline_breakdown.png"
base=$(git -C "$repo" rev-parse HEAD)
git_dir=$(git -C "$repo" rev-parse --absolute-git-dir)
export GIT_INDEX_FILE="$run/baseline-git-index"
cd "$bundle"
git_archive() { git --git-dir="$git_dir" --work-tree="$bundle" "$@"; }
git_archive read-tree "$base"
git_archive add -f -- "$path"
git_archive diff --cached --quiet -- . ":(exclude)$path"
git_archive diff --cached --check -- "$path/scripts" "$path/README.md" "$path/BASELINE_REPORT.md"
tree=$(git_archive write-tree)
export GIT_AUTHOR_NAME=Codex GIT_COMMITTER_NAME=Codex
export GIT_AUTHOR_EMAIL=codex@users.noreply.github.com GIT_COMMITTER_EMAIL=codex@users.noreply.github.com
commit=$(git_archive commit-tree "$tree" -p "$base" -m "Measure Wan2.1 text-to-video pipeline baseline")
git -C "$repo" update-ref "$branch" "$commit" ""
unset GIT_INDEX_FILE
git -C "$repo" log -1 --format='%H %s' "$branch"
git -C "$repo" diff-tree --no-commit-id --name-only -r "$commit" | wc -l
cp "$bundle/$path/BASELINE_REPORT.md" "$run/BASELINE_REPORT.md"
git -C "$repo" status --short
