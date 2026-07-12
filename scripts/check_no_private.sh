#!/usr/bin/env bash
# Leak guard (git-aware): fail if any PRIVATE/proprietary content would be COMMITTED
# or PUSHED -- i.e. is tracked or staged. The CUDA solver SOURCE (*.cu) and compiled
# BINARIES (bench_*) must NEVER be published. They MAY exist locally (git-ignored --
# e.g. a built bin/ for local runs) without tripping this guard; only what git would
# actually publish is checked. Run before committing and in CI.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
fail=0
say() { echo "LEAK: $*"; fail=1; }

if git rev-parse --git-dir >/dev/null 2>&1; then
  files=$(git ls-files)                 # tracked + staged (the index) = what git publishes
  scope="tracked/staged"
else
  files=$(find . -type f -not -path './.git/*' | sed 's#^\./##')   # no git repo: scan on-disk
  scope="on-disk (no git repo)"
fi

# 1. CUDA source
found=$(printf '%s\n' "$files" | grep -E '\.cu$' || true)
[ -n "$found" ] && say "CUDA source ($scope):"$'\n'"$found"

# 2. compiled solver binaries OUTSIDE bin/ (the bin/ binaries are shipped on purpose;
#    a bench_* anywhere else in the tree is a mistake)
found=$(printf '%s\n' "$files" | grep -E '(^|/)bench_[0-9a-z_]+$' | grep -vE '^bin/' || true)
[ -n "$found" ] && say "solver binary outside bin/ ($scope):"$'\n'"$found"

# 3. private artifacts / build tooling that must never be published
found=$(printf '%s\n' "$files" | grep -E '(^|/)(src/|docker/|docs/deployment\.md|docs/gotchas\.md|CMakeLists\.txt)' || true)
[ -n "$found" ] && say "private artifact ($scope):"$'\n'"$found"

# 4. oversized tracked files (binaries belong in private releases, not git)
while IFS= read -r f; do
  [ -n "$f" ] && [ -f "$f" ] || continue
  sz=$(wc -c <"$f" 2>/dev/null || echo 0)
  [ "$sz" -gt $((50 * 1024 * 1024)) ] && say "file > 50 MiB ($scope): $f ($((sz / 1024 / 1024)) MiB)"
done <<< "$files"

# 5. a submodule or remote pointing at the private repo
[ -f .gitmodules ] && say ".gitmodules present (could reference the private repo)"
if git rev-parse --git-dir >/dev/null 2>&1 && git remote -v 2>/dev/null | grep -qiE 'fullwave2-ultra-private|:.*-private'; then
  say "git remote references a private repo"
fi

# 6. MATLAB source (the .m lineage must never be re-added)
found=$(printf '%s\n' "$files" | grep -E '\.m$' || true)
[ -n "$found" ] && say "MATLAB source ($scope):"$'\n'"$found"

# 7. a matlab/ directory (the scrubbed reference helpers must never come back)
found=$(printf '%s\n' "$files" | grep -E '(^|/)matlab/' || true)
[ -n "$found" ] && say "matlab/ path ($scope):"$'\n'"$found"

# 8. the scrubbed numerical-method module (stencil*.py)
found=$(printf '%s\n' "$files" | grep -E '(^|/)stencil[^/]*\.py$' || true)
[ -n "$found" ] && say "stencil module ($scope):"$'\n'"$found"

# 9. .py/.md whose CONTENT leaks the scrubbed numerical-method coefficients
while IFS= read -r f; do
  [ -n "$f" ] && [ -f "$f" ] || continue
  case "$f" in
    *.py|*.md|*.h|*.hpp|*.cuh|*.cu)
      if grep -iEq 'drp_d|drp_dmap|dcmap_from_c|tan.?coef|tan.?huang' "$f" 2>/dev/null; then
        say "coefficient leak in content ($scope): $f"
      fi ;;
  esac
done <<< "$files"

# 10. C/CUDA headers (pure-Python package; the private drp_stencil.h and any
#     solver headers must never be published here)
found=$(printf '%s\n' "$files" | grep -E '\.(h|hpp|cuh)$' || true)
[ -n "$found" ] && say "C/CUDA header ($scope):"$'\n'"$found"

# 11. EMBARGOED SOURCE (PML/absorbing-boundary + differentiability + attenuation
#     calibration). The embargo covers SOURCE, not the shipped bin/ binaries
#     (maintainer decision 2026-07): none of the private-side modules, the
#     validation/optimizer scripts, their tests, or any manuscript material may
#     be (re-)added until the corresponding papers post.
found=$(printf '%s\n' "$files" | grep -E '(^|/)(fullwave2_ultra/(diff|adjoint|boundaries|attenuation)\.py|validation/(absorb|diff|stretch)_[^/]*|tests/test_(diff|adjoint|stretch|attenuation|fatt)[^/]*|tests/test_batch_eq_solo_fatt\.sh|manuscript/|scripts/gen_adjoint_taps\.py)' || true)
[ -n "$found" ] && say "embargoed source ($scope):"$'\n'"$found"

# 12. docs/README content leaking the embargoed feature CONTRACTS (a doc copied
#     from the private tree carries the USE_STRETCH/stretch.dat and
#     USE_FATT/fatt_coefs.dat sections; scrub them before publishing)
while IFS= read -r f; do
  [ -n "$f" ] && [ -f "$f" ] || continue
  case "$f" in
    docs/*|README.md)
      if grep -Eq 'USE_STRETCH|stretch\.dat|write_stretch_rundir|USE_FATT|fatt_coefs|simulate3d\(|_cpml_profiles' "$f" 2>/dev/null; then
        say "embargoed feature contract in docs ($scope): $f"
      fi ;;
  esac
done <<< "$files"

if [ "$fail" -eq 0 ]; then
  echo "OK ($scope): nothing private would be committed/published"
  echo "  (no .cu / .cuh / .h / bench_* / >50MiB / src / docker / deployment / gotchas / private remote;"
  echo "   no .m / matlab/ / stencil*.py / scrubbed-coefficient content;"
  echo "   no embargoed diff/adjoint/boundaries/attenuation source, manuscripts, or feature contracts)."
else
  echo "FAILED leak-check -- the above is tracked/staged and would be published."
  echo "  Unstage/remove it (local git-ignored copies are fine) before commit/push."
  exit 1
fi
