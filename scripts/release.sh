#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/release.sh <commit> [--fluidaudio PATH]

Installs Voxis from one exact commit of this repository as a global uv tool,
with dependencies constrained to that commit's uv.lock. The installed tool is
a snapshot: later edits or commits in this checkout do not reach it until the
script is run again.

  <commit>           commit to release (sha, tag, or branch; resolved to a sha)
  --fluidaudio PATH  also install this FluidAudio CLI binary as `fluidaudio`
                     next to the installed `voxis` (needed for --diarize). Only
                     the binary is copied, so it must run standalone; use
                     ./scripts/install.sh --with-diarize to build one with its
                     frameworks.

Requires macOS on Apple Silicon, uv, git, and ffmpeg.
USAGE
}

commit=""
fluidaudio_src=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fluidaudio)
      [[ $# -ge 2 ]] || { echo "--fluidaudio needs a path" >&2; exit 2; }
      fluidaudio_src="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      [[ -z "$commit" ]] || { echo "Only one commit may be given" >&2; exit 2; }
      commit="$1"
      shift
      ;;
  esac
done
if [[ -z "$commit" ]]; then
  usage >&2
  exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Voxis requires macOS on Apple Silicon (arm64)." >&2
  exit 1
fi
for tool in uv git ffmpeg; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required." >&2
    exit 1
  fi
done
if [[ -n "$fluidaudio_src" && ! -x "$fluidaudio_src" ]]; then
  echo "FluidAudio binary not found or not executable: $fluidaudio_src" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="$(git -C "$repo_root" rev-parse --verify --quiet "${commit}^{commit}")" || {
  echo "Not a commit in $repo_root: $commit" >&2
  exit 1
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Constraints come from the released commit's lock file, not from the working tree.
mkdir "$work/src"
git -C "$repo_root" archive "$sha" | tar -x -C "$work/src"
uv export --quiet --project "$work/src" --frozen --no-dev --no-emit-project --no-hashes \
  --output-file "$work/constraints.txt"

# Stage FluidAudio before touching the tool dir: the source may be the installed copy itself
# or live inside the uv environment that the reinstall replaces.
if [[ -n "$fluidaudio_src" ]]; then
  cp -L "$fluidaudio_src" "$work/fluidaudio"
fi

uv tool install --force --python 3.12 --constraints "$work/constraints.txt" \
  "voxis @ git+file://$repo_root@$sha"

tool_bin="$(uv tool dir --bin)"
tool_python="$(uv tool dir)/voxis/bin/python"

if [[ -n "$fluidaudio_src" ]]; then
  rm -f "$tool_bin/fluidaudio"
  install -m 755 "$work/fluidaudio" "$tool_bin/fluidaudio"
fi

# Verify from outside the repository, with only the tool bin dir and system paths on PATH.
verify_path="$tool_bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
doctor=("$tool_bin/voxis" doctor)
if [[ -x "$tool_bin/fluidaudio" ]]; then
  doctor+=(--require-diarize)
fi
(cd / && PATH="$verify_path" "${doctor[@]}")
# doctor only locates FluidAudio; make sure the installed binary actually starts.
if [[ -x "$tool_bin/fluidaudio" ]]; then
  (cd / && PATH="$verify_path" "$tool_bin/fluidaudio" --help >/dev/null)
fi

installed_url="$("$tool_python" -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("voxis").read_text("direct_url.json"))["vcs_info"]["commit_id"])')"
if [[ "$installed_url" != "$sha" ]]; then
  echo "Installed commit $installed_url does not match requested $sha" >&2
  exit 1
fi
mlx_audio_version="$("$tool_python" -c 'import importlib.metadata as m; print(m.version("mlx-audio"))')"

echo
echo "Released voxis commit: $sha"
echo "Binary: $tool_bin/voxis"
echo "mlx-audio: $mlx_audio_version"
if [[ -x "$tool_bin/fluidaudio" ]]; then
  echo "FluidAudio: $tool_bin/fluidaudio"
else
  echo "FluidAudio: not installed in $tool_bin (diarization needs --fluidaudio PATH)"
fi
