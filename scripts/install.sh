#!/usr/bin/env bash
set -euo pipefail

FLUIDAUDIO_TAG="v0.9.1"
FLUIDAUDIO_REPO="https://github.com/FluidInference/FluidAudio.git"

usage() {
  cat <<'USAGE'
Usage: ./scripts/install.sh [--with-diarize]

Installs Voxis as a global uv tool. With --with-diarize, also builds the pinned
FluidAudio CLI. Requires macOS on Apple Silicon, uv, ffmpeg, and (for
diarization) Git plus a working Xcode/Swift C++ toolchain.

This script does not install Homebrew, ffmpeg, or Xcode automatically.
USAGE
}

with_diarize=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-diarize)
      with_diarize=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Voxis requires macOS on Apple Silicon (arm64)." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: brew install uv" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it with: brew install ffmpeg" >&2
  exit 1
fi
if [[ "$with_diarize" -eq 1 ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "Git is required for --with-diarize. Install the Xcode command-line tools with: xcode-select --install" >&2
    exit 1
  fi
  if ! command -v swift >/dev/null 2>&1; then
    echo "Swift is required for --with-diarize. Install the Xcode command-line tools with: xcode-select --install" >&2
    exit 1
  fi
  if ! command -v xcrun >/dev/null 2>&1 || \
    ! printf '#include <cmath>\nint main() { return 0; }\n' | xcrun clang++ -x c++ - -fsyntax-only >/dev/null 2>&1; then
    echo "The Xcode C++ toolchain is not usable (failed to compile <cmath>)." >&2
    echo "Install or update full Xcode. If it is already installed, run:" >&2
    echo "  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer" >&2
    exit 1
  fi
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_dir="$HOME/.config/voxis"
config_path="$config_dir/.env"

uv tool install --force --python 3.12 "$repo_root"
tool_bin="$(uv tool dir --bin)"

if [[ ! -f "$config_path" ]]; then
  mkdir -p "$config_dir"
  install -m 600 "$repo_root/.env.example" "$config_path"
  echo "Created config: $config_path"
else
  echo "Kept existing config: $config_path"
fi

if [[ "$with_diarize" -eq 1 ]]; then
  data_root="${XDG_DATA_HOME:-$HOME/.local/share}/voxis"
  fluid_checkout="$data_root/FluidAudio-$FLUIDAUDIO_TAG"
  if [[ ! -d "$fluid_checkout/.git" ]]; then
    mkdir -p "$data_root"
    git clone --depth 1 --branch "$FLUIDAUDIO_TAG" "$FLUIDAUDIO_REPO" "$fluid_checkout"
  fi
  swift build --package-path "$fluid_checkout" --configuration release --product fluidaudiocli
  fluid_bin_dir="$(swift build --package-path "$fluid_checkout" --configuration release --show-bin-path)"
  built_fluidaudio="$fluid_bin_dir/fluidaudiocli"
  built_espeak="$fluid_bin_dir/ESpeakNG.framework"
  if [[ ! -x "$built_fluidaudio" ]]; then
    echo "Built FluidAudio CLI was not found or not executable: $built_fluidaudio" >&2
    exit 1
  fi
  if [[ ! -d "$built_espeak" ]]; then
    echo "Built FluidAudio dependency was not found: $built_espeak" >&2
    exit 1
  fi
  rm -f "$tool_bin/fluidaudio"
  rm -rf "$tool_bin/ESpeakNG.framework"
  install -m 755 "$built_fluidaudio" "$tool_bin/fluidaudio"
  cp -R "$built_espeak" "$tool_bin/ESpeakNG.framework"
  "$tool_bin/fluidaudio" --help >/dev/null
fi

doctor=("$tool_bin/voxis" doctor)
if [[ "$with_diarize" -eq 1 ]]; then
  doctor+=(--require-diarize)
fi
PATH="$tool_bin:$PATH" "${doctor[@]}"

echo
echo "Installed Voxis command: $tool_bin/voxis"
if [[ ":$PATH:" != *":$tool_bin:"* ]]; then
  echo "Add that directory to PATH with: uv tool update-shell"
fi
echo "First transcription downloads several GB of MLX models and may take a while."
