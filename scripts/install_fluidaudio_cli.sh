#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install_fluidaudio_cli.sh <FluidAudio checkout> [--configuration debug|release] [--install-dir DIR] [--skip-build]

Builds FluidAudio's fluidaudiocli product and installs it as `fluidaudio`.
Default install dir is this repo's .venv/bin so `uv run voxis ...` can launch it.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fluid_audio_checkout=""
configuration="release"
install_dir="$repo_root/.venv/bin"
skip_build=0

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

fluid_audio_checkout="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configuration)
      configuration="${2:-}"
      shift 2
      ;;
    --install-dir)
      install_dir="${2:-}"
      shift 2
      ;;
    --skip-build)
      skip_build=1
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

case "$configuration" in
  debug|release) ;;
  *)
    echo "--configuration must be debug or release" >&2
    exit 2
    ;;
esac

if [[ ! -f "$fluid_audio_checkout/Package.swift" ]]; then
  echo "FluidAudio checkout must contain Package.swift: $fluid_audio_checkout" >&2
  exit 2
fi

mkdir -p "$install_dir"

if [[ "$skip_build" -eq 0 ]]; then
  swift build --package-path "$fluid_audio_checkout" --configuration "$configuration" --product fluidaudiocli
  bin_dir="$(swift build --package-path "$fluid_audio_checkout" --configuration "$configuration" --show-bin-path)"
  built_cli="$bin_dir/fluidaudiocli"
else
  built_cli="$(find "$fluid_audio_checkout/.build" -path "*/$configuration/fluidaudiocli" -type f -perm -111 | head -n 1)"
fi

if [[ ! -x "$built_cli" ]]; then
  echo "Built CLI was not found or not executable: $built_cli" >&2
  exit 1
fi

installed_cli="$install_dir/fluidaudio"
rm -f "$installed_cli"
install -m 755 "$built_cli" "$installed_cli"
"$installed_cli" --help >/dev/null

echo "Installed fluidaudio from: $built_cli"
echo "Installed fluidaudio to: $installed_cli"
echo "Install dir: $install_dir"
echo "Verify PATH: uv run which fluidaudio"
echo "Smoke with audio: uv run voxis-bench run-sortformer --audio /path/to/short-audio.m4a --out /tmp/voxis-fluidaudio-smoke --model-id smoke"
