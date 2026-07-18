from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_installs_global_tool_and_preserves_existing_config(tmp_path: Path) -> None:
    env, log, tool_bin = install_fake_commands(tmp_path)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "ignored-xdg-config")

    first = subprocess.run(["bash", str(INSTALL_SCRIPT)], check=False, capture_output=True, text=True, env=env)

    config = tmp_path / "home" / ".config" / "voxis" / ".env"
    assert first.returncode == 0, first.stderr
    assert config.exists()
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert "VOXIS_LLM_API_KEY=" in config.read_text(encoding="utf-8")
    assert (tool_bin / "voxis").exists()
    assert "tool install --force --python 3.12" in log.read_text(encoding="utf-8")
    assert "doctor" in log.read_text(encoding="utf-8")
    assert not (tmp_path / "ignored-xdg-config" / "voxis" / ".env").exists()

    config.write_text("KEEP_ME=1\n", encoding="utf-8")
    second = subprocess.run(["bash", str(INSTALL_SCRIPT)], check=False, capture_output=True, text=True, env=env)

    assert second.returncode == 0, second.stderr
    assert config.read_text(encoding="utf-8") == "KEEP_ME=1\n"
    assert f"Kept existing config: {config}" in second.stdout


def test_install_script_with_diarize_builds_pinned_fluidaudio_into_tool_bin(tmp_path: Path) -> None:
    env, log, tool_bin = install_fake_commands(tmp_path, with_build_tools=True)

    completed = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--with-diarize"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tool_bin / "fluidaudio").exists()
    assert (tool_bin / "ESpeakNG.framework").is_dir()
    assert "clone --depth 1 --branch v0.9.1" in log.read_text(encoding="utf-8")
    assert "doctor --require-diarize" in log.read_text(encoding="utf-8")


def test_install_script_with_diarize_preflights_swift_before_mutation(tmp_path: Path) -> None:
    env, log, tool_bin = install_fake_commands(tmp_path, with_git=True)
    env["PATH"] = f"{tmp_path / 'fake-bin'}:/bin"

    completed = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--with-diarize"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 1
    assert "Swift is required" in completed.stderr
    assert not log.exists()
    assert not (tool_bin / "voxis").exists()
    assert not (tmp_path / "home" / ".config" / "voxis" / ".env").exists()


def test_install_script_with_diarize_preflights_cxx_before_mutation(tmp_path: Path) -> None:
    env, log, tool_bin = install_fake_commands(tmp_path, with_build_tools=True)
    write_executable(tmp_path / "fake-bin" / "xcrun", "#!/bin/sh\nexit 1\n")

    completed = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--with-diarize"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 1
    assert "Xcode C++ toolchain is not usable" in completed.stderr
    assert "sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer" in completed.stderr
    assert not log.exists()
    assert not (tool_bin / "voxis").exists()
    assert not (tmp_path / "home" / ".config" / "voxis" / ".env").exists()


def test_install_script_help_and_shell_syntax() -> None:
    syntax = subprocess.run(["bash", "-n", str(INSTALL_SCRIPT)], check=False, capture_output=True, text=True)
    help_result = subprocess.run(["bash", str(INSTALL_SCRIPT), "--help"], check=False, capture_output=True, text=True)

    assert syntax.returncode == 0, syntax.stderr
    assert help_result.returncode == 0
    assert "--with-diarize" in help_result.stdout


def install_fake_commands(
    tmp_path: Path,
    *,
    with_build_tools: bool = False,
    with_git: bool = False,
) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    tool_bin = tmp_path / "tool-bin"
    swift_bin = tmp_path / "swift-bin"
    home = tmp_path / "home"
    log = tmp_path / "commands.log"
    for directory in (fake_bin, tool_bin, swift_bin, home):
        directory.mkdir(parents=True)

    write_executable(
        fake_bin / "uname",
        """#!/bin/sh
if [ "$1" = "-s" ]; then echo Darwin; else echo arm64; fi
""",
    )
    write_executable(fake_bin / "ffmpeg", "#!/bin/sh\nexit 0\n")
    write_executable(
        fake_bin / "uv",
        """#!/bin/sh
echo "$*" >> "$INSTALL_TEST_LOG"
if [ "$1 $2 $3" = "tool dir --bin" ]; then
  echo "$INSTALL_TEST_TOOL_BIN"
  exit 0
fi
if [ "$1 $2" = "tool install" ]; then
  cat > "$INSTALL_TEST_TOOL_BIN/voxis" <<'TOOL'
#!/bin/sh
echo "$*" >> "$INSTALL_TEST_LOG"
exit 0
TOOL
  chmod +x "$INSTALL_TEST_TOOL_BIN/voxis"
  exit 0
fi
exit 2
""",
    )
    if with_build_tools or with_git:
        write_executable(
            fake_bin / "git",
            """#!/bin/sh
echo "$*" >> "$INSTALL_TEST_LOG"
destination=""
for argument in "$@"; do destination="$argument"; done
mkdir -p "$destination/.git"
touch "$destination/Package.swift"
""",
        )
    if with_build_tools:
        write_executable(
            fake_bin / "swift",
            """#!/bin/sh
echo "$*" >> "$INSTALL_TEST_LOG"
case " $* " in
  *" --show-bin-path "*) echo "$INSTALL_TEST_SWIFT_BIN" ;;
  *)
    mkdir -p "$INSTALL_TEST_SWIFT_BIN/ESpeakNG.framework/Versions/A"
    touch "$INSTALL_TEST_SWIFT_BIN/ESpeakNG.framework/Versions/A/ESpeakNG"
    cat > "$INSTALL_TEST_SWIFT_BIN/fluidaudiocli" <<'TOOL'
#!/bin/sh
exit 0
TOOL
    chmod +x "$INSTALL_TEST_SWIFT_BIN/fluidaudiocli"
    ;;
esac
""",
        )
        write_executable(fake_bin / "xcrun", "#!/bin/sh\ncat >/dev/null\nexit 0\n")

    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "INSTALL_TEST_LOG": str(log),
            "INSTALL_TEST_TOOL_BIN": str(tool_bin),
            "INSTALL_TEST_SWIFT_BIN": str(swift_bin),
        }
    )
    return env, log, tool_bin


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
