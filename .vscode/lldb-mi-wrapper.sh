#!/usr/bin/env bash
set -euo pipefail

latest_cpptools_dir="$(ls -d "$HOME/.vscode/extensions/ms-vscode.cpptools-"*"-darwin-arm64" 2>/dev/null | sort -V | tail -n 1)"
if [[ -z "${latest_cpptools_dir:-}" ]]; then
  echo "cpptools extension not found under $HOME/.vscode/extensions" >&2
  exit 1
fi

lldb_mi_path="$latest_cpptools_dir/debugAdapters/lldb-mi/bin/lldb-mi"
if [[ ! -x "$lldb_mi_path" ]]; then
  echo "lldb-mi not found or not executable: $lldb_mi_path" >&2
  exit 1
fi

exec "$lldb_mi_path" "$@"
