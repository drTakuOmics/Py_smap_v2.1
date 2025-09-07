#!/bin/bash
# Wrapper for running SMAP Python modules
# usage: run_smappoi.sh <paramsFile> [boardIndex]
paramsFile=$1
board=${2:-1}
fxnToRun=$(grep 'function' "$paramsFile" | grep '^[^#;]' | sed 's/^.* //')
python -m smap_tools_python.smappoi_${fxnToRun} "$paramsFile" "$board"
