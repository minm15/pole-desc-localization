#!/usr/bin/env bash
set -euo pipefail

find . -type f -name 'localization*.npz' -print -exec rm -f {} +
