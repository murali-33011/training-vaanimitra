#!/bin/bash
# 00_clean.sh — Run this FIRST. Wipes all previous vanimitra work.
set -e
echo "=== Cleaning previous vanimitra work ==="
cd ~
rm -rf vanimitra/
echo "=== Creating clean directory structure ==="
mkdir -p ~/vanimitra
cd ~/vanimitra
echo "=== Done. Now run: cd ~/vanimitra ==="
