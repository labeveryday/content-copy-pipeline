#!/bin/bash
###############################################################################
# Content Copy Pipeline - Quick Start
#
# Simple script that uses all defaults from config files.
# Just run: ./quick_run.sh
###############################################################################

echo "=========================================="
echo "Content Copy Pipeline - Quick Start"
echo "=========================================="
echo ""
echo "Using default settings from:"
echo "  - config.json (preprocessing, paths)"
echo "  - config/models.yaml (AI models)"
echo ""
echo "To customize, edit config files or use run.sh"
echo ""
echo "=========================================="
echo ""

python run_pipeline.py

echo ""
echo "=========================================="
echo "✅ Pipeline complete!"
echo "=========================================="

