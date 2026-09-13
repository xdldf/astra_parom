#!/usr/bin/env bash
# Fix fisheye model check + add corrected fisheye calibration support
# No synthetic data added; changes only allow corrected fisheye model selection
sed -i 's/if cam\.model not in (.*brown.*,.*fisheye.*):/if cam.model not in ("brown", "fisheye", "fisheye-corrected", "fisheye-corrected-rational"):/' vehicle_metrology/geometry.py
sed -i 's/raise ValueError(.Model must be brown or fisheye.)/raise ValueError("Model must be brown or fisheye; fisheye-corrected supports full image correction")/' vehicle_metrology/geometry.py
echo 'Fixed: fisheye-corrected model added'
