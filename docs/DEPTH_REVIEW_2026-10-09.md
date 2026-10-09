# Depth calibration review — 9 October 2026

The user supplied a screenshot showing **4.723 m** and a **4.950 m catalogue/specification reference**, explicitly not a physical measurement. The difference is −0.227 m (−4.59% of the reference). Matching this one observation would require multiplying its length by 1.04806. That number is a diagnostic ratio, not a fitted depth curve, and has not been applied to the production profile.

The local bbox calibration already uses `pixels_per_metre = 61.69849 + 98.22797 × road_depth`. Its eight references occupy relative road depths 0.549–0.714; five lie between 0.688 and 0.715. Only four of the eight training references reproduce within 10 cm. The current local profile subsequently replaces that bbox result with an outline/wheel regression, which has its own silhouette-depth and wheel-depth features. Editing only the bbox slope would not recalibrate those final outputs.

The screenshot's deployed profile, source frame and exact detector boundary have not been identified. The enlarged crop has lost the full-frame road coordinates and scale. Matching against the cached vehicle images did not establish a reliable source match. A video filename and timestamp or saved measurement record are needed to trace this observation, check the depth/contact proxy, and bind the 4.950 m catalogue reference to the actual car before fitting.

The next calibration comparison should retain complete centred observations across near, middle and far road positions, and evaluate separate passages that did not fit the coefficients. This reference can support catalogue-based development once linked; it must not be presented as physically measured ground truth. The centre-capture fixes are independent of this unresolved depth-calibration example.

Local diagnostics: `runs/depth_review_20261009/diagnostics.json`. No calibration coefficients or historical records were changed for this screenshot.
