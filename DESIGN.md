# Vehicle length metrology: design and experiment

## Decision and scope

Build a calibrated, object-centric, multi-frame 3D measurement system, not a pixel-to-meter lookup. The first experiment uses manually identified road-contact and rigid body points in recorded video. Automation follows only after this experiment establishes geometric accuracy on real footage. A detector rectangle alone does not determine physical length. The runnable proof of concept is deliberately conditional on richer observations; its foreground detector is a diagnostic baseline, not a trained vehicle classifier.

No footage, survey, intrinsic calibration, known-length labels, or required numerical tolerance was supplied. Thus no real-world accuracy or tariff readiness can be claimed. Synthetic tests validate implementation and controlled failure modes only.

## 1. Fundamental geometry and observability

Use metric road coordinates (X,Y,Z), with the local road plane Z=0. A central camera has pixel projection u = pi_theta(Rcw P + tcw). theta includes intrinsic and lens parameters; Rcw and tcw map world to camera. Calibrate the actual recorded resolution, crop, focus, zoom, stabilization and dewarping settings. Lock them thereafter.

After inverse distortion, a pixel defines a world ray C + lambda d. Intersection with a surveyed plane n.P+d0=0 is lambda=-(n.C+d0)/(n.d). This recovers a point ONLY if that physical point lies on the plane. Reject nearly parallel or backwards intersections. For Z=0, the familiar homography applies to undistorted ground coordinates, not directly to distorted pixels or the whole vehicle.

If a point at height h is instead projected onto Z=0 by a camera at height H, its ground intersection is Cxy + H/(H-h)*(Pxy-Cxy). For two points at the same height this scales their horizontal separation by H/(H-h); for different heights it also creates a position-dependent error. The approximation 'high camera means a flat vehicle' is not valid for precision metrology. Wheelbase also is not bumper-to-bumper length.

An arbitrary unknown 3D vehicle cannot be uniquely recovered from one view. Hidden front/rear extents, shape/depth ambiguities and absolute scale ambiguity remain. Surveyed metric geometry fixes scene scale, but not all unknown object geometry. A high camera improves ground visibility, not universal identifiability. A fixed camera gives no static-scene stereo baseline; a translating rigid vehicle DOES give object-relative viewpoint diversity if its metric motion is known or jointly constrained.

## 2. Chosen measurement model

For a rigid body, P_tj = T_t + Rz(psi_t) q_j, where T_t=(x_t,y_t,0), psi_t is heading, and q_j is a constant point in vehicle coordinates. Actual length is max(q.x)-min(q.x) over the specified physical body envelope, not Euclidean distance between two arbitrary corners.

Production: jointly estimate trajectory, heading, rigid shape/extreme landmarks and shared length over a track. Fit distorted-image keypoints, genuine ground contacts and silhouette observations with visibility-aware robust residuals. Survey/intrinsic calibration remains anchored by independent data, with uncertainties propagated. Use calibrated shape priors only to resolve ambiguities; report when the answer is prior-dominated. A cuboid is a useful initialization/diagnostic envelope, not proof that a curved bumper or silhouette fits a cuboid exactly. Articulated trucks require separate rigid parts and a defined overall-length policy.

Proof of concept: mark rear and front wheel-road contacts on the SAME side in each selected frame. The rear contact defines T_t; the vector to the front defines heading. The longitudinal positions of these ideal contact points are fixed in a non-steering, non-pitching rigid-body approximation. Mark the SAME visible physical frontmost and rearmost body points across frames. These need not be on the ground or at equal heights. In body coordinates each endpoint lies on rays o_t + lambda d_t, with o_t=Rz(-psi_t)(C-T_t). Least-squares intersection followed by robust image reprojection refinement estimates its 3D position. Length is the front minus rear longitudinal coordinate. This uses measured motion, not assumed constant speed, so moderate acceleration is allowed.

The annotation contract is restrictive: do NOT track a rotating tire tread mark; do NOT substitute the lowest segmentation pixel for wheel-road contact; do NOT change the selected bumper point as the silhouette tangent changes; do NOT use the near-side front wheel with the far-side rear wheel. Steering/contact geometry, yaw, body pitch/roll, suspension travel and uncertain contact visibility can invalidate this simplified pose estimator. A surveyed static pose or tracked wheel-center model with measured radius is an alternative for manual experiments.

Reject insufficient angular baseline, ill-conditioned rays, high reprojection error, nonphysical endpoint heights, truncated/occluded required landmarks, inconsistent wheelbase, and contact points outside the surveyed usable road polygon. End-on motion, very short tracks and distant small vehicles may be unobservable. There is no requirement for exactly side-on viewing; favor views with high longitudinal information and complementary baseline. Near side-on is often helpful but not sufficient by itself.

## 3. Alternatives and classification

- Theoretically sound under stated conditions: calibrated rigid multi-view reconstruction with metric road/motion anchors and genuinely observed extremes; uncertainty-aware joint optimization.
- Practical approximation: road-constrained 3D cuboid fitting to segments/keypoints; useful initialization, but class- and view-dependent shape bias needs validation.
- Theoretically valid envelope: a multi-view visual hull intersects silhouette cones in the common metric body frame. Exact complete masks/poses contain the true object, so finite longitudinal hull extent is an upper bound on full-object extent, not automatically exact bumper length. Limited viewpoints may leave it loose/unbounded; noisy masks break the bound. Use contour constraints without assuming contour pixels are persistent material points.
- Practical approximation: speed times crossing duration; requires correct 3D gate, trajectory, timing and longitudinal extremes. A ground image line is not a vertical physical gate; overhead pixel crossings remain height-biased.
- Empirical correction: held-out-calibrated residual correction after geometry. Useful only inside supported classes/positions with propagated uncertainty. Never hide geometric errors with a per-pixel scale table.
- Insufficient: box width times scale, diagonal length on a ground homography, wheelbase as body length, generic monocular depth with an unvalidated size prior, or averaging biased frame estimates.
- If stringent accuracy/coverage is non-negotiable and this experiment fails: add a calibrated second camera with useful baseline or surveyed LiDAR/light-curtain sensing. Additional hardware is an explicit alternative, not assumed available in this single-camera deliverable.

## 4. Calibration, recorded files only

### Intrinsics and distortion

Record a rigid accurately dimensioned checkerboard/ChArUco target at many tilts, rotations, depths and positions using the identical video mode. A practical starting point is 25-50 diverse usable board views, not adjacent near-duplicates. Fill corners/edges as well as center. Avoid blur and near-frontal-only boards. If the board is too small at road distance, calibrate nearby at locked focus and validate distant surveyed landmarks. Target flexibility and print scaling become measurement errors.

Compare pinhole radial+tangential (Brown), rational when justified, and angular fisheye models. Wide appearance alone does not choose a model. Select using spatially held-out views/landmarks, residual vector fields, ray-inversion stability and metric check distances, not training reprojection RMS alone. Higher-order coefficients can overfit. Noncentral lenses or undocumented in-camera dewarp may need a calibrated ray lookup/generalized model; do not force a central model onto systematic residuals. The first implementation supports Brown and angular fisheye; rational/generalized models are follow-ups only if residuals demand them.

### Extrinsics, scale, road surface

Establish surveyed road X/Y/Z coordinates in meters using a total station where accuracy requires it, or a checked tape/laser survey for initial trials. Use approximately 12-20 well-spread control points plus 6-10 separate check points as a starting design; include longitudinal extent, every lane, near/far zones and boundaries. Their coordinates, not just isolated segment lengths, are needed. Four noncollinear points on a plane are the algebraic homography minimum AFTER known intrinsics/distortion; this is not an adequate metrology procedure. Use redundant pose estimation, positive-depth/above-road checks and independent metric holdouts. Measured camera height is a cross-check, not an approximate replacement for calibration.

Measure road elevation at a grid along/across lanes. A tilted plane is fine in its own coordinates; crown, grade changes and humps require a piecewise surveyed surface/mesh when induced error exceeds the budget. Plane residuals alone do not set length error: propagate them through rays and trajectories. The POC only supports a plane and must not silently flatten a nonplanar site.

Save calibration IDs, source-file hashes, units, resolution, K, distortion model/coefficients, Rcw/tcw, control/check errors, survey uncertainty, road polygon and camera settings. Independent landmarks in each video detect pole drift, wind motion, changed crop/zoom or stabilization. Frames failing registration need corrected poses or rejection, not a length correction.

## 5. Known vehicles

Directly measure actual bumper-to-bumper longitudinal extent under a documented policy (tow hooks, load, trailer, mirrors, accessories), with ground-truth uncertainty. Catalog dimensions are not sufficient unless configuration is verified. Repeated runs of the same vehicle are valuable because true length is constant. Reserve whole physical vehicles and recording sessions for validation; never split adjacent frames between calibration and test.

Known lengths can anchor a remaining global scale after projective/Euclidean reconstruction, constrain shared geometric parameters in bundle adjustment, identify residual bias and train landmark/shape models. Length-only labels attached to 2D boxes do NOT in general identify intrinsics, distortion, height, pose and vehicle shape uniquely. Spatial and heading diversity plus independent scene evidence are required. Separate survey-based calibration from any vehicle-based refinement and compare held-out results before accepting refinement. Maintain held-out vehicle labels untouched by tuning.

## 6. Calibration/validation experiment before automation

1. Define required absolute/relative error, allowed reject rate and tariff boundaries BEFORE evaluation. Derive a survey, pixel-localization, model-bias and timing error budget from these. No requested tolerance has yet been given.
2. Calibrate lens; survey road/holdouts; test metric segments across the full usable region. Record residuals by image radius, ground position and direction.
3. Drive known vehicles repeatedly across controlled near/middle/far lateral tracks, both permitted directions and realistic heading changes, covering center and edges. Cross the same physical vehicles over the same position bins to avoid confounding vehicle type with distance. Include cars/SUVs/vans/long rigid vehicles; analyze articulated vehicles separately. Use only safe, authorized traffic control.
4. Select sharp, unoccluded frame windows along every track. Manually annotate endpoints and ground anchors, repeat annotations with different operators, estimate pixel uncertainty. This establishes a best-observable geometric baseline independently of detector quality.
5. Estimate one shared length per track and prefix/local-window estimates to expose drift. Use disjoint windows or account for their correlation. Compare oracle/manual poses with detected contacts, manual features with automatic features, lens models, planar versus surveyed road, and one-frame cuboid versus multi-view fits. Controlled ablations reveal the dominant error source rather than presuming it.
6. Freeze calibration/model/thresholds and run a vehicle- and session-held-out test. Include day/night, weather, glare, speed, blur, occlusion, different lanes, image edges and all required classes. Match identity to ground truth; report missed/false/merged/split tracks as well as measurement quality.

Report signed bias, mean/median absolute error, RMSE, mean/median relative absolute error, observed maximum absolute error, p90/p95/p99 absolute error where sample size supports them. The maximum observed error is not a worst-case guarantee. Report signed and absolute errors binned by world X/Y, radial image position, heading/view angle, class, speed, pixel length, visibility and recording conditions. Pair same-vehicle differences between spatial bins. Report per-vehicle and per-passage statistics (repeated passages are correlated), bootstrap by physical vehicle/session, and report uncertainty with sample counts. For tails, dozens of examples are insufficient for a reliable p99 claim.

Also report within-track local-window MAD/range, drift against position, annotation repeatability, accepted coverage and all rejection reasons, conditional interval coverage, and accuracy-versus-coverage. Very stable estimates can still have shared calibration bias. Detection confidence is not metrological confidence. A tariff boundary inside a validated uncertainty interval must trigger review, not silent rounding.

## 7. Uncertainty and likely error mechanisms

Use Jacobian conditioning/profile likelihood to detect underconstrained geometry. Propagate pixel/contact/pose uncertainty jointly within a frame, and common intrinsic/survey parameters once per sequence in Monte Carlo. Do not shrink calibration/model bias by sqrt(number of frames). Temporal correlation calls for block resampling, not independent-frame bootstrap. Validate predicted interval coverage and inflate/recalibrate or reject unsupported cases.

Expected risks, not established findings: incorrect endpoint/contact localization and changing silhouette correspondences; lens/pose errors at edges; hidden overhangs; weak baseline/end-on viewing; road crown; rolling shutter and blur; pole vibration; shared detector/shape priors. Diagnose with residual-vector patterns and the ablations above. A synthetic noise interval is conditional repeatability, not a certified error bound.

Occluded endpoints may be measured from other genuinely visible frames of the same intact track. Fully hidden extremes remain unmeasurable without priors/additional sensing. Shadows/reflections require instance masks and semantic landmark visibility. Truncated edge detections should be rejected unless earlier/later frames supply sufficient observations. Trucks with articulated trailers are not one rigid body; distinguish body-part dimensions, straightened combination length and instantaneous envelope.

## 8. Implementation selection

Python is appropriate for an inspectable offline research harness. NumPy/SciPy provide linear algebra and robust optimization; OpenCV provides central-camera Brown/fisheye models, video decoding, annotation and overlays. These libraries implement the required geometry; no detector framework is mandatory. Recorded-video background subtraction plus deterministic association is a no-weights diagnostic detector only. Production requires validated instance segmentation, persistent keypoints/ground-contact estimation and occlusion-aware tracking, trained/evaluated at this site. Bounding boxes remain useful for detection/association, not meter conversion.

Implement calibration artifacts, manual observations, a multi-view geometry kernel, local-file runner, structured per-track and per-frame diagnostics, saved overlays, pause/step/replay, known-length evaluation and deterministic synthetic regression tests. An unannotated/insufficient track must output null length with a reason, not plausible fabricated meters. Timestamp from decoded presentation time when available; fixed frame/FPS time is only a fallback. Archive source hashes, configuration, dependency versions and seeds; repeatability is within a pinned decode/numerical environment, not a promise of identical codecs across platforms.

## Sources

OpenCV camera models/calibration: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
OpenCV angular fisheye model: https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
Vehicle-part annotation/3D reconstruction comparison (includes scale ambiguity and priors): https://arxiv.org/abs/2506.21358
These references establish available models/related approaches, not accuracy evidence for this installation.
