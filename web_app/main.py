#!/usr/bin/env python
"""All-in-one web app for vehicle length measurement from fixed camera."""
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from vehicle_metrology.geometry import Camera, measure_track
from vehicle_metrology.video import local_video, sha256_file

app = FastAPI(title="Vehicle Length Metrology Web App")
from web_app.workbench import router as workbench_router
app.include_router(workbench_router)
from web_app.station import router as station_router
app.include_router(station_router)
from web_app.video_stream import router as stream_router
app.include_router(stream_router)
from web_app.ip_cameras import router as ip_router
app.include_router(ip_router)

@app.on_event('startup')
def resume_plate_jobs():
    from web_app.plates import resume_pending
    resume_pending()
    from web_app.ip_cameras import startup
    startup()


@app.on_event('shutdown')
def stop_ip_cameras():
    from web_app.ip_cameras import stop
    stop()

# Configuration
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CALIBRATION_DIR = DATA_DIR / "calibrations"
CALIBRATION_DIR.mkdir(exist_ok=True)
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)

# Session storage (in production use proper database)
sessions: Dict[str, Dict] = {}


# ============ Models ============

class Point(BaseModel):
    x: float
    y: float


class RoadPolygon(BaseModel):
    points: List[Point]


class CameraIntrinsics(BaseModel):
    model: str  # "brown" or "fisheye"
    K: List[List[float]]  # 3x3
    D: List[float]
    image_size: List[int]  # [w, h]


class CameraExtrinsics(BaseModel):
    Rcw: List[List[float]]  # 3x3
    tcw: List[float]  # 3


class CalibrationData(BaseModel):
    calibration_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    road_polygon: RoadPolygon


class AnnotationPoint(BaseModel):
    frame: int
    rear_contact_uv: List[float]
    front_contact_uv: List[float]
    rear_uv: List[float]
    front_uv: List[float]
    sigma_px: float = 1.0


class AnnotationData(BaseModel):
    track_id: str
    observations: List[AnnotationPoint]


class MeasurementResult(BaseModel):
    track_id: str
    status: str
    length_m: Optional[float]
    reasons: List[str]
    diagnostics: Dict[str, Any]


class SessionState(BaseModel):
    session_id: str
    video_path: Optional[str] = None
    video_info: Optional[Dict] = None
    calibration: Optional[CalibrationData] = None
    annotations: List[AnnotationData] = []


# ============ Helper Functions ============

def compute_position_coefficient(bbox_bottom_y: float, road_polygon: np.ndarray, image_size: tuple) -> float:
    """
    Compute a coefficient based on where the bottom of the bbox falls on the road polygon.

    Logic:
    - If bbox bottom is near the bottom edge of road polygon (closer to camera) -> coefficient closer to 1.0 (larger)
    - If bbox bottom is near the top/horizon edge of road polygon (farther from camera) -> coefficient smaller

    The coefficient scales the measured length to account for perspective.
    """
    h, w = image_size[1], image_size[0]

    # Find the vertical bounds of the road polygon in image space
    poly_y = road_polygon[:, 1]
    poly_y_min = float(np.min(poly_y))  # Top of polygon (horizon/far)
    poly_y_max = float(np.max(poly_y))  # Bottom of polygon (near camera)

    # Normalize bbox bottom position within road polygon vertical range
    # Clamp to polygon bounds
    y_clamped = max(poly_y_min, min(poly_y_max, bbox_bottom_y))

    # Normalized position: 0 = at horizon (far), 1 = at bottom (near)
    if poly_y_max - poly_y_min < 1:
        return 1.0

    normalized_pos = (y_clamped - poly_y_min) / (poly_y_max - poly_y_min)

    # Coefficient curve:
    # At normalized_pos = 0 (horizon) -> coeff = 0.5 (compressed)
    # At normalized_pos = 1 (bottom) -> coeff = 1.0 (full scale)
    # Quadratic curve for smooth transition
    coeff = 0.5 + 0.5 * (normalized_pos ** 2)

    return float(coeff)


def dedistort_image(image: np.ndarray, camera: Camera) -> np.ndarray:
    """Apply lens distortion correction to image."""
    if camera.model in ('fisheye', 'fisheye-corrected', 'fisheye-corrected-rational'):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera.K, camera.D, np.eye(3), camera.K, camera.image_size, cv2.CV_16SC2
        )
    else:
        map1, map2 = cv2.initUndistortRectifyMap(
            camera.K, camera.D, np.eye(3), camera.K, camera.image_size, cv2.CV_16SC2
        )
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def draw_road_polygon(image: np.ndarray, polygon: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Draw road polygon on image."""
    pts = polygon.astype(np.int32).reshape(-1, 1, 2)
    return cv2.polylines(image, [pts], True, color, thickness)


def bbox_bottom_on_polygon(bbox: List[float], polygon: np.ndarray) -> tuple:
    """Get the bottom center point of bbox and check if it's inside polygon."""
    x, y, w, h = bbox
    bottom_center = np.array([x + w/2, y + h])
    inside = cv2.pointPolygonTest(polygon.astype(np.float32), (float(bottom_center[0]), float(bottom_center[1])), False) >= 0
    return bottom_center, inside


# ============ API Routes ============

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web app."""
    return FileResponse(Path(__file__).parent / "static" / "station.html")


@app.get('/calibration', response_class=HTMLResponse)
def calibration_studio():
    return FileResponse(Path(__file__).parent / 'static' / 'index.html')


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file."""
    if not file.filename:
        raise HTTPException(400, "No filename")

    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    video_path = session_dir / file.filename
    content = await file.read()
    video_path.write_bytes(content)

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Cannot decode video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    sessions[session_id] = {
        "video_path": str(video_path),
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "filename": file.filename
        },
        "calibration": None,
        "annotations": []
    }

    return {"session_id": session_id, "video_info": sessions[session_id]["video_info"]}


@app.get("/api/video-frame/{session_id}/{frame_idx}")
async def get_video_frame(session_id: str, frame_idx: int):
    """Extract and return a specific frame as JPEG."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    video_path = session["video_path"]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise HTTPException(404, "Frame not found")

    # Apply dedistortion if calibration exists
    if session.get("calibration"):
        try:
            cam = Camera.from_dict(session["calibration"])
            frame = dedistort_image(frame, cam)
        except Exception:
            pass  # If dedistortion fails, return original

    # Draw road polygon if exists
    if session.get("calibration") and session["calibration"].get("road_polygon"):
        poly = np.array(session["calibration"]["road_polygon"]["points"], dtype=np.float32)
        frame = draw_road_polygon(frame, poly)

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return JSONResponse(content={"frame": buffer.tobytes().hex()})


@app.post("/api/save-calibration/{session_id}")
async def save_calibration(session_id: str, calibration: CalibrationData):
    """Save camera calibration (intrinsics, extrinsics, road polygon)."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    sessions[session_id]["calibration"] = calibration.dict()

    # Save to file
    cal_path = CALIBRATION_DIR / f"{session_id}_calibration.json"
    cal_path.write_text(json.dumps(calibration.dict(), indent=2))

    return {"status": "saved", "path": str(cal_path)}


@app.get("/api/get-calibration/{session_id}")
async def get_calibration(session_id: str):
    """Get current calibration."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return sessions[session_id].get("calibration")


@app.post("/api/save-road-polygon/{session_id}")
async def save_road_polygon(session_id: str, polygon: RoadPolygon):
    """Save road polygon points."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    if "calibration" not in sessions[session_id] or not sessions[session_id]["calibration"]:
        sessions[session_id]["calibration"] = {
            "calibration_id": f"session-{session_id}",
            "intrinsics": {"model": "brown", "K": [[1,0,0],[0,1,0],[0,0,1]], "D": [0,0,0,0], "image_size": [1920, 1080]},
            "extrinsics": {"Rcw": [[1,0,0],[0,1,0],[0,0,1]], "tcw": [0,0,0]},
            "road_polygon": {"points": []}
        }

    sessions[session_id]["calibration"]["road_polygon"] = polygon.dict()
    return {"status": "saved", "points": len(polygon.points)}


@app.post("/api/save-annotations/{session_id}")
async def save_annotations(session_id: str, annotations: AnnotationData):
    """Save manual annotations for a track."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    sessions[session_id]["annotations"].append(annotations.dict())

    # Save to file
    ann_path = ANNOTATIONS_DIR / f"{session_id}_{annotations.track_id}.json"
    ann_path.write_text(json.dumps(annotations.dict(), indent=2))

    return {"status": "saved", "track_id": annotations.track_id, "count": len(annotations.observations)}


@app.post("/api/measure/{session_id}/{track_id}")
async def measure_vehicle(session_id: str, track_id: str, min_frames: int = 4, pixel_sigma: float = 1.0, mc_samples: int = 10):
    """Run measurement on annotated track."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]

    if not session.get("calibration"):
        raise HTTPException(400, "Calibration required")

    # Find annotations for this track
    track_anns = [a for a in session["annotations"] if a["track_id"] == track_id]
    if not track_anns:
        raise HTTPException(404, "No annotations for this track")

    observations = track_anns[0]["observations"]

    # Build camera object
    cal = session["calibration"]
    camera = Camera.from_dict({
        "schema_version": 1,
        "image_size": cal["intrinsics"]["image_size"],
        "model": cal["intrinsics"]["model"],
        "K": cal["intrinsics"]["K"],
        "D": cal["intrinsics"]["D"],
        "Rcw": cal["extrinsics"]["Rcw"],
        "tcw": cal["extrinsics"]["tcw"],
        "road_polygon": np.array(cal["road_polygon"]["points"]).tolist(),
        "calibration_id": cal["calibration_id"]
    })

    # Run measurement
    result = measure_track(camera, observations, min_frames=min_frames, pixel_sigma=pixel_sigma, mc_samples=mc_samples)

    return MeasurementResult(
        track_id=track_id,
        status=result["status"],
        length_m=result["length_m"],
        reasons=result["reasons"],
        diagnostics=result["diagnostics"]
    )


@app.post("/api/detect-and-coefficient/{session_id}")
async def detect_and_coefficient(session_id: str, frame_idx: int, bbox: List[float]):
    """
    Detect vehicles and compute position-based coefficient.

    Logic:
    - bbox: [x, y, w, h] in pixels
    - Find where bbox bottom center falls on road polygon
    - Near bottom of polygon (close to camera) -> coefficient ~1.0
    - Near top of polygon (far from camera/horizon) -> coefficient ~0.5
    """
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]

    if not session.get("calibration") or not session["calibration"].get("road_polygon"):
        raise HTTPException(400, "Road polygon required")

    cal = session["calibration"]
    road_polygon = np.array(cal["road_polygon"]["points"], dtype=np.float32)
    image_size = cal["intrinsics"]["image_size"]

    # Get bbox bottom center
    bottom_center, inside = bbox_bottom_on_polygon(bbox, road_polygon)

    # Compute coefficient
    coeff = compute_position_coefficient(bottom_center[1], road_polygon, image_size)

    # Also compute world position if calibration available
    world_pos = None
    if inside:
        try:
            camera = Camera.from_dict({
                "schema_version": 1,
                "image_size": cal["intrinsics"]["image_size"],
                "model": cal["intrinsics"]["model"],
                "K": cal["intrinsics"]["K"],
                "D": cal["intrinsics"]["D"],
                "Rcw": cal["extrinsics"]["Rcw"],
                "tcw": cal["extrinsics"]["tcw"],
                "road_polygon": road_polygon.tolist(),
                "calibration_id": cal["calibration_id"]
            })
            world_pt = camera.ground(bottom_center.reshape(1, 2))[0]
            world_pos = {"x": float(world_pt[0]), "y": float(world_pt[1]), "z": float(world_pt[2])}
        except Exception:
            pass

    return {
        "bbox_bottom": {"x": float(bottom_center[0]), "y": float(bottom_center[1])},
        "inside_polygon": bool(inside),
        "coefficient": coeff,
        "world_position": world_pos,
        "road_polygon_bounds": {
            "y_min": float(np.min(road_polygon[:, 1])),
            "y_max": float(np.max(road_polygon[:, 1]))
        }
    }


@app.post("/api/auto-detect/{session_id}/{frame_idx}")
async def auto_detect(session_id: str, frame_idx: int):
    """Run MOG2 background subtraction to detect vehicles (diagnostic only)."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    video_path = session["video_path"]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise HTTPException(404, "Frame not found")

    # Apply dedistortion if calibration exists
    if session.get("calibration"):
        try:
            cal = session["calibration"]
            camera = Camera.from_dict({
                "schema_version": 1,
                "image_size": cal["intrinsics"]["image_size"],
                "model": cal["intrinsics"]["model"],
                "K": cal["intrinsics"]["K"],
                "D": cal["intrinsics"]["D"],
                "Rcw": cal["extrinsics"]["Rcw"],
                "tcw": cal["extrinsics"]["tcw"],
                "road_polygon": np.array(cal["road_polygon"]["points"]).tolist(),
                "calibration_id": cal["calibration_id"]
            })
            frame = dedistort_image(frame, camera)
        except Exception:
            pass

    # MOG2 detection (diagnostic only)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=False)
    mask = fgbg.apply(frame)

    # Simple: just return the mask as reference; real detection needs training
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if 500 <= area <= w * h * 0.5:
            x, y, bw, bh = cv2.boundingRect(c)
            detections.append({"bbox": [x, y, bw, bh], "area": float(area)})

    return {"detections": detections, "frame_idx": frame_idx}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session state."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    return {
        "session_id": session_id,
        "video_info": sessions[session_id].get("video_info"),
        "has_calibration": sessions[session_id].get("calibration") is not None,
        "annotations": sessions[session_id].get("annotations", [])
    }


# ============ Static Files ============

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
