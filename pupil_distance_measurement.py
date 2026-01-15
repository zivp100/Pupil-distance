"""
Pupil Distance Measurement Script

This script measures inter-pupillary distance with sub-millimeter accuracy
using MediaPipe Face Mesh for iris detection and a credit card as a reference scale.
"""

import cv2
import numpy as np
import json
import mediapipe as mp
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import sys
import urllib.request
import os

# Constants
CREDIT_CARD_WIDTH_MM = 85.60  # ISO 7810 ID-1 standard full card width
MAGNETIC_STRIPE_WIDTH_MM = 79.0  # Magnetic stripe width (narrower than full card)
LEFT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
NOSE_TIP_LANDMARK = 4
NOSE_BRIDGE_LANDMARK = 6

# MediaPipe model URL
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_MODEL_NAME = "face_landmarker.task"


def _download_model(model_path: Path) -> Path:
    """Download the face landmarker model if it doesn't exist."""
    if model_path.exists():
        return model_path
    
    print(f"Downloading face landmarker model to {model_path}...")
    try:
        # Create SSL context that doesn't verify certificates (for Mac SSL issues)
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Download with SSL context
        req = urllib.request.Request(FACE_LANDMARKER_MODEL_URL)
        with urllib.request.urlopen(req, context=ssl_context) as response:
            with open(model_path, 'wb') as f:
                f.write(response.read())
        print("Model downloaded successfully.")
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}\n"
                          f"You can manually download it from:\n"
                          f"{FACE_LANDMARKER_MODEL_URL}\n"
                          f"and place it at: {model_path}")


def _get_face_landmarker(model_path: Optional[str] = None):
    """Get MediaPipe Face Landmarker for MediaPipe 0.10+."""
    try:
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import face_landmarker
        from mediapipe.tasks.python.vision.core import vision_task_running_mode
        
        # Determine model path
        if model_path is None:
            # Use current directory for model storage
            model_dir = Path.cwd() / ".mediapipe_models"
            try:
                model_dir.mkdir(exist_ok=True)
            except PermissionError:
                # Fallback to temp directory
                import tempfile
                model_dir = Path(tempfile.gettempdir()) / "mediapipe_models"
                model_dir.mkdir(exist_ok=True)
            model_path = model_dir / FACE_LANDMARKER_MODEL_NAME
        
        model_path = Path(model_path)
        model_path = _download_model(model_path)
        
        # Create base options
        base_opts = base_options.BaseOptions(model_asset_path=str(model_path))
        
        # Create face landmarker options
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        # Create face landmarker
        landmarker = face_landmarker.FaceLandmarker.create_from_options(options)
        
        return landmarker, True
    except ImportError as e:
        raise ImportError(f"Failed to import MediaPipe tasks API: {e}")


class PupilDistanceMeasurer:
    """Main class for measuring inter-pupillary distance."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize MediaPipe Face Landmarker."""
        self.face_landmarker, self.use_new_api = _get_face_landmarker(model_path)
        
    def detect_face_iris(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face mesh and extract iris landmarks.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with face landmarks and iris data, or None if not detected
        """
        from mediapipe.tasks.python.vision.core import image as mp_image
        
        h, w = image.shape[:2]
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face landmarks
        detection_result = self.face_landmarker.detect(mp_img)
        
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return None
        
        face_landmarks = detection_result.face_landmarks[0]
        
        # Extract iris landmarks
        left_iris_points = []
        right_iris_points = []
        
        for idx in LEFT_IRIS_LANDMARKS:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                left_iris_points.append([landmark.x * w, landmark.y * h])
        
        for idx in RIGHT_IRIS_LANDMARKS:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                right_iris_points.append([landmark.x * w, landmark.y * h])
        
        if len(left_iris_points) == 0 or len(right_iris_points) == 0:
            return None
        
        # Calculate pupil centers as centroids
        left_pupil = np.mean(left_iris_points, axis=0)
        right_pupil = np.mean(right_iris_points, axis=0)
        
        # Calculate iris diameter (using the 5-point iris landmarks)
        # Points 0 is center, 1-4 are boundary points (typically arranged as: center, right, top, left, bottom)
        # Diameter is approximately 2x the distance from center to boundary
        left_iris_arr = np.array(left_iris_points)
        right_iris_arr = np.array(right_iris_points)
        
        # Calculate diameter as the max span in X or Y direction
        left_iris_diameter = max(
            np.max(left_iris_arr[:, 0]) - np.min(left_iris_arr[:, 0]),  # X span
            np.max(left_iris_arr[:, 1]) - np.min(left_iris_arr[:, 1])   # Y span
        )
        right_iris_diameter = max(
            np.max(right_iris_arr[:, 0]) - np.min(right_iris_arr[:, 0]),
            np.max(right_iris_arr[:, 1]) - np.min(right_iris_arr[:, 1])
        )
        avg_iris_diameter_pixels = (left_iris_diameter + right_iris_diameter) / 2
        
        
        # Get nose position for credit card detection
        if NOSE_TIP_LANDMARK < len(face_landmarks) and NOSE_BRIDGE_LANDMARK < len(face_landmarks):
            nose_tip = face_landmarks[NOSE_TIP_LANDMARK]
            nose_bridge = face_landmarks[NOSE_BRIDGE_LANDMARK]
            nose_y = (nose_tip.y + nose_bridge.y) / 2 * h
        else:
            # Fallback: use approximate nose position
            nose_y = h * 0.4
        
        # Find chin (bottom of face) - look for the lowest Y coordinate in face outline
        # Face outline landmarks are typically in the lower part of the face
        chin_y = nose_y  # Initialize with nose_y as fallback
        if len(face_landmarks) > 0:
            # Find the landmark with the maximum Y (lowest on face)
            max_y = 0
            for landmark in face_landmarks:
                if landmark.y > max_y:
                    max_y = landmark.y
                    chin_y = landmark.y * h
        
        # Find face width - use leftmost and rightmost face landmarks
        min_x = w
        max_x = 0
        for landmark in face_landmarks:
            x_coord = landmark.x * w
            if x_coord < min_x:
                min_x = x_coord
            if x_coord > max_x:
                max_x = x_coord
        
        face_left = int(min_x)
        face_right = int(max_x)
        face_width = face_right - face_left
        
        return {
            'face_landmarks': face_landmarks,
            'left_pupil': left_pupil,
            'right_pupil': right_pupil,
            'left_iris_points': left_iris_points,
            'right_iris_points': right_iris_points,
            'iris_diameter_pixels': avg_iris_diameter_pixels,
            'nose_y': nose_y,
            'chin_y': chin_y,
            'face_left': face_left,
            'face_right': face_right,
            'face_width': face_width,
            'image_shape': (h, w)
        }
    
    def __del__(self):
        """Clean up face landmarker."""
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            try:
                self.face_landmarker.close()
            except:
                pass
    
    def detect_credit_card(self, image: np.ndarray, nose_y: float, chin_y: float,
                          face_left: int, face_right: int, face_width: int,
                          left_pupil: np.ndarray, right_pupil: np.ndarray,
                          output_path: Optional[str] = None) -> Optional[Dict]:
        """
        Detect credit card below the nose and measure its width.
        
        Crops the image from nose to chin, full face width, and searches for card there.
        
        Args:
            image: Input image as numpy array
            nose_y: Y coordinate of nose (start of crop)
            chin_y: Y coordinate of chin (end of crop)
            face_left: Left edge of face (start of horizontal crop)
            face_right: Right edge of face (end of horizontal crop)
            face_width: Width of face
            left_pupil: Left pupil position [x, y] for size reference
            right_pupil: Right pupil position [x, y] for size reference
            output_path: Optional path to save the cropped image for verification
            
        Returns:
            Dictionary with card bounding box and width, or None if not detected
        """
        h, w = image.shape[:2]
        
        # Crop from below nose to chin, full face width
        # Start below the nose (add a small margin to ensure nose is not included)
        nose_margin = (chin_y - nose_y) * 0.1  # 10% of nose-to-chin distance as margin
        roi_y_start = max(0, int(nose_y + nose_margin))
        roi_y_end = min(h, int(chin_y))
        roi_x_start = max(0, face_left)
        roi_x_end = min(w, face_right)
        
        # Ensure we have a valid crop
        if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end:
            return None
        
        roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        if roi.size == 0:
            return None
        
        roi_h, roi_w = roi.shape[:2]
        
        # Credit card dimensions: 85.6mm x 53.98mm (ratio ~1.586:1)
        # Card should be about as wide as the face
        expected_card_width = roi_w * 0.5  # About 50% of face width (card is typically narrower)
        min_card_width = max(15, roi_w * 0.20)  # At least 20% of ROI width or 15px
        max_card_width = roi_w  # At most 100% of face width (very lenient)
        
        
        # ============================================================
        # METHOD: Magnetic stripe detection (multi-method approach)
        # Try dark pixel detection first, then fall back to edge detection
        # for stripes that aren't pure black.
        # ============================================================
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Convert to HSV - the magnetic stripe is LOW SATURATION (gray/black)
        # while skin has HIGHER SATURATION (warm tones)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]  # Saturation channel
        value = hsv[:, :, 2]  # Value (brightness) channel
        
        # The magnetic stripe has low saturation (it's gray, not colorful)
        # Use this to filter out skin which has higher saturation
        # Be lenient - some stripes have slight color tint
        low_saturation_mask = saturation < 150  # Low saturation = gray/black (very relaxed for stripe edges)
        
        
        
        # Combine with darkness check
        # Stripe is both LOW SATURATION and DARK
        
        # Also compute horizontal edges for fallback detection
        # The stripe should have strong horizontal edges at top and bottom
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y_abs = np.abs(sobel_y)
        
        # The black stripe is very dark - find dark pixels
        # Use adaptive thresholding based on the image
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Try multiple dark thresholds to find the stripe
        # Start with stricter (darker) and relax if needed
        dark_thresholds = [
            mean_intensity * 0.3,  # Very dark
            mean_intensity * 0.4,
            mean_intensity * 0.5,
            mean_intensity - std_intensity,  # Below average
            100,  # Fixed threshold
            115,  # Higher threshold for lighter stripe edges
        ]
        
        # Skip the very top rows (might be artifacts from cropping)
        skip_top_rows = max(5, int(roi_h * 0.05))  # Skip top 5% or at least 5 rows
        
        # Find horizontal bands of dark pixels
        # The stripe should span a significant portion of the width
        best_stripe_y = None
        best_stripe_left_x = None
        best_stripe_right_x = None
        best_stripe_height = 0
        best_score = 0
        
        # Try each threshold and collect all candidates
        # Pick the best one across all thresholds (not just the first found)
        for dark_threshold in dark_thresholds:
                
            # Create mask of dark pixels with LOW SATURATION
            # This filters out dark skin (which has color/saturation) from the stripe (neutral gray)
            dark_mask = (gray < dark_threshold) & low_saturation_mask
            
            # Scan for horizontal dark bands
            y = skip_top_rows
            while y < roi_h - 5:
                row = dark_mask[y, :]
                dark_indices = np.where(row)[0]
                
                if len(dark_indices) > 0:
                    # Find the LONGEST contiguous run of dark pixels
                    # This avoids including gaps between separate dark regions
                    runs = []
                    run_start = dark_indices[0]
                    run_end = dark_indices[0]
                    
                    for i in range(1, len(dark_indices)):
                        if dark_indices[i] - dark_indices[i-1] <= 10:  # Allow small gaps (10px)
                            run_end = dark_indices[i]
                        else:
                            runs.append((run_start, run_end, run_end - run_start))
                            run_start = dark_indices[i]
                            run_end = dark_indices[i]
                    runs.append((run_start, run_end, run_end - run_start))
                    
                    # Find the run that is CENTERED and reasonably wide
                    # The magnetic stripe should be in the center of the ROI
                    roi_center = roi_w // 2
                    
                    # Use the longest contiguous run that meets minimum width
                    # This is more reliable than center-preference which can select partial stripes
                    best_run = None
                    sorted_runs = sorted(runs, key=lambda r: r[2], reverse=True)
                    for run_start, run_end, run_width in sorted_runs:
                        if run_width >= min_card_width:
                            best_run = (run_start, run_end, run_width)
                            break
                    
                    
                    if best_run is None:
                        y += 1
                        continue
                        
                    left_x = best_run[0]
                    right_x = best_run[1]
                    stripe_width = right_x - left_x
                    coverage = stripe_width / roi_w
                    
                    # Check if this could be the stripe (reasonable width, about face width)
                    # Check if this could be the stripe (reasonable width, about face width)
                    max_reasonable_width = roi_w * 0.90  # Stripe shouldn't be more than 90% of ROI
                    if stripe_width >= min_card_width and stripe_width <= max_card_width and coverage > 0.3:
                        # Found a potential stripe row - now find the full height of the stripe
                        stripe_start_y = y
                        stripe_end_y = y
                        
                        # Scan down to find the bottom of the stripe
                        for check_y in range(y + 1, min(y + 50, roi_h)):  # Stripe shouldn't be more than 50px tall
                            check_row = dark_mask[check_y, :]
                            check_dark_indices = np.where(check_row)[0]
                            
                            if len(check_dark_indices) > 0:
                                # Use contiguous run detection for this row too
                                check_runs = []
                                check_run_start = check_dark_indices[0]
                                check_run_end = check_dark_indices[0]
                                for ci in range(1, len(check_dark_indices)):
                                    if check_dark_indices[ci] - check_dark_indices[ci-1] <= 3:
                                        check_run_end = check_dark_indices[ci]
                                    else:
                                        check_runs.append((check_run_start, check_run_end))
                                        check_run_start = check_dark_indices[ci]
                                        check_run_end = check_dark_indices[ci]
                                check_runs.append((check_run_start, check_run_end))
                                
                                # Find the run that overlaps most with the current stripe
                                best_overlap_run = None
                                best_overlap = 0
                                for cr_start, cr_end in check_runs:
                                    overlap_start = max(cr_start, left_x)
                                    overlap_end = min(cr_end, right_x)
                                    overlap = max(0, overlap_end - overlap_start)
                                    if overlap > best_overlap:
                                        best_overlap = overlap
                                        best_overlap_run = (cr_start, cr_end)
                                
                                if best_overlap_run is None:
                                    break
                                
                                check_left, check_right = best_overlap_run
                                check_width = check_right - check_left
                                check_coverage = check_width / roi_w
                                
                                # Check if this row is still part of the stripe
                                # Allow slightly wider rows but limit to 1.05x to avoid edge creep
                                if (check_width <= stripe_width * 1.05 and 
                                    check_width >= stripe_width * 0.5 and
                                    check_coverage > 0.2):
                                    stripe_end_y = check_y
                                    # Update width only if check row is similar width (within 3%)
                                    if check_width > (right_x - left_x) and check_width <= stripe_width * 1.03:
                                        left_x = check_left
                                        right_x = check_right
                                else:
                                    break
                            else:
                                break
                        
                        stripe_height = stripe_end_y - stripe_start_y + 1
                        stripe_width = right_x - left_x
                        
                        # The magnetic stripe has a known aspect ratio:
                        # Standard: ~79mm width / ~12.7mm height = ~6.2:1
                        aspect_ratio = stripe_width / max(stripe_height, 1)
                        
                        # The stripe should be roughly 55-65% of face/ROI width
                        # (card width is similar to face width, stripe is ~90% of card)
                        stripe_face_ratio = stripe_width / roi_w
                        
                        # The magnetic stripe is typically 10-15 pixels tall in typical photos
                        # and spans most of the card width
                        if stripe_height >= 5 and stripe_width >= min_card_width:
                            # Calculate how dark the stripe region is
                            stripe_region = gray[stripe_start_y:stripe_end_y+1, left_x:right_x]
                            darkness = 1.0 - (np.mean(stripe_region) / 255.0)  # 0=white, 1=black
                            
                            # Score based on multiple factors:
                            # 1. Darkness (darker is better)
                            # 2. Height bonus (prefer taller stripes)
                            # 3. Aspect ratio penalty (penalize ratios far from ideal 6.2:1)
                            # 4. Face ratio penalty (penalize if stripe is too wide relative to face)
                            # 5. Position bonus: prefer stripes near the TOP of the ROI (magnetic stripe is at top of card)
                            height_bonus = min(1.0, stripe_height / 20.0)
                            
                            # Aspect ratio penalty: ideal is 6.2, penalize deviation
                            aspect_penalty = 1.0 - min(0.5, abs(aspect_ratio - 6.2) / 10.0)
                            
                            # Face ratio penalty: ideal is ~0.72 (stripe is ~92% of card, card is ~78% of face)
                            # Favor wider stripes over narrower ones
                            face_ratio_penalty = 1.0 - min(0.5, abs(stripe_face_ratio - 0.72) / 0.3)
                            
                            # Position bonus: magnetic stripe is at the TOP of the card
                            # Stripes in top 30% of ROI get full bonus, lower ones get penalized
                            y_position_ratio = stripe_start_y / roi_h
                            position_bonus = max(0.3, 1.0 - y_position_ratio * 1.5)
                            
                            # Width bonus: slightly prefer wider stripes (but don't overweight)
                            # Bonus is between 0.9 and 1.1 to slightly favor wider stripes
                            width_bonus = 0.9 + 0.2 * (stripe_face_ratio / 0.8)
                            
                            # Include position bonus and width bonus in final score
                            score = darkness * height_bonus * aspect_penalty * face_ratio_penalty * position_bonus * width_bonus
                            
                            if score > best_score:
                                best_stripe_y = stripe_start_y
                                best_stripe_left_x = left_x
                                best_stripe_right_x = right_x
                                best_stripe_height = stripe_height
                                best_score = score
                        
                            # Skip past this stripe region
                            y = stripe_end_y + 1
                            continue
                
                y += 1
        
        # ============================================================
        # FALLBACK: Center-region row intensity detection
        # If dark pixel detection failed or detected full width,
        # analyze the center 70% of ROI (excluding hair/face edges)
        # ============================================================
        
        if best_score == 0 or (best_stripe_right_x is not None and best_stripe_left_x is not None and 
                                (best_stripe_right_x - best_stripe_left_x) > roi_w * 0.85):
            # Reset for fallback
            best_stripe_y = None
            best_stripe_left_x = None
            best_stripe_right_x = None
            best_stripe_height = 0
            best_score = 0
            
            # Analyze center 70% of ROI to avoid hair/face edges
            margin = int(roi_w * 0.15)
            center_gray = gray[:, margin:roi_w-margin]
            center_w = center_gray.shape[1]
            
            # Calculate mean intensity per row in center region
            row_means = np.mean(center_gray, axis=1)
            
            # Find rows that are significantly darker than average
            overall_mean = np.mean(row_means)
            dark_rows = row_means < overall_mean * 0.85
            
            # Find contiguous bands of dark rows
            dark_indices = np.where(dark_rows)[0]
            if len(dark_indices) > 5:
                # Find contiguous runs
                runs = []
                run_start = dark_indices[0]
                for i in range(1, len(dark_indices)):
                    if dark_indices[i] - dark_indices[i-1] > 3:  # Gap of more than 3 rows
                        runs.append((run_start, dark_indices[i-1]))
                        run_start = dark_indices[i]
                runs.append((run_start, dark_indices[-1]))
                
                # Find the run with highest contrast
                best_run = None
                best_contrast = 0
                for start_y, end_y in runs:
                    if end_y - start_y >= 5:  # At least 5 rows tall
                        band_mean = np.mean(row_means[start_y:end_y+1])
                        above_mean = np.mean(row_means[max(0,start_y-10):start_y]) if start_y > 10 else 255
                        below_mean = np.mean(row_means[end_y+1:min(roi_h, end_y+11)]) if end_y < roi_h-10 else 255
                        contrast = (above_mean + below_mean) / 2 - band_mean
                        
                        if contrast > best_contrast and contrast > 10:  # Must be at least 10 intensity units darker
                            best_run = (start_y, end_y)
                            best_contrast = contrast
                
                if best_run:
                    start_y, end_y = best_run
                    # Now find the horizontal extent of the stripe in this band
                    stripe_band = gray[start_y:end_y+1, :]
                    band_mean = np.mean(stripe_band)
                    
                    # Use vertical edge detection to find card boundaries
                    # The card has distinct vertical edges (card -> hair transition)
                    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_x_abs = np.abs(sobel_x)
                    
                    # Sum vertical edge strength across the stripe band rows
                    band_edges = sobel_x_abs[start_y:end_y+1, :]
                    col_edge_strength = np.sum(band_edges, axis=0)
                    
                    # Smooth the edge signal
                    kernel = np.ones(5) / 5
                    col_edge_smooth = np.convolve(col_edge_strength, kernel, mode='same')
                    
                    # Find strong vertical edges (card boundaries)
                    edge_threshold = np.percentile(col_edge_smooth, 85)
                    strong_edges = col_edge_smooth > edge_threshold
                    edge_indices = np.where(strong_edges)[0]
                    
                    roi_center = roi_w // 2
                    left_edge = None
                    right_edge = None
                    
                    # Find leftmost strong edge in the center-left region
                    center_left = roi_center - int(roi_w * 0.35)
                    for idx in edge_indices:
                        if center_left - 30 < idx < center_left + 30:
                            left_edge = idx
                            break
                    
                    # If not found, use first edge after margin
                    if left_edge is None:
                        for idx in edge_indices:
                            if 30 < idx < roi_center:
                                left_edge = idx
                                break
                    
                    # Find rightmost strong edge in the center-right region
                    center_right = roi_center + int(roi_w * 0.35)
                    for idx in reversed(edge_indices):
                        if center_right - 30 < idx < center_right + 30:
                            right_edge = idx
                            break
                    
                    # If not found, use last edge before margin
                    if right_edge is None:
                        for idx in reversed(edge_indices):
                            if roi_center < idx < roi_w - 30:
                                right_edge = idx
                                break
                    
                    if left_edge and right_edge and right_edge - left_edge >= min_card_width:
                        # We found the CARD edges, but we need the STRIPE edges
                        # The magnetic stripe is ~92% of the card width, centered
                        card_width = right_edge - left_edge
                        card_center = (left_edge + right_edge) // 2
                        
                        # The magnetic stripe is 79mm on a 85.6mm card = 92.3%
                        stripe_ratio = 0.923
                        stripe_width = int(card_width * stripe_ratio)
                        stripe_left = card_center - stripe_width // 2
                        stripe_right = card_center + stripe_width // 2
                        
                        best_gap = (stripe_left, stripe_right, stripe_width)
                        gaps = [(stripe_left, stripe_right, stripe_width, card_center)]
                    else:
                        gaps = []
                        best_gap = None
                    
                    if best_gap:
                            best_stripe_y = start_y
                            best_stripe_left_x = best_gap[0]
                            best_stripe_right_x = best_gap[1]
                            best_stripe_height = end_y - start_y + 1
                            best_score = 0.5
            
            # If center-region detection didn't work, try edge-based as final fallback
            if best_score == 0:
                # Look for pairs of strong horizontal edges (top and bottom of stripe)
                # Sum horizontal edge strength across rows
                row_edge_strength = np.sum(sobel_y_abs, axis=1)
                
                # Find peaks in edge strength (potential stripe boundaries)
                # Smooth the signal first
                kernel_size = 5
                smoothed = np.convolve(row_edge_strength, np.ones(kernel_size)/kernel_size, mode='same')
                
                # Find rows with high edge strength
                edge_threshold = np.percentile(smoothed, 80)
                
                # Look for stripe: two strong edges separated by 10-50 pixels
                for start_y in range(skip_top_rows, roi_h - 15):
                    if smoothed[start_y] > edge_threshold:
                        # Found potential top edge, look for bottom edge
                        for end_y in range(start_y + 8, min(start_y + 50, roi_h)):
                            if smoothed[end_y] > edge_threshold:
                                # Found potential stripe region
                                stripe_region = gray[start_y:end_y, :]
                                
                                # The stripe should be darker than surrounding areas
                                region_mean = np.mean(stripe_region)
                                above_mean = np.mean(gray[max(0, start_y-10):start_y, :]) if start_y > 10 else 255
                                below_mean = np.mean(gray[end_y:min(roi_h, end_y+10), :]) if end_y < roi_h - 10 else 255
                                
                                # Stripe should be darker than area above (face/skin) 
                                # The area below might be card which is also light
                                if region_mean < above_mean * 0.95:
                                    # Find the horizontal extent of the stripe using intensity profile
                                    # Average intensity across rows in the stripe region
                                    stripe_profile = np.mean(gray[start_y:end_y, :], axis=0)
                                    
                                    # Find where the stripe is (darker than threshold)
                                    stripe_threshold = region_mean + 10
                                    stripe_cols = stripe_profile < stripe_threshold
                                    
                                    # Find contiguous runs in the stripe
                                    dark_cols = np.where(stripe_cols)[0]
                                    if len(dark_cols) > 0:
                                        # Find contiguous runs
                                        runs = []
                                        run_start = dark_cols[0]
                                        run_end = dark_cols[0]
                                        for i in range(1, len(dark_cols)):
                                            if dark_cols[i] - dark_cols[i-1] <= 5:  # Allow small gaps
                                                run_end = dark_cols[i]
                                            else:
                                                runs.append((run_start, run_end, run_end - run_start))
                                                run_start = dark_cols[i]
                                                run_end = dark_cols[i]
                                        runs.append((run_start, run_end, run_end - run_start))
                                        
                                        # Find the widest centered run
                                        roi_center = roi_w // 2
                                        best_run = None
                                        for run_start, run_end, run_width in runs:
                                            run_center = (run_start + run_end) // 2
                                            if run_width >= min_card_width and abs(run_center - roi_center) < roi_w * 0.3:
                                                if best_run is None or run_width > best_run[2]:
                                                    best_run = (run_start, run_end, run_width)
                                        
                                        if best_run and best_run[2] < max_reasonable_width:
                                            left_x, right_x, stripe_width = best_run[0], best_run[1], best_run[2]
                                            best_stripe_y = start_y
                                            best_stripe_left_x = left_x
                                            best_stripe_right_x = right_x
                                            best_stripe_height = end_y - start_y
                                            best_score = 0.5  # Indicate fallback method was used
                                            break
                        if best_score > 0:
                            break
        
        # Use the stripe detection results
        best_card_top_y = best_stripe_y
        best_card_left_x = best_stripe_left_x
        best_card_right_x = best_stripe_right_x
        
        # Use white card body to refine stripe edges (helps when dark skin touches stripe)
        if best_card_left_x is not None and best_card_right_x is not None and best_stripe_y is not None:
            # Find the white card body below the stripe
            card_body_start_y = best_stripe_y + best_stripe_height + 5
            card_body_end_y = min(roi_h, card_body_start_y + 80)
            
            if card_body_end_y > card_body_start_y:
                card_body_region = gray[card_body_start_y:card_body_end_y, :]
                # Find columns where the card body is bright (white card)
                col_brightness = np.mean(card_body_region, axis=0)
                
                # Use high threshold for white card and find contiguous bright region
                bright_threshold = 200
                bright_mask = col_brightness > bright_threshold
                bright_indices = np.where(bright_mask)[0]
                
                if len(bright_indices) > 50:
                    # Find the longest contiguous run of bright pixels (the card body)
                    runs = []
                    run_start = bright_indices[0]
                    run_end = bright_indices[0]
                    for i in range(1, len(bright_indices)):
                        if bright_indices[i] - bright_indices[i-1] <= 5:  # Allow small gaps
                            run_end = bright_indices[i]
                        else:
                            runs.append((run_start, run_end, run_end - run_start))
                            run_start = bright_indices[i]
                            run_end = bright_indices[i]
                    runs.append((run_start, run_end, run_end - run_start))
                    
                    # Select the longest run (should be the card body)
                    if runs:
                        best_run = max(runs, key=lambda r: r[2])
                        card_body_left = best_run[0]
                        card_body_right = best_run[1]
                        
                        # Only apply constraint if card body is reasonably sized
                        if best_run[2] > roi_w * 0.5:
                            # Clip stripe edges to card body edges (stripe can't extend beyond card)
                            if best_card_left_x < card_body_left:
                                best_card_left_x = card_body_left
                            if best_card_right_x > card_body_right:
                                best_card_right_x = card_body_right
        
        # Store visualization data
        all_contours = []
        
        # If we found the stripe, store its bounding box
        if best_card_top_y is not None:
            card_width = best_card_right_x - best_card_left_x
            
            # Store the stripe bounding box
            x = int(best_card_left_x)
            y = int(best_card_top_y)
            rect_w = int(card_width)
            rect_h = int(best_stripe_height)
            
            
            # Stripe bounding box (adjusted to full image coordinates)
            stripe_bbox = {
                'x': x + roi_x_start,
                'y': y + roi_y_start,
                'width': rect_w,
                'height': rect_h,
                'width_pixels': float(rect_w)
            }
            
            # Top edge line for compatibility
            top_edge_line = {
                'x1': x + roi_x_start,
                'y1': y + roi_y_start,
                'x2': x + rect_w + roi_x_start,
                'y2': y + roi_y_start,
                'width_pixels': float(rect_w)
            }
            
            
            best = {
                'stripe_bbox': stripe_bbox,
                'top_edge': top_edge_line,
                'width': rect_w,
                'height': rect_h,
                'method': 'black_stripe',
                'top_edge_y': best_card_top_y
            }
            
            result = {
                'stripe_bbox': stripe_bbox,
                'top_edge': top_edge_line,
                'width_pixels': float(rect_w),
                'height_pixels': float(rect_h),
                'roi_offset': roi_y_start,
                'all_candidates': [best],
                'selected_candidate': best,
                'all_contours': all_contours
            }
        else:
            # No card detected
            result = None
        
        roi_coords = {'y_start': roi_y_start, 'x_start': roi_x_start}
        
        return result, roi_coords
        
        roi_coords = {'y_start': roi_y_start, 'x_start': roi_x_start}
        
        return result, roi_coords
    
    def calculate_pupil_distance(self, left_pupil: np.ndarray, right_pupil: np.ndarray,
                                card_width_pixels: float) -> Dict:
        """
        Calculate inter-pupillary distance in pixels and convert to millimeters.
        
        Args:
            left_pupil: Left pupil center coordinates [x, y]
            right_pupil: Right pupil center coordinates [x, y]
            card_width_pixels: Credit card width in pixels
            
        Returns:
            Dictionary with distance measurements
        """
        # Calculate Euclidean distance in pixels
        pupil_distance_pixels = np.linalg.norm(right_pupil - left_pupil)
        
        # Calculate conversion factor using credit card width
        mm_per_pixel = MAGNETIC_STRIPE_WIDTH_MM / card_width_pixels
        
        # Convert to millimeters
        pupil_distance_mm = pupil_distance_pixels * mm_per_pixel
        
        return {
            'pupil_distance_pixels': float(pupil_distance_pixels),
            'pupil_distance_mm': float(pupil_distance_mm),
            'card_width_pixels': card_width_pixels,
            'card_width_mm': MAGNETIC_STRIPE_WIDTH_MM,
            'mm_per_pixel': float(mm_per_pixel)
        }
    
    def create_pupil_visualization(self, image: np.ndarray, left_pupil: np.ndarray,
                                   right_pupil: np.ndarray, distance_mm: float) -> np.ndarray:
        """
        Create visualization image with pupils marked and distance labeled.
        
        Args:
            image: Original image
            left_pupil: Left pupil center [x, y]
            right_pupil: Right pupil center [x, y]
            distance_mm: Inter-pupillary distance in mm
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        # Draw pupils (red circles)
        cv2.circle(vis_image, (int(left_pupil[0]), int(left_pupil[1])), 5, (0, 0, 255), -1)
        cv2.circle(vis_image, (int(right_pupil[0]), int(right_pupil[1])), 5, (0, 0, 255), -1)
        
        # Draw connecting line (green)
        cv2.line(vis_image, 
                (int(left_pupil[0]), int(left_pupil[1])),
                (int(right_pupil[0]), int(right_pupil[1])),
                (0, 255, 0), 2)
        
        return vis_image
    
    def create_combined_visualization(self, image: np.ndarray, left_pupil: np.ndarray,
                                     right_pupil: np.ndarray, distance_mm: float,
                                     card_data: Dict) -> np.ndarray:
        """
        Create combined visualization with both card detection (yellow) and pupil detection (bright green).
        Adds distance text at the bottom of the image.
        
        Args:
            image: Original image
            left_pupil: Left pupil center [x, y]
            right_pupil: Right pupil center [x, y]
            distance_mm: Inter-pupillary distance in mm
            card_data: Dictionary with card detection data
            
        Returns:
            Annotated image with both card and pupil detection
        """
        vis_image = image.copy()
        
        # Draw card detection in yellow
        if 'stripe_bbox' in card_data:
            bbox = card_data['stripe_bbox']
            x, y = int(bbox['x']), int(bbox['y'])
            w, h = int(bbox['width']), int(bbox['height'])
            # Yellow color in BGR: (0, 255, 255)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        elif 'top_edge' in card_data:
            # Fallback to top edge line in yellow
            top_edge = card_data['top_edge']
            x1, y1 = int(top_edge['x1']), int(top_edge['y1'])
            x2, y2 = int(top_edge['x2']), int(top_edge['y2'])
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        # Draw pupils in bright green (BGR: (0, 255, 0) is bright green)
        # Make circles slightly larger for visibility
        cv2.circle(vis_image, (int(left_pupil[0]), int(left_pupil[1])), 6, (0, 255, 0), -1)
        cv2.circle(vis_image, (int(right_pupil[0]), int(right_pupil[1])), 6, (0, 255, 0), -1)
        
        # Draw connecting line in bright green
        cv2.line(vis_image, 
                (int(left_pupil[0]), int(left_pupil[1])),
                (int(right_pupil[0]), int(right_pupil[1])),
                (0, 255, 0), 3)
        
        # Add text at the bottom of the image with distance
        h, w = vis_image.shape[:2]
        label = f"PD distance: {distance_mm:.2f} mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Smaller font for better fit
        thickness = 1
        color = (255, 255, 255)  # White text
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position text at bottom center with some padding
        text_x = (w - text_width) // 2
        text_y = h - 15  # 15 pixels from bottom
        
        # Draw background rectangle for better text visibility
        cv2.rectangle(vis_image,
                     (text_x - 8, text_y - text_height - baseline - 8),
                     (text_x + text_width + 8, text_y + baseline + 8),
                     (0, 0, 0), -1)  # Black background
        
        # Draw text
        cv2.putText(vis_image, label,
                   (text_x, text_y),
                   font, font_scale, color, thickness)
        
        return vis_image
    
    def create_card_visualization(self, image: np.ndarray, card_data: Dict) -> np.ndarray:
        """
        Create visualization image with detected black stripe rectangle.
        
        Args:
            image: Original image
            card_data: Dictionary with card detection data (must include 'stripe_bbox' or 'top_edge')
            
        Returns:
            Annotated image with stripe rectangle
        """
        vis_image = image.copy()
        
        # Draw the stripe bounding box if available
        if 'stripe_bbox' in card_data:
            bbox = card_data['stripe_bbox']
            x, y = int(bbox['x']), int(bbox['y'])
            w, h = int(bbox['width']), int(bbox['height'])
            
            # Draw rectangle around the stripe (blue)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        elif 'top_edge' in card_data:
            # Fallback to top edge line
            top_edge = card_data['top_edge']
            x1, y1 = int(top_edge['x1']), int(top_edge['y1'])
            x2, y2 = int(top_edge['x2']), int(top_edge['y2'])
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        return vis_image
    
    def create_contours_visualization(self, image: np.ndarray, card_data: Dict, 
                                     roi_y_start: int, roi_x_start: int) -> np.ndarray:
        """
        Create visualization image showing all detected contours, lines, and the selected one.
        
        Args:
            image: Original image
            card_data: Dictionary with card detection data
            roi_y_start: Y offset of the ROI
            roi_x_start: X offset of the ROI
            
        Returns:
            Annotated image with all contours and lines
        """
        vis_image = image.copy()
        
        # Draw all contours in light gray
        if 'all_contours' in card_data:
            for contour_data in card_data['all_contours']:
                contour = contour_data['contour']
                offset_x, offset_y = contour_data['roi_offset']
                # Adjust contour coordinates to full image
                contour_adjusted = contour.copy()
                if len(contour_adjusted.shape) == 3 and contour_adjusted.shape[1] == 1:
                    contour_adjusted[:, 0, 0] += offset_x
                    contour_adjusted[:, 0, 1] += offset_y
                    cv2.drawContours(vis_image, [contour_adjusted], -1, (128, 128, 128), 1)
        
        # Draw all candidates' top edges in cyan
        if 'all_candidates' in card_data:
            for candidate in card_data['all_candidates']:
                if 'top_edge' in candidate:
                    top_edge = candidate['top_edge']
                    x1, y1 = int(top_edge['x1']), int(top_edge['y1'])
                    x2, y2 = int(top_edge['x2']), int(top_edge['y2'])
                    cv2.line(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Draw selected candidate's top edge in red (thicker)
        if 'selected_candidate' in card_data:
            selected = card_data['selected_candidate']
            if 'top_edge' in selected:
                top_edge = selected['top_edge']
                x1, y1 = int(top_edge['x1']), int(top_edge['y1'])
                x2, y2 = int(top_edge['x2']), int(top_edge['y2'])
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        return vis_image
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Process a single image and generate all outputs.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output files (default: same as input)
            
        Returns:
            Dictionary with all measurements and output paths
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Setup output paths
        input_path = Path(image_path)
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = input_path.stem
        
        # Detect face and iris
        face_data = self.detect_face_iris(image)
        if face_data is None:
            raise ValueError("Face or iris not detected in image")
        
        # Detect credit card
        card_data, roi_coords = self.detect_credit_card(
            image, 
            face_data['nose_y'],
            face_data['chin_y'],
            face_data['face_left'],
            face_data['face_right'],
            face_data['face_width'],
            face_data['left_pupil'],
            face_data['right_pupil'],
            None
        )
        if card_data is None:
            raise ValueError("Credit card not detected in image")
        
        # Calculate distances using card-based calibration
        measurements = self.calculate_pupil_distance(
            face_data['left_pupil'],
            face_data['right_pupil'],
            card_data['width_pixels']
        )
        
        # Create visualizations
        pupil_vis = self.create_pupil_visualization(
            image,
            face_data['left_pupil'],
            face_data['right_pupil'],
            measurements['pupil_distance_mm']
        )
        
        # Create combined visualization with both card and pupil detection
        combined_vis = self.create_combined_visualization(
            image,
            face_data['left_pupil'],
            face_data['right_pupil'],
            measurements['pupil_distance_mm'],
            card_data
        )
        
        # Save visualization images
        pupil_vis_path = output_dir / f"{base_name}_pupils_marked.jpg"
        combined_vis_path = output_dir / f"{base_name}_output.jpg"
        
        cv2.imwrite(str(pupil_vis_path), pupil_vis)
        cv2.imwrite(str(combined_vis_path), combined_vis)
        
        # Extract card bounding box
        card_bbox = None
        if 'stripe_bbox' in card_data:
            bbox = card_data['stripe_bbox']
            card_bbox = {
                'x': float(bbox['x']),
                'y': float(bbox['y']),
                'width': float(bbox['width']),
                'height': float(bbox['height'])
            }
        
        # Prepare results
        results = {
            'input_image': str(image_path),
            'output_image': str(combined_vis_path),
            'pupil_distance_pixels': measurements['pupil_distance_pixels'],
            'pupil_distance_mm': measurements['pupil_distance_mm'],
            'card_width_pixels': measurements['card_width_pixels'],
            'card_width_mm': measurements['card_width_mm'],
            'mm_per_pixel': measurements['mm_per_pixel'],
            'left_pupil': [float(face_data['left_pupil'][0]), float(face_data['left_pupil'][1])],
            'right_pupil': [float(face_data['right_pupil'][0]), float(face_data['right_pupil'][1])],
            'card_bbox': card_bbox,
            'output_images': {
                'pupils_marked': str(pupil_vis_path)
            }
        }
        
        # Save JSON results
        json_path = output_dir / f"{base_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['results_json'] = str(json_path)
        
        return results


def main():
    """Main function to process images."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Measure inter-pupillary distance using face detection and credit card scale'
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output directory (default: same as input image)')
    
    args = parser.parse_args()
    
    measurer = PupilDistanceMeasurer()
    
    try:
        results = measurer.process_image(args.image, args.output)
        
        print("Measurement completed successfully!")
        print(f"Inter-pupillary distance: {results['pupil_distance_mm']:.2f} mm")
        print(f"Credit card width: {results['card_width_pixels']:.1f} pixels")
        print(f"Conversion factor: {results['mm_per_pixel']:.4f} mm/pixel")
        print(f"\nOutput files:")
        print(f"  - Pupils visualization: {results['output_images']['pupils_marked']}")
        print(f"  - Results JSON: {results['results_json']}")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
