"""
SIH 2026 - Problem Statement 166
Multi-modal, Sun-angle and Scale-invariant Image Correspondence (Prototype)

Pipeline: CLAHE (illumination/sun-angle normalization) -> ORB (scale/rotation
invariant features) -> BFMatcher + Lowe ratio test -> RANSAC homography
(geometric consistency) -> Match Score + visualization.

Run: python app.py   then open http://127.0.0.1:5000
"""

import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_image_from_filestorage(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def to_base64_png(img_bgr):
    ok, buffer = cv2.imencode('.png', img_bgr)
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode('utf-8')


def resize_max(im, max_dim=800):
    h, w = im.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale < 1:
        im = cv2.resize(im, (int(w * scale), int(h * scale)))
    return im


def make_synthetic_pair(img):
    """Creates a rotated + scaled + illumination-shifted copy of img.
    Used as a demo fallback and to visibly demonstrate scale/sun-angle
    invariance when only one sample image is available."""
    h, w = img.shape[:2]
    angle = 25
    scale = 0.8
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Simulate a different "sun angle" by lowering brightness / adding offset
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.6 + 15, 0, 255)
    synthetic = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return synthetic


def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/match', methods=['POST'])
def match():
    try:
        img1_file = request.files.get('image1')
        img2_file = request.files.get('image2')
        force_synthetic = request.form.get('synthetic') == 'true'

        if img1_file is None or img1_file.filename == '':
            return jsonify({'error': 'Image 1 is required'}), 400

        img1 = read_image_from_filestorage(img1_file)
        if img1 is None:
            return jsonify({'error': 'Could not read Image 1. Use JPG/PNG.'}), 400

        no_second_image = (img2_file is None or img2_file.filename == '')
        used_synthetic = force_synthetic or no_second_image

        if used_synthetic:
            img2 = make_synthetic_pair(img1)
        else:
            img2 = read_image_from_filestorage(img2_file)
            if img2 is None:
                return jsonify({'error': 'Could not read Image 2. Use JPG/PNG.'}), 400

        img1 = resize_max(img1)
        img2 = resize_max(img2)

        gray1 = preprocess_gray(img1)
        gray2 = preprocess_gray(img2)

        orb = cv2.ORB_create(nfeatures=3000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return jsonify({'error': 'Not enough visual features detected. Try clearer/higher-contrast images.'}), 400

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        inlier_count = 0
        matches_mask = None

        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                matches_mask = mask.ravel().tolist()
                inlier_count = int(mask.sum())

        total_good = len(good)
        inlier_ratio = (inlier_count / total_good * 100) if total_good > 0 else 0.0
        feature_ratio = (total_good / max(1, min(len(kp1), len(kp2)))) * 100

        match_score = round(min(inlier_ratio * 0.7 + feature_ratio * 0.3, 100.0), 2)
        correspondence_detected = inlier_count >= 10 and inlier_ratio >= 40

        if matches_mask is not None:
            draw_params = dict(
                matchColor=(60, 220, 60),
                singlePointColor=None,
                matchesMask=matches_mask,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            result_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        else:
            result_img = cv2.drawMatches(
                img1, kp1, img2, kp2, good[:30], None,
                matchColor=(0, 165, 255),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        return jsonify({
            'result_image': to_base64_png(result_img),
            'image1': to_base64_png(img1),
            'image2': to_base64_png(img2),
            'keypoints1': len(kp1),
            'keypoints2': len(kp2),
            'good_matches': total_good,
            'inliers': inlier_count,
            'match_score': match_score,
            'correspondence_detected': correspondence_detected,
            'used_synthetic': used_synthetic
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.envirn.get("PORT",5000))
    app.run(hst="0.0.0.0",port=port)
