#!/usr/bin/env python3
"""
Web-based Camera Streaming Application

This script creates a Flask web server that streams camera feeds to a web browser.
It avoids the OpenCV GUI dependency issues by serving frames via HTTP.

Usage:
    python web_camera_stream.py                    # Start web server with camera detection
    python web_camera_stream.py --camera 8         # Start with specific camera
    python web_camera_stream.py --port 5000        # Use custom port
"""

import argparse
import base64
import io
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for camera management
current_camera = None
camera_lock = threading.Lock()
camera_info = {}
available_cameras = []


def detect_all_cameras() -> List[Dict[str, Any]]:
    """
    Detect all available OpenCV cameras in the system.
    
    Returns:
        List of dictionaries containing camera information
    """
    all_cameras = []
    
    # Detect OpenCV cameras by trying different indices
    logger.info("Detecting OpenCV cameras...")
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            camera_info = {
                'id': i,
                'name': f'Camera {i}',
                'type': 'OpenCV',
                'width': width,
                'height': height,
                'fps': fps,
                'backend_api': cap.getBackendName()
            }
            all_cameras.append(camera_info)
            logger.info(f"Found camera {i}: {width}x{height} @ {fps:.1f}fps")
        cap.release()
        # Small delay to ensure camera is properly released
        time.sleep(0.1)
    
    # Also check common device paths
    device_paths = ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3']
    for path in device_paths:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            camera_info = {
                'id': path,
                'name': f'Camera {path}',
                'type': 'OpenCV',
                'width': width,
                'height': height,
                'fps': fps,
                'backend_api': cap.getBackendName()
            }
            all_cameras.append(camera_info)
            logger.info(f"Found camera {path}: {width}x{height} @ {fps:.1f}fps")
        cap.release()
        # Small delay to ensure camera is properly released
        time.sleep(0.1)
    
    logger.info(f"Found {len(all_cameras)} total cameras")
    return all_cameras


def create_camera_instance(camera_id: str) -> Optional[cv2.VideoCapture]:
    """
    Create a camera instance based on the provided camera identifier.
    
    Args:
        camera_id: Camera identifier (index or path)
        
    Returns:
        OpenCV VideoCapture instance or None if creation fails
    """
    logger.info(f"Creating camera instance for ID: {camera_id}")
    
    try:
        # Convert string to int if it's a numeric camera index
        if camera_id.isdigit():
            camera_id = int(camera_id)
        
        # Try to create VideoCapture with the given ID
        camera = cv2.VideoCapture(camera_id)
        
        # Give it a moment to initialize
        time.sleep(0.5)
        
        if not camera.isOpened():
            logger.error(f"Failed to open camera with ID: {camera_id}")
            camera.release()
            return None
        
        # Test if we can actually read a frame
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.error(f"Camera opened but cannot read frames from ID: {camera_id}")
            camera.release()
            return None
        
        logger.info(f"Successfully connected to camera: {camera_id}")
        return camera
        
    except Exception as e:
        logger.error(f"Failed to create camera instance: {e}")
        return None


def get_camera_frame():
    """
    Get a frame from the current camera.
    
    Returns:
        Tuple of (success, frame) where frame is numpy array or None
    """
    global current_camera
    
    with camera_lock:
        if current_camera is None or not current_camera.isOpened():
            return False, None
        
        ret, frame = current_camera.read()
        if not ret or frame is None:
            return False, None
        
        return True, frame


def generate_frames():
    """
    Generator function to yield camera frames for streaming.
    """
    while True:
        success, frame = get_camera_frame()
        
        if not success or frame is None:
            # Create a black frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main page with camera stream."""
    return render_template('camera_stream.html', cameras=available_cameras)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras')
def api_cameras():
    """API endpoint to get list of available cameras."""
    return jsonify(available_cameras)


@app.route('/api/camera/<camera_id>', methods=['POST'])
def switch_camera(camera_id):
    """API endpoint to switch to a different camera."""
    global current_camera, camera_info
    
    logger.info(f"Switching to camera: {camera_id}")
    
    with camera_lock:
        # Release current camera
        if current_camera and current_camera.isOpened():
            current_camera.release()
        
        # Create new camera instance
        new_camera = create_camera_instance(camera_id)
        if new_camera:
            current_camera = new_camera
            # Find camera info
            for cam in available_cameras:
                if str(cam['id']) == str(camera_id):
                    camera_info = cam
                    break
            return jsonify({'success': True, 'message': f'Switched to camera {camera_id}'})
        else:
            return jsonify({'success': False, 'message': f'Failed to connect to camera {camera_id}'})


@app.route('/api/capture', methods=['POST'])
def capture_image():
    """API endpoint to capture and save current frame."""
    success, frame = get_camera_frame()
    
    if not success or frame is None:
        return jsonify({'success': False, 'message': 'No camera available'})
    
    # Save frame
    timestamp = int(time.time())
    filename = f"captured_frame_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    
    # Also encode as base64 for web display
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'success': True, 
            'message': f'Frame saved as {filename}',
            'image': frame_base64
        })
    
    return jsonify({'success': False, 'message': 'Failed to encode frame'})


def create_html_template():
    """Create the HTML template for the camera streaming interface."""
    template_dir = '/home/p/Desktop/lerobot-yam/templates'
    import os
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .video-container {
            text-align: center;
            margin-bottom: 30px;
        }
        #videoStream {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        label {
            font-weight: bold;
            color: #555;
        }
        select, button {
            padding: 8px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        select {
            min-width: 150px;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .status {
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .camera-info {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .camera-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .camera-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background: white;
        }
        .camera-card h3 {
            margin-top: 0;
            color: #333;
        }
        .camera-card p {
            margin: 5px 0;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📹 Camera Stream</h1>
        
        <div class="video-container">
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Camera Stream">
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="cameraSelect">Select Camera:</label>
                <select id="cameraSelect">
                    {% for camera in cameras %}
                    <option value="{{ camera.id }}">{{ camera.name }} ({{ camera.id }})</option>
                    {% endfor %}
                </select>
            </div>
            <div class="control-group">
                <button id="switchCamera">Switch Camera</button>
            </div>
            <div class="control-group">
                <button id="captureBtn">Capture Image</button>
            </div>
        </div>
        
        <div id="status"></div>
        
        <div class="camera-info">
            <h3>Available Cameras</h3>
            <div class="camera-list">
                {% for camera in cameras %}
                <div class="camera-card">
                    <h3>{{ camera.name }}</h3>
                    <p><strong>ID:</strong> {{ camera.id }}</p>
                    <p><strong>Resolution:</strong> {{ camera.width }}x{{ camera.height }}</p>
                    <p><strong>FPS:</strong> {{ "%.1f"|format(camera.fps) }}</p>
                    <p><strong>Backend:</strong> {{ camera.backend_api }}</p>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>

    <script>
        const cameraSelect = document.getElementById('cameraSelect');
        const switchBtn = document.getElementById('switchCamera');
        const captureBtn = document.getElementById('captureBtn');
        const statusDiv = document.getElementById('status');
        const videoStream = document.getElementById('videoStream');

        function showStatus(message, isError = false) {
            statusDiv.innerHTML = `<div class="status ${isError ? 'error' : 'success'}">${message}</div>`;
            setTimeout(() => {
                statusDiv.innerHTML = '';
            }, 3000);
        }

        switchBtn.addEventListener('click', async () => {
            const cameraId = cameraSelect.value;
            switchBtn.disabled = true;
            switchBtn.textContent = 'Switching...';
            
            try {
                const response = await fetch(`/api/camera/${cameraId}`, {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.success) {
                    showStatus(data.message);
                    // Refresh the video stream
                    videoStream.src = videoStream.src + '?t=' + new Date().getTime();
                } else {
                    showStatus(data.message, true);
                }
            } catch (error) {
                showStatus('Error switching camera: ' + error.message, true);
            } finally {
                switchBtn.disabled = false;
                switchBtn.textContent = 'Switch Camera';
            }
        });

        captureBtn.addEventListener('click', async () => {
            captureBtn.disabled = true;
            captureBtn.textContent = 'Capturing...';
            
            try {
                const response = await fetch('/api/capture', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.success) {
                    showStatus(data.message);
                    // Show captured image in a new window
                    if (data.image) {
                        const newWindow = window.open();
                        newWindow.document.write(`
                            <html>
                                <head><title>Captured Image</title></head>
                                <body style="margin:0; text-align:center; background:#000;">
                                    <img src="data:image/jpeg;base64,${data.image}" style="max-width:100%; max-height:100vh;">
                                </body>
                            </html>
                        `);
                    }
                } else {
                    showStatus(data.message, true);
                }
            } catch (error) {
                showStatus('Error capturing image: ' + error.message, true);
            } finally {
                captureBtn.disabled = false;
                captureBtn.textContent = 'Capture Image';
            }
        });

        // Auto-refresh camera list every 30 seconds
        setInterval(async () => {
            try {
                const response = await fetch('/api/cameras');
                const cameras = await response.json();
                // Update camera select options if needed
                // This is a simple implementation - you might want to make it more sophisticated
            } catch (error) {
                console.log('Failed to refresh camera list:', error);
            }
        }, 30000);
    </script>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'camera_stream.html'), 'w') as f:
        f.write(html_content)


def main():
    """Main function to start the web server."""
    global available_cameras, current_camera
    
    parser = argparse.ArgumentParser(
        description="Web-based Camera Streaming Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python web_camera_stream.py                    # Start web server with camera detection
  python web_camera_stream.py --camera 8         # Start with specific camera
  python web_camera_stream.py --port 5000        # Use custom port
        """
    )
    
    parser.add_argument(
        '--camera',
        type=str,
        help='Camera identifier to start with (index or path)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the web server on (default: 5000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Detect available cameras
    available_cameras = detect_all_cameras()
    
    if not available_cameras:
        logger.error("No cameras detected. Exiting.")
        return
    
    # Create HTML template
    create_html_template()
    
    # Initialize camera if specified
    if args.camera:
        current_camera = create_camera_instance(args.camera)
        if current_camera:
            logger.info(f"Started with camera: {args.camera}")
        else:
            logger.warning(f"Failed to initialize camera {args.camera}, starting without camera")
    else:
        # Try to start with the first available camera
        if available_cameras:
            first_camera = available_cameras[0]
            current_camera = create_camera_instance(str(first_camera['id']))
            if current_camera:
                logger.info(f"Auto-started with camera: {first_camera['id']}")
    
    # Start the web server
    logger.info(f"Starting web server on http://{args.host}:{args.port}")
    logger.info("Open your web browser and navigate to the URL above")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        # Cleanup
        if current_camera and current_camera.isOpened():
            current_camera.release()
            logger.info("Camera disconnected")


if __name__ == "__main__":
    main()
