# Import required libraries
import datetime
import eventlet
import orjson
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from eventlet import wsgi
from eventlet.websocket import WebSocketWSGI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from scipy.spatial.distance import cdist

# Monkey patch standard library for async compatibility
eventlet.monkey_patch()

# Initialize Flask application
app = Flask(__name__)
# Enable CORS for cross-origin requests
CORS(app)

# MediaPipe setup for hand detection
mp_hands = mp.solutions.hands
# Configure hand detector
hands = mp_hands.Hands(
    static_image_mode=False,  # For video streams
    max_num_hands=2,  # Maximum number of hands to detect
    min_detection_confidence=0.65,  # Minimum confidence for detection
    min_tracking_confidence=0.65  # Minimum confidence for tracking
)

# Store saved hand symbols/gestures
hand_symbols = []

# Route for main page
@app.route('/')
def index():
    # Serve the main HTML page
    return render_template('index.html')

# Route to save new hand symbols
@app.route('/save_handsymbol', methods=['POST'])
def save_handsymbol():
    # Extract data from POST request
    data = request.json
    name, handedness, landmarks = data['name'], data['handedness'], data['landmarks']
    
    # Normalize landmarks relative to wrist
    wrist = landmarks[0]
    normalized_landmarks = np.array(landmarks) - wrist
    
    # Calculate rotation angle using middle finger base
    middle_finger_mcp = normalized_landmarks[9]
    angle = np.arctan2(middle_finger_mcp[1], middle_finger_mcp[0])
    # Create rotation matrix for alignment
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    # Apply rotation to landmarks (only x,y coordinates)
    rotated_landmarks = np.hstack((normalized_landmarks[:, :2] @ rotation_matrix.T, normalized_landmarks[:, 2:]))
    
    # Store the normalized and rotated symbol
    hand_symbols.append({
        'name': name,
        'handedness': handedness,
        'landmarks': rotated_landmarks.flatten()  # Flatten to 1D array for distance calculation
    })

    return jsonify({'status': 'success'})

# WebSocket handler for real-time hand tracking
@WebSocketWSGI
def handle_websocket(ws):
    # Main WebSocket loop
    while True:
        message = ws.wait()
        if message is None:
            break
        try:
            # Start timing for performance monitoring
            start_time = datetime.datetime.now()

            # Process incoming message (could be binary or JSON)
            if isinstance(message, bytes):
                frame_bytes = message
            else:
                payload = json.loads(message)
                image_data = payload.get("image", "")
                if image_data.startswith("data:image"):
                    # Extract base64 image data
                    header, encoded = image_data.split(",", 1)
                    frame_bytes = base64.b64decode(encoded)
                else:
                    frame_bytes = None
            
            if frame_bytes is not None:
                # Decode image from bytes
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                # Resize for consistent processing
                frame = cv2.resize(frame, (640, 360))
                h, w = frame.shape[:2]

                # Performance check: skip if pre-processing is too slow
                if (datetime.datetime.now() - start_time).total_seconds() * 1000 > 50:
                    print("Skipping frame: Pre-processing too slow")
                    ws.send(orjson.dumps({'status': 'dropped'}))
                    continue

                # Process frame for hand detection
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Performance check: skip if hand detection is too slow
                if (datetime.datetime.now() - start_time).total_seconds() * 1000 > 50:
                    print("Skipping frame: Hand processing too slow")
                    ws.send(orjson.dumps({'status': 'dropped'}))
                    continue

                hands_data = []
                detected = {"Left": False, "Right": False}

                # Process detected hands
                if results.multi_hand_landmarks:
                    for idx, landmarks in enumerate(results.multi_hand_landmarks):
                        # Performance check for each hand
                        if (datetime.datetime.now() - start_time).total_seconds() * 1000 > 50:
                            print("Skipping frame: Exceeded time limit in loop")
                            ws.send(orjson.dumps({'status': 'dropped'}))
                            break
                        
                        # Get handedness (left or right)
                        handedness = results.multi_handedness[idx].classification[0].label
                        # Skip if already detected this hand type (max 1 per type)
                        if detected[handedness]:
                            continue
                        detected[handedness] = True
                        
                        # Convert normalized coordinates to pixel coordinates
                        hand_landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in landmarks.landmark])
                        # Normalize relative to wrist
                        wrist = hand_landmarks[0]
                        normalized_landmarks = hand_landmarks - wrist
                        
                        # Calculate rotation for alignment
                        middle_finger_mcp = normalized_landmarks[9]
                        angle = np.arctan2(middle_finger_mcp[1], middle_finger_mcp[0])
                        rotation_matrix = np.array([
                            [np.cos(-angle), -np.sin(-angle)],
                            [np.sin(-angle),  np.cos(-angle)]
                        ])
                        # Apply rotation
                        rotated_landmarks = np.hstack((normalized_landmarks[:, :2] @ rotation_matrix.T, normalized_landmarks[:, 2:]))
                        flattened_landmarks = rotated_landmarks.flatten()

                        # Match against saved symbols
                        detected_symbols = []
                        if hand_symbols:
                            # Get landmarks for symbols of same handedness
                            symbol_landmarks = np.array([
                                symbol['landmarks']
                                for symbol in hand_symbols if symbol['handedness'] == handedness
                            ])
                            if symbol_landmarks.size > 0:
                                # Calculate cosine similarity for matching
                                similarities = (1 - cdist([flattened_landmarks], symbol_landmarks, metric='cosine')[0]).tolist()
                                # Get top 3 matches
                                detected_symbols = sorted(
                                    zip([s['name'] for s in hand_symbols if s['handedness'] == handedness], similarities),
                                    key=lambda x: x[1],
                                    reverse=True
                                )[:3]
                        
                        # Add hand data to response
                        hands_data.append({
                            'handedness': handedness,
                            'landmarks': hand_landmarks.round(3).tolist(),
                            'connections': [[conn[0], conn[1]] for conn in mp_hands.HAND_CONNECTIONS],
                            'detected_symbols': detected_symbols
                        })
                
                # Final performance check before sending
                if (datetime.datetime.now() - start_time).total_seconds() * 1000 > 50:
                    print("Skipping frame: Final check exceeded 50ms")
                    ws.send(orjson.dumps({'status': 'dropped'}))
                    continue

                # Send processed data back via WebSocket
                print(datetime.datetime.now().strftime("%H:%M:%S") + " returned")
                ws.send(orjson.dumps({'status': 'success', 'hands': hands_data, 'image_size': {'width': w, 'height': h}}))
        except Exception as e:
            print("WebSocket error:", str(e))

# Combined application handler for both HTTP and WebSocket
def combined_app(environ, start_response):
    path = environ['PATH_INFO']
    # Route WebSocket requests to handle_websocket
    if path == '/ws':
        return handle_websocket(environ, start_response)
    # Route all other requests to Flask app
    return app(environ, start_response)

# Main entry point
if __name__ == '__main__':
    # Start combined HTTP/WebSocket server
    wsgi.server(eventlet.listen(('0.0.0.0', 6969), reuse_port=True), combined_app)
