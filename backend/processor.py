# Import and apply eventlet monkey patching for asynchronous networking
import eventlet
eventlet.monkey_patch()

# Import necessary libraries
import cv2
from webserver import socketio, app  # Import socketio and app from webserver module
from detector import HandDetector  # Import HandDetector class from detector module

class VisionProcessor:
    def __init__(self):
        # Initialize video capture from default camera (camera index 0)
        self.cap = cv2.VideoCapture(0)
        # Create instance of HandDetector for hand tracking
        self.detector = HandDetector()
        # Store last detected hands to track changes
        self.last_hands = []

    def process_video(self):
        # Create application context for Flask-SocketIO operations
        with app.app_context():
            # Main video processing loop
            while True:
                # Read frame from camera
                success, frame = self.cap.read()
                # Skip to next iteration if frame capture failed
                if not success:
                    continue
                
                # Flip frame horizontally for mirror effect (more intuitive)
                frame = cv2.flip(frame, 1)
                # Detect hands in the current frame
                hands_data = self.detector.detect(frame)
                
                # Only emit data if there's a change from previous detection
                if hands_data != self.last_hands:
                    try:
                        # Emit hand data to connected WebSocket clients
                        socketio.emit('hand_data', hands_data)
                        # Update last_hands with current detection data
                        self.last_hands = hands_data
                    except Exception as e:
                        # Print error if emit fails
                        print("Emit error:", str(e))
                
                # Small sleep to yield control and allow other operations
                eventlet.sleep(0.001)

# Main execution block
if __name__ == '__main__':
    # Create VisionProcessor instance
    processor = VisionProcessor()
    # Start video processing
    processor.process_video()
