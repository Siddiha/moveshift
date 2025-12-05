import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self):
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        # Create hands detector with specified parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Process video frames (not static images)
            max_num_hands=2,  # Maximum number of hands to detect
            min_detection_confidence=0.75,  # Minimum confidence for hand detection
            min_tracking_confidence=0.75)  # Minimum confidence for hand tracking
    
    def detect(self, frame):
        # Convert BGR image to RGB (MediaPipe requires RGB input)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame to detect hands
        results = self.hands.process(rgb)
        # Convert results to serializable format
        return self._serialize_results(results)

    def _serialize_results(self, results):
        # Initialize empty list to store hand data
        hands = []
        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # Iterate through each detected hand
            for idx, landmarks in enumerate(results.multi_hand_landmarks):
                try:
                    # Get handedness (left or right hand)
                    handedness = results.multi_handedness[idx].classification[0].label
                    # Create hand dictionary with landmarks and connections
                    hand = {
                        "handedness": handedness,  # "Left" or "Right"
                        # Convert landmarks to list of (x, y, z) coordinates
                        "landmarks": [(float(lm.x), float(lm.y), float(lm.z)) for lm in landmarks.landmark],
                        # Get hand skeleton connections
                        "connections": [[int(conn[0]), int(conn[1])] for conn in self.mp_hands.HAND_CONNECTIONS]
                    }
                    # Add hand data to the list
                    hands.append(hand)
                except (IndexError, AttributeError):
                    # Skip if there's an error with this hand's data
                    continue
        # Return list of detected hands
        return hands
