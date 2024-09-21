import cv2
from fer import FER

# Initialize the FER detector
detector = FER()

def process_frame(frame):
    # Detect emotions in the frame
    result = detector.detect_emotions(frame)
    if result:
        # Get the most prominent emotion
        emotions = result[0]['emotions']
        max_emotion = max(emotions, key=emotions.get)
        return max_emotion, emotions
    return None, {}

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access the webcam. Please make sure your webcam is plugged in and accessible.")
        return

    
    print("Webcam is successfully opened. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from webcam.")
            break
        
        # Get frame dimensions
        height, width, _ = frame.shape
        
        # Process the frame
        emotion, emotions = process_frame(frame)
        
        # Display the resulting frame with detected emotion
        # Set default emotion text
        text = 'Emotion: Happy'
        if emotion:
            text = f'Emotion: {emotion}'
            
        # Define font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # Green color
        font_thickness = 2
        
        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = width - text_size[0] - 10  # 10 pixels from the right edge
        text_y = height // 2 + text_size[1] // 2  # Centered vertically

        # Put text on the frame
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        
        # Create a window with proper size
        cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emotion Detection', width, height)
        
        # Show the full video frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
