import cv2

# Initialize CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Open the webcam (0 is the default camera, change it if you have multiple cameras)
video = cv2.VideoCapture("E:\\Double1.mp4")

# Read the first frame
success, frame = video.read()
if not success:
    print("Error accessing the camera.")
    exit()

# Select the object to track
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)  # Initialize tracker with the selected bounding box

while True:
    # Read frames from the camera
    success, frame = video.read()
    if not success:
        print("Failed to read frame from the camera.")
        break

    # Update the tracker with the current frame
    success, bbox = tracker.update(frame)

    # Draw the bounding box if tracking is successful
    if success:
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()