import cv2
import numpy as np
from keras.models import load_model
from tensorflow import keras

model = keras.models.load_model('saved_model_format')

# Load the trained gender classification model
model = load_model("model.h5")  # 0: Male, 1: Female

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Preprocess face image for model input
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # 1. Grayscale
        resized_face = cv2.resize(gray_face, (110, 80))         # 2. Resize (width=110, height=80)
        normalized_face = resized_face / 255.0                   # 3. Normalize
        input_face = np.expand_dims(normalized_face, axis=0)     # 4. Add batch dim
        input_face = np.expand_dims(input_face, axis=-1)         # 5. Add channel dim, shape=(1, 80, 110, 1)

        # Predict gender
        prediction = model.predict(input_face)[0][0]
        gender = "Female" if prediction > 0.5 else "Male"

        # Draw bounding box and label on original frame
        color = (255, 0, 255) if gender == "Female" else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, gender, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Gender Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
