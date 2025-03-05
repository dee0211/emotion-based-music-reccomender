from flask import Flask, render_template, Response, request, redirect
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

app = Flask(__name__)

# Load model and labels
model = load_model("model.h5")
label = np.load("labels.npy", allow_pickle=True)

# Mediapipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize global variables
emotion = ""
lang = ""
singer = ""

# Function to process video frames
def process_frame():
    global emotion
    cap = cv2.VideoCapture(0)
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            emotion = pred
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        _, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recommend', methods=['POST'])
def recommend():
    global lang, singer, emotion
    lang = request.form.get("language")
    singer = request.form.get("singer")

    if not emotion:
        return "<h1>Please let the system capture your emotion first. Go back and try again.</h1>", 400

    query = f"{lang}+{emotion}+song+playlist+{singer}"
    webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
    emotion = ""
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
