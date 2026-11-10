import numpy as np
import cv2
import time
import os
import pyttsx3
from gtts import gTTS
import tempfile
import pygame
import asyncio
from googletrans import Translator
from flask import Flask, Response, render_template
import threading

app = Flask(__name__)

pygame.mixer.init()
translator = Translator()

print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg")
labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

cap = cv2.VideoCapture(0)

detection_counts = {}
frame_threshold = 5

def speak_in_thread(text, lang="en"):
    try:
        if text.strip():
            tts = gTTS(text=text, lang=lang, slow=False)
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmpfile.name)
            tmpfile.close()
            pygame.mixer.music.load(tmpfile.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.remove(tmpfile.name)
    except Exception as e:
        print(f"Error while generating speech: {e}")

async def translate_label(label):
    try:
        translated_tamil = await translator.translate(label, src='en', dest='ta')
        translated_hindi = await translator.translate(label, src='en', dest='hi')
        return translated_tamil.text, translated_hindi.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "", ""

def handle_translation(label):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(translate_label(label))
    return result if result else ("", "")

def generate_frames():
    global detection_counts
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes, confidences, classIDs = [], [], []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(idxs) > 0:
            for i in idxs.flatten():
                classID = classIDs[i]
                label = LABELS[classID]
                color = [int(c) for c in COLORS[classID]]
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}".format(label)
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detection_counts[label] = detection_counts.get(label, 0) + 1
                if detection_counts[label] >= frame_threshold:
                    print(f"Detected: {label}")
                    speak_in_thread(label, lang="en")
                    tamil_translation, hindi_translation = handle_translation(label)
                    print(f"Tamil: {tamil_translation}, Hindi: {hindi_translation}")
                    speak_in_thread(tamil_translation, lang="ta")
                    speak_in_thread(hindi_translation, lang="hi")
                    detection_counts[label] = 0  # Reset count after speech
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)