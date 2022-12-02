from flask import Flask, render_template, Response, request, redirect
import cv2
import numpy as np
import argparse
import time
import os
import beepy

app = Flask(__name__)

logined = False

camera = cv2.VideoCapture(1)


def gen_frames():
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", type=str, default='yolo',
        help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    labelsPath = os.path.sep.join([args["yolo"], "custom.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

    weightsPath = os.path.sep.join([args["yolo"], "yolov3_custom_last.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3-custom.cfg"])

    cap = cv2.VideoCapture(1)

    print("[INFO] loading YOLO from disk")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    while True :
        # Capture frame-by-frame
        ret, frame = camera.read()  # read the camera frame
        frame = cv2.resize(frame, dsize=(400,400), interpolation=cv2.INTER_CUBIC)
        ret = cv2.imencode('.jpg', frame)[1].tobytes()
    
        (H, W) = frame.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

            if len(idxs) > 0:
                for i in idxs.flatten():
                    try:
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                                print(text)
                                if(LABELS[classIDs[i]] == 'no-mask'):
                                    beepy.beep(sound=1)
                                    print('Beep')
                                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + ret + b'\r\n')
                                    
                                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                    except IndexError:
                        pass


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/login', methods=['POST','GET'])
def login():
    global logined
    if request.method == 'POST':
        uname = request.form['username']
        passwd = request.form['password']
        if uname == 'kunal' and passwd == 'password':
            logined = True 
    return redirect('/')


@app.route('/')
def index():
    if(logined == False):
        return render_template('login.html')
    else:
        return render_template('stream.html')


if __name__ == '__main__':
    app.run(debug=True)