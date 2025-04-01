import cv2 as cv
import numpy as np

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
thr = 0.8
width, height = 368, 368

def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    confidences = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        if conf > thr:
            points.append((int(x), int(y)))
            confidences.append(conf)
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.circle(frame, points[idFrom], 3, (0, 0, 255), -1)
            cv.circle(frame, points[idTo], 3, (0, 0, 255), -1)

    if confidences:
        accuracy = (sum(confidences) / len(confidences)) * 100
        accuracy_text = f"Pose Accuracy: {accuracy:.2f}%"
    else:
        accuracy = 0.0
        accuracy_text = "Pose Accuracy: 0.00%"

    cv.putText(frame, accuracy_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 105), 2)

    return frame, accuracy

image_path = "download.jpg"
input_img = cv.imread(image_path)

if input_img is None:
    print("Error: Image not found!")
else:
    output_img, accuracy = poseDetector(input_img)
    output_path = "D:/open_pose/pose_output.jpg"
    cv.imwrite(output_path, output_img)
    print(f"Estimated Pose Accuracy: {accuracy:.2f}%")
    print(f"Pose estimated image saved as: {output_path}")

    cv.namedWindow("Pose Detection", cv.WINDOW_NORMAL)
    cv.resizeWindow("Pose Detection", output_img.shape[1], output_img.shape[0])
    cv.imshow("Pose Detection", output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
