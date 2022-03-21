# USAGE
# python ChosenV1.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import time
import cv2

# Отвечает за удаление накладывающихся друг на друга детекций. В данном случае если прямоугольники
# имееют 20ти % пересечение, один из них не будет выводиться
threshold = 0.2

# Выводим только результаты распознавания вероятность верности которых >20%
confidenceIdx = 0.2

labelsPath = "yolo_trained_6/obj.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(57)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = "yolo_trained_6/yolov4-obj_best.weights"
configPath = "yolo_trained_6/yolov4-obj.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet(configPath, weightsPath)

cap = cv2.VideoCapture(0)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

while True:
	_, image = cap.read()
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
			if confidence > confidenceIdx:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceIdx, threshold)

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	cv2.imshow("img", image)
	if ord("q") == cv2.waitKey(1):
		break

cap.release()
cv2.destroyAllWindows()