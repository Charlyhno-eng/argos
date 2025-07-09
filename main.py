import cv2
import onnxruntime as onnxr
import numpy as np

IMAGE_SIZE = 416
ONNX_PATH = "human_detection.onnx"
PERSON_CLASS_ID = 0

session = onnxr.InferenceSession(ONNX_PATH)

input_name = session.get_inputs()[0].name
output_names = [out.name for out in session.get_outputs()]

def compute_iou(box1, box2):
    box1_w, box1_h = box1[2] / 2.0, box1[3] / 2.0
    box2_w, box2_h = box2[2] / 2.0, box2[3] / 2.0

    b1_1, b1_2 = box1[0] - box1_w, box1[1] - box1_h
    b1_3, b1_4 = box1[0] + box1_w, box1[1] + box1_h
    b2_1, b2_2 = box2[0] - box2_w, box2[1] - box2_h
    b2_3, b2_4 = box2[0] + box2_w, box2[1] + box2_h

    x1, y1 = max(b1_1, b2_1), max(b1_2, b2_2)
    x2, y2 = min(b1_3, b2_3), min(b1_4, b2_4)

    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1_3 - b1_1) * (b1_4 - b1_2)
    area2 = (b2_3 - b2_1) * (b2_4 - b2_2)
    union = area1 + area2 - intersect

    return intersect / union if union > 0 else 0

def run_onnx_inference(frame):
    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return session.run(output_names, {input_name: input_tensor})

def postprocess_detections(outputs, conf_thres, iou_thres):
    detections = []
    for detection in outputs[0]:
        boxes = detection[:4, :]
        scores = detection[4:, :]
        class_ids = np.argmax(scores, axis=0)
        confs = scores[class_ids, np.arange(scores.shape[1])]
        valid = (confs >= conf_thres)
        indices = np.where(valid)[0]

        flags = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            if flags[i]:
                continue
            box = boxes[:, idx]
            score = confs[idx]

            for j, idx2 in enumerate(indices):
                if idx2 < idx:
                    continue
                if compute_iou(box, boxes[:, idx2]) >= iou_thres:
                    flags[j] = True

            detections.append({"bbox": box, "confidence": score})
            flags[i] = True
    return detections

def extract_box(frame, detection, x_scale, y_scale):
    x, y, w, h = detection["bbox"]
    x1, y1 = int((x - w / 2) * x_scale), int((y - h / 2) * y_scale)
    x2, y2 = int((x + w / 2) * x_scale), int((y + h / 2) * y_scale)
    if x2 - x1 < 60 or y2 - y1 < 20:
        return None, None
    return (x1, y1, x2, y2), frame[y1:y2, x1:x2]

def display_camera_with_detection():
    cap = cv2.VideoCapture(0)
    conf_thres, iou_thres = 0.4, 0.5
    x_scale, y_scale = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if x_scale is None or y_scale is None:
            h, w = frame.shape[:2]
            x_scale = w / IMAGE_SIZE
            y_scale = h / IMAGE_SIZE

        outputs = run_onnx_inference(frame)
        detections = postprocess_detections(outputs, conf_thres, iou_thres)

        for det in detections:
            key, crop = extract_box(frame, det, x_scale, y_scale)
            if key:
                x1, y1, x2, y2 = key
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Conf: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Camera with Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera_with_detection()
