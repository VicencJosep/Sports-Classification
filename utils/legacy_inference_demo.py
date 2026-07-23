from ultralytics import YOLO
import supervision as sv
import cv2


model = YOLO("yolov8n.pt")

# Abrir vídeo
cap = cv2.VideoCapture("corner.mp4")

# Información del vídeo
video_info = sv.VideoInfo.from_video_path("corner.mp4")

# Guardar resultado
sink = sv.VideoSink("output.mp4", video_info)

# Dibujar cajas
box_annotator = sv.BoxAnnotator()

with sink:
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Inferencia
        result = model(frame)[0]

        # Convertir formato Ultralytics -> Supervision
        detections = sv.Detections.from_ultralytics(result)

        # Dibujar bounding boxes
        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        sink.write_frame(annotated)

cap.release()

print("Vídeo generado correctamente.")