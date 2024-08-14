from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
from datetime import timedelta
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def processVideoTrackYoloCSV(str videoName, str videoFolder, str outputFolder='', int save_interval=60):
    cdef str outputCSVFile = outputFolder + videoName + '-output.csv'
    cdef str cacheCSVFile = outputFolder + videoName + '-cached-output.csv'

    cdef str video_path = videoFolder + videoName
    cdef cv2.VideoCapture cap = cv2.VideoCapture(video_path)

    cdef float fps = cap.get(cv2.CAP_PROP_FPS)
    cdef int frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cdef int frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cdef int fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cdef str output_path = outputFolder + videoName + '_output.mp4'
    cdef cv2.VideoWriter out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    cdef dict track_history = defaultdict(lambda: [])
    cdef int frame_number = 0
    cdef list all_detections = []

    while cap.isOpened():
        cdef bool success
        cdef np.ndarray frame
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
            frame_number += 1
            if results[0].boxes.id is not None:
                cdef np.ndarray boxes = results[0].boxes.xywh.cpu()
                cdef list track_ids = results[0].boxes.id.int().cpu().tolist()

                annotated_frame = results[0].plot()

                for box, track_id in zip(boxes, track_ids):
                    cdef float x, y, w, h
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    cdef np.ndarray points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 170, 29), thickness=15)

                finished_tracks = track_history.keys() - track_ids
                for ft_id in finished_tracks:
                    ft = track_history.pop(ft_id)

                cv2.imshow("YOLOv8 Tracking Vehicles", annotated_frame)

                out.write(annotated_frame)

                detections = results[0].summary(normalize=False, decimals=5)
                millis = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = str(timedelta(milliseconds=millis))

                for detection in detections:
                    detection['frame'] = frame_number
                    detection['timestamp'] = timestamp
                    detection['x1'] = detection['box']['x1']
                    detection['y1'] = detection['box']['y1']
                    detection['x2'] = detection['box']['x2']
                    detection['y2'] = detection['box']['y2']
                    del detection['box']
                    all_detections.append(detection)

            if frame_number % save_interval == 0:
                df = pd.DataFrame(all_detections)
                df.to_csv(cacheCSVFile, index=False)
                print(f"Cached results saved as {cacheCSVFile} at frame {frame_number}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    df = pd.DataFrame(all_detections)
    df.to_csv(outputCSVFile, index=False)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved as {output_path}")
