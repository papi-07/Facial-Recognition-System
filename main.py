import sys, os, cv2, time, subprocess
import numpy as np
import threading
from PySide6 import QtWidgets, QtCore, QtGui
from utils import mtcnn, resnet, image_to_rgb, load_known_embeddings, save_known_embeddings, build_embeddings_from_known, find_best_match, mark_attendance, send_email_alert
from datetime import datetime
import torch


GMAIL_SENDER = "raghavsanoria007@gmail.com"
GMAIL_APP_PASSWORD = "mncrymesobjuojdr"  
ALERT_RECIPIENT = "raghavsanoria007@gmail.com"  
MATCH_THRESHOLD = 1.0

class VideoThread(QtCore.QThread):
    change_pixmap = QtCore.Signal(np.ndarray)
    info_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.siren_process = None 
        self.known_embeddings, self.known_names = load_known_embeddings()
        if not self.known_embeddings:
            self.known_embeddings, self.known_names = build_embeddings_from_known()
        self.known_embeddings = np.array(self.known_embeddings) if self.known_embeddings else []
        self.device = torch.device("cpu")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.info_signal.emit("Cannot open camera. Close other apps or try replugging.")
            return
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = image_to_rgb(frame) 
            boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)  
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    if landmarks is not None:
                        lm = landmarks[i] 
                        for (x,y) in lm:
                            cv2.circle(frame, (int(x),int(y)), 2, (0,255,255), -1)
                        le = tuple(lm[0].astype(int))
                        re = tuple(lm[1].astype(int))
                        nose = tuple(lm[2].astype(int))
                        lm_left = tuple(lm[3].astype(int))
                        lm_right = tuple(lm[4].astype(int))
                        cv2.line(frame, le, nose, (255,0,0), 1)
                        cv2.line(frame, re, nose, (255,0,0), 1)
                        cv2.line(frame, nose, lm_left, (255,0,0), 1)
                        cv2.line(frame, nose, lm_right, (255,0,0), 1)
                    face_tensor = mtcnn.extract(rgb, [box], None)
                    if face_tensor is None or len(face_tensor) == 0:
                        continue
                    with torch.no_grad():
                        emb = resnet(face_tensor[0].unsqueeze(0)).numpy()[0]
                    name, dist = find_best_match(emb, list(self.known_embeddings) if len(self.known_embeddings) > 0 else [], list(self.known_names), threshold=MATCH_THRESHOLD)
                    label = f"{name} ({dist:.2f})" if dist is not None else name
                    cv2.putText(frame, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    if name != "Unknown":
                        newly_marked = mark_attendance(name)
                        if newly_marked:
                            self.info_signal.emit(f"Marked attendance for {name} at {datetime.now().strftime('%H:%M:%S')}")
                    else:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outname = f"unknown_{ts}.jpg"
                        cv2.imwrite(outname, frame)
                        try:
                            if self.siren_process and self.siren_process.poll() is None:
                                self.siren_process.terminate()
                            self.siren_process = subprocess.Popen(["afplay", "siren.mp3"])
                        except Exception as e:
                            print("afplay error", e)
                        subject = f"Unknown person detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        body = f"An unknown person was detected. See attached image {outname}."
                        threading.Thread(target=send_email_alert, args=(GMAIL_SENDER, GMAIL_APP_PASSWORD, ALERT_RECIPIENT, subject, body, outname)).start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.siren_process and self.siren_process.poll() is None:
                    self.siren_process.terminate()
                    self.info_signal.emit("Siren stopped manually")

            self.change_pixmap.emit(frame)
            time.sleep(0.02)
        cap.release()

    def stop(self):
        self._run_flag = False
        if self.siren_process and self.siren_process.poll() is None:
            self.siren_process.terminate()
        self.wait()

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Face Recognition System")
        self.resize(900,700)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(800,600)
        self.layout.addWidget(self.image_label)
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.register_btn = QtWidgets.QPushButton("Register New Person")
        self.reload_btn = QtWidgets.QPushButton("Reload Known Embeddings")
        self.status = QtWidgets.QLabel("Status: Ready")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.register_btn)
        btn_layout.addWidget(self.reload_btn)
        btn_layout.addWidget(self.status)
        self.layout.addLayout(btn_layout)
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.register_btn.clicked.connect(self.open_register)
        self.reload_btn.clicked.connect(self.reload_embeddings)
        self.thread = None

        QtCore.QTimer.singleShot(500, self.start_camera)

    def start_camera(self):
        if self.thread and self.thread.isRunning():
            return
        self.thread = VideoThread()
        self.thread.change_pixmap.connect(self.update_image)
        self.thread.info_signal.connect(self.set_status)
        self.thread.start()
        self.set_status("Camera started")

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.set_status("Camera stopped")

    def open_register(self):
        import subprocess
        subprocess.Popen([sys.executable, "register.py"])
        self.set_status("Opened register window")

    def reload_embeddings(self):
        self.set_status("Rebuilding embeddings ...")
        from utils import build_embeddings_from_known
        build_embeddings_from_known()
        self.set_status("Embeddings rebuilt. Restart camera to load changes.")

    def set_status(self, text):
        self.status.setText(f"Status: {text}")

    @QtCore.Slot(np.ndarray)
    def update_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pix)

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
