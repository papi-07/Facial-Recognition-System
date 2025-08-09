import sys, os, cv2
from PySide6 import QtWidgets, QtCore, QtGui
from pathlib import Path
import numpy as np

class RegisterWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Register New Person - Face Capture")
        self.resize(800, 600)
        self.layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        form.addRow("Person Name:", self.name_edit)
        self.capture_btn = QtWidgets.QPushButton("Start Camera & Capture Images")
        self.capture_btn.clicked.connect(self.capture_images)
        self.info_label = QtWidgets.QLabel("Instructions: 8-20 images. Vary angles & lighting.")
        self.layout.addLayout(form)
        self.layout.addWidget(self.capture_btn)
        self.layout.addWidget(self.info_label)

    def capture_images(self):
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Name required", "Enter the person's name (no spaces preferred).")
            return
        person_dir = Path("known_faces") / name
        person_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Camera error", "Cannot open webcam. Quit other apps using camera.")
            return
        count = 0
        QtWidgets.QMessageBox.information(self, "Capture",
            "Press SPACE to capture a frame, ESC to finish capturing.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(640,480, QtCore.Qt.KeepAspectRatio)
            label = QtWidgets.QLabel()
            label.setPixmap(pix)

            preview = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(preview)
            v.addWidget(QtWidgets.QLabel(f"Capture for: {name} — saved {count}"))
            v.addWidget(QtWidgets.QLabel())
            img_label = QtWidgets.QLabel()
            img_label.setPixmap(pix)
            v.addWidget(img_label)
            preview.setWindowTitle("Live Capture - Press SPACE to capture, ESC to quit")
            preview.show()

            key = cv2.waitKey(1)

            cv2.imshow("Live - Press Space to capture, Esc to exit", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  
                break
            if k == 32:  
                count += 1
                fname = person_dir / f"img_{count:03d}.jpg"

                out = cv2.resize(frame, (640,480))
                cv2.imwrite(str(fname), out)
                print("Saved", fname)
        cap.release()
        cv2.destroyAllWindows()
        QtWidgets.QMessageBox.information(self, "Done", f"Saved {count} images to {person_dir}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ensure = Path("known_faces")
    ensure.mkdir(exist_ok=True)
    from pathlib import Path
    win = RegisterWindow()
    win.show()
    sys.exit(app.exec())
