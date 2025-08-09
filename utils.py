import os
import cv2
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path
import smtplib
from email.message import EmailMessage

EMBEDDINGS_FILE = "known_embeddings.pkl"
ATTENDANCE_FILE = "attendance.csv"

mtcnn = MTCNN(keep_all=True)  
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def ensure_dirs():
    Path("known_faces").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

def image_to_rgb(cv2_img):
    
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

def load_known_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            return data.get("embeddings", []), data.get("names", [])
    return [], []

def save_known_embeddings(embeddings, names):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": names}, f)

def build_embeddings_from_known():
    """
    Walk known_faces/<name>/ and build embeddings
    """
    embeddings = []
    names = []
    base = Path("known_faces")
    for person_dir in base.iterdir():
        if person_dir.is_dir():
            for img_path in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = image_to_rgb(img)
                
                boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)
                if boxes is None:
                    continue
                
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                crop = rgb[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                face_tensor = mtcnn.extract(rgb, [boxes[0]], None)  
                if face_tensor is None or len(face_tensor) == 0:
                    continue
                with torch.no_grad():
                    emb = resnet(face_tensor[0].unsqueeze(0)).numpy()[0]
                embeddings.append(emb)
                names.append(person_dir.name)
    save_known_embeddings(embeddings, names)
    return embeddings, names

import torch
from numpy.linalg import norm

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def find_best_match(face_emb, known_embeddings, known_names, threshold=0.85):
    """
    threshold: smaller -> more strict
    returns (name, distance) or ("Unknown", None)
    """
    if not known_embeddings:
        return "Unknown", None
    dists = [l2_distance(face_emb, emb) for emb in known_embeddings]
    idx = int(np.argmin(dists))
    best = dists[idx]
    if best < threshold:
        return known_names[idx], float(best)
    return "Unknown", float(best)

def mark_attendance(name):
    from datetime import datetime
    today = datetime.now().date().isoformat()  
    time_str = datetime.now().strftime("%H:%M:%S")
    
    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    
    df = pd.read_csv(ATTENDANCE_FILE)
    
    if ((df["Name"] == name) & (df["Date"] == today)).any():
        return False  
    
    df = pd.concat([df, pd.DataFrame([{"Name": name, "Date": today, "Time": time_str}])], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    return True


def send_email_alert(sender_email, app_password, recipient_email, subject, body, image_path=None):
    """
    sender_email: your Gmail address
    app_password: 16-char app password (recommended) OR your SMTP password
    recipient_email: where to send alert
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content(body)
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_data = f.read()
            import imghdr
            img_type = imghdr.what(None, img_data)
            msg.add_attachment(img_data, maintype="image", subtype=img_type, filename=os.path.basename(image_path))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)
