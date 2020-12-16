import insightface
import cv2, imutils
import queue, threading
import numpy as np
import time, os, json
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
from io import BytesIO
import telegram

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        #get supported video formats -> "v4l2-ctl --list-formats-ext"
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()

def resize_max(img, max_h, max_w):
    #resize image to improve interference speed
    (h, w) = img.shape[:2]
    if h > max_h:
        img = imutils.resize(img, height=max_h)
    (h, w) = img.shape[:2]
    if w > max_w:
        img = imutils.resize(img, width=max_w)
    return img

def sendtelegramfoto(face_frame, bot, chatid):
    ret, face_frame_jpg = cv2.imencode('.jpg', face_frame)
    bio = BytesIO(face_frame_jpg)
    bio.name = 'image.jpg'
    bio.seek(0)
    bot.send_photo(chatid, photo=bio)

#load models... face detection is done with a faster model
#no age and gender model req.
model = insightface.app.FaceAnalysis(det_name="retinaface_mnet025_v2", ga_name=None)

#-1 setup with cpu only - 0 = gpu 0 is used
ctx_id = -1
model.prepare(ctx_id = ctx_id, nms=0.4)

face_db = {}

start_time = time.time()

#home folder Pictures/FaceDB
path = os.path.expanduser("~") + "/Pictures/FaceDB"
for folder in os.listdir(path):
    folderjoin = os.path.join(path, folder)
    if os.path.isdir(folderjoin):
        face_db[folder] = {"embeddings":[]}

        #check if we have a json file with embeddings and we could skip the picture scan process
        #to rescan the directory, simple delete this file
        if 'embeddings.json' in os.listdir(folderjoin):
            with open(os.path.join(path, folder, 'embeddings.json')) as json_data_file:
                try:
                    data = json.load(json_data_file)
                    face_db[folder]["embeddings"] = data["embeddings"]
                except Exception as e:
                    pass
                else:
                    continue

        for filename in os.listdir(folderjoin):
            filenamejoin = os.path.join(path, folder, filename)

            ext = os.path.splitext(filenamejoin)[-1].lower()
            if ext == '.jpg' or ext == '.jpeg':
                img = resize_max(cv2.imread(filenamejoin), 480, 640)

                face_data = model.get(img)
                if len(face_data) != 1:
                    print(filenamejoin + " less or more then one face found in picture")
                    exit()
                face_db[folder]["embeddings"].append(face_data[0].embedding.tolist())

        #save embeddings to skip this process next time
        with open(os.path.join(path, folder, "embeddings.json"), 'w') as outfile:
            json.dump(face_db[folder], outfile)

print("Time for scanning face database: %d s" % (time.time() - start_time))

tele_data = {}
with open('telegram.json') as json_data_file:
    try:
        tele_data = json.load(json_data_file)
    except Exception as e:
        tele_data = {}
        pass

bot = telegram.Bot(token=tele_data["key"])

if tele_data["enabled"]:
    bot.send_message(chat_id=tele_data["chatid"], text='Start face recognition cam...')

cap = VideoCapture(0)


send_picture = {
    "fixtimeout_before": 15.0,
    "fixtimeout_after": 15.0,
    "timeout":time.time(),
    "state":"DISCARD",
    "det_score":0.0,
    "image":None
}


while(True):
    # Capture frame-by-frame
    frame = resize_max(cap.read(), 480, 640)

    start_time = time.time()
    faces = model.get(frame)
    #print("Interference time %d ms" % ((time.time() - start_time) * 1000))

    name_list = []
    #iter over faces in recorded frame
    for idx_face, face in enumerate(faces):

        if face.det_score < 0.85:
            continue

        #draw box around face
        box = face.bbox.astype(np.int).flatten()
        #widen the box to use the captured images as face database as well (if only one face is present)
        per_x = int((box[2] - box[0]) * 0.25)
        per_y = int((box[2] - box[0]) * 0.25)
        box = [box[0] - per_x, box[1] - per_y, box[2] + per_x, box[3] + per_y]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        frame = cv2.putText(frame, "{:.2f}".format(face.det_score), (box[0], box[3] + 22), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv2.LINE_AA)

        for name, value in face_db.items():
            #use cosinus similarity cause arc face is trained to maximize this distance between faces
            dist = np.max(pairwise.cosine_similarity(value["embeddings"], [face.embedding]))
            if(dist > 0.35):
                name_list.append(name)
                frame = cv2.putText(frame, name, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv2.LINE_AA)
                #face is known, reset alarm
                send_picture["timeout"] = time.time()
                send_picture["state"] = "DISCARD"

            else:
                if send_picture["state"] == "DETECT":
                    if send_picture["det_score"] <= 0.0:
                        send_picture["timeout"] = time.time()
                        print("Unknown face found")
                    if send_picture["det_score"] < face.det_score:
                        send_picture["det_score"] = face.det_score
                        send_picture["image"] = frame.copy()

    if send_picture["state"] == "DISCARD" and (time.time() - send_picture["timeout"]) > send_picture["fixtimeout_before"]:
        print("Detect state")
        send_picture["state"] = "DETECT"
        send_picture["det_score"] = 0.0
    if send_picture["det_score"] > 0.0 and send_picture["state"] == "DETECT" and (time.time() - send_picture["timeout"]) > send_picture["fixtimeout_after"]:
        send_picture["det_score"] = 0.0
        if tele_data["enabled"]:
            print("Send to telegram bot")
            sendtelegramfoto(send_picture["image"], bot, tele_data["chatid"])

    # Display the resulting frame if your ubuntu issn't headless only
    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
