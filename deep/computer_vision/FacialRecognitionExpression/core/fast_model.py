# -*- coding: utf-8 -*-"""@author: JulienWuthrich"""import osimport cv2import globimport dlibimport pygameimport numpy as npimport face_recognitionfrom threading import Threadfrom imutils import face_utilsfrom scipy.spatial import distance as distimport kerasfrom keras.models import Sequentialfrom keras.layers import Conv2D, MaxPooling2D, AveragePooling2Dfrom keras.layers import Dense, Dropout, Flattenfrom keras.preprocessing import imagepygame.mixer.init()class FacialRecognitionExpression(object):    """"""    def __init__(self,                 facial_expression_model="../../../../models/facial_expression_model_weights.h5",                 images_paths="../../../../data/raw/face_recognition/renault_digital/train/*",                 cascade="../../../../models/haarcascade_frontalface_default.xml",                 landmark="../../../../models/shape_predictor_68_face_landmarks.dat",                 alarm="../metal_gear.wav"):        """Init.        :param facial_expression_model: (str) path of the model weights        :param images_paths: (str) path of the pictures        :param cascade: (str) cascade path        :param alarm: (str) alarm song        """        self.num_classes = 7        self.classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]        self.alarm = alarm        self.facial_expression_model = facial_expression_model        self.face_cascade = cv2.CascadeClassifier(cascade)        self.images_paths = images_paths        #self.model_exp = self.model_expression()        self.known_face_encodings, self.known_face_names = self.load_and_recognize_pictures()        self.eyes_closed_tresh = 0.3        self.frames_closed_tresh = 10        self.detector = dlib.get_frontal_face_detector()        self.predictor = dlib.shape_predictor(landmark)        self.left_s, self.left_e = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]        self.right_s, self.right_e = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]        print(self.known_face_names)    def model_expression(self):        """Load model for expression recognition        :return: (keras.models.Sequential) modelBi-Monthly meetings - PoC Renault DigitalDate : jeudi 15 novembre 2018 à 16:00 - 17:00.Lieu : Réunion en ligne ou Présentiel pour ceux qui le peuvent!        """        model = Sequential()        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))        model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))        model.add(Conv2D(64, (3, 3), activation='relu'))        model.add(Conv2D(64, (3, 3), activation='relu'))        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))        model.add(Conv2D(128, (3, 3), activation='relu'))        model.add(Conv2D(128, (3, 3), activation='relu'))        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))        model.add(Flatten())        model.add(Dense(1024, activation='relu'))        model.add(Dropout(0.2))        model.add(Dense(1024, activation='relu'))        model.add(Dropout(0.2))        model.add(Dense(self.num_classes, activation='softmax'))        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])        model.load_weights(self.facial_expression_model)        return model    def sound_alarm(self):        """        :return:        """        pygame.mixer.music.load(self.alarm)        pygame.mixer.music.play()    @staticmethod    def eye_aspect_ratio(eye):        """        :param eye:        :return:        """        A = dist.euclidean(eye[1], eye[5])        B = dist.euclidean(eye[2], eye[4])        C = dist.euclidean(eye[0], eye[3])        return (A + B) / (2.0 * C)    def load_and_recognize_pictures(self):        """        :param path: (str) path of the pictures        :return:        """        known_face_encodings = list()        known_face_names = list()        images_paths = glob.glob(self.images_paths)        for image_path in images_paths:            name = image_path.split(os.path.sep)[-1].split(".")[0]            vars()[name] = face_recognition.load_image_file(image_path)            vars()[name + "_encoding"] = face_recognition.face_encodings(vars()[name])[0]            known_face_encodings.append(vars()[name + "_encoding"])            known_face_names.append(name)        return known_face_encodings, known_face_names    @staticmethod    def visualize_eyes(left, right, frame, show=False):        """        :param left:        :param right:        :param frame:        :return:        """        if show:            left_eye_hull = cv2.convexHull(left)            right_eye_hull = cv2.convexHull(right)            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)    def detect_emotion(self, x, y, w, h, img):        """        :param x:        :param y:        :param w:        :param h:        :param img:        :return:        """        detected_face = img[int(y):int(y+h), int(x):int(x+w)]        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)        detected_face = cv2.resize(detected_face, (48, 48))        img_pixels = image.img_to_array(detected_face)        img_pixels = np.expand_dims(img_pixels, axis=0)        img_pixels /= 255        predictions = self.model_exp.predict(img_pixels)        max_index = np.argmax(predictions[0])        emotion = self.classes[max_index]        self.show_emotion(img, emotion, x, y)        return emotion    def show_emotion(self, frame, emotion, x, y, show=True):        """        :param frame:        :param emotion:        :param x:        :param y:        :return:        """        if show:            cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)    def get_eyes(self, face, gray):        """        :param face:        :param gray:        :return:        """        shape = self.predictor(gray, face)        shape = face_utils.shape_to_np(shape)        return shape[self.left_s:self.left_e], shape[self.right_s:self.right_e]    def recognize_faces(self, rgb_small_frame):        """        :param rgb_small_frame:        :return:        """        face_names = []        face_locations = face_recognition.face_locations(rgb_small_frame)        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)        for face_encoding in face_encodings:            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)            name = "Unknown"            if True in matches:                first_match_index = matches.index(True)                name = self.known_face_names[first_match_index]            face_names.append(name)        return face_locations, face_names    @staticmethod    def show_face(top, left, right, bottom, frame, name):        """        :param top:        :param left:        :param right:        :param bottom:        :param frame:        :param name:        :return:        """        font = cv2.FONT_HERSHEY_DUPLEX        cv2.putText(frame, name, (left*4 + 6, bottom*4 - 6), font, 1.0, (255, 255, 0), 1)    def tiredness(self, face, gray, frame, frames_closed):        """        :param face:        :param gray:        :param frame:        :param frames_closed:        :return:        """        left_eyes, right_eyes = self.get_eyes(face, gray)        self.visualize_eyes(left_eyes, right_eyes, frame)        lratio = self.eye_aspect_ratio(left_eyes)        rratio = self.eye_aspect_ratio(right_eyes)        alarm = 0        if lratio < self.eyes_closed_tresh and rratio < self.eyes_closed_tresh:            frames_closed += 1            if frames_closed >= self.frames_closed_tresh:                alarm = 1                Thread(target=self.sound_alarm).start()                cv2.putText(frame, "ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)        else:            frames_closed = 0        self.show_tiredness(frame, lratio, rratio)        return frames_closed, alarm    def show_tiredness(self, frame, lratio, rratio, show=True):        """        :param frame:        :param lratio:        :param rratio:        :return:        """        if show:            cv2.putText(frame, """Opened left_eyes: {:.2f} \n            Opened right_eyes: {:.2f}""".format(lratio, rratio), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    def main(self):        """"""        name = "unknow"        emotion = "unknow"        alarm = 0        frames_closed = 0        video_capture = cv2.VideoCapture(1)        process_this_frame = True        while True:            ret, frame = video_capture.read()            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)            rgb_small_frame = small_frame[:, :, ::-1]            faces_eyes = self.detector(gray, 0)            # tiredness            for face in faces_eyes:                frames_closed, alarm = self.tiredness(face, gray, frame, frames_closed)            # emotions            #for (x, y, w, h) in faces:            #    emotion = self.detect_emotion(x, y, w, h, frame)            # face            if process_this_frame:                face_locations, face_names = self.recognize_faces(rgb_small_frame)            process_this_frame = not process_this_frame            for (top, right, bottom, left), name in zip(face_locations, face_names):                self.show_face(top, left, right, bottom, frame, name)            cv2.imshow('Video', frame)            # print(name, emotion, alarm)            if cv2.waitKey(1) & 0xFF == ord('q'):                break        video_capture.release()        cv2.destroyAllWindows()if __name__ == '__main__':    FacialRecognitionExpression().main()