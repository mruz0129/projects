from flask import Flask, render_template, Response
import threading
import face_recognition
import os, sys
import cv2
import numpy as np
import math
import mysql.connector
import time
import datetime
from datetime import timedelta

app = Flask(__name__)

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"
    
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    frame_frequency = 10
    last_verified_name = None
    is_verified = False

    def __init__(self):
        self.encode_faces()
        self.connect_db()
        self.recognized_frames = 0
        self.verify_start_time = None

    def encode_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        for person in os.listdir('faces'):
            if not os.path.isdir(f'faces/{person}'):
                continue
            for image in os.listdir(f'faces/{person}'):
                if not image.endswith(('.png', '.jpg', '.jpeg')):  # Add or remove file extensions as needed
                    continue
                face_image = face_recognition.load_image_file(f'faces/{person}/{image}')
                face_encodings = face_recognition.face_encodings(face_image, model="cnn")
                if face_encodings:
                    face_encoding = face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(person)

        print(self.known_face_names)


    def run_recognition(self):
        self.connect_db()
        
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Camera could not be opened...')

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.4)
                    name = 'Unknown'
                    confidence = 'Unknown'
                    display_name = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                        
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        
                        display_name = name

                        if name != 'Unknown':
                            self.recognized_frames += 1
                            query = ("SELECT full_name FROM member WHERE id = %s")
                            self.db_cursor.execute(query, (name,))
                            name_from_db = self.db_cursor.fetchone()
                            if name_from_db is not None:
                                display_name = name_from_db[0]
                                
                            # get enrolled classes and check if there're ongoing classes
                            ongoing_classes = []
                            query = ("SELECT class_id FROM enroll_class_member WHERE student_id = %s")
                            self.db_cursor.execute(query, (name,))
                            enrolled_classes = [row[0] for row in self.db_cursor.fetchall()]
                            
                            current_time = datetime.datetime.now()
                            current_date = current_time.date()
                            current_day_of_week = current_date.strftime('%A')
                            current_time = current_time.time()
        
                            adjusted_start_time = (datetime.datetime.combine(datetime.date(1, 1, 1), current_time) + timedelta(minutes=30)).time()

                            self.db_cursor.execute("""
                                SELECT class_id FROM class
                                WHERE start_date <= %s AND end_date >= %s
                                AND start_time <= %s AND end_time >= %s
                                AND day_of_week = %s
                            """, (current_date, current_date, adjusted_start_time, current_time, current_day_of_week))
        
                            ongoing_classes = [row[0] for row in self.db_cursor.fetchall()]

                            if self.recognized_frames >= self.frame_frequency:
                                if self.verify_start_time is None:
                                    self.verify_start_time = time.time()

                                    if ongoing_classes:
                                        for enrolled_class in enrolled_classes:
                                            if enrolled_class in ongoing_classes:
                                                query = ("INSERT INTO attendances (student_id, class_id) VALUES (%s, %s)")
                                                self.db_cursor.execute(query, (name, enrolled_class))
                                                self.db_connection.commit()
                                                display_name = f'{display_name} (Verified)'
                                                self.is_verified = True
                                            
                                elif time.time() - self.verify_start_time >= 10:
                                    self.recognized_frames = 0
                                    self.verify_start_time = None
                                    self.last_verified_name = None
                                    self.is_verified = False
                                    
                    elif self.last_verified_name == display_name:
                        display_name = f'{display_name}'

                    self.face_names.append(f'{display_name} {"(Verified)" if self.is_verified and display_name != 'Unknown' else ""}')
            
            self.process_current_frame = not self.process_current_frame

            # Display Annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # If the face is recognized, draw a green rectangle. Otherwise, draw a red rectangle.
                if name.split(' ')[0] != 'Unknown':
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), -1)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue

            # Yield the frame in the streaming format
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    def connect_db(self):
        self.db_connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='horng0129',
            database='tuitioncenter'
        )
        self.db_cursor = self.db_connection.cursor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    fr = FaceRecognition()
    return Response(fr.run_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)