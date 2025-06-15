from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pygame
import os
import struct
from collections import deque
import threading
import time

class DrowsinessDetector:
    def __init__(self, 
                 eye_ar_thresh=0.19,
                 eye_ar_consec_frames=25,
                 mouth_ar_thresh=0.6,
                 mouth_ar_consec_frames=15,
                 head_movement_thresh=8,
                 head_movement_frames=10,
                 drowsiness_threshold=70,
                 max_drowsiness_score=100,
                 model_path="models/shape_predictor_68_face_landmarks.dat"):
        
        # Parameter deteksi kantuk
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.mouth_ar_thresh = mouth_ar_thresh
        self.mouth_ar_consec_frames = mouth_ar_consec_frames
        self.head_movement_thresh = head_movement_thresh
        self.head_movement_frames = head_movement_frames
        self.drowsiness_threshold = drowsiness_threshold
        self.max_drowsiness_score = max_drowsiness_score
        
        # Inisialisasi dlib detector dan predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        
        # Landmark untuk mata dan mulut
        self.left_eye_start, self.left_eye_end = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        self.right_eye_start, self.right_eye_end = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.mouth_start, self.mouth_end = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
        
        # Variabel untuk melacak status
        self.reset_counters()
        
        # Untuk menghaluskan perhitungan
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        self.head_movement_history = deque(maxlen=5)
        
        # Inisialisasi pygame untuk alarm suara
        self.init_audio()
        
        # Status alarm
        self.alarm_thread = None
        self.is_alarm_playing = False
    
    def reset_counters(self):
        """Reset semua counter dan variabel status"""
        self.eye_counter = 0
        self.mouth_counter = 0
        self.head_movement_counter = 0
        self.alarm_on = False
        self.prev_shape = None
        self.drowsiness_score = 0
    
    def init_audio(self):
        """Inisialisasi pygame mixer untuk alarm"""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        except Exception as e:
            print(f"Warning: Tidak dapat menginisialisasi audio: {e}")
    
    def eye_aspect_ratio(self, eye):
        """Menghitung Eye Aspect Ratio (EAR)"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Menghitung Mouth Aspect Ratio (MAR) untuk deteksi menguap"""
        A = distance.euclidean(mouth[2], mouth[10])
        B = distance.euclidean(mouth[4], mouth[8])
        C = distance.euclidean(mouth[0], mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar
    
    def head_movement(self, shape, prev_shape):
        """Mengukur pergerakan kepala berdasarkan perubahan posisi landmark wajah"""
        if prev_shape is None:
            return 0
        
        nose_tip = shape[30]
        prev_nose_tip = prev_shape[30]
        movement = distance.euclidean(nose_tip, prev_nose_tip)
        return movement
    
    def _play_simple_beep(self):
        """Memainkan beep sederhana menggunakan pygame"""
        try:
            duration = 2.0
            sample_rate = 44100
            bits = 16
            frequency = 440
            period = int(sample_rate / frequency)
            amplitude = 2**(bits-1) - 1
            
            buffer = bytearray()
            for i in range(0, int(sample_rate * duration)):
                value = amplitude * np.sin(2.0 * np.pi * float(i) / float(period))
                buffer.extend(struct.pack('h', int(value)))
            
            sound = pygame.mixer.Sound(buffer=buffer)
            sound.play()
            return True
        except Exception as e:
            print(f"Error saat memainkan beep: {e}")
            return False
    
    def _sound_alarm_thread(self):
        """Thread untuk memainkan alarm tanpa blocking"""
        self.is_alarm_playing = True
        
        # Reset dan inisialisasi ulang mixer audio
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        except:
            pass
        
        if os.path.exists("alarm.wav"):
            try:
                alarm_sound = pygame.mixer.Sound("alarm.wav")
                alarm_sound.play()
            except Exception as e:
                print(f"Error memainkan alarm.wav: {e}")
                self._play_simple_beep()
        else:
            self._play_simple_beep()
        
        time.sleep(3)
        self.is_alarm_playing = False
    
    def sound_alarm(self):
        """Membunyikan alarm dalam thread terpisah"""
        if not self.is_alarm_playing:
            self.alarm_thread = threading.Thread(target=self._sound_alarm_thread)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
    
    def detect_drowsiness(self, frame):
        """
        Fungsi utama untuk mendeteksi kantuk dari frame
        
        Args:
            frame: Frame gambar dari kamera (numpy array)
            
        Returns:
            dict: {
                'frame': frame dengan annotation,
                'drowsiness_score': skor kantuk (0-100),
                'is_drowsy': boolean apakah mengantuk,
                'status': string status deteksi,
                'metrics': dict dengan nilai EAR, MAR, head_movement
            }
        """
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = self.detector(gray, 0)
        
        current_ear = 0
        current_mar = 0
        current_head_movement = 0
        status = "Normal"
        
        # Jika tidak ada wajah terdeteksi
        if len(faces) == 0:
            self.drowsiness_score = max(0, self.drowsiness_score - 1)
            cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            status = "No Face Detected"
        
        # Proses setiap wajah yang terdeteksi
        for face in faces:
            # Ekstrak landmark wajah
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Ekstrak daerah mata dan mulut
            left_eye = shape[self.left_eye_start:self.left_eye_end]
            right_eye = shape[self.right_eye_start:self.right_eye_end]
            mouth = shape[self.mouth_start:self.mouth_end]
            
            # Hitung metrik
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(ear)
            current_ear = sum(self.ear_history) / len(self.ear_history)
            
            mar = self.mouth_aspect_ratio(mouth)
            self.mar_history.append(mar)
            current_mar = sum(self.mar_history) / len(self.mar_history)
            
            movement = self.head_movement(shape, self.prev_shape)
            self.head_movement_history.append(movement)
            current_head_movement = sum(self.head_movement_history) / len(self.head_movement_history)
            
            self.prev_shape = shape.copy()
            
            # Visualisasi landmark
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            mouth_hull = cv2.convexHull(mouth)
            
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
            
            # Analisis mata tertutup
            if current_ear < self.eye_ar_thresh:
                self.eye_counter += 1
                self.drowsiness_score = min(self.max_drowsiness_score, self.drowsiness_score + 2)
                cv2.putText(frame, "MATA SAYU/TERTUTUP!", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                status = "Eyes Closed"
            else:
                self.eye_counter = max(0, self.eye_counter - 1)
                self.drowsiness_score = max(0, self.drowsiness_score - 1)
            
            # Analisis menguap
            if current_mar > self.mouth_ar_thresh:
                self.mouth_counter += 1
                if self.mouth_counter >= self.mouth_ar_consec_frames:
                    self.drowsiness_score = min(self.max_drowsiness_score, self.drowsiness_score + 15)
                    cv2.putText(frame, "MENGUAP TERDETEKSI!", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status = "Yawning"
            else:
                self.mouth_counter = 0
            
            # Analisis gerakan kepala
            if current_head_movement > self.head_movement_thresh:
                self.head_movement_counter += 1
                if self.head_movement_counter >= self.head_movement_frames:
                    self.drowsiness_score = min(self.max_drowsiness_score, self.drowsiness_score + 10)
                    cv2.putText(frame, "GERAKAN KEPALA BERLEBIHAN!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status = "Head Movement"
            else:
                self.head_movement_counter = max(0, self.head_movement_counter - 1)
            
            # Tampilkan nilai metrik
            cv2.putText(frame, f"EAR: {current_ear:.2f}/{self.eye_ar_thresh:.2f}", (450, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {current_mar:.2f}/{self.mouth_ar_thresh:.2f}", (450, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Head: {current_head_movement:.2f}/{self.head_movement_thresh:.2f}", (450, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Tampilkan skor kantuk
        cv2.putText(frame, f"Drowsiness Score: {self.drowsiness_score}/{self.drowsiness_threshold}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Progress bar untuk skor kantuk
        bar_length = int((self.drowsiness_score / self.max_drowsiness_score) * 400)
        cv2.rectangle(frame, (10, 350), (10 + bar_length, 370), (0, 0, 255), -1)
        cv2.rectangle(frame, (10, 350), (410, 370), (255, 255, 255), 2)
        
        # Cek status mengantuk dan alarm
        is_drowsy = self.drowsiness_score >= self.drowsiness_threshold
        
        if is_drowsy:
            if not self.alarm_on:
                self.alarm_on = True
                self.sound_alarm()
            
            cv2.putText(frame, "AWAS! ANDA MENGANTUK", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "SEGERA MENEPI UNTUK BERISTIRAHAT", (10, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
            status = "DROWSY - ALERT!"
        else:
            self.alarm_on = False
        
        # Return hasil deteksi
        return {
            'frame': frame,
            'drowsiness_score': self.drowsiness_score,
            'is_drowsy': is_drowsy,
            'status': status,
            'metrics': {
                'ear': current_ear,
                'mar': current_mar,
                'head_movement': current_head_movement,
                'eye_counter': self.eye_counter,
                'mouth_counter': self.mouth_counter,
                'head_movement_counter': self.head_movement_counter
            }
        }
    
    def get_status(self):
        """Mendapatkan status saat ini tanpa memproses frame"""
        return {
            'drowsiness_score': self.drowsiness_score,
            'is_drowsy': self.drowsiness_score >= self.drowsiness_threshold,
            'alarm_on': self.alarm_on,
            'counters': {
                'eye_counter': self.eye_counter,
                'mouth_counter': self.mouth_counter,
                'head_movement_counter': self.head_movement_counter
            }
        }
    
    def set_thresholds(self, **kwargs):
        """Mengatur threshold secara dinamis"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Set {key} = {value}")
    
    def cleanup(self):
        """Membersihkan resources"""
        try:
            pygame.mixer.quit()
        except:
            pass


if __name__ == "__main__":
    # Inisialisasi detector
    detector = DrowsinessDetector()
    
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    
    print("[INFO] Starting drowsiness detection...")
    print("Tekan 'q' untuk keluar, 'r' untuk reset counters")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Tidak dapat membaca frame dari kamera.")
            break
        
        # Deteksi kantuk
        result = detector.detect_drowsiness(frame)
        
        # Tampilkan frame
        cv2.imshow("Drowsiness Detection", result['frame'])
        
        # Print status (optional)
        if result['is_drowsy']:
            print(f"DROWSY DETECTED! Score: {result['drowsiness_score']}")
        
        # Kontrol keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset_counters()
            print("Counters reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
    print("[INFO] Program selesai.")