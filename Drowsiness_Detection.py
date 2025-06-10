from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import time
import pygame
import os
import struct  # Tambahkan import struct untuk packing bytes
from collections import deque

# Inisialisasi pygame untuk alarm suara
pygame.mixer.init(frequency=44100, size=-16, channels=1)  # Gunakan mono untuk sederhana

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Jarak vertikal antara landmark mata
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Jarak horizontal antara landmark mata
    C = distance.euclidean(eye[0], eye[3])
    # Hitung EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Fungsi untuk menghitung Mouth Aspect Ratio (MAR) untuk deteksi menguap
def mouth_aspect_ratio(mouth):
    # Jarak vertikal antara landmark mulut
    A = distance.euclidean(mouth[2], mouth[10])  # Vertikal 1
    B = distance.euclidean(mouth[4], mouth[8])   # Vertikal 2
    C = distance.euclidean(mouth[0], mouth[6])   # Horizontal
    # Hitung MAR
    mar = (A + B) / (2.0 * C)
    return mar

# Fungsi untuk mengukur pergerakan kepala berdasarkan perubahan posisi landmark wajah
def head_movement(shape, prev_shape):
    if prev_shape is None:
        return 0
    
    # Gunakan landmark hidung sebagai referensi untuk pergerakan kepala
    nose_tip = shape[30]
    prev_nose_tip = prev_shape[30]
    
    # Hitung perpindahan hidung
    movement = distance.euclidean(nose_tip, prev_nose_tip)
    return movement

def _play_simple_beep():
    try:
        # Metode sederhana: Gunakan PyGame untuk membuat beep mono
        duration = 15.0  # durasi dalam detik
        sample_rate = 44100
        bits = 16
        
        # Buat array mono sederhana
        frequency = 440  # A4 note
        period = int(sample_rate / frequency)
        amplitude = 2**(bits-1) - 1
        
        # Buat gelombang sinus mono sederhana sebagai bytearray
        buffer = bytearray()
        for i in range(0, int(sample_rate * duration)):
            value = amplitude * np.sin(2.0 * np.pi * float(i) / float(period))
            # Packing value sebagai signed 16-bit
            buffer.extend(struct.pack('h', int(value)))
        
        # Buat objek Sound dari buffer
        sound = pygame.mixer.Sound(buffer=buffer)
        sound.play()
    except Exception as e:
        print(f"Error saat memainkan beep: {e}")
        # Jika PyGame gagal, coba gunakan winsound (Windows) jika tersedia
        try:
            import winsound
            winsound.Beep(440, 1000)  # 440Hz selama 1 detik (Windows only)
        except:
            print("Tidak dapat memainkan suara alarm dengan cara apapun.")

def sound_alarm():
    """
    Fungsi untuk membunyikan alarm suara.
    Menggunakan file alarm.wav jika tersedia, atau menghasilkan beep sederhana.
    """
    # Reset dan inisialisasi ulang mixer audio
    pygame.mixer.quit()
    pygame.mixer.init(frequency=44100, size=-16, channels=1)  # Gunakan mono untuk sederhana
    
    if os.path.exists("alarm.wav"):
        print("Memainkan alarm.wav...")
        try:
            alarm_sound = pygame.mixer.Sound("alarm.wav")
            alarm_sound.play()
        except Exception as e:
            print(f"Error memainkan alarm.wav: {e}")
            # Jika gagal memainkan file, gunakan beep sederhana
            _play_simple_beep()
    else:
        print("alarm.wav tidak ditemukan, memainkan nada beep...")
        _play_simple_beep()

# Parameter deteksi kantuk yang dapat disesuaikan
# Parameter ini dapat disesuaikan berdasarkan pengujian untuk mengakomodasi orang dengan mata sipit
EYE_AR_THRESH = 0.19      # Nilai lebih rendah untuk mengakomodasi mata sipit
EYE_AR_CONSEC_FRAMES = 25  # Jumlah frame dengan mata tertutup sebelum alarm
MOUTH_AR_THRESH = 0.6     # Threshold untuk menguap
MOUTH_AR_CONSEC_FRAMES = 15  # Jumlah frame dengan mulut terbuka untuk mendeteksi menguap
HEAD_MOVEMENT_THRESH = 8  # Threshold gerakan kepala
HEAD_MOVEMENT_FRAMES = 10  # Jumlah frame untuk memantau gerakan kepala

# Inisialisasi detector dan predictor dari dlib
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Definisikan landmark untuk mata dan mulut
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Inisialisasi kamera
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Variabel untuk melacak status
eye_counter = 0
mouth_counter = 0
head_movement_counter = 0
alarm_on = False
prev_shape = None

# Untuk menghaluskan perhitungan
ear_history = deque(maxlen=5)
mar_history = deque(maxlen=5)
head_movement_history = deque(maxlen=5)

# Skor kantuk total
drowsiness_score = 0
MAX_DROWSINESS_SCORE = 100
DROWSINESS_THRESHOLD = 70  # Nilai di mana alarm akan diaktifkan

# Loop utama
while True:
    # Ambil frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Tidak dapat membaca frame dari kamera.")
        break
        
    # Resize frame untuk mempercepat pemrosesan
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = detector(gray, 0)
    
    # Status untuk frame ini
    current_ear = 0
    current_mar = 0
    current_head_movement = 0
    
    # Jika tidak ada wajah terdeteksi, kurangi skor kantuk
    if len(faces) == 0:
        drowsiness_score = max(0, drowsiness_score - 1)
        cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Proses setiap wajah yang terdeteksi
    for face in faces:
        # Ekstrak landmark wajah
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Ekstrak daerah mata dan mulut
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        
        # Hitung Eye Aspect Ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        ear_history.append(ear)
        current_ear = sum(ear_history) / len(ear_history)
        
        # Hitung Mouth Aspect Ratio
        mar = mouth_aspect_ratio(mouth)
        mar_history.append(mar)
        current_mar = sum(mar_history) / len(mar_history)
        
        # Hitung pergerakan kepala
        movement = head_movement(shape, prev_shape)
        head_movement_history.append(movement)
        current_head_movement = sum(head_movement_history) / len(head_movement_history)
        
        # Simpan shape saat ini untuk perbandingan di frame berikutnya
        prev_shape = shape.copy()
        
        # Visualisasi landmark mata dan mulut
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        # Cek mata tertutup
        if current_ear < EYE_AR_THRESH:
            eye_counter += 1
            # Tambah skor kantuk berdasarkan durasi mata tertutup
            drowsiness_score = min(MAX_DROWSINESS_SCORE, drowsiness_score + 2)
            cv2.putText(frame, "MATA SAYU/TERTUTUP!", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Reset counter jika mata terbuka
            eye_counter = max(0, eye_counter - 1)
            # Kurangi skor kantuk sedikit
            drowsiness_score = max(0, drowsiness_score - 1)
        
        # Cek menguap
        if current_mar > MOUTH_AR_THRESH:
            mouth_counter += 1
            # Tambah skor kantuk jika menguap
            if mouth_counter >= MOUTH_AR_CONSEC_FRAMES:
                drowsiness_score = min(MAX_DROWSINESS_SCORE, drowsiness_score + 15)
                cv2.putText(frame, "MENGUAP TERDETEKSI!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            mouth_counter = 0
        
        # Cek gerakan kepala
        if current_head_movement > HEAD_MOVEMENT_THRESH:
            head_movement_counter += 1
            # Gerakan kepala bisa menandakan kantuk jika terlalu banyak gerakan (anggukan)
            if head_movement_counter >= HEAD_MOVEMENT_FRAMES:
                drowsiness_score = min(MAX_DROWSINESS_SCORE, drowsiness_score + 10)
                cv2.putText(frame, "GERAKAN KEPALA BERLEBIHAN!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            head_movement_counter = max(0, head_movement_counter - 1)
        
        # Tampilkan nilai metrik
        cv2.putText(frame, f"EAR: {current_ear:.2f}/{EYE_AR_THRESH:.2f}", (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {current_mar:.2f}/{MOUTH_AR_THRESH:.2f}", (450, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Head: {current_head_movement:.2f}/{HEAD_MOVEMENT_THRESH:.2f}", (450, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Tampilkan skor kantuk
    cv2.putText(frame, f"Drowsiness Score: {drowsiness_score}/{DROWSINESS_THRESHOLD}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Progress bar untuk skor kantuk
    bar_length = int((drowsiness_score / MAX_DROWSINESS_SCORE) * 400)
    cv2.rectangle(frame, (10, 350), (10 + bar_length, 370), (0, 0, 255), -1)
    cv2.rectangle(frame, (10, 350), (410, 370), (255, 255, 255), 2)
    
    # Aktivasi alarm jika skor kantuk melebihi threshold
    if drowsiness_score >= DROWSINESS_THRESHOLD:
        # Jika alarm belum aktif, nyalakan
        if not alarm_on:
            alarm_on = True
            # Bunyikan alarm
            sound_alarm()
        
        # Tampilkan peringatan
        cv2.putText(frame, "AWAS! ANDA MENGANTUK", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "SEGERA MENEPI UNTUK BERISTIRAHAT", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Berikan tampilan visual yang lebih menarik perhatian
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
    else:
        alarm_on = False
    
    # Tampilkan frame
    cv2.imshow("Drowsiness Detection", frame)
    
    # Keluar jika tekan 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Bersihkan
cv2.destroyAllWindows()
cap.release()
print("[INFO] Program selesai.")