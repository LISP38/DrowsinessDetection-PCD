# Drowsiness Detection OpenCV 😴 🚫 🚗

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

This code can detect your eyes and alert when the user is drowsy.

## Applications 🎯
This can be used by riders who tend to drive for a longer period of time that may lead to accidents

### Code Requirements 🦄
The example code is in Python ([version 2.7](https://www.python.org/download/releases/2.7/) or higher will work). Sistem Deteksi Kantuk
Ringkasan
Proyek ini adalah sistem deteksi kantuk berbasis visi komputer dan peringatan suara secara real-time. Sistem ini memantau wajah pengguna melalui webcam untuk mendeteksi tanda-tanda kantuk, seperti mata tertutup lama, menguap, dan gerakan kepala berlebihan. Skor kantuk diakumulasikan berdasarkan indikator ini dan memicu alarm suara saat skor mencapai ambang batas (3.0), mengingatkan pengguna untuk beristirahat. Proyek ini menggunakan MediaPipe untuk deteksi landmark wajah, OpenCV untuk pemrosesan video, dan Pygame untuk peringatan suara.
Kode ini cocok untuk aplikasi seperti pemantauan pengemudi untuk meningkatkan keselamatan dengan mendeteksi tanda awal kelelahan.
Fitur

Deteksi Mata Tertutup: Melacak Eye Aspect Ratio (EAR) untuk mendeteksi mata tertutup lama, menambahkan hingga 0.5 poin ke skor kantuk berdasarkan durasi (maks. 2 detik).
Deteksi Menguap: Memantau Mouth Aspect Ratio (MAR) untuk mendeteksi menguap, menambahkan 0.3 poin per kejadian.
Deteksi Gerakan Kepala: Mengukur gerakan kepala yang dinormalisasi untuk mendeteksi anggukan atau kemiringan berlebihan, menambahkan 0.2 poin per kejadian.
Skor Kantuk Akumulatif: Bertahan antar frame, menjumlahkan kontribusi hingga mencapai ambang batas (3.0) atau direset manual.
Alarm Suara: Berbunyi saat skor kantuk mencapai 3.0, menggunakan bunyi beep atau suara kustom (alarm.wav), berulang setiap 3 detik hingga direset.
Umpan Balik Visual: Menampilkan metrik real-time (EAR, MAR, gerakan kepala), skor kantuk, dan bilah kemajuan pada feed video.
Reset Manual: Tekan 'r' untuk mengatur ulang skor; tekan 'q' untuk keluar.

Persyaratan

Python 3.8+
Pustaka:
mediapipe>=0.10.0
scipy>=1.10.0
numpy>=1.22.0
opencv-python>=4.7.0
pygame>=2.1.0


Perangkat Keras:
Webcam (USB atau bawaan)
Speaker atau headphone untuk peringatan suara


Opsional:
File alarm.wav di direktori proyek untuk suara alarm kustom (default ke beep jika tidak ada)



Instalasi

Klon atau Unduh Proyek:
git clone <url-repositori>
cd deteksi-kantuk

Atau unduh dan ekstrak file proyek.

Instal Dependensi:
pip install mediapipe scipy numpy opencv-python pygame


Siapkan Audio (Opsional):

Tempatkan file alarm.wav di direktori proyek untuk suara alarm kustom.
Jika tidak ada, sistem akan menghasilkan beep 440 Hz.


Verifikasi Webcam:

Pastikan webcam terhubung dan berfungsi.
Uji dengan skrip OpenCV sederhana jika diperlukan:import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imshow("Uji", frame)
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()





Penggunaan

Jalankan Skrip:
python drowsiness_detector.py


Feed webcam akan terbuka dengan anotasi real-time.
Log konsol menampilkan kejadian deteksi dan pembaruan skor kantuk.


Berinteraksi dengan Sistem:

Pantau Kantuk: Sistem melacak mata tertutup, menguap, dan gerakan kepala, memperbarui skor kantuk (maks. 5.0).
Menguap: +0.3 per kejadian (cooldown 2 detik).
Mata Tertutup: Hingga +0.5 berdasarkan durasi (cooldown 1 detik, kontribusi maks. setelah 2 detik).
Gerakan Kepala: +0.2 per kejadian (cooldown 1.5 detik).


Alarm: Berbunyi saat skor mencapai 3.0, berulang setiap 3 detik hingga direset.
Reset: Tekan 'r' untuk mengatur ulang skor ke 0.0.
Keluar: Tekan 'q' untuk keluar.


Contoh Keluaran:

Konsol:[DETECTION] yawning: +0.30 → Total: 0.30/3.00
[DETECTION] eyes_closed: +0.40 → Total: 0.55/3.00
[DETECTION] head_movement: +0.20 → Total: 0.75/3.00
DROWSY DETECTED! Level: 3.10, Indicators: {'ear': 0.20, 'mar': 0.50, 'head_movement': 0.12}


Feed Video: Menampilkan metrik EAR, MAR, gerakan kepala, skor kantuk, bilah kemajuan, dan peringatan seperti “MATA TERTUTUP (1.2s)!” atau “KANTUK! ISTIRAHAT”.



Penyesuaian Sistem
Untuk mengoptimalkan deteksi sesuai setup Anda (kamera, pencahayaan, pengguna), sesuaikan parameter di inisialisasi DrowsinessDetector:
detector = DrowsinessDetector(
    eye_ar_thresh=0.40,  # Turunkan (0.2) untuk deteksi mata sensitif; naikkan (0.3) untuk kurang sensitif
    mouth_ar_thresh=0.7,  # Turunkan (0.6) untuk deteksi menguap sensitif; naikkan (0.8) untuk kurang
    head_movement_thresh=0.1,  # Turunkan (0.05) untuk gerakan kecil; naikkan (0.15) untuk besar
    eye_weight=0.5,  # Naikkan (0.7) untuk dampak mata tertutup lebih besar
    yawn_weight=0.3,  # Sesuaikan dampak menguap
    head_movement_weight=0.2,  # Sesuaikan dampak gerakan kepala
    drowsiness_threshold=3.0,  # Turunkan (2.5) untuk alarm lebih cepat
    max_drowsiness_level=5.0  # Batas skor maksimum
)


Kalibrasi:
Jalankan skrip dan catat nilai EAR/MAR saat waspada (ditampilkan di feed video).
Atur eye_ar_thresh ke ~70% dari EAR normal (mis., jika EAR=0.35, set 0.40).
Atur mouth_ar_thresh ke ~120% dari MAR normal (mis., jika MAR=0.5, set 0.6).


Cooldown:
Kurangi cooldowns['eyes_closed'] (mis., 0.5) untuk deteksi mata tertutup lebih sering.
Tambah cooldowns['yawning'] (mis., 3.0) jika menguap terdeteksi terlalu sering.


Lingkungan:
Pastikan pencahayaan baik dan wajah berada di tengah frame.
Minimalkan gerakan latar belakang untuk mengurangi deteksi gerakan kepala salah.



Pemecahan Masalah

Wajah Tidak Terdeteksi:
Periksa koneksi webcam dan pencahayaan.
Pastikan wajah berada dalam frame dan tidak terhalang.


Tidak Ada Deteksi:
Verifikasi nilai EAR/MAR (di feed video) melampaui ambang batas.
Turunkan eye_ar_thresh atau mouth_ar_thresh jika nilai selalu di atas/bawah.


Alarm Tidak Berbunyi:
Pastikan skor kantuk mencapai 3.0 (cek log konsol).
Turunkan drowsiness_threshold atau tambah bobot untuk akumulasi lebih cepat.


Deteksi Salah:
Naikkan ambang batas atau perpanjang cooldown.
Sesuaikan head_movement_thresh jika kemiringan kepala normal terdeteksi.


Masalah Performa:
Ubah ukuran frame sebelum diproses:frame = cv2.resize(frame, (640, 480))


Lewati frame dengan menambahkan penghitung di loop utama.



Struktur Proyek
deteksi-kantuk/
├── drowsiness_detector.py  # Skrip utama
└── README.md              # File ini

Lisensi
Proyek ini disediakan untuk tujuan pendidikan di bawah Lisensi MIT. Lihat LICENSE untuk detail.
Penghargaan

Dibuat dengan MediaPipe untuk deteksi landmark wajah.
Terinspirasi dari penelitian deteksi kantuk menggunakan metrik EAR dan MAR.



### Dependencies

1) import cv2
2) import imutils
3) import dlib
4) import scipy


### Description 📌

A computer vision system that can automatically detect driver drowsiness in a real-time video stream and then play an alarm if the driver appears to be drowsy.

### Algorithm 👨‍🔬

Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person), and then working clockwise around the eye.

It checks 20 consecutive frames and if the Eye Aspect ratio is less than 0.40, Alert is generated.

<img src="https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/assets/eye1.jpg">


#### Relationship

<img src="https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/assets/eye2.png">

#### Summing up

<img src="https://github.com/akshaybahadur21/Drowsiness_Detection/blob/master/assets/eye3.jpg">


For more information, [see](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)

### Results 📊

<img src="https://github.com/akshaybahadur21/BLOB/blob/master/drowsy.gif">


### Execution 🐉
To run the code, type `python Drowsiness_Detection.py`

```
python Drowsiness_Detection.py
```

###### Made with ❤️ and 🦙 by Akshay Bahadur

## 📌 Cite Us

To cite this guide, use the below format:
```
@article{Drowsiness_Detection,
author = {Bahadur, Akshay},
journal = {https://github.com/akshaybahadur21/Drowsiness_Detection},
month = {01},
title = {{Drowsiness_Detection}},
year = {2018}
}
```

## References 🔱
 
 -   Adrian Rosebrock, [PyImageSearch Blog](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)

