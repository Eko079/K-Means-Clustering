# Proyek Data Mining: Analisis Pola Kecelakaan Lalu Lintas Global dengan K-Means Clustering

![Data Mining Banner](https://img.shields.io/badge/Data%20Mining-Clustering-blue?style=for-the-badge&logo=apachespark)
![Python](https://img.shields.io/badge/Python-3.x-informational?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KMeans-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-red?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-green?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-purple?style=for-the-badge&logo=seaborn&logoColor=white)

## Daftar Isi

1.  [Pendahuluan Proyek](#1-pendahuluan-proyek)
2.  [Tujuan Analisis](#2-tujuan-analisis)
3.  [Dataset](#3-dataset)
4.  [Struktur Proyek dan Implementasi Kode](#4-struktur-proyek-dan-implementasi-kode)
    * [Bagian 1: Inisialisasi dan Persiapan Lingkungan](#bagian-1-inisialisasi-dan-persiapan-lingkungan)
    * [Bagian 2: Eksplorasi Data Awal (EDA)](#bagian-2-eksplorasi-data-awal-eda)
    * [Bagian 3: Pra-pemrosesan Data](#bagian-3-pra-pemrosesan-data)
    * [Bagian 4: Menentukan Jumlah Cluster Optimal (Metode Elbow)](#bagian-4-menentukan-jumlah-cluster-optimal-metode-elbow)
    * [Bagian 5: Implementasi K-Means Clustering](#bagian-5-implementasi-k-means-clustering)
    * [Bagian 6: Analisis dan Interpretasi Hasil Clustering](#bagian-6-analisis-dan-interpretasi-hasil-clustering)
    * [Bagian 7: Kesimpulan dan Rekomendasi](#bagian-7-kesimpulan-dan-rekomendasi)
5.  [Cara Menjalankan Proyek](#5-cara-menjalankan-proyek)
6.  [Kontributor](#6-kontributor)
7.  [Lisensi](#7-lisensi)

---

## 1. Pendahuluan Proyek

Proyek ini merupakan bagian dari tugas akhir mata kuliah Data Mining yang berfokus pada penerapan teknik *unsupervised learning*, khususnya **K-Means Clustering**. Dalam era di mana data terus bertumbuh secara eksponensial, kemampuan untuk menemukan pola tersembunyi dan struktur dalam data yang tidak berlabel menjadi sangat berharga.

Kecelakaan lalu lintas adalah masalah global yang menyebabkan kerugian besar, baik dari segi nyawa maupun ekonomi. Memahami pola-pola di balik kecelakaan ini dapat menjadi kunci untuk mengembangkan strategi pencegahan yang lebih efektif dan alokasi sumber daya yang lebih efisien. Dengan menggunakan algoritma K-Means, proyek ini berupaya mengelompokkan insiden kecelakaan lalu lintas yang memiliki karakteristik serupa, memberikan wawasan baru yang mungkin tidak terlihat dari analisis permukaan.

## 2. Tujuan Analisis

Tujuan utama proyek ini adalah untuk **mengidentifikasi dan mengkarakterisasi segmen-segmen kecelakaan lalu lintas yang berbeda berdasarkan fitur-fitur yang tersedia dalam dataset `global_traffic_accidents.csv`**. Secara spesifik, analisis ini bertujuan untuk menjawab pertanyaan-pertanyaan berikut:

* **P1: Apakah terdapat kelompok kecelakaan yang cenderung terjadi pada kondisi cuaca atau jalan tertentu?**
* **P2: Apakah terdapat cluster yang terkait dengan jumlah kendaraan yang terlibat atau tingkat keparahan korban?**
* **P3: Apakah terdapat pola waktu tertentu dalam kecelakaan (berdasarkan jam, hari, atau bulan)?**
* **P4: Bagaimana karakteristik umum dari masing-masing cluster?**
    * Misalnya, "Cluster X menunjukkan kecelakaan parah yang sering terjadi di musim dingin."
    * Misalnya, "Cluster Y merupakan kecelakaan ringan yang sering terjadi di siang hari pada kondisi jalan normal."

## 3. Dataset

Dataset yang digunakan dalam proyek ini adalah `global_traffic_accidents.csv`. Dataset ini berisi catatan insiden kecelakaan lalu lintas global dengan berbagai atribut, termasuk informasi ID kecelakaan, tanggal dan waktu, lokasi (lintang dan bujur), kondisi cuaca, kondisi jalan, jumlah kendaraan yang terlibat, jumlah korban, dan penyebab kecelakaan.

Berikut adalah cuplikan data dari dataset:

| Accident ID | Date       | Time    | Location        | Latitude   | Longitude   | Weather Condition | Road Condition | Vehicles Involved | Casualties | Cause              |
| ----------- | ---------- | ------- | --------------- | ---------- | ----------- | ----------------- | -------------- | ----------------- | ---------- | ------------------ |
| b0d66167    | 2023-04-19 | 06:39   | Mumbai, India   | 13.488432  | 73.290682   | Snow              | Snowy          | 5                 | 7          | Reckless Driving   |
| debfa0a9    | 2023-01-17 | 02:47   | São Paulo, Brazil | -37.798317 | -32.242412  | Clear             | Icy            | 4                 | 1          | Drunk Driving      |
| 6d68aa36    | 2024-04-09 | 02:55   | Sydney, Australia | 33.767869  | 104.869018  | Rain              | Snowy          | 1                 | 7          | Reckless Driving   |
| 425b01fd    | 2023-10-10 | 11:23   | Tokyo, Japan    | -0.378031  | -165.825855 | Storm             | Wet            | 4                 | 0          | Drunk Driving      |
| 90d50f62    | 2023-01-02 | 12:07   | Beijing, China  | 41.254079  | -30.776959  | Storm             | Snowy          | 3                 | 9          | Reckless Driving   |

**Catatan:** Kolom 'Cause' tidak akan digunakan sebagai fitur input untuk proses clustering karena clustering adalah teknik *unsupervised learning* yang tidak memerlukan label target. Namun, kolom ini dapat digunakan untuk memberikan wawasan tambahan setelah cluster terbentuk.

## 4. Struktur Proyek dan Implementasi Kode

Proyek ini diimplementasikan menggunakan Google Colaboratory, yang memungkinkan eksekusi kode Python secara interaktif dan terintegrasi dengan Google Drive. Kode dipecah menjadi beberapa bagian utama untuk memfasilitasi pemahaman dan presentasi.

### Bagian 1: Inisialisasi dan Persiapan Lingkungan

* **Konteks Analisis:** Memastikan semua alat dan data siap diakses sebelum memulai analisis.
* **Tujuan:**
    * Mengimpor semua library Python yang dibutuhkan (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).
    * Menghubungkan Google Colab dengan Google Drive untuk mengakses dataset.
    * Memuat dataset `global_traffic_accidents.csv` ke dalam DataFrame Pandas.
* **Langkah-langkah Implementasi:**
    1.  Import `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `StandardScaler`, `LabelEncoder`, `KMeans`, `silhouette_score`, `PCA`, `os`, dan `warnings`.
    2.  Set `PATH` ke direktori tempat dataset disimpan di Google Drive.
    3.  Gunakan `drive.mount()` untuk mengaitkan Google Drive.
    4.  Muat `global_traffic_accidents.csv` menggunakan `pd.read_csv()`, dengan penanganan error dasar jika file tidak ditemukan.

### Bagian 2: Eksplorasi Data Awal (EDA)

* **Konteks Analisis:** Memahami karakteristik dasar data adalah langkah pertama dalam setiap proyek data mining.
* **Tujuan:**
    * Mendapatkan gambaran awal tentang struktur dan isi dataset.
    * Mengidentifikasi tipe data setiap kolom.
    * Mendeteksi keberadaan *missing values* dan potensi anomali.
    * Memvisualisasikan distribusi fitur-fitur kunci untuk wawasan awal.
* **Langkah-langkah Implementasi:**
    1.  Tampilkan beberapa baris pertama data (`data.head()`).
    2.  Gunakan `data.info()` untuk ringkasan non-null count dan tipe data.
    3.  Gunakan `data.describe(include='all')` untuk statistik deskriptif.
    4.  Hitung dan visualisasikan *missing values* (`data.isnull().sum()`, `sns.heatmap()`).
    5.  Visualisasikan distribusi fitur-fitur kunci seperti `Casualties`, `Vehicles Involved` (menggunakan `histplot`), serta `Weather Condition`, `Road Condition`, dan `Cause` (menggunakan `countplot`).

### Bagian 3: Pra-pemrosesan Data

* **Konteks Analisis:** K-Means adalah algoritma berbasis jarak yang sensitif terhadap skala fitur dan memerlukan input numerik. Oleh karena itu, data perlu dibersihkan dan ditransformasi.
* **Tujuan:**
    * Menangani *missing values* untuk memastikan kelengkapan data.
    * Mengekstrak fitur yang lebih bermakna dari kolom tanggal dan waktu.
    * Mengubah variabel kategorikal menjadi representasi numerik.
    * Melakukan penskalaan fitur numerik untuk menstandardisasi rentang nilainya.
* **Langkah-langkah Implementasi:**
    1.  **Penanganan Missing Values:** Mengisi nilai yang hilang: modus untuk kolom kategorikal, median untuk kolom numerik.
    2.  **Ekstraksi Fitur Waktu:** Mengkonversi kolom 'Date' dan 'Time' menjadi objek datetime, lalu mengekstrak `DayOfWeek`, `Month`, dan `Hour`. Kolom asli kemudian dihapus.
    3.  **Pemilihan Fitur untuk Clustering:** Memilih kolom-kolom yang relevan (`Latitude`, `Longitude`, `Weather Condition`, `Road Condition`, `Vehicles Involved`, `Casualties`, `DayOfWeek`, `Month`, `Hour`). Kolom seperti `Accident ID`, `Location`, dan `Cause` diabaikan karena alasan relevansi clustering (ID unik, variasi tinggi, atau merupakan label target potensial).
    4.  **Encoding Variabel Kategorikal:** Menggunakan `LabelEncoder` dari Scikit-learn untuk mengkonversi kolom kategorikal (`Weather Condition`, `Road Condition`) menjadi representasi numerik.
    5.  **Scaling Fitur Numerik:** Menerapkan `StandardScaler` pada semua fitur yang dipilih untuk clustering (`X_scaled`). Ini memastikan semua fitur memiliki rata-rata nol dan standar deviasi satu, mencegah fitur dengan rentang nilai besar mendominasi perhitungan jarak.

### Bagian 4: Menentukan Jumlah Cluster Optimal (Metode Elbow)

* **Konteks Analisis:** Salah satu tantangan K-Means adalah menentukan nilai `k` (jumlah cluster) yang optimal. Metode Elbow adalah heuristik populer untuk membantu dalam pengambilan keputusan ini.
* **Tujuan:**
    * Menemukan nilai `k` yang paling sesuai untuk dataset, yang menyeimbangkan kepadatan intra-cluster dengan pemisahan antar-cluster.
* **Langkah-langkah Implementasi:**
    1.  Menjalankan algoritma K-Means untuk berbagai nilai `k` (dari 1 hingga 10).
    2.  Menghitung *Within-Cluster Sum of of Squares* (WCSS), atau `inertia_`, untuk setiap nilai `k`. WCSS mengukur seberapa padat cluster tersebut (semakin kecil WCSS, semakin padat).
    3.  Memvisualisasikan hasil WCSS terhadap `k` dalam sebuah *line plot* (grafik Elbow). Titik "siku" pada grafik (di mana penurunan WCSS melambat secara signifikan) menunjukkan nilai `k` yang optimal.

### Bagian 5: Implementasi K-Means Clustering

* **Konteks Analisis:** Setelah menentukan jumlah cluster yang optimal, langkah selanjutnya adalah menerapkan algoritma K-Means untuk mengelompokkan data.
* **Tujuan:**
    * Menginisialisasi dan melatih model K-Means dengan `k` optimal yang telah ditentukan.
    * Menugaskan setiap data poin ke cluster yang sesuai.
* **Langkah-langkah Implementasi:**
    1.  Menginisialisasi objek `KMeans` dengan parameter `n_clusters` (berdasarkan hasil Elbow Method), `init='k-means++'` (untuk inisialisasi centroid yang cerdas), `max_iter`, `n_init`, dan `random_state` (untuk reproduktifitas).
    2.  Melatih model (`.fit()`) pada data yang sudah di-scaled (`X_scaled`).
    3.  Menambahkan label cluster yang dihasilkan (`kmeans.labels_`) sebagai kolom baru bernama `Cluster` ke DataFrame asli (`data`) dan DataFrame fitur (`X`).

### Bagian 6: Analisis dan Interpretasi Hasil Clustering

* **Konteks Analisis:** Ini adalah tahap paling penting di mana kita menarik wawasan dari hasil clustering. Kita akan mengidentifikasi karakteristik unik dari setiap cluster dan menjawab pertanyaan-pertanyaan penelitian kita.
* **Tujuan:**
    * Memahami ukuran dan distribusi data di setiap cluster.
    * Mengidentifikasi karakteristik rata-rata atau modus dari fitur-fitur penting dalam setiap cluster.
    * Memvisualisasikan pemisahan cluster dalam ruang dimensi rendah.
    * Menjawab pertanyaan-pertanyaan kunci analisis (P1, P2, P3).
    * Mengevaluasi kualitas clustering secara kuantitatif.
* **Langkah-langkah Implementasi:**
    1.  **Ringkasan Ukuran Cluster:** Menampilkan jumlah data poin di setiap cluster (`data['Cluster'].value_counts()`).
    2.  **Karakteristik Rata-rata/Modus per Cluster:**
        * Menghitung rata-rata fitur numerik (`cluster_summary_numeric`) untuk setiap cluster.
        * Menghitung modus fitur kategorikal asli (`Weather Condition`, `Road Condition`) untuk setiap cluster, memberikan gambaran karakteristik dominan.
        * (Tambahan) Menampilkan distribusi kolom 'Cause' per cluster untuk wawasan tambahan, meskipun ini bukan fitur input clustering.
    3.  **Visualisasi Cluster:**
        * **PCA Scatter Plot:** Menggunakan Principal Component Analysis (PCA) untuk mereduksi data dimensi tinggi menjadi 2 komponen utama. Plot scatter kemudian dibuat dengan titik-titik yang diwarnai berdasarkan clusternya, memungkinkan visualisasi pemisahan cluster.
        * **Box Plots/Count Plots per Cluster:** Membuat serangkaian plot untuk secara visual menunjukkan distribusi fitur-fitur kunci (`Casualties`, `Vehicles Involved`, `Weather Condition`, `Road Condition`, `Hour`, `DayOfWeek`, `Month`) di setiap cluster. Ini secara langsung mendukung interpretasi dan menjawab pertanyaan P1, P2, P3.
    4.  **Evaluasi Kuantitatif (Silhouette Score):** Menghitung `silhouette_score` untuk mengukur seberapa baik setiap objek berada di dalam clusternya dibandingkan dengan cluster tetangga. Skor yang lebih tinggi (mendekati 1) menunjukkan cluster yang lebih padat dan terpisah.

### Bagian 7: Kesimpulan dan Rekomendasi

* **Konteks Analisis:** Mengkonsolidasikan semua temuan dan mengkomunikasikan nilai praktis dari analisis clustering.
* **Tujuan:**
    * Merangkum karakteristik utama dari setiap cluster yang teridentifikasi.
    * Memberikan jawaban eksplisit terhadap pertanyaan-pertanyaan kunci analisis.
    * Mengusulkan potensi rekomendasi atau tindakan berdasarkan wawasan yang diperoleh.
* **Langkah-langkah Implementasi:**
    1.  Sajikan ringkasan deskriptif untuk setiap cluster, jelaskan karakteristik dominan (misalnya, "Cluster 0 mewakili kecelakaan parah yang terjadi pada kondisi cuaca ekstrem...").
    2.  Secara langsung jawab pertanyaan P1, P2, dan P3 berdasarkan interpretasi dari Bagian 6.
    3.  Berikan rekomendasi yang dapat diterapkan, misalnya, kebijakan keselamatan lalu lintas yang ditargetkan, alokasi sumber daya darurat yang lebih baik, atau peningkatan infrastruktur di area/waktu tertentu.
    4.  (Opsional) Simpan dataset yang sudah diberi label cluster ke file CSV baru untuk penggunaan di masa mendatang.

## 5. Cara Menjalankan Proyek

Untuk menjalankan proyek ini di Google Colaboratory:

1.  **Siapkan Dataset:** Unggah file `global_traffic_accidents.csv` ke dalam folder bernama `Dataset` di Google Drive Anda. Pastikan jalur `PATH='/content/drive/My Drive/Dataset/'` di kode sudah sesuai.
2.  **Buka di Google Colab:** Buka file `ipynb` ini di Google Colab.
3.  **Jalankan Sel Secara Berurutan:** Jalankan setiap sel kode secara berurutan, mulai dari Bagian 1 hingga Bagian 7.
4.  **Interaksi dengan Output:** Perhatikan output teks dan grafik yang dihasilkan. Khususnya, amati grafik Elbow Method di Bagian 4 untuk menentukan nilai `optimal_k`, dan sesuaikan kode jika diperlukan.
5.  **Isi Kesimpulan:** Setelah semua kode dijalankan, Anda perlu mengisi bagian **Kesimpulan dan Rekomendasi** (Bagian 7) secara manual berdasarkan interpretasi visual dan statistik dari hasil clustering Anda.

## 6. Kontributor

* [Nama Lengkap Anda] - [Peran Anda dalam Proyek, misalnya: Data Analyst, Developer]
* [Nama Anggota Kelompok Lainnya (jika ada)] - [Peran mereka]
    * *(Tambahkan nama-nama anggota kelompok Anda di sini)*

## 7. Lisensi

Proyek ini dilisensikan di bawah lisensi MIT. Lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.
