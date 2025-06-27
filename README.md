Proyek Data Mining: Analisis Pola Kecelakaan Lalu Lintas Global dengan K-Means Clustering

Gambar: Representasi Visual Proyek Analisis Data Mining

Daftar Isi
Pendahuluan

Apa Itu Clustering dalam Data Mining?

Tujuan Proyek

Pertanyaan Kunci Analisis

Dataset

Struktur Proyek (Google Colab Notebook)

Tahapan Analisis dan Implementasi Kode

4.1 Bagian 1: Inisialisasi dan Persiapan Lingkungan

4.2 Bagian 2: Eksplorasi Data Awal (EDA)

4.3 Bagian 3: Pra-pemrosesan Data

4.4 Bagian 4: Menentukan Jumlah Cluster Optimal (Metode Elbow)

4.5 Bagian 5: Implementasi K-Means Clustering

4.6 Bagian 6: Analisis dan Interpretasi Hasil Clustering

4.7 Bagian 7: Kesimpulan dan Rekomendasi

Cara Menjalankan Proyek

Kontributor

Lisensi

1. Pendahuluan
Proyek ini merupakan implementasi teknik Data Clustering dalam konteks tugas akhir mata kuliah Data Mining. Kami berfokus pada aplikasi algoritma K-Means untuk mengungkap pola-pola tersembunyi dalam data kecelakaan lalu lintas global.

Apa Itu Clustering dalam Data Mining?
Clustering adalah salah satu teknik Unsupervised Learning dalam data mining. Berbeda dengan klasifikasi yang membutuhkan data berlabel untuk memprediksi kategori, clustering berfungsi untuk mengelompokkan data poin yang serupa (homogen) ke dalam kelompok-kelompok (cluster) berdasarkan kemiripan karakteristiknya, tanpa adanya label yang ditentukan sebelumnya. Tujuannya adalah untuk menemukan struktur atau pola alami dalam data.

Tujuan Proyek
Tujuan utama proyek ini adalah mengidentifikasi dan mengkarakterisasi segmen-segmen kecelakaan lalu lintas yang berbeda berdasarkan fitur-fitur yang ada, seperti kondisi lingkungan, tingkat keparahan insiden, dan pola waktu, tanpa mengandalkan label penyebab kecelakaan yang sudah ada.

Pertanyaan Kunci Analisis
Melalui analisis clustering, kami berusaha menjawab pertanyaan-pertanyaan kunci berikut:

Apakah terdapat kelompok kecelakaan yang cenderung terjadi pada kondisi cuaca atau jalan tertentu?

Apakah terdapat cluster yang terkait dengan jumlah kendaraan yang terlibat atau tingkat keparahan korban?

Apakah terdapat pola waktu tertentu dalam kecelakaan (berdasarkan jam, hari dalam seminggu, atau bulan)?

Bagaimana karakteristik umum dari masing-masing cluster yang ditemukan? (Misalnya: "Cluster X menunjukkan kecelakaan parah di musim dingin," atau "Cluster Y merupakan kecelakaan ringan yang sering terjadi di siang hari.")

2. Dataset
Dataset yang digunakan dalam proyek ini adalah global_traffic_accidents.csv. Dataset ini berisi informasi mengenai berbagai insiden kecelakaan lalu lintas global.

Nama File: global_traffic_accidents.csv
Sumber: (Jika ada link sumber, tambahkan di sini, misalnya: Kaggle, UCI Machine Learning Repository, dll.)

Kolom-kolom Penting dalam Dataset:

Accident ID: Pengidentifikasi unik untuk setiap kecelakaan.

Date: Tanggal terjadinya kecelakaan.

Time: Waktu terjadinya kecelakaan.

Location: Lokasi geografis kecelakaan.

Latitude, Longitude: Koordinat geografis lokasi kecelakaan.

Weather Condition: Kondisi cuaca saat kecelakaan terjadi (misal: Clear, Rain, Snow, Storm).

Road Condition: Kondisi jalan saat kecelakaan terjadi (misal: Dry, Wet, Icy, Under Construction).

Vehicles Involved: Jumlah kendaraan yang terlibat dalam kecelakaan.

Casualties: Jumlah korban dalam kecelakaan.

Cause: Penyebab utama kecelakaan (meskipun ada, tidak digunakan sebagai input clustering karena ini adalah tugas unsupervised).

3. Struktur Proyek (Google Colab Notebook)
Proyek ini diimplementasikan dalam lingkungan Google Colab, dan kodenya telah dibagi menjadi beberapa sel terpisah. Pembagian ini bertujuan untuk:

Keterbacaan yang Lebih Baik: Memudahkan pemahaman alur analisis langkah demi langkah.

Demostrasi yang Terstruktur: Memungkinkan presentasi yang rapi dan logis di setiap tahap pengerjaan.

Modularitas: Memisahkan setiap tahapan (inisialisasi, EDA, pra-pemrosesan, implementasi model, analisis hasil) ke dalam blok kode yang lebih kecil.

Setiap sel kode disertai dengan penjelasan dalam format Markdown yang relevan, menjelaskan tujuan dan langkah-langkah yang dilakukan dalam sel tersebut.

4. Tahapan Analisis dan Implementasi Kode
Berikut adalah rincian setiap bagian dalam notebook Google Colab dan fungsi kodenya:

4.1 Bagian 1: Inisialisasi dan Persiapan Lingkungan
Tujuan:
Bagian ini bertanggung jawab untuk mengatur lingkungan kerja di Google Colab. Ini termasuk mengimpor semua library Python yang dibutuhkan untuk analisis data, machine learning, dan visualisasi, serta menghubungkan Google Drive untuk akses mudah ke dataset.

Proses:

Import Library: Mengimpor pustaka fundamental seperti pandas untuk manipulasi data, numpy untuk operasi numerik, matplotlib.pyplot dan seaborn untuk plotting, serta modul-modul spesifik dari scikit-learn untuk pra-pemrosesan (StandardScaler, LabelEncoder), clustering (KMeans), evaluasi (silhouette_score), dan reduksi dimensi (PCA).

Mount Google Drive: Menghubungkan Google Colab dengan akun Google Drive pengguna agar dataset yang tersimpan dapat diakses.

Memuat Dataset: Membaca file global_traffic_accidents.csv dari lokasi yang ditentukan di Google Drive ke dalam DataFrame Pandas.

Cuplikan Kode (Awal Cell):

# Import Library yang Diperlukan
import numpy as np
import pandas as pd
# ... (library lainnya)

# Mengatur Google Drive untuk Akses Dataset
PATH='/content/drive/My Drive/Dataset/'
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Memuat Dataset
try:
    data = pd.read_csv(PATH + 'global_traffic_accidents.csv')
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("Error: File tidak ditemukan.")

4.2 Bagian 2: Eksplorasi Data Awal (EDA)
Tujuan:
Memahami karakteristik awal dataset, termasuk struktur, tipe data, distribusi nilai, dan keberadaan missing values. Tahap EDA sangat penting untuk menginformasikan keputusan pra-pemrosesan yang akan datang.

Proses:

Melihat Cuplikan Data: Menampilkan beberapa baris pertama DataFrame (data.head()) untuk mendapatkan gambaran sekilas tentang data.

Informasi Dataset: Menggunakan data.info() untuk memeriksa tipe data setiap kolom dan jumlah nilai non-null, yang membantu mengidentifikasi missing values dan tipe data yang tidak sesuai.

Statistik Deskriptif: Menggunakan data.describe(include='all') untuk mendapatkan ringkasan statistik (rata-rata, standar deviasi, min/max, kuartil) untuk kolom numerik, dan frekuensi untuk kolom kategorikal.

Penanganan Missing Values: Mengecek jumlah missing values per kolom dan secara opsional memvisualisasikannya menggunakan heatmap untuk melihat pola missingness.

Analisis Distribusi Fitur Kunci: Memvisualisasikan distribusi fitur-fitur penting seperti Casualties, Vehicles Involved (menggunakan histogram), serta Weather Condition, Road Condition, dan Cause (menggunakan countplot) untuk memahami sebaran data.

Cuplikan Kode & Visualisasi yang Diharapkan:

# Cuplikan Data Awal
print(data.head())

# Informasi Dataset
data.info()

# Statistik Deskriptif Dataset
print(data.describe(include='all'))

# Heatmap Missing Values (jika ada)
# 

# Distribusi Casualties dan Vehicles Involved
# 

# Distribusi Weather Condition dan Road Condition
# 

# Distribusi Cause of Accident
# 

4.3 Bagian 3: Pra-pemrosesan Data
Tujuan:
Mengubah data mentah menjadi format yang siap untuk algoritma K-Means. K-Means adalah algoritma berbasis jarak yang sangat sensitif terhadap skala fitur dan memerlukan input numerik.

Proses:

Penanganan Missing Values: Mengisi nilai yang hilang (NaN) menggunakan strategi yang sesuai: mode (nilai paling sering muncul) untuk kolom kategorikal dan median (nilai tengah) untuk kolom numerik.

Ekstraksi Fitur Waktu: Menguraikan kolom Date dan Time menjadi fitur-fitur yang lebih bermakna seperti DayOfWeek (hari dalam seminggu), Month (bulan), dan Hour (jam). Kolom asli Date dan Time kemudian dihapus.

Pemilihan Fitur untuk Clustering: Memilih subset kolom yang paling relevan untuk proses clustering. Kolom seperti Accident ID, Location (karena terlalu bervariasi), dan Cause (karena clustering adalah unsupervised) dihapus dari set fitur yang akan diproses.

Encoding Variabel Kategorikal: Mengubah kolom teks kategorikal (misalnya, Weather Condition, Road Condition) menjadi representasi numerik menggunakan LabelEncoder. Ini diperlukan karena K-Means hanya bekerja dengan data numerik.

Scaling Fitur Numerik: Menggunakan StandardScaler untuk menstandarisasi semua fitur numerik. Proses ini mengubah data sehingga memiliki rata-rata nol dan standar deviasi satu, memastikan bahwa semua fitur berkontribusi secara adil terhadap perhitungan jarak tanpa ada dominasi oleh fitur dengan rentang nilai yang lebih besar.

Cuplikan Kode:

# Penanganan Missing Values
for column in data.columns:
    # ... (kode penanganan missing values)

# Ekstraksi Fitur Waktu
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
# ... (ekstraksi Hour, Month)
data = data.drop(['Date', 'Time', 'Time_processed'], axis=1)

# Pemilihan Fitur & Encoding Kategorikal
features_for_clustering = ['Latitude', 'Longitude', 'Weather Condition', 'Road Condition',
                           'Vehicles Involved', 'Casualties', 'DayOfWeek', 'Month', 'Hour']
X = data[features_for_clustering].copy()
for col in categorical_cols_to_encode:
    # ... (kode LabelEncoder)

# Scaling Fitur Numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

4.4 Bagian 4: Menentukan Jumlah Cluster Optimal (Metode Elbow)
Tujuan:
Menemukan nilai k (jumlah cluster) yang paling tepat untuk model K-Means. Pemilihan k yang optimal adalah kunci untuk mendapatkan hasil clustering yang bermakna.

Proses:

Perhitungan WCSS (Within-Cluster Sum of Squares): Algoritma K-Means dijalankan untuk berbagai nilai k (misalnya, dari 1 hingga 10). Untuk setiap k, nilai WCSS dihitung. WCSS mengukur jumlah kuadrat jarak antara setiap titik data dan centroid clusternya. Semakin kecil WCSS, semakin padat cluster tersebut.

Visualisasi Elbow Plot: WCSS diplot terhadap jumlah cluster (k). Titik "siku" (elbow) pada grafik, di mana penurunan WCSS mulai melambat secara signifikan, seringkali dianggap sebagai nilai k yang optimal. Ini menunjukkan adanya titik di mana penambahan cluster lebih lanjut tidak memberikan banyak manfaat tambahan dalam mengurangi variabilitas dalam cluster.

Cuplikan Kode & Visualisasi yang Diharapkan:

# Menentukan Jumlah Cluster Optimal (Metode Elbow)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # inertia_ adalah WCSS

# Visualisasi Elbow Method
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Metode Elbow untuk Menentukan K Optimal')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('WCSS')
plt.show()

Gambar: Contoh Grafik Metode Elbow. Carilah 'siku' pada kurva untuk menentukan K optimal.

4.5 Bagian 5: Implementasi K-Means Clustering
Tujuan:
Menerapkan algoritma K-Means pada data yang telah diproses menggunakan nilai k optimal yang telah ditentukan.

Proses:

Inisialisasi Model: Objek KMeans diinisialisasi dengan n_clusters (jumlah cluster optimal), init='k-means++' (metode cerdas untuk pemilihan centroid awal), max_iter (iterasi maksimum), n_init (jumlah percobaan dengan centroid awal berbeda), dan random_state (untuk reproduktifitas hasil).

Pelatihan Model: Model K-Means dilatih (.fit()) pada data yang telah di-scaled (X_scaled).

Penambahan Label Cluster: Label cluster yang dihasilkan oleh model K-Means (kmeans.labels_) ditambahkan sebagai kolom baru ('Cluster') ke DataFrame asli (data) dan DataFrame fitur yang digunakan (X).

Cuplikan Kode:

# Implementasi K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_
X['Cluster'] = kmeans.labels_

4.6 Bagian 6: Analisis dan Interpretasi Hasil Clustering
Tujuan:
Mengkarakterisasi setiap cluster yang terbentuk dan menginterpretasikannya untuk menjawab pertanyaan-pertanyaan kunci yang diajukan di awal proyek.

Proses:

Ringkasan Ukuran Cluster: Menampilkan jumlah data poin yang termasuk dalam setiap cluster untuk memahami distribusi cluster.

Karakteristik Rata-rata/Modus per Cluster: Menghitung rata-rata untuk fitur numerik dan modus untuk fitur kategorikal untuk setiap cluster. Ini membantu mengidentifikasi profil "tipikal" dari kecelakaan dalam setiap kelompok.

Visualisasi Cluster:

PCA (Principal Component Analysis): Karena dataset memiliki banyak dimensi, PCA digunakan untuk mereduksi data menjadi 2 komponen utama untuk visualisasi 2D. Scatter plot dari komponen PCA ini kemudian diwarnai berdasarkan label cluster, menunjukkan bagaimana cluster-cluster tersebut terpisah secara visual.

Box Plots/Count Plots per Cluster: Membuat visualisasi spesifik (boxplot untuk numerik, countplot untuk kategorikal) yang membandingkan distribusi fitur-fitur kunci (Casualties, Vehicles Involved, Weather Condition, Road Condition, Hour, DayOfWeek, Month) di antara setiap cluster. Visualisasi ini secara langsung membantu menjawab pertanyaan kunci analisis.

Evaluasi Kuantitatif (Silhouette Score): Menghitung Silhouette Score untuk mengukur seberapa baik data dikelompokkan. Nilai mendekati 1 menunjukkan cluster yang padat dan terpisah dengan baik, sementara nilai mendekati 0 menunjukkan tumpang tindih.

Cuplikan Kode & Visualisasi yang Diharapkan:

# Ringkasan Ukuran Cluster
print(data['Cluster'].value_counts())

# Karakteristik Rata-rata Fitur Numerik per Cluster
print(data.groupby('Cluster')[['Casualties', 'Vehicles Involved', 'Hour']].mean())

# Visualisasi PCA
# 

# Visualisasi Kondisi Cuaca dan Jalan per Cluster
# 

# Visualisasi Casualties dan Vehicles Involved per Cluster
# 

# Visualisasi Pola Waktu per Cluster (Jam, Hari, Bulan)
# 

# Evaluasi Silhouette Score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"Rata-rata Silhouette Score: {silhouette_avg:.3f}")

4.7 Bagian 7: Kesimpulan dan Rekomendasi
Tujuan:
Merangkum temuan-temuan utama dari analisis clustering, secara eksplisit menjawab pertanyaan-pertanyaan kunci yang diajukan, dan memberikan rekomendasi praktis berdasarkan wawasan yang diperoleh.

Penting untuk Diisi (Oleh Anda):
Bagian ini adalah tempat Anda akan menginterpretasikan hasil dari Bagian 6. Anda perlu menganalisis ringkasan statistik (rata-rata/modus per cluster) dan visualisasi (box plot, count plot) untuk setiap cluster dan merumuskan kesimpulan yang bermakna.

Struktur Kesimpulan yang Direkomendasikan:

Karakteristik Umum Setiap Cluster: Deskripsikan profil unik dari setiap cluster yang teridentifikasi. Contoh: "Cluster 0 cenderung menunjukkan kecelakaan parah di malam hari pada kondisi jalan bersalju, sementara Cluster 1 adalah kecelakaan ringan di siang hari pada kondisi cuaca cerah."

Menjawab Pertanyaan Kunci: Kaitkan temuan Anda secara langsung dengan pertanyaan-pertanyaan yang diajukan di awal proyek.

Pertanyaan 1 (Kondisi Cuaca/Jalan): Jelaskan apakah ada cluster yang dominan pada kondisi cuaca atau jalan tertentu.

Pertanyaan 2 (Kendaraan Terlibat/Korban): Identifikasi cluster dengan tingkat keparahan korban atau jumlah kendaraan terlibat yang tinggi/rendah.

Pertanyaan 3 (Pola Waktu): Temukan pola jam, hari, atau bulan tertentu yang terkait dengan cluster-cluster tertentu.

Potensi Rekomendasi: Berikan saran atau rekomendasi praktis yang dapat diambil dari wawasan clustering. Contoh: "Pemerintah dapat memfokuskan kampanye keselamatan berkendara di musim dingin untuk daerah dan waktu yang terkait dengan Cluster X," atau "Alokasi sumber daya darurat dapat dioptimalkan berdasarkan karakteristik kecelakaan di Cluster Z."

Penyimpanan Hasil:
Pada akhir bagian ini, dataset dengan label cluster yang baru ditambahkan akan disimpan ke file CSV baru, global_traffic_accidents_clustered.csv, di Google Drive Anda.

5. Cara Menjalankan Proyek
Untuk menjalankan proyek ini di Google Colab, ikuti langkah-langkah berikut:

Persyaratan:

Akun Google.

Akses ke Google Drive.

File dataset global_traffic_accidents.csv diunggah ke folder My Drive/Dataset/ di Google Drive Anda. Pastikan nama file dan path folder sudah benar.

Langkah-langkah:

Buka Google Colab: Kunjungi colab.research.google.com.

Buat Notebook Baru: Pilih File > New notebook.

Salin dan Tempel Kode: Salin setiap blok kode dan teks Markdown dari dokumentasi ini ke dalam sel-sel terpisah di notebook Colab Anda. Pastikan sel Markdown sebagai 'Teks' dan sel Python sebagai 'Kode'.

Jalankan Sel: Jalankan setiap sel secara berurutan, mulai dari atas. Pastikan untuk mengizinkan Google Colab mengakses Google Drive Anda saat diminta.

Sesuaikan optimal_k: Setelah menjalankan Bagian 4 (Metode Elbow), amati grafik dan sesuaikan nilai optimal_k di sel Bagian 4 dan Bagian 5 sesuai dengan titik "siku" yang Anda temukan.

Analisis Hasil: Amati output dan visualisasi dari Bagian 6 untuk menginterpretasikan cluster.

Isi Kesimpulan: Lengkapi Bagian 7 (Kesimpulan dan Rekomendasi) di notebook atau laporan Anda berdasarkan analisis yang telah dilakukan.

6. Kontributor
[Nama Anda] - (NIM Anda / Peran Anda dalam Proyek)

[Nama Anggota Kelompok Lain (jika ada)]

7. Lisensi
Proyek ini dilisensikan di bawah lisensi MIT. Lihat file LICENSE untuk detail lebih lanjut.
