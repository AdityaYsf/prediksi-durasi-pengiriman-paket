import pandas as pd
import numpy as np

# Set random seed untuk reproducibility
np.random.seed(42)

# Jumlah data
n_samples = 75

# Generate fitur
data = {
    'Jarak': [],
    'Berat': [],
    'Layanan': [],
    'Cuaca': [],
    'Wilayah': [],
    'Durasi': []
}

# Definisi kategori
layanan_options = ['Reguler', 'Express', 'Same Day']
cuaca_options = ['Cerah', 'Berawan', 'Hujan']
wilayah_options = ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Semarang']

# Faktor pengali untuk setiap kategori
layanan_factor = {
    'Reguler': 1.5,
    'Express': 1.0,
    'Same Day': 0.6
}

cuaca_factor = {
    'Cerah': 1.0,
    'Berawan': 1.15,
    'Hujan': 1.4
}

wilayah_factor = {
    'Jakarta': 0.9,      # Infrastruktur bagus
    'Bandung': 1.0,      # Standar
    'Surabaya': 0.95,    # Infrastruktur bagus
    'Medan': 1.2,        # Lebih jauh, infrastruktur sedang
    'Semarang': 1.1      # Infrastruktur sedang
}

berat_factor = {
    'ringan': 1.0,       # < 5kg
    'sedang': 1.1,       # 5-15kg
    'berat': 1.25        # > 15kg
}

# Generate data
for i in range(n_samples):
    # Random features
    jarak = np.random.randint(10, 201)  # 10-200 km
    berat = np.random.randint(1, 51)     # 1-50 kg
    layanan = np.random.choice(layanan_options)
    cuaca = np.random.choice(cuaca_options)
    wilayah = np.random.choice(wilayah_options)
    
    # Tentukan kategori berat
    if berat < 5:
        berat_cat = 'ringan'
    elif berat <= 15:
        berat_cat = 'sedang'
    else:
        berat_cat = 'berat'
    
    # Hitung durasi dasar (dalam jam)
    # Formula: Base = (Jarak / 40) * 24 jam
    # Asumsi: kecepatan rata-rata 40 km/hari untuk pengiriman darat
    base_duration = (jarak / 40) * 24
    
    # Apply faktor-faktor
    duration = base_duration
    duration *= layanan_factor[layanan]
    duration *= cuaca_factor[cuaca]
    duration *= wilayah_factor[wilayah]
    duration *= berat_factor[berat_cat]
    
    # Tambahkan random noise (±15%)
    noise = np.random.uniform(0.85, 1.15)
    duration *= noise
    
    # Minimum 2 jam untuk same day, 6 jam untuk express, 12 jam untuk reguler
    if layanan == 'Same Day':
        duration = max(duration, 2)
    elif layanan == 'Express':
        duration = max(duration, 6)
    else:
        duration = max(duration, 12)
    
    # Bulatkan ke 1 desimal
    duration = round(duration, 1)
    
    # Simpan data
    data['Jarak'].append(jarak)
    data['Berat'].append(berat)
    data['Layanan'].append(layanan)
    data['Cuaca'].append(cuaca)
    data['Wilayah'].append(wilayah)
    data['Durasi'].append(duration)

# Buat DataFrame
df = pd.DataFrame(data)

# Tampilkan statistik
print("=" * 60)
print("DATASET PENGIRIMAN PAKET - REALISTIS")
print("=" * 60)
print(f"\nJumlah Data: {len(df)}")
print(f"\nKolom: {list(df.columns)}")

print("\n" + "=" * 60)
print("PREVIEW DATA (10 baris pertama)")
print("=" * 60)
print(df.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("STATISTIK DESKRIPTIF")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("DISTRIBUSI KATEGORI")
print("=" * 60)
print("\nLayanan:")
print(df['Layanan'].value_counts())
print("\nCuaca:")
print(df['Cuaca'].value_counts())
print("\nWilayah:")
print(df['Wilayah'].value_counts())

print("\n" + "=" * 60)
print("VALIDASI LOGIKA DATA")
print("=" * 60)

# Cek korelasi jarak dengan durasi
correlation = df['Jarak'].corr(df['Durasi'])
print(f"\nKorelasi Jarak vs Durasi: {correlation:.3f}")
print("✅ Korelasi positif tinggi = Makin jauh makin lama (LOGIS!)")

# Cek rata-rata durasi per layanan
print("\nRata-rata Durasi per Layanan:")
for layanan in layanan_options:
    avg = df[df['Layanan'] == layanan]['Durasi'].mean()
    print(f"  {layanan}: {avg:.1f} jam")

# Cek rata-rata durasi per cuaca
print("\nRata-rata Durasi per Cuaca:")
for cuaca in cuaca_options:
    avg = df[df['Cuaca'] == cuaca]['Durasi'].mean()
    print(f"  {cuaca}: {avg:.1f} jam")

print("\n" + "=" * 60)
print("CONTOH PREDIKSI LOGIS")
print("=" * 60)

# Contoh kasus
examples = [
    {'Jarak': 50, 'Layanan': 'Same Day', 'Cuaca': 'Cerah'},
    {'Jarak': 50, 'Layanan': 'Express', 'Cuaca': 'Cerah'},
    {'Jarak': 50, 'Layanan': 'Reguler', 'Cuaca': 'Cerah'},
    {'Jarak': 100, 'Layanan': 'Express', 'Cuaca': 'Hujan'},
    {'Jarak': 150, 'Layanan': 'Reguler', 'Cuaca': 'Cerah'},
]

print("\nBerdasarkan dataset ini, durasi yang wajar:")
for ex in examples:
    matching = df[(df['Jarak'].between(ex['Jarak']-10, ex['Jarak']+10)) & 
                  (df['Layanan'] == ex['Layanan']) & 
                  (df['Cuaca'] == ex['Cuaca'])]
    if len(matching) > 0:
        avg = matching['Durasi'].mean()
        print(f"  Jarak {ex['Jarak']}km, {ex['Layanan']}, {ex['Cuaca']}: ~{avg:.1f} jam")

print("\n" + "=" * 60)
print("SAVE FILE")
print("=" * 60)

# Save to CSV
df.to_csv('pengiriman.csv', index=False)
print("\n✅ Dataset berhasil disimpan ke 'pengiriman.csv'")
print("\n📦 Siap digunakan untuk training model Decision Tree!")
print("=" * 60)

# Tampilkan semua data untuk verifikasi manual
print("\n" + "=" * 60)
print("SEMUA DATA (untuk verifikasi)")
print("=" * 60)
print(df.to_string(index=False))