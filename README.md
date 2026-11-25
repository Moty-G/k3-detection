# Sistem Deteksi Keselamatan Kerja (K3) Konstruksi

Aplikasi Streamlit satu halaman untuk menilai pekerja **Aman** / **Tidak Aman** berdasarkan helm dan rompi keselamatan, dengan konfirmasi temporal dan cooldown per ID pelacakan pada video.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur

- `app.py` — antarmuka Streamlit
- `core/config.py` — konstanta & nilai default (Single Source of Truth)
- `core/detector.py` — muat model & inferensi/tracking CPU
- `core/rules.py` — evaluasi APD, filter temporal, cooldown
- `core/processor.py` — pipeline gambar & video H.264
- `models/best.pt` — model deteksi (kelas `person`, `helmet`, `safety-vest` dipakai)

Dataset: *Personal Protective Equipment — Combined Model* (Roboflow Universe, CC BY 4.0).
