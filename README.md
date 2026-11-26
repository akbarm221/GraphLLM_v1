# Chatbot WDB di Streamlit Community Cloud

Aplikasi membaca skema dan data dari Neo4j online. Azure OpenAI menyusun kueri Cypher dan merangkai jawaban dari hasil baca. Setiap pertanyaan diproses sendiri. Riwayat dipakai untuk tampilan percakapan.

## Deploy

1. Ekstrak ZIP dan unggah `app.py`, `requirements.txt`, `.gitignore`, `secrets.example.toml`, dan `README.md` ke satu repositori GitHub. Untuk pengujian dengan karyawan, gunakan repositori privat dan pastikan pengaturan berbagi aplikasi juga privat.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io/), pilih **Create app**, lalu pilih repositori dan branch yang kamu unggah. Isi **Main file path** dengan `app.py`.
3. Buka **Advanced settings**. Pilih Python **3.12** dan salin isi `secrets.example.toml` ke **Secrets**. Ganti nilai contoh dengan konfigurasi Neo4j dan Azure OpenAI yang sudah berjalan secara lokal. Untuk `AZURE_OPENAI_API_VERSION`, salin nilai dari konfigurasi lokal jika ada. Biarkan nama variabel berada di tingkat atas TOML.
4. Klik **Deploy**. Setelah muncul pesan **Terhubung ke Neo4j**, coba pertanyaan **“Status kawasan apa yang tercatat untuk Taman Nasional Berbak?”**. Untuk pengujian versi privat, undang peserta melalui menu **Share**.

Repo deploy cukup berisi aplikasi dan dependensinya. Simpan nilai `.env`, `.streamlit/secrets.toml`, dan kredensial di luar GitHub. Jika paket Neo4j mendukung pengaturan hak akses, gunakan akun aplikasi dengan hak baca data dan skema.

## Uji lokal di PowerShell

Letakkan `.env` yang biasa kamu gunakan di folder `app.py`, lalu jalankan perintah ini dari folder tersebut.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Panel kueri, skema, dan hasil mentah disembunyikan secara bawaan. Untuk pemeriksaan lokal, setel `WDB_SHOW_DEBUG=1` di `.env` dan jalankan ulang aplikasi. Saat menggunakan aplikasi untuk peserta, pertahankan `WDB_SHOW_DEBUG` pada nilai `0`.
