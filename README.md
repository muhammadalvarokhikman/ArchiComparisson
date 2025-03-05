# Architecture Comparison for Image Classification

## Overview
Proyek ini bertujuan untuk membandingkan performa berbagai arsitektur deep learning dalam tugas klasifikasi citra. Model yang diuji mencakup CNN berbasis ImageNet, CNN berbasis ResNet, dan Transformer berbasis MLP-Mixer. Perbandingan dilakukan berdasarkan akurasi pelatihan dan validasi.

## Dataset
Dataset yang digunakan dalam proyek ini merupakan kumpulan gambar dari berbagai kategori, yang telah diproses menjadi format yang dapat digunakan oleh model deep learning. Dataset ini telah dibagi menjadi data latih dan validasi untuk mengevaluasi performa model dengan lebih akurat.
Dataset Information

Name: Heart Disease Risk Prediction Dataset <br>
Source: EarlyMed, Vellore Institute of Technology (VIT-AP) <br>
Contributors: Mahatir Ahmed Tusher, Saket Choudary Kongara, Vangapalli Sivamani

Description:

Dataset sintetis yang dirancang untuk memprediksi risiko penyakit jantung berdasarkan gejala, faktor gaya hidup, dan riwayat medis.
Berisi 70.000 sampel dengan fitur biner (Ya/Tidak) dan label risiko penyakit jantung (0: Low Risk, 1: High Risk).
Dataset dibuat menggunakan pendekatan statistik dan berdasarkan referensi dari penelitian serta pedoman medis.

License:

This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
You are free to use, modify, and distribute the dataset, provided you give appropriate credit to the original authors.
<br>Original dataset and documentation available at: https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset/data.

Attribution:

If you use this dataset in research, projects, or publications, please cite the contributors as follows:

Mahatir Ahmed Tusher, Saket Choudary Kongara, Vangapalli Sivamani. Heart Disease Risk Prediction Dataset. EarlyMed, Vellore Institute of Technology (VIT-AP), 2025. Released under CC BY 4.0.


## Models Used
1. **CNN + ImageNet**: Model CNN yang menggunakan pretrained weights dari ImageNet sebagai fitur awal.
2. **CNN + ResNet**: Model CNN berbasis ResNet yang memiliki arsitektur lebih dalam dan dapat menangkap fitur yang lebih kompleks.
3. **Transformer (MLP-Mixer)**: Model berbasis transformer yang menggunakan MLP-Mixer untuk menangkap pola dalam data tanpa menggunakan konvolusi.

## Training & Evaluation
- Model dilatih selama 100 epoch dengan optimasi yang sesuai untuk setiap arsitektur.
- Akurasi pelatihan dan validasi dicatat untuk setiap epoch.
- Grafik perbandingan akurasi digunakan untuk analisis performa.

## Results & Conclusion
Hasil perbandingan model dapat disimpulkan sebagai berikut:
<img height="300" src="C:\Users\alvar\OneDrive\Desktop\ArchiComparisson\result\Figure_4.png" width="500"/>
- **CNN + ResNet** menunjukkan performa terbaik dalam generalisasi dengan akurasi validasi yang stabil.
- **CNN + ImageNet** memiliki performa yang cukup baik tetapi lebih fluktuatif dibandingkan ResNet.
- **Transformer (MLP-Mixer)** mencapai akurasi pelatihan yang hampir sempurna (~1.0), tetapi akurasi validasinya lebih rendah dan fluktuatif, yang mengindikasikan overfitting.

Jika fokus utama adalah generalisasi yang baik, **CNN + ResNet** menjadi pilihan optimal. Jika dataset lebih besar, Transformer bisa lebih unggul dengan teknik regulasi tambahan.

## How to Run
1. Clone repository ini:
   ```bash
   git clone https://github.com/muhammadalvarokhikman/ArchiComparisson.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan skrip utama:
   ```bash
   python main.py
   ```

## Author
Project ini dikembangkan oleh **Alvaro**, seorang Machine Learning Engineer yang suka melakukan eksperimen dalam Machine Learning.
