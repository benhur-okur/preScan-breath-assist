# preScan-breath-assist

IMU (accelerometer) + 2 kamera (front/side) kayıtlarını **clap ile senkronlayıp**, IMU’dan **hold/breathing label** üretip, videoları **window-level** dataset formatına paketleyen proje.

> Not: Kod tabanı `moviepy.editor` import yolunu kullanır. Bu nedenle **MoviePy==1.0.3** zorunludur (MoviePy 2.x ile `moviepy.editor` bulunmayabilir).

---

## Repo Yapısı

data/
raw/
imu_zip/ # orijinal Sensor Logger zip’leri
videos/ # data/raw/videos/<Person>/<person><caseIdx><front|side>.mp4
synced/ # sync_manager outputları
labeled_safe/ # label_from_imu + clamp sonrası güvenli labeled csv’ler
windows/ # window-level manifest.csv

tools/
sync_manager.py
analyze_synced.py
label_from_imu.py
qa_frames_in_range_batch.py
clamp_frames.py
window_dataset_builder.py

markdown
Kodu kopyala

**Video isim standardı:**
- `caseIdx`: 1=normal, 2=inhale_hold, 3=exhale_hold, 4=irregular
- Örn: `data/raw/videos/Sinan/sinan_2_front.mp4`, `sinan_2_side.mp4`

---

## Ortam Kurulumu

Bu repo **pip-tools** ile yönetilir:
- `requirements.in` → top-level bağımlılıklar
- `requirements.lock.windows.txt` / `requirements.lock.macos.txt` → platform lock dosyaları

### Windows (GTX 4060)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.lock.windows.txt
macOS (M1)
bash
Kodu kopyala
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.lock.macos.txt
Hızlı Bağımlılık Testi
bash
Kodu kopyala
python -c "import numpy,pandas,cv2,matplotlib; import moviepy.editor as mp; print('deps OK')"
Bağımlılık / Lock Güncelleme (pip-tools)
Lock dosyalarını elle editlemeyin. Sadece requirements.in güncellenir.

Windows’ta:

powershell
Kodu kopyala
pip install pip-tools
pip-compile requirements.in -o requirements.lock.windows.txt
pip-sync requirements.lock.windows.txt
macOS’ta:

bash
Kodu kopyala
pip install pip-tools
pip-compile requirements.in -o requirements.lock.macos.txt
pip-sync requirements.lock.macos.txt
Pipeline (Özet)
1) Sync (IMU + Front + Side)
IMU ve videolarda iki clap arası (start/end) segmenti alır ve zaman eksenlerini map’ler.

bash
Kodu kopyala
python tools/sync_manager.py \
  --imu data/raw/imu_zip/<rec>.zip \
  --front data/raw/videos/<Person>/<person>_<idx>_front.mp4 \
  --side  data/raw/videos/<Person>/<person>_<idx>_side.mp4 \
  --out data/synced/<person>_<case>_synced.csv
2) QA: Senkron Kontrol
bash
Kodu kopyala
python tools/analyze_synced.py data/synced/<file>.csv --case-type normal
3) IMU’dan Label Üretimi (hold=1 / breathing=0)
bash
Kodu kopyala
python tools/label_from_imu.py data/synced/<file>.csv --case-type inhale_hold --plot
4) Frame QA + Clamp (güvenlik)
Önce frame index’ler video range içinde mi kontrol edilir, sonra gerekiyorsa clamp yapılır.

bash
Kodu kopyala
python tools/qa_frames_in_range_batch.py \
  --labeled-dir data/labeled_safe \
  --videos-root data/raw/videos

python tools/clamp_frames.py ...
5) Window-level Dataset Manifest Üretimi
30 FPS için default:

window = 0.5s (15 frame)

stride = 0.25s (~8 frame)

bash
Kodu kopyala
python tools/window_dataset_builder.py \
  --labeled-dir data/labeled_safe \
  --videos-root data/raw/videos \
  --out data/windows/manifest.csv \
  --window-sec 0.5 \
  --stride-sec 0.25 \
  --min-coverage 0.60
Manifest satırı bir window’u tanımlar:

video path (front/side)

start/end frame

label (0/1)

fps, window/stride metadata

video duration / total frames (frame taşması riskini azaltmak için)
```

### PyThorch 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


 python -m tools.predict_video `
>>   --manifest-csv runs\video_demo\manifest_one_video.csv `                                                                                                                        
>>   --pooling max `                                                                                                                                                                
>>   --ckpt-dir runs\loso_max_diff_final `                                                                                                                                          
>>   --threshold 0.35 `                                                                                                                                                             
>>   --min-hold-sec 1.0 `                                                                                                                                                           
>>   --out-dir runs\video_demo/preds_thr035_min1s `                                                                                                                                 
>>   --plot                                                            

python -c "import pandas as pd; df=pd.read_csv('runs/video_demo/pred_thr_042/predictions.csv'); print('thr=0.42 hold_windows=',(df.pred_hold==1).sum());"

runs\loso_max_diff_final\fold_01_fold_1_test_benhur\best.pt `

python -m tools.make_windows_for_video `
  --video "data\raw\Videos/Benhur/benhur_2_front.mp4" `
  --out-csv "runs\video_demo\manifest_hold_video.csv" `
  --window-frames 15 `
  --stride-frames 5


Yani predict komutunu standardize et:

python -m tools.predict_video `
  --manifest-csv "runs\video_demo\manifest_one_video.csv" `
  --ckpt-dir "runs\loso_max_diff_final" `
  --pooling max `
  --threshold 0.25 `
  --min-hold-sec 1.0 `
  --out-dir "runs\video_demo\preds_thr025_min1s" `
  --plot


python -m tools.choose_global_threshold `
  --manifest-final data\windows\manifest_final_train.csv `
  --final-fold-id final `
  --ckpt runs\final_model\final_best.pt `
  --split val `
  --pooling max `
  --out runs\final_model\final_threshold.json


python -m tools.make_windows_for_video `
  --video "data\raw\Videos/Ecenaz/ecenaz_3_front.mp4" `
  --out-csv "runs\final_demo\ecenaz_exhale_hold_manifest.csv" `
  --window-frames 15 `
  --stride-frames 5


python -m tools.predict_video `
  --manifest-csv runs/final_demo/ecenaz_exhale_hold_manifest.csv `
  --ckpt "runs/final_model_v4_rgb_bce/final_best.pt" `
  --pooling max `
  --input-mode rgb `
  --threshold 0.90 `
  --min-hold-sec 2.0 `
  --out-dir runs/final_demo/v4_ecenaz_exhale_hold_thr090_min2s `
  --plot

python -m tools.make_windows_for_video `
  --video "data\raw\Videos/Doga/doga_2_front.mp4" `
  --window-frames 15 `
  --stride-frames 5 `
  --out-csv "data/windows/manifest_infer_doga_2.csv" `
  --fold-id "inference" `
  --split "test"

python -m tools.eval_single_video_from_manifest `
  --manifest "data/windows/manifest_final_train_v3.csv" `
  --video "doga" `
  --ckpt "runs/final_model_v4_rgb_bce/final_best.pt" `
  --input-mode "rgb" `
  --threshold 0.90 `
  --min-consecutive 13 `
  --plot `
  --out-dir "runs/debug_doga/train_windows"

python -m tools.compare_windowing_train_vs_infer `
  --train-manifest "data/windows/manifest_final_train_v3.csv" `
  --infer-manifest "data/windows/manifest_infer_doga_2.csv" `
  --video "doga"


