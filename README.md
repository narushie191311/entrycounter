# entreecounter

入場カウント・追跡・顔ベスト保存・DeepFace年齢/性別・Excel出力を行うスクリプト。

## セットアップ
```bash
./scripts/setup.sh
```
- ffmpeg は別途インストール（macOS: `brew install ffmpeg`）
- OpenCV DNNモデル（任意）: `models/age_gender/` に caffemodel/prototxt を配置

## 実行例
GPU自動切替（CUDA/MPS/CPU）に対応。Linux+NVIDIAならCUDA、Apple SiliconならMPSを自動選択。

Google Driveのテスト動画を取得:
```bash
bash scripts/download_gdrive.sh input.mkv
```

実行:
```bash
python entreecounter.py \
  --video input.mkv \
  --outdir output \
  --model yolov8n.pt --conf 0.35 --cam-id camA \
  --df-warmup-frames 2 --df-reinfer-interval 60 \
  --autosave-sec 20 \
  --log-detections output/detections.csv
```

## 出力
- entries_detail.csv / counts_15min.csv / stays.csv / overlay.mp4
- faces/*, faces_index_live.csv / faces_index.csv / faces_embeddings.npz
- faces_sheet.jpg / faces_report.xlsx
  - FaceSheetは入退場時刻を分まで（秒切捨て）で2行表示

## オフラインID統合
```bash
python id_merge.py --in output/faces_index.csv --map mapping.csv --out output/faces_index_merged.csv
```
