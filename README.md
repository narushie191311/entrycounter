# entreecounter

入場カウント・追跡・顔ベスト保存・DeepFace年齢/性別・Excel出力を行うスクリプト。

## セットアップ
```bash
./scripts/setup.sh
```
- ffmpeg は別途インストール（macOS: `brew install ffmpeg`）
- OpenCV DNNモデル（任意）: `models/age_gender/` に caffemodel/prototxt を配置

## 実行例
```bash
source .venv/bin/activate
python entreecounter.py \
  --video path/to/merged.mkv \
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

## オフラインID統合
```bash
python id_merge.py --in output/faces_index.csv --map mapping.csv --out output/faces_index_merged.csv
```
