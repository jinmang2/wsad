#!/bin/bash
D="$HOME/data/wsad/ucf_crime/features/i3d_1024_raw"
for z in RGB0 Training_Normal_Videos_Anomaly1 Training_Normal_Videos_Anomaly2; do
  echo "=== unzip $z $(date +%H:%M:%S) ==="
  unzip -oq "$D/train/$z.zip" -d "$D/train_npy/"
done
echo "=== UNZIP DONE $(date +%H:%M:%S); files: $(find $D/train_npy -name '*.npy' | wc -l) ==="
