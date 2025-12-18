<div align="center">
  <h1> RMFAT </h1>
  <h2>Recurrent Multi-scale Feature Atmospheric Turbulence Mitigator [AAAI 2026]</h2>
</div>

<p align="center">Zhiming Liu, Nantheera Anantrasirichai</p>


## 📑 Contents
- [Environment Installation](#environment-installation)
- [My Dataset](#Datasets-prepare)
- [Training](#Training)
- [Performance Evaluation](#Performance)


<h2 id="environment-installation">🔨 Environment Installation</h2>

```shell
conda create -n RMFAT python=3.11
conda activate RMFAT
cd code
pip install -r requirements.txt
```

<h2 id="Datasets-prepare">🧩 My Datasets</h2>

This is a public dataset for turbulence migitation tasks, containing two subsets: **static** and **dynamic**.

### 📦 Download

You can download each dataset subset as a ZIP file below:

- 👉 [Download Static Dataset](https://drive.google.com/file/d/1JZCslKDHqjTHnUowgYlFja_ewejMtZfJ/view?usp=drive_link)  
- 👉 [Download Dynamic Dataset](https://drive.google.com/file/d/1RZeK-e-MfPt56feIUsowtr_oKVLFqH3J/view?usp=drive_link)

<h2 id="Training">🛠️ Training</h2>
For the training on dynamic scene data, run the following:

```
python recursive_train.py --train_path "/path/to/dynamic/train/data" --val_path "/path/to/dynamic/val/data" --batch_size 1 --patch_size 256 --num_frames 10 --tmt_dims 16 --log_path "/path/to/save/logs/dynamic" run_name "train_dynamic" --resume_ckpt "/path/to/dynamic/checkpoint.pth"
```

For the training on static scene data, run the following:
```
python recursive_train.py --train_path "/path/to/static/train/data" --val_path "/path/to/static/val/data" --batch_size 1 --patch_size 256 --num_frames 10 --tmt_dims 16 --log_path "/path/to/save/logs/dynamic" --run_name "train_dynamic" --resume_ckpt "/path/to/dynamic/checkpoint.pth"
```

<h2 id="Performance">🚀 Performance Evaluation</h2>
For the single video inference, run the following:

```
python restore_video.py --video_path /path/to/video.mp4 --output_dir /path/to/output --restoration_ckpt /path/to/model.pth --tmt_dim 32 --warp_mode enc --n_frames 3 --save_raw
```

<h2 id="Citation">Citation</h2>

```shell
@inproceedings{liu2026rmfat,
  title     = {{RMFAT}: Recurrent Multi-scale Feature Atmospheric Turbulence Mitigator},
  author    = {Liu, Zhiming and Anantrasirichai, Nantheera},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```


