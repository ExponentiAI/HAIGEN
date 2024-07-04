# HAIGEN
## [Part 01: Image-to-Sketch Generation]

***
A Creative Style Generation Framework for Fashion Design

Our model weights is avialable [**checkpoint**](https://drive.google.com/drive/folders/1-_ts9fbZsR7ZMy6I9fqu_hKV1hho_Ufa?usp=drive_link)

## Data Preparation
For datasets that have paired sketch-image data, the path should be formatted as:
```yaml
./data/rgb/  # training reference (image)
./data/skt/  # training ground truth (sketch)
```
Our Image-to-Sketch Generation dataset is available [**Clothes-V1**](https://drive.google.com/file/d/1dPNQOaFYMRY2sxREs8Is-G3ltpi61Ftm/view?usp=drive_link).


## Train and Test
### Train
```yaml
python main.py
```

### test
```yaml
python evaluate.py.py
```
Note: you should change your checkpoint path.
