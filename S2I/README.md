# HAIGEN 
## [Part 02: Sketch-to-Image Style Transfer]

***
A Creative Style Generation Framework for Fashion Design

Our model weights is avialable [**checkpoint**](https://drive.google.com/drive/folders/1DW2O9xIiL_wb4BDz06PflUqSq_n9v-Lf?usp=drive_link)

## Data Preparation
For datasets that have paired sketch-image data, the path should be formatted as:
```yaml
./dataset/trainA/  # training reference (sketch)
./dataset/trainB/  # training ground truth (image)
./dataset/testA/  # testing reference (sketch)
./dataset/testB/  # testing ground truth (image)
```

Our Sketch-to-Image synthesis dataset is available [**HAIFashion**](https://drive.google.com/file/d/1Cy8I92VYnBEgWbpIvLsy5VcYPliJ1PON/view?usp=drive_link).


## Train and Test
### Train
run:
```yaml
python main.py
```

### test
run:
```yaml
python infer.py
```
Note: you should change your checkpoint path.
