# MI-EEG-1D-CNN-Pytorch

This is unofficial code of MI-EEG-1D-CNN for pytorch. Please see https://github.com/Kubasinska/MI-EEG-1D-CNN.



## Dataset

Download the EEG Motor Movement/Imagery Dataset [here](https://physionet.org/content/eegmmidb/1.0.0/) or command on terminal `wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/`.

## Training

1. Generate the dataset; generator.py. Change the dataset path to the path of the dataset you downloaded.
2. Run main.py `python main.py`

## Result


## Citation
```bibtex
@article{mattioli20211d,
  title={A 1D CNN for high accuracy classification and transfer learning in motor imagery EEG-based brain-computer interface},
  author={Mattioli, Francesco and Porcaro, Camillo and Baldassarre, Gianluca},
  journal={Journal of Neural Engineering},
  year={2021},
  publisher={IOP Publishing}
}
```
