# Open-rppg

Open-rppg is a comprehensive Python toolbox designed for Remote Photoplethysmography (rPPG) inference. It provides a unified interface for state-of-the-art deep learning models, enabling physiological signal measurement (such as heart rate and heart rate variability) from facial videos. The toolkit supports both offline video processing and low-latency real-time inference using JAX and Keras.

## Installation

This package requires Python 3.9 through 3.13.

To install the standard version:

```bash
pip install open-rppg
```

To enable GPU acceleration (Linux/CUDA), install the CUDA-supported version of JAX:

```bash
pip install jax[cuda]
```

## Quick Start

The core interface is managed through the `Model` class. By default, it initializes a robust, general-purpose model (`FacePhys.rlap`).

```python
import rppg

# Initialize the model
model = rppg.Model()

# Process a video file
results = model.process_video("path/to/video.mkv")

# Display the heart rate
print(f"Estimated Heart Rate: {results['hr']} BPM")
```

## Usage Guide

### 1. Offline Video Processing
To analyze a pre-recorded video file, use the `process_video` method. This method handles frame extraction, face detection, and signal inference automatically.

```python
results = model.process_video("subject_test.mkv")
```

**Output Structure:**
The returned dictionary contains the following keys:

* **hr**: Heart Rate estimated via the frequency domain (FFT).
* **SQI**: Signal Quality Index (0.0 to 1.0), indicating the reliability of the measurement.
* **latency**: Inference latency (primarily relevant for real-time streams).
* **hrv**: A dictionary of Heart Rate Variability metrics calculated in the time domain:
    * *bpm*: Heart rate derived from peak detection.
    * *ibi*: Inter-Beat Interval (milliseconds).
    * *sdnn*: Standard deviation of NN intervals.
    * *rmssd*: Root mean square of successive differences.
    * *pnn50*: Proportion of NN50 > 50ms.
    * *LF/HF*: Ratio of Low Frequency to High Frequency power.
    * *breathingrate*: Estimated respiration rate.

### 2. Real-Time Inference
Open-rppg includes a threaded pipeline optimized for real-time webcam inference. Use the `video_capture` context manager to handle the video stream safely.

```python
import rppg
import time
import cv2

model = rppg.Model()
# Open the default camera (index 0)
with model.video_capture(0):
    last_process_time = 0
    current_hr = None
    
    # Iterate through the preview generator (this is the main loop)
    for frame, box in model.preview:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 1. Calculate HR every 1 second to avoid lag
        now = time.time()
        if now - last_process_time > 1.0:
            result = model.hr(start=-10)
            if result and result['hr']:
                current_hr = result['hr']
                print(f"Real-time HR: {current_hr:.1f} BPM")
            last_process_time = now
            
        # 2. Visualization
        if box is not None:
            # box format: [[row_min, row_max], [col_min, col_max]]
            y1, y2 = box[0]
            x1, x2 = box[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display HR on the frame if available
            if current_hr is not None:
                cv2.putText(frame, f"HR: {current_hr:.1f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("rPPG Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

## Advanced API

### Retrieving Raw Signals
You can extract the underlying Blood Volume Pulse (BVP) waveform for further analysis or plotting.

```python
# Retrieve the full BVP signal and corresponding timestamps
bvp, timestamps = model.bvp()

# Retrieve the raw, unfiltered BVP signal
raw_bvp, timestamps = model.bvp(raw=True)
```

### Time Slicing
The toolbox allows specific time windows to be analyzed within the buffer.

```python
# Get signal from t=10s to t=20s
bvp_slice, ts_slice = model.bvp(start=10, end=20)

# Get metrics for the last 15 seconds
metrics = model.hr(start=-15)
```

### Tensor Inputs
For integration into existing pipelines where frames are already loaded as memory arrays, use the tensor processing methods.

* **Input Format:** `uint8` array with shape `(Frames, Height, Width, 3)`.

```python
import numpy as np

# tensor shape: (T, H, W, 3)
video_tensor = np.zeros((300, 480, 640, 3), dtype='uint8') # 480p video

result = model.process_video_tensor(video_tensor, fps=30.0)

faces_tensor = np.zeros((300, 128, 128, 3), dtype='uint8') # face array

result = model.process_faces_tensor(faces_tensor, fps=30.0)

```

### Model Selection
You can specify different architectures during initialization. The models are categorized by architecture and training configuration (`rlap` or `pure`).

```python
# Example: Initialize the PhysMamba model
model = rppg.Model('PhysMamba.pure')
```

## Model Zoo

The following architectures are supported. 

| Model Name | Description | Reference |
| :--- | :--- | :--- |
| **ME-chunk** | State-space model rPPG (chunk inference) | arXiv 2025 |
| **ME-flow** | State-space model rPPG (low-latency flow) | arXiv 2025 |
| **PhysMamba** | Dual-branch Mamba architecture | CCBR 2024 |
| **RhythmMamba**| Frequency-domain constrained Mamba | AAAI 2025 |
| **PhysFormer** | Temporal Difference Transformer | CVPR 2022 |
| **TSCAN** | Temporal Shift Convolutional Attention Network | NeurIPS 2020 |
| **EfficientPhys**| Self-attention variant of TSCAN | WACV 2023 |
| **PhysNet** | 3D Convolutional Encoder-Decoder | BMVC 2019 |
| **FacePhys** | Optimized state-space model | - |

*Note: Suffixes `.rlap` and `.pure` indicate different training protocols/weights.*

## Licensing

The source code and tools in this repository are released under the **MIT License**.

**Important:** Pretrained models and model configurations provided in this repository are derived from academic research. They are the intellectual property of their respective authors and are subject to the license terms specified in their original publications. Please refer to the citations below for details.

## Citation

If you use this toolkit or the included models in your research, please cite the relevant papers:

```bibtex
@article{yu2019remote,
  title={Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks},
  author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
  journal={arXiv preprint arXiv:1905.02419},
  year={2019}
}

@article{liu2020multi,
  title={Multi-task temporal shift attention networks for on-device contactless vitals measurement},
  author={Liu, Xin and Fromm, Josh and Patel, Shwetak and McDuff, Daniel},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={19400--19411},
  year={2020}
}

@inproceedings{liu2023efficientphys,
  title={Efficientphys: Enabling simple, fast and accurate camera-based cardiac measurement},
  author={Liu, Xin and Hill, Brian and Jiang, Ziheng and Patel, Shwetak and McDuff, Daniel},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={5008--5017},
  year={2023}
}

@inproceedings{yu2022physformer,
  title={Physformer: Facial video-based physiological measurement with temporal difference transformer},
  author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip HS and Zhao, Guoying},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4186--4196},
  year={2022}
}

@inproceedings{luo2024physmamba,
  title={PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba},
  author={Luo, Chaoqi and Xie, Yiping and Yu, Zitong},
  booktitle={Chinese Conference on Biometric Recognition},
  pages={248--259},
  year={2024},
  organization={Springer}
}

@inproceedings{zou2025rhythmmamba,
  title={RhythmMamba: Fast, Lightweight, and Accurate Remote Physiological Measurement},
  author={Zou, Bochao and Guo, Zizheng and Hu, Xiaocheng and Ma, Huimin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={11077--11085},
  year={2025}
}

@article{wang2025memory,
  title={Memory-efficient Low-latency Remote Photoplethysmography through Temporal-Spatial State Space Duality},
  author={Wang, Kegang and Tang, Jiankai and Fan, Yuxuan and Ji, Jiatong and Shi, Yuanchun and Wang, Yuntao},
  journal={arXiv preprint arXiv:2504.01774},
  year={2025}
}
```
