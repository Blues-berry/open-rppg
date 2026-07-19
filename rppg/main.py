import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import logging
logger = logging.getLogger('open-rppg')
handler = logging.StreamHandler()
formatter = logging.Formatter('OPEN-RPPG:%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from .models import * 

from functools import partial
import onnxruntime as ort
import av
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import pkg_resources
from .signal import SQI, bandpass_filter, detrend, get_hr, get_prv, norm_bvp

def validate_param(**kw):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            all_args = bound_args.arguments
            for param_name, value in all_args.items():
                if param_name in kw:
                    allowed = kw[param_name]
                    if value not in allowed:
                        raise ValueError(
                            f"Invalid value for '{param_name}': {value}. "
                            f"Allowed values: {allowed}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

class KalmanFilter1D:
    def __init__(self, process_noise, measurement_noise, initial_state, initial_estimate_error, reference_interval=1/30):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate = initial_state
        self.estimate_error = initial_estimate_error
        self.reference_interval = reference_interval
    
    def update(self, measurement, dt=None):
        if dt is None:
            dt = self.reference_interval
        time_scale = dt / self.reference_interval
        adjusted_process_noise = self.process_noise * (time_scale ** 2)
        prediction = self.estimate
        prediction_error = self.estimate_error + adjusted_process_noise
        kalman_gain = prediction_error / (prediction_error + self.measurement_noise)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        return self.estimate

class FaceDetector:
    def __init__(self, model_path, score_threshold=0.5, iou_threshold=0.3):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = input_shape[2]
        
        if self.input_size == 128:
            self.strides = [8, 16, 16, 16]
            self.anchor_offset = 0.5
            self.anchors = self._generate_anchors_short()
        else:
            self.strides = [4]
            self.anchor_offset = 0.5
            self.num_layers = 1
            self.interpolated_scale_aspect_ratio = 0.0
            self.anchors = self._generate_anchors_full()
        
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
    
    def _generate_anchors_short(self):
        anchors = []
        for stride in self.strides:
            feature_map_size = self.input_size // stride
            for y in range(feature_map_size):
                for x in range(feature_map_size):
                    for _ in range(2):
                        x_center = (x + self.anchor_offset) / feature_map_size
                        y_center = (y + self.anchor_offset) / feature_map_size
                        anchors.append([x_center, y_center])
        return np.array(anchors, dtype=np.float32)
    
    def _generate_anchors_full(self):
        anchors = []
        layer_id = 0
        
        while layer_id < self.num_layers:
            last_same_stride_layer = layer_id
            repeats = 0
            
            while (last_same_stride_layer < self.num_layers and 
                   self.strides[last_same_stride_layer] == self.strides[layer_id]):
                last_same_stride_layer += 1
                repeats += 2 if self.interpolated_scale_aspect_ratio == 1.0 else 1
            
            stride = self.strides[layer_id]
            feature_map_height = self.input_size // stride
            feature_map_width = self.input_size // stride
            
            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    y_center = (y + self.anchor_offset) / feature_map_height
                    x_center = (x + self.anchor_offset) / feature_map_width
                    
                    for _ in range(repeats):
                        anchors.append([x_center, y_center])
            
            layer_id = last_same_stride_layer
        
        return np.array(anchors, dtype=np.float32)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))
    
    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.input_size, self.input_size, 3), 0, dtype=np.uint8)
        y_offset = (self.input_size - new_h) // 2
        x_offset = (self.input_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        normalized = (padded.astype(np.float32) / 255.0 - 0.5) / 0.5
        return normalized[None,], (x_offset, y_offset, new_w, new_h, scale)
    
    def _decode_boxes(self, raw_boxes, valid_indices):
        num_boxes = len(valid_indices)
        boxes_xyxy = np.zeros((num_boxes, 4))
        
        for i, idx in enumerate(valid_indices):
            anchor = self.anchors[idx]
            
            dx = raw_boxes[idx, 0] / self.input_size
            dy = raw_boxes[idx, 1] / self.input_size
            dw = raw_boxes[idx, 2] / self.input_size
            dh = raw_boxes[idx, 3] / self.input_size
            
            cx = dx + anchor[0]
            cy = dy + anchor[1]
            
            w = dw
            h = dh
            
            xmin = cx - w*0.45
            ymin = cy - h*0.6
            xmax = cx + w*0.45
            ymax = cy + h*0.5
            
            boxes_xyxy[i] = [xmin, ymin, xmax, ymax]
        
        return boxes_xyxy
    
    def _decode_keypoints(self, raw_boxes, valid_indices):
        keypoints_list = []
        
        for idx in valid_indices:
            anchor = self.anchors[idx]
            keypoints = np.zeros((6, 2))
            
            for k in range(6):
                kx = raw_boxes[idx, 4 + 2*k] / self.input_size + anchor[0]
                ky = raw_boxes[idx, 4 + 2*k + 1] / self.input_size + anchor[1]
                keypoints[k] = [kx, ky]
            
            keypoints_list.append(keypoints)
        return keypoints_list
    
    def _nms(self, boxes, scores):
        if len(boxes) == 0:
            return []
        
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]
        
        selected = []
        
        while len(order) > 0:
            i = order[0, 0]
            selected.append(i)
            
            if len(order) == 1:
                break
            
            current_box = boxes[0]
            other_boxes = boxes[1:]
            
            x1 = np.maximum(current_box[0,0], other_boxes[:,0, 0])
            y1 = np.maximum(current_box[0,1], other_boxes[:,0, 1])
            x2 = np.minimum(current_box[0,2], other_boxes[:,0, 2])
            y2 = np.minimum(current_box[0,3], other_boxes[:,0, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            current_box = current_box[0]
            other_boxes = other_boxes[0]
            area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            union = area_current + area_others - intersection
            
            iou = intersection / (union + 1e-8)
            
            keep_indices = np.where(iou <= self.iou_threshold)[0]
            order = order[keep_indices + 1]
            boxes = boxes[keep_indices + 1]
            scores = scores[keep_indices + 1]
        
        return selected
    
    def _remove_padding(self, detections, padding_info):
        x_offset, y_offset, new_w, new_h, scale = padding_info
        
        results = []
        for bbox, keypoints, score in detections:
            xmin = (bbox[0] * self.input_size - x_offset) / scale
            ymin = (bbox[1] * self.input_size - y_offset) / scale
            xmax = (bbox[2] * self.input_size - x_offset) / scale
            ymax = (bbox[3] * self.input_size - y_offset) / scale
            
            bbox_relative = np.array([xmin, ymin, xmax, ymax])
            
            keypoints_relative = np.zeros_like(keypoints)
            for i, kp in enumerate(keypoints):
                kx = (kp[0] * self.input_size - x_offset) / scale
                ky = (kp[1] * self.input_size - y_offset) / scale
                keypoints_relative[i] = [kx, ky]
            
            results.append((bbox_relative, keypoints_relative, score))
        
        return results
    
    def detect(self, image):
        input_tensor, padding_info = self.preprocess(image)
        
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        raw_boxes, raw_scores = outputs
        
        raw_boxes = raw_boxes[0] 
        raw_scores = raw_scores[0]
        
        scores = self._sigmoid(raw_scores)
        
        valid_indices = np.where(scores > self.score_threshold)[0]
        if len(valid_indices) == 0:
            return []
        
        valid_scores = scores[valid_indices]
        
        boxes_xyxy = self._decode_boxes(raw_boxes, valid_indices)
        keypoints_list = self._decode_keypoints(raw_boxes, valid_indices)
        
        selected_indices = self._nms(boxes_xyxy, valid_scores)
        results = []
        for idx in selected_indices:
            results.append((
                boxes_xyxy[idx],
                keypoints_list[idx],
                valid_scores[idx]
            ))
        
        results = self._remove_padding(results, padding_info)
        
        return results

supported_models = ['FacePhys.rlap', 'ME-chunk.rlap', 'ME-flow.rlap', 'ME-chunk.pure', 'ME-flow.pure',
                    'PhysMamba.pure', 'PhysMamba.rlap', 'RhythmMamba.rlap', 'RhythmMamba.pure',
                    'PhysFormer.pure', 'PhysFormer.rlap', 'TSCAN.rlap', 'TSCAN.pure',
                    'PhysNet.rlap', 'PhysNet.pure', 'EfficientPhys.pure', 'EfficientPhys.rlap']

class Model:
    
    @validate_param(model=supported_models)
    def __init__(self, model='FacePhys.rlap'):
        if model == 'FacePhys.rlap':
            f, state, meta = load_FacePhys_rlap()
        if model == 'ME-chunk.rlap':
            f, state, meta = load_ME_chunk_rlap()
        if model == 'ME-chunk.pure':
            f, state, meta = load_ME_chunk_pure()
        if model == 'ME-flow.rlap':
            f, state, meta = load_ME_rlap()
        if model == 'ME-flow.pure':
            f, state, meta = load_ME_pure()
        if model == 'PhysMamba.pure':
            f, state, meta = load_PhysMamba_pure()
        if model == 'PhysMamba.rlap':
            f, state, meta = load_PhysMamba_rlap()
        if model == 'RhythmMamba.rlap':
            f, state, meta = load_RhythmMamba_rlap()
        if model == 'RhythmMamba.pure':
            f, state, meta = load_RhythmMamba_pure()
        if model == 'PhysFormer.rlap':
            f, state, meta = load_PhysFormer_rlap()
        if model == 'PhysFormer.pure':
            f, state, meta = load_PhysFormer_pure()
        if model == 'TSCAN.rlap':
            f, state, meta = load_TSCAN_rlap()
        if model == 'TSCAN.pure':
            f, state, meta = load_TSCAN_pure()
        if model == 'PhysNet.rlap':
            f, state, meta = load_PhysNet_rlap()
        if model == 'PhysNet.pure':
            f, state, meta = load_PhysNet_pure()
        if model == 'EfficientPhys.rlap':
            f, state, meta = load_EfficientPhys_rlap()
        if model == 'EfficientPhys.pure':
            f, state, meta = load_EfficientPhys_pure()
        self.__load(f, state, meta)
        with self:
            pass
    
    def __load(self, func, state, meta):
        self.state = state 
        self.meta = meta 
        self.fps = meta['fps'] 
        self.input = meta['input'] 
        self.face_mode = 'Near'
        self.face_detection_threads = max(os.cpu_count()//2, 1)
        self.face_resampling_threads = max(os.cpu_count()//2, 1)
        self.face_detect_per_n = 5
        self.call = func 
        self.run = None 
        self.frame = None 
        self.alive = False
        self.frame_buffer_size = 10
        self.preview_lock = threading.Lock()
        self.preview_lock.acquire()
    
    def __enter__(self):
        if self.alive:
            raise RuntimeError('A task is currently running!')
        self.boxkf = None
        self.ts = []
        self.n_frame = 0 
        self.n_signal = 0
        self.box = None 
        self.rbox = None 
        self.hasface = 0
        self.face_buff = []
        self.signal_buff = {}
        self.statistic = {'frames':0, 'key':0, 'skipped':0, 'filled':0, 'dependent':0, 'null':0}
        self.sp = threading.Semaphore(0)
        self.frame_lock = threading.Lock()
        self.face_detection_semaphore = threading.Semaphore(self.face_detection_threads+self.frame_buffer_size)
        self.face_detection_pool = ThreadPoolExecutor(max_workers=self.face_detection_threads)
        self.face_resampling_pool = ThreadPoolExecutor(max_workers=self.face_resampling_threads)
        self.face_detection_task = None
        self.face_detection_chain_lock = None
        self.face_detection_chain_ts = 0
        self.face_detect_count = 0
        if self.face_mode == 'Near':
            self.detector = FaceDetector(pkg_resources.resource_filename('rppg','weights/blaze_face.onnx'))
        else:
            self.detector = FaceDetector(pkg_resources.resource_filename('rppg','weights/blaze_face_full.onnx'))
        self.alive = True
        def inference():
            try:
                while self.alive or (len(self.face_buff)>=self.meta['input'][0]):
                    self.sp.acquire()
                    if len(self.face_buff)<self.meta['input'][0]:
                        continue 
                    face_imgs = self.face_buff[:self.meta['input'][0]]
                    face_imgs = list(self.face_resampling_pool.map(lambda x:(x[0](), x[1]), face_imgs))
                    ipt = np.array([i[0] for i in face_imgs])
                    msk = np.array([i[1] for i in face_imgs], bool)
                    r, self.state = self.call(ipt, self.state)
                    r = {k:v*msk for k,v in r.items() if v.shape[-1]==len(msk)}
                    with self.frame_lock:
                        for i in range(self.meta['input'][0]):
                            self.face_buff.pop(0)
                    self.n_signal += self.meta['input'][0]
                    if not self.signal_buff:
                        self.signal_buff = {k:list(v) for k,v in r.items()}
                    else:
                        for k, v in self.signal_buff.items():
                            v.extend(r[k])
                if len(self.face_buff):
                    face_imgs = self.face_buff + [self.face_buff[-1]]*(self.meta['input'][0]-len(self.face_buff))
                    face_imgs = list(self.face_resampling_pool.map(lambda x:(x[0](), x[1]), face_imgs))
                    ipt = np.array([i[0] for i in face_imgs])
                    msk = np.array([i[1] for i in face_imgs], bool)
                    r, _ = self.call(ipt, self.state)
                    r = {k:v*msk for k,v in r.items() if v.shape[-1]==len(msk)}
                    self.n_signal += len(self.face_buff)
                    for k, v in self.signal_buff.items():
                        v.extend(r[k][:len(self.face_buff)])
                    self.face_buff.clear()
            except Exception as e:
                import sys
                sys.excepthook(*sys.exc_info())
        self.ift = threading.Thread(target=inference,daemon=True)
        self.ift.start()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type:
                raise exc_value
        finally:
            if self.face_detection_task is not None:
                self.face_detection_task.result()
            self.alive = False
            self.sp.release()
            self.ift.join()
            self.wait_completion()
            self.face_detection_pool.shutdown()
                
    def collect_signals(self, start=None, end=None):
        if not start:
            start = 0 
        if not end:
            end = self.now
        if start<0:
            start += self.now
        if end<0:
            end += self.now
        if not self.signal_buff:
            return {}, None
        if not start<=end:
            raise ValueError('Start must be less than end')
        signals = self.signal_buff
        start_n, end_n = 0, None
        for n, i in enumerate(reversed(self.ts)):
            i -= self.ts[0]
            n = len(self.ts)-n-1
            if start and i<start:
                start_n = n 
                break 
            if end and i<=end and not end_n: 
                end_n = n+1
        signals = {k:v[start_n:end_n] for k,v in signals.items()}
        ts = np.array(self.ts[start_n:end_n])
        return signals, ts - self.ts[0]
    
    @property
    def now(self):
        return self.ts[self.n_signal-1]-self.ts[0] if self.ts else 0
    
    @property
    def latency(self):
        return self.ts[-1]-self.ts[self.n_signal-1] if self.ts else 0
    
    @property
    def has_signal(self):
        return bool(self.n_signal)
    
    def process_bvp(self, bvp):
        try:
            bvp = bandpass_filter(bvp, fs=self.fps)
            bvp = norm_bvp(bvp, sr=self.fps)
            return bvp
        except Exception as exc:
            logger.debug("BVP filtering skipped: %s", exc)
            return bvp 
    
    @property
    def video_statistic(self):
        data = self.statistic
        return f"Total Frames: {data['frames']}\nKey Frames: {data['key']}\nNon-Key Frames: {data['dependent']}\nSkipped Frames: {data['skipped']}\nForward Filled Frames: {data['filled']}\nNo Face Detected Frames: {data['null']}"
        
    def bvp(self, start=0, end=None, raw=False):
        signals, ts = self.collect_signals(start, end)
        if 'bvp' not in signals or ts[-1]-ts[0]<2:
            return [], []
        bvp = signals['bvp']
        if len(bvp)<self.fps*2:
            return [], []
        if self.meta.get('cumsum_output'):
            bvp = np.cumsum(bvp)
            bvp = detrend(bvp, sr=self.fps)
        if not raw:
            bvp = self.process_bvp(bvp)
        return bvp, ts
        
    def hr(self, start=0, end=None, return_hrv=True):
        if self.has_signal:
            bvp, ts = self.bvp(start, end)
            try:
                sqi = SQI(self.bvp(start, end, raw=True)[0], sr=self.fps)
                hrv = {}
                if return_hrv and sqi>0.5:
                    hrv = get_prv(bvp, ts, self.fps)
                hr  = get_hr(bvp, self.fps)
            except:
                hr, sqi, hrv = None, None, {}
            return {'hr':hr, 'SQI':sqi, 'hrv':hrv, 'latency':self.latency}
        return None
    
    def update_face(self, face_img, ts=None, hasface=True):
        if face_img is not None:
            resolution = self.input[1:3]
            face_img = cv2.resize(face_img, resolution, interpolation=cv2.INTER_AREA)
            face_img = partial(lambda x:x, face_img) # Instant calculation (realtime)
        self.update_face_resized(face_img, ts=ts, hasface=hasface)
    
    def update_face_resized(self, face_img, ts=None, hasface=True):
        if face_img is None:
            if not hasface:
                face_img = lambda :np.zeros(self.input[1:], dtype='uint8')
                self.statistic['null'] += 1
            else:
                return
        if ts is None:
            ts = time.time()
        n = 0
        while (self.n_frame-0.3)/self.fps<=ts-(self.ts+[ts])[0]:
            with self.frame_lock:
                self.ts.append(ts)
                self.statistic['frames'] += 1
                if n>0:
                    self.statistic['filled'] += 1
                self.face_buff.append((face_img, hasface))
            self.n_frame += 1
            self.sp.release()
            n += 1
        if n==0:
            self.statistic['skipped'] += 1
        
    def update_frame(self, frame, ts=None):
        if ts is None:
            ts = time.time()
        def detect(n, img, lock1, lock2, ts, dt):
            try:
                if not n%self.face_detect_per_n:
                    r = self.detector.detect(img)
                    if len(r):
                        r, _, _ = r[0]
                        r = np.round(r).astype('int')
                        r = np.array(((r[1],r[3]),(r[0],r[2])))
                else:
                    r = 'skipped'
                if lock1 is not None:
                    lock1.acquire()
                box = self.__update_Kalman(img, ts, r, dt)
                r = self.__update_frame_box(img, ts, box, dt)
                self.update_face_resized(*r)
            except:
                import sys
                sys.excepthook(*sys.exc_info())
            finally:
                lock2.release()
                self.face_detection_semaphore.release()
        lock1 = self.face_detection_chain_lock 
        self.face_detection_chain_lock = threading.Lock()
        self.face_detection_chain_lock.acquire()
        lock2 = self.face_detection_chain_lock 
        self.face_detection_semaphore.acquire()
        dt = ts - self.face_detection_chain_ts
        self.face_detection_chain_ts = ts
        self.face_detection_task = self.face_detection_pool.submit(lambda:detect(self.face_detect_count, frame, lock1, lock2, ts, dt))
        self.face_detect_count += 1
    
    def __update_Kalman(self, frame, ts, box=np.array([]), dt=1/30):
        if not isinstance(box, str):
            if len(box):
                box[box<0] = 0
                self.hasface = ts + 1
            elif self.box is None or ts+1-self.hasface>1 or (self.rbox[:,0]<=np.array(0.05)*self.input[1:3]).any() or (self.rbox[:,1]>=np.array(0.95)*self.input[1:3]).any():
                self.hasface = 0
            if len(box):
                if self.boxkf is None:
                    kbox, self.boxkf = box, [KalmanFilter1D(0.01,0.5,i,1) for i in box.reshape(-1)]
                else:
                    kbox = np.array([round(k.update(i, dt)) for k, i in zip(self.boxkf, box.reshape(-1))]).reshape((2,2))
                self.rbox = box
                if self.box is None or max(np.abs((np.mean(kbox, axis=1)-np.mean(self.box, axis=1)))/np.mean(self.box, axis=0))>0.02:
                    self.box = kbox
        else:
            if self.boxkf is not None:
                kbox = np.array([round(k.update(i, dt)) for k, i in zip(self.boxkf, self.rbox.reshape(-1))]).reshape((2,2))
        return self.box
    
    def __update_frame_box(self, frame, ts=None, box=np.array([]), dt=1/30):
        self.frame = frame
        if box is not None:
            img = np.ascontiguousarray(frame[slice(*box[0]), slice(*box[1])])
        else:
            img = None 
        if self.preview_lock.locked():
            self.preview_lock.release()
        if img is not None:
            resolution = self.input[1:3] # Delayed calculation (parallel)
            img = partial(cv2.resize, img, resolution, interpolation=cv2.INTER_AREA)
        return img, ts, self.hasface
    
    def video_capture(self, vid_path=0):
        if self.run is not None:
            raise RuntimeError('A task is currently running!')
        import sys 
        api = 0
        if sys.platform.startswith('win32'):
            api = 700
        self.run = threading.Thread(target=lambda:self.__process_video_capture(vid_path, api))
        self.run.start()
        stop = self.stop
        class _:
            def __enter__(self):
                return self 
            def __exit__(self, *k):
                stop()
        self.preview_lock.acquire()
        return _()
    
    def wait_completion(self):
        if self.run is None:
            return
        self.run.join()
    
    @property
    def preview(self):
        def f():
            while 1:
                if self.preview_lock is None:
                    return
                self.preview_lock.acquire()
                yield self.frame, self.box 
        return f()
        
    def stop(self):
        self.alive = False
        self.wait_completion()
        self.run = None
        
    def process_video_tensor(self, tensor, fps=30.):
        # tensor shape (length, H, W, RGB) uint8
        if tensor.dtype != 'uint8' or len(tensor.shape) != 4 or tensor.shape[-1] != 3:
            raise TypeError("Only processes uint8 tensors with shape (length, H, W, RGB) and values in the range of 0 to 255.")
        logger.warning('Tensor mode, video quality check disabled.')
        with self:
            ts = 0
            for i in range(len(tensor)):
                self.update_frame(tensor[i], ts)
                ts += 1/fps
        return self.hr()
    
    def process_faces_tensor(self, tensor, fps=30.):
        # tensor shape (length, H, W, RGB) uint8
        if tensor.dtype != 'uint8' or len(tensor.shape) != 4 or tensor.shape[-1] != 3:
            raise TypeError("Only processes uint8 tensors with shape (length, H, W, RGB) and values in the range of 0 to 255.")
        logger.warning('Tensor mode, video quality check disabled.')
        with self:
            ts = 0
            for i in range(len(tensor)):
                self.update_face(tensor[i], ts)
                ts += 1/fps
        return self.hr()
    
    def process_video(self, vid_path):
        container = av.open(vid_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        tsarr = []
        with self:
            for frame in container.decode(stream):
                if frame.key_frame:
                    self.statistic['key'] += 1
                else:
                    self.statistic['dependent'] += 1
                rotation = -frame.rotation%360
                ts = frame.time
                tsarr.append(ts)
                img = frame.to_ndarray(format='rgb24')
                if rotation == 90:
                    img = img.swapaxes(0, 1)[:, ::-1, :]
                elif rotation == 180:
                    img = img[::-1, ::-1, :]
                elif rotation == 270:
                    img = img.swapaxes(0, 1)[::-1, :, :]
                self.update_frame(img, ts)
        container.close()
        if len(tsarr)>2:
            goodvid = True
            fps = 1/np.diff(tsarr)
            fps_std = np.std(fps)
            fps = np.mean(fps)
            if not (self.fps*0.95<fps<self.fps*1.05):
                logger.warning('Frame rate mismatch, performing nearest neighbor sampling.')
                goodvid = False
            if fps_std>0.05*fps:
                logger.warning('Frame rate is unstable, performing nearest neighbor sampling.')
                goodvid = False
            if self.statistic['dependent']>0:
                logger.warning('Detected non-key frames, this will damage the rPPG signal. For information on key frames, see https://en.wikipedia.org/wiki/I-frame. \nPlease use https://github.com/KegangWangCCNU/PhysRecorder to record videos that only contain key frames.')
                goodvid = False
            if not goodvid:
                logger.info('\n'+self.video_statistic)
        return self.hr()
            
    
    def __process_video_capture(self, vid_path, api=None):
        cap = cv2.VideoCapture(vid_path, api)
        orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        with self:
            while self.alive:
                _, img = cap.read()
                if isinstance(vid_path, str):
                    ts = round(cap.get(cv2.CAP_PROP_POS_MSEC))%1000000000/1000
                else:
                    ts = time.time()
                if not _:
                    break
                if orientation>0:
                    if orientation == 90:
                        rotate_code = cv2.ROTATE_90_CLOCKWISE
                    elif orientation == 180:
                        rotate_code = cv2.ROTATE_180
                    elif orientation == 270:
                        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                    img = cv2.rotate(img, rotate_code)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.update_frame(img, ts)
        cap.release()
        return self
