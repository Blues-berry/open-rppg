API Reference
=============

This section provides detailed documentation for the core API of the **Open-rppg** toolbox.

Core Interface
--------------

The ``Model`` class is the primary entry point for all rPPG tasks. It encapsulates the complexity of different neural network architectures (like PhysMamba, TSCAN, etc.) into a unified interface.

.. py:class:: rppg.Model(model='FacePhys.rlap')

   The main wrapper class for rPPG inference. It handles the full pipeline: face detection, ROI signal extraction, and physiological metric calculation.

   :param str model: The name of the neural network architecture to use. Defaults to ``'FacePhys.rlap'``.
   
       **Supported Models:**
   
       * **State-Space Models (SSM):** ``'FacePhys.rlap'`` (Default), ``'ME-chunk'``, ``'ME-flow'``, ``'PhysMamba'``, ``'RhythmMamba'``.
       * **Transformer / Attention:** ``'PhysFormer'``, ``'TSCAN'``, ``'EfficientPhys'``.
       * **Convolutional:** ``'PhysNet'``.
       
       *Note: Append* ``.rlap`` *or* ``.pure`` *to the model name to specify weights (e.g.,* ``'PhysNet.pure'`` *).*

   .. py:method:: process_video(vid_path)

      Processes a video file from start to finish to extract physiological signals.

      :param str vid_path: Path to the input video file (e.g., ``.mp4``, ``.avi``, ``.mkv``).
      :return: A dictionary containing the estimated heart rate (``'hr'``), signal quality (``'SQI'``), and other metrics.
      :rtype: dict

   .. py:method:: video_capture(vid_path=0)

      A context manager for processing real-time video streams or video files frame-by-frame. 
      It initializes a background thread for non-blocking inference.

      :param vid_path: The device index (int) for webcams or the file path (str) for video files. Defaults to ``0``.
      :return: The instance itself (context manager).

   .. py:method:: process_video_tensor(tensor, fps=30.0)

      Processes a pre-loaded video tensor (uint8). Useful for integration with existing data pipelines where the video is already in memory.

      :param numpy.ndarray tensor: A 4D tensor of shape ``(Frames, Height, Width, 3)``.
      :param float fps: The frame rate of the video data.
      :return: A dictionary containing the analysis results.

   .. py:method:: process_faces_tensor(tensor, fps=30.0)

      Processes a pre-cropped face tensor, skipping the internal face detection step.

      :param numpy.ndarray tensor: A 4D tensor of shape ``(Frames, Height, Width, 3)``.
      :param float fps: The frame rate of the video data.
      :return: A dictionary containing the analysis results.

   .. py:method:: bvp(start=0, end=None, raw=False)

      Retrieves the Blood Volume Pulse (BVP) signal from the internal buffer.

      :param float start: Start time in seconds.
      :param float end: End time in seconds. If ``None``, returns data up to the current timestamp.
      :param bool raw: If ``True``, returns the raw model output. If ``False``, applies bandpass filtering and normalization.
      :return: A tuple ``(signal, timestamps)``.
      :rtype: tuple

   .. py:method:: hr(start=0, end=None, return_hrv=True)

      Calculates heart rate and HRV metrics from the buffered signal.

      :param float start: Start time in seconds.
      :param float end: End time in seconds.
      :param bool return_hrv: Whether to compute detailed HRV metrics (SDNN, RMSSD, etc.). Defaults to ``True``.
      :return: A dictionary with keys ``'hr'``, ``'SQI'``, ``'hrv'``, and ``'latency'``.

   .. py:attribute:: preview

      A generator property used in the ``video_capture`` loop.

      :yields: ``(frame, box)`` where ``frame`` is the current RGB image and ``box`` is the face bounding box coordinates.

   .. py:method:: stop()

      Stops the running inference thread. Called automatically when exiting the context manager.

Signal Processing Utilities
---------------------------

These helper functions are available in ``rppg.main`` for post-processing physiological signals.

.. py:function:: rppg.main.SQI(signal, sr=30, min_freq=0.5, max_freq=3.0, window_size=10)

   Calculates the Signal Quality Index (SQI) to assess the reliability of the rPPG signal. 
   It uses an autocorrelation-based method to determine if the signal has a periodic physiological structure.

   :param numpy.ndarray signal: The input BVP signal array.
   :param int sr: Sampling rate in Hz. Defaults to 30.
   :param float min_freq: Minimum expected human pulse frequency (Hz).
   :param float max_freq: Maximum expected human pulse frequency (Hz).
   :param int window_size: Window size in seconds for sliding window calculation.
   :return: A quality score between 0.0 (noise) and 1.0 (clean signal).
   :rtype: float

.. py:function:: rppg.main.get_hr(y, sr=30, min=30, max=180)

   Estimates the Heart Rate (HR) from a BVP signal using the frequency domain approach.
   It calculates the Power Spectral Density (PSD) using Welch's method and finds the peak frequency.

   :param numpy.ndarray y: The input BVP signal.
   :param int sr: Sampling rate in Hz.
   :param int min: Minimum valid heart rate (BPM).
   :param int max: Maximum valid heart rate (BPM).
   :return: Estimated Heart Rate in BPM.
   :rtype: float

.. py:function:: rppg.main.get_prv(y, ts=None, sr=30)

   Computes Pulse Rate Variability (PRV) metrics (time-domain and frequency-domain) using peak detection.
   This function wraps ``heartpy`` analysis to provide standard HRV metrics.

   :param numpy.ndarray y: The input BVP signal.
   :param numpy.ndarray ts: Corresponding timestamps (optional).
   :param int sr: Sampling rate in Hz.
   :return: A dictionary containing metrics such as ``'sdnn'``, ``'rmssd'``, ``'pnn50'``, ``'LF/HF'``, and ``'breathingrate'``.
   :rtype: dict

.. py:function:: rppg.main.detrend(signal, min_freq=0.5, sr=30)

   Applies Smoothness Priors Detrending (SPD) to remove low-frequency non-stationary trends (such as motion artifacts) from the signal.

   :param numpy.ndarray signal: The raw input signal.
   :param float min_freq: The cut-off frequency for the trend. Trends slower than this will be removed.
   :param int sr: Sampling rate in Hz.
   :return: The detrended signal.
   :rtype: numpy.ndarray

.. py:function:: rppg.main.bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3)

   Applies a standard Butterworth bandpass filter to the signal.

   :param numpy.ndarray data: Input signal.
   :param float lowcut: Low frequency cutoff (Hz). Defaults to 0.5 (30 BPM).
   :param float highcut: High frequency cutoff (Hz). Defaults to 3.0 (180 BPM).
   :param int fs: Sampling frequency (Hz).
   :param int order: Order of the filter.
   :return: Filtered signal.
   :rtype: numpy.ndarray