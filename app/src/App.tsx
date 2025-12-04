// src/App.tsx
import React, { useEffect, useRef, useState } from "react";
import { InferenceSession } from "onnxruntime-web";
import { loadModel, classifyImage } from "./face_classifier";
import { preprocessImageData } from "./preprocess";
// Simple center-crop pipeline (face detector reverted)

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [result, setResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { notFace: number; face: number };
  } | null>(null);

  // Load model once
  useEffect(() => {
    console.log("Loading ONNX model...");
    loadModel()
      .then((loadedSession) => {
        console.log("Model loaded successfully:", loadedSession);
        setSession(loadedSession);
      })
      .catch((err) => {
        console.error("Failed to load model:", err);
      });
  }, []);

  // No external face detector. Using center-crop only.

  // Start camera on user interaction (required for iOS Safari)
  const startCamera = async () => {
    let stream: MediaStream | null = null;
    try {
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: "user" },
        },
        audio: false,
      };
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        // iOS Safari requires these to render inline and autoplay
        videoRef.current.setAttribute("playsinline", "true");
        videoRef.current.setAttribute("autoplay", "true");
        videoRef.current.muted = true;
        videoRef.current.srcObject = stream;
        // Attempt to play; sometimes needs a second attempt on iOS
        try {
          await videoRef.current.play();
        } catch (e) {
          setTimeout(() => {
            videoRef.current && videoRef.current.play().catch(() => {});
          }, 100);
        }
      }
    } catch (err) {
      console.error("Error accessing camera", err);
      alert("Could not access camera. Please allow permission and try tapping the Start Camera button.");
    }
  };

  const handleCapture = async () => {
    if (!videoRef.current || !session) return;

    const size = 224;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Center crop (80% of frame) to focus on face region
    const srcW = videoRef.current.videoWidth || 320;
    const srcH = videoRef.current.videoHeight || 240;
    const cropScale = 0.8;
    const cropW = Math.floor(srcW * cropScale);
    const cropH = Math.floor(srcH * cropScale);
    const cropX = Math.floor((srcW - cropW) / 2);
    const cropY = Math.floor((srcH - cropH) / 2);

    console.log("Using center crop.");

    // Draw cropped region scaled to 224x224
    canvas.width = size;
    canvas.height = size;
    ctx.drawImage(videoRef.current, cropX, cropY, cropW, cropH, 0, 0, size, size);
    const imageData = ctx.getImageData(0, 0, size, size);

    // Preprocess to NCHW float32 [1,3,224,224], normalized like in Python
    const inputData = preprocessImageData(imageData);
    renderPreprocessedToCanvas(inputData, size);

    const prediction = await classifyImage(session, inputData);
    setResult(prediction);
  };

  function renderPreprocessedToCanvas(inputData: Float32Array, size = 224) {
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const channelSize = size * size;

    const debugCanvas = document.getElementById("debug-preprocessed") as HTMLCanvasElement | null;
    if (!debugCanvas) return;
    debugCanvas.width = size;
    debugCanvas.height = size;
    const ctx = debugCanvas.getContext("2d");
    if (!ctx) return;

    const imgData = ctx.createImageData(size, size);
    const data = imgData.data;

    for (let i = 0; i < channelSize; i++) {
      const rNorm = inputData[0 * channelSize + i];
      const gNorm = inputData[1 * channelSize + i];
      const bNorm = inputData[2 * channelSize + i];

      const r = rNorm * std[0] + mean[0];
      const g = gNorm * std[1] + mean[1];
      const b = bNorm * std[2] + mean[2];

      const idx = i * 4;
      data[idx + 0] = Math.max(0, Math.min(255, Math.round(r * 255)));
      data[idx + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
      data[idx + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
      data[idx + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  return (
    <div style={{ padding: 16 }}>
      <h1>Face / Not Face PoC 3</h1>
      <video ref={videoRef} playsInline muted autoPlay style={{ width: 320, height: 240, backgroundColor: "#ccc" }} />
      <div>
        <button onClick={startCamera} style={{ marginRight: 8 }}>
          Start Camera
        </button>
        <button onClick={handleCapture} disabled={!session}>
          Capture & Classify
        </button>
      </div>
      <div style={{ marginTop: 16 }}>
        <h3>Preprocessed Input (what the model sees)</h3>
        <canvas id="debug-preprocessed" style={{ width: 224, height: 224, border: "1px solid #ccc" }} />
      </div>
      {result && (
        <div style={{ marginTop: 16 }}>
          <h2>Prediction: {result.label}</h2>
          <p style={{ fontSize: 18 }}>
            Confidence: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
          </p>
          <div style={{ fontSize: 14, color: "#666" }}>
            <div>Not Face: {(result.probabilities.notFace * 100).toFixed(2)}%</div>
            <div>Face: {(result.probabilities.face * 100).toFixed(2)}%</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
