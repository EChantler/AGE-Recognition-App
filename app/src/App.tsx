// src/App.tsx
import React, { useCallback, useEffect, useRef, useState } from "react";
import { InferenceSession } from "onnxruntime-web";
import {
  loadAgeModel,
  loadGenderModel,
  loadExpressionModel,
  loadExpressionEfficientModel,
  classifyAge,
  classifyGender,
  classifyExpression,
  classifyExpressionEfficient,
} from "./face_classifier";
import { preprocessImageData, initializeFaceDetector, extractFaceFrame } from "./preprocess";

type ModelKey = "age" | "gender" | "expression" | "expressionEfficient" | "mediapipe";

type ModelLoadState = {
  label: string;
  status: "idle" | "loading" | "loaded" | "error";
  error?: string;
};

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const [ageSession, setAgeSession] = useState<InferenceSession | null>(null);
  const [genderSession, setGenderSession] = useState<InferenceSession | null>(null);
  const [expressionSession, setExpressionSession] = useState<InferenceSession | null>(null);
  const [expressionEfficientSession, setExpressionEfficientSession] = useState<InferenceSession | null>(null);
  const [faceDetectorReady, setFaceDetectorReady] = useState(false);
  const [loadState, setLoadState] = useState<Record<ModelKey, ModelLoadState>>({
    age: { label: "Age", status: "idle" },
    gender: { label: "Gender", status: "idle" },
    expression: { label: "Expression", status: "idle" },
    expressionEfficient: { label: "Expression (EffNet)", status: "idle" },
    mediapipe: { label: "MediaPipe", status: "idle" },
  });
  const [ageResult, setAgeResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { [key: string]: number };
    duration: number;
  } | null>(null);
  const [genderResult, setGenderResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { female: number; male: number };
    duration: number;
  } | null>(null);
  const [expressionResult, setExpressionResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { [key: string]: number };
    duration: number;
  } | null>(null);
  const [expressionEfficientResult, setExpressionEfficientResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { [key: string]: number };
    duration: number;
  } | null>(null);
  const [detectionMessage, setDetectionMessage] = useState<string | null>(null);

  const updateLoadState = useCallback((key: ModelKey, partial: Partial<ModelLoadState>) => {
    setLoadState((prev) => ({
      ...prev,
      [key]: { ...prev[key], ...partial },
    }));
  }, []);

  // Load models once
  useEffect(() => {
    console.log("Loading ONNX models and MediaPipe...");

    const startLoad = async () => {
      updateLoadState("age", { status: "loading" });
      updateLoadState("gender", { status: "loading" });
      updateLoadState("expression", { status: "loading" });
      updateLoadState("expressionEfficient", { status: "loading" });
      updateLoadState("mediapipe", { status: "loading" });

      const agePromise = (async () => {
        try {
          const loadedSession = await loadAgeModel();
          console.log("Age model loaded successfully:", loadedSession);
          setAgeSession(loadedSession);
          updateLoadState("age", { status: "loaded" });
        } catch (err) {
          console.error("Failed to load age model:", err);
          updateLoadState("age", {
            status: "error",
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const genderPromise = (async () => {
        try {
          const loadedSession = await loadGenderModel();
          console.log("Gender model loaded successfully:", loadedSession);
          setGenderSession(loadedSession);
          updateLoadState("gender", { status: "loaded" });
        } catch (err) {
          console.error("Failed to load gender model:", err);
          updateLoadState("gender", {
            status: "error",
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const expressionPromise = (async () => {
        try {
          const loadedSession = await loadExpressionModel();
          console.log("Expression model loaded successfully:", loadedSession);
          setExpressionSession(loadedSession);
          updateLoadState("expression", { status: "loaded" });
        } catch (err) {
          console.error("Failed to load expression model:", err);
          updateLoadState("expression", {
            status: "error",
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const expressionEfficientPromise = (async () => {
        try {
          const loadedSession = await loadExpressionEfficientModel();
          console.log("Expression EfficientNet model loaded successfully:", loadedSession);
          setExpressionEfficientSession(loadedSession);
          updateLoadState("expressionEfficient", { status: "loaded" });
        } catch (err) {
          console.error("Failed to load expression EfficientNet model:", err);
          updateLoadState("expressionEfficient", {
            status: "error",
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const mediapipePromise = (async () => {
        try {
          const detector = await initializeFaceDetector();
          console.log("MediaPipe face detector ready:", detector);
          setFaceDetectorReady(true);
          updateLoadState("mediapipe", { status: "loaded" });
        } catch (err) {
          console.error("Failed to load MediaPipe face detector:", err);
          updateLoadState("mediapipe", {
            status: "error",
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      await Promise.all([agePromise, genderPromise, expressionPromise, expressionEfficientPromise, mediapipePromise]);
    };

    startLoad();
  }, []);

  // MediaPipe face detector is preloaded at startup for quicker captures.

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
    if (!videoRef.current || !faceDetectorReady) return;

    const size = 224;

    try {
      // Initialize face detector and extract face frame
      await initializeFaceDetector();
      const faceData = await extractFaceFrame(videoRef.current, size);

      if (!faceData) {
        console.log("No face detected");
        setDetectionMessage("No face detected");
        setAgeResult(null);
        setGenderResult(null);
        setExpressionResult(null);
        setExpressionEfficientResult(null);
        return;
      }

      const { imageData, multipleFaces } = faceData;
      if (multipleFaces) {
        setDetectionMessage("Multiple faces detected, using the largest one");
      } else {
        setDetectionMessage(null);
      }
      console.log("Face detected and extracted using MediaPipe.");

      // Preprocess to NCHW float32 [1,3,224,224], normalized like in Python
      const inputData = preprocessImageData(imageData);
      renderPreprocessedToCanvas(inputData, size);

      // Run age classification if age model is loaded
      if (ageSession) {
        const agePrediction = await classifyAge(ageSession, inputData);
        setAgeResult(agePrediction);
      } else {
        setAgeResult(null);
      }

      // Run gender classification if gender model is loaded
      if (genderSession) {
        const genderPrediction = await classifyGender(genderSession, inputData);
        setGenderResult(genderPrediction);
      } else {
        setGenderResult(null);
      }

      // Run expression classification if expression model is loaded
      if (expressionSession) {
        const expressionPrediction = await classifyExpression(expressionSession, inputData);
        setExpressionResult(expressionPrediction);
      } else {
        setExpressionResult(null);
      }

      // Run EfficientNet expression classification if loaded
      if (expressionEfficientSession) {
        const expressionEfficientPrediction = await classifyExpressionEfficient(expressionEfficientSession, inputData);
        setExpressionEfficientResult(expressionEfficientPrediction);
      } else {
        setExpressionEfficientResult(null);
      }

      // Scroll to results after successful classification
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      console.error("Error during face capture or classification:", err);
      setDetectionMessage("Error during capture");
      setAgeResult(null);
      setGenderResult(null);
      setExpressionResult(null);
      setExpressionEfficientResult(null);
    }
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
      <h1>Face Detection & Age & Gender & Expression Recognition</h1>
      <div style={{ marginBottom: 16, display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
        {Object.entries(loadState).map(([key, info]) => {
          const statusIcon =
            info.status === "loaded" ? "✓" : info.status === "error" ? "✗" : info.status === "loading" ? "⟳" : "○";
          const border = info.status === "loaded" ? "#16a34a" : info.status === "error" ? "#dc2626" : "#f59e0b";
          const bg =
            info.status === "loaded"
              ? "#ecfdf5"
              : info.status === "error"
              ? "#fef2f2"
              : info.status === "loading"
              ? "#fffbeb"
              : "#f3f4f6";
          const iconColor = info.status === "loaded" ? "#16a34a" : info.status === "error" ? "#dc2626" : "#d97706";
          return (
            <div
              key={key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 14,
                padding: "6px 12px",
                backgroundColor: bg,
                color: "#111827",
                borderRadius: 6,
                border: `1px solid ${border}`,
              }}
            >
              <span
                style={{
                  color: iconColor,
                  fontSize: 16,
                  fontWeight: 700,
                  display: "inline-block",
                  animation: info.status === "loading" ? "spin 1s linear infinite" : "none",
                }}
                aria-label={info.status}
                title={info.status}
              >
                {statusIcon}
              </span>
              <span style={{ lineHeight: 1 }}>{info.label}</span>
              {info.error && <span style={{ color: "#991b1b", fontSize: 12 }}>({info.error})</span>}
            </div>
          );
        })}
      </div>
      <div
        role="note"
        aria-label="privacy disclaimer"
        style={{
          marginBottom: 16,
          padding: 12,
          backgroundColor: "#eef2ff",
          border: "1px solid #6366f1",
          borderRadius: 6,
          color: "#1f2937",
          fontSize: 13,
          lineHeight: 1.4,
        }}
      >
        <strong style={{ color: "#3730a3" }}>Disclaimer:</strong> This is an Edge AI project. Images and video are not
        saved or uploaded; all processing happens on-device in your browser.
      </div>
      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
      <video ref={videoRef} playsInline muted autoPlay style={{ width: 320, height: 240, backgroundColor: "#ccc" }} />
      <div>
        <button onClick={startCamera} style={{ marginRight: 8 }}>
          Start Camera
        </button>
        <button
          onClick={handleCapture}
          disabled={
            !ageSession || !genderSession || !expressionSession || !expressionEfficientSession || !faceDetectorReady
          }
        >
          Capture & Classify
        </button>
      </div>
      <div style={{ marginTop: 16 }}>
        <h3>Preprocessed Input (what the model sees)</h3>
        <canvas id="debug-preprocessed" style={{ width: 224, height: 224, border: "1px solid #ccc" }} />
      </div>
      {detectionMessage && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            backgroundColor: "#fef3c7",
            borderRadius: 6,
            border: "1px solid #f59e0b",
          }}
        >
          <p style={{ margin: 0, fontSize: 14, color: "#92400e" }}>{detectionMessage}</p>
        </div>
      )}
      <div ref={resultsRef}>
        {ageResult && (
          <div style={{ marginTop: 16, borderTop: "2px solid #333", paddingTop: 16 }}>
            <h2>Age Group: {ageResult.label}</h2>
            <p style={{ fontSize: 18 }}>
              Confidence: <strong>{(ageResult.confidence * 100).toFixed(2)}%</strong>
            </p>
            <p style={{ fontSize: 14, color: "#0066cc" }}>
              Inference time: <strong>{ageResult.duration.toFixed(2)} ms</strong>
            </p>
            <div style={{ fontSize: 14, color: "#666" }}>
              <div>Young: {(ageResult.probabilities.YOUNG * 100).toFixed(2)}%</div>
              <div>Middle: {(ageResult.probabilities.MIDDLE * 100).toFixed(2)}%</div>
              <div>Old: {(ageResult.probabilities.OLD * 100).toFixed(2)}%</div>
            </div>
          </div>
        )}
        {genderResult && (
          <div style={{ marginTop: 16, borderTop: "2px solid #333", paddingTop: 16 }}>
            <h2>Gender: {genderResult.label}</h2>
            <p style={{ fontSize: 18 }}>
              Confidence: <strong>{(genderResult.confidence * 100).toFixed(2)}%</strong>
            </p>
            <p style={{ fontSize: 14, color: "#0066cc" }}>
              Inference time: <strong>{genderResult.duration.toFixed(2)} ms</strong>
            </p>
            <div style={{ fontSize: 14, color: "#666" }}>
              <div>Female: {(genderResult.probabilities.female * 100).toFixed(2)}%</div>
              <div>Male: {(genderResult.probabilities.male * 100).toFixed(2)}%</div>
            </div>
          </div>
        )}
        {expressionResult && (
          <div style={{ marginTop: 16, borderTop: "2px solid #333", paddingTop: 16 }}>
            <h2>Expression: {expressionResult.label}</h2>
            <p style={{ fontSize: 18 }}>
              Confidence: <strong>{(expressionResult.confidence * 100).toFixed(2)}%</strong>
            </p>
            <p style={{ fontSize: 14, color: "#0066cc" }}>
              Inference time: <strong>{expressionResult.duration.toFixed(2)} ms</strong>
            </p>
            <div style={{ fontSize: 14, color: "#666" }}>
              <div>Angry: {(expressionResult.probabilities.angry * 100).toFixed(2)}%</div>
              <div>Disgust: {(expressionResult.probabilities.disgust * 100).toFixed(2)}%</div>
              <div>Fear: {(expressionResult.probabilities.fear * 100).toFixed(2)}%</div>
              <div>Happy: {(expressionResult.probabilities.happy * 100).toFixed(2)}%</div>
              <div>Neutral: {(expressionResult.probabilities.neutral * 100).toFixed(2)}%</div>
              <div>Sad: {(expressionResult.probabilities.sad * 100).toFixed(2)}%</div>
              <div>Surprise: {(expressionResult.probabilities.surprise * 100).toFixed(2)}%</div>
            </div>
          </div>
        )}
        {expressionEfficientResult && (
          <div style={{ marginTop: 16, borderTop: "2px solid #333", paddingTop: 16 }}>
            <h2>Expression (EffNet): {expressionEfficientResult.label}</h2>
            <p style={{ fontSize: 18 }}>
              Confidence: <strong>{(expressionEfficientResult.confidence * 100).toFixed(2)}%</strong>
            </p>
            <p style={{ fontSize: 14, color: "#0066cc" }}>
              Inference time: <strong>{expressionEfficientResult.duration.toFixed(2)} ms</strong>
            </p>
            <div style={{ fontSize: 14, color: "#666" }}>
              <div>Angry: {(expressionEfficientResult.probabilities.angry * 100).toFixed(2)}%</div>
              <div>Disgust: {(expressionEfficientResult.probabilities.disgust * 100).toFixed(2)}%</div>
              <div>Fear: {(expressionEfficientResult.probabilities.fear * 100).toFixed(2)}%</div>
              <div>Happy: {(expressionEfficientResult.probabilities.happy * 100).toFixed(2)}%</div>
              <div>Neutral: {(expressionEfficientResult.probabilities.neutral * 100).toFixed(2)}%</div>
              <div>Sad: {(expressionEfficientResult.probabilities.sad * 100).toFixed(2)}%</div>
              <div>Surprise: {(expressionEfficientResult.probabilities.surprise * 100).toFixed(2)}%</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
