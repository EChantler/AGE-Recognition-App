// src/App.tsx
import React, { useCallback, useEffect, useRef, useState } from "react";
import { InferenceSession } from "onnxruntime-web";
import {
  loadModel,
  loadAgeModel,
  loadGenderModel,
  loadExpressionModel,
  classifyImage,
  classifyAge,
  classifyGender,
  classifyExpression,
} from "./face_classifier";
import { preprocessImageData, initializeFaceDetector, extractFaceFrame } from "./preprocess";

type ModelKey = "face" | "age" | "gender" | "expression" | "mediapipe";

type ModelLoadState = {
  label: string;
  status: "idle" | "loading" | "loaded" | "error";
  progress: number;
  error?: string;
};

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [ageSession, setAgeSession] = useState<InferenceSession | null>(null);
  const [genderSession, setGenderSession] = useState<InferenceSession | null>(null);
  const [expressionSession, setExpressionSession] = useState<InferenceSession | null>(null);
  const [faceDetectorReady, setFaceDetectorReady] = useState(false);
  const [loadState, setLoadState] = useState<Record<ModelKey, ModelLoadState>>({
    face: { label: "Face Detector", status: "idle", progress: 0 },
    age: { label: "Age", status: "idle", progress: 0 },
    gender: { label: "Gender", status: "idle", progress: 0 },
    expression: { label: "Expression", status: "idle", progress: 0 },
    mediapipe: { label: "MediaPipe", status: "idle", progress: 0 },
  });
  const [result, setResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { notFace: number; face: number };
  } | null>(null);
  const [ageResult, setAgeResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { [key: string]: number };
  } | null>(null);
  const [genderResult, setGenderResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { female: number; male: number };
  } | null>(null);
  const [expressionResult, setExpressionResult] = useState<{
    label: string;
    confidence: number;
    probabilities: { [key: string]: number };
  } | null>(null);

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
      updateLoadState("face", { status: "loading", progress: 25 });
      updateLoadState("age", { status: "loading", progress: 25 });
      updateLoadState("gender", { status: "loading", progress: 25 });
      updateLoadState("expression", { status: "loading", progress: 25 });
      updateLoadState("mediapipe", { status: "loading", progress: 25 });

      const facePromise = (async () => {
        try {
          const loadedSession = await loadModel();
          console.log("Face model loaded successfully:", loadedSession);
          setSession(loadedSession);
          updateLoadState("face", { status: "loaded", progress: 100 });
        } catch (err) {
          console.error("Failed to load face model:", err);
          updateLoadState("face", {
            status: "error",
            progress: 100,
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const agePromise = (async () => {
        try {
          const loadedSession = await loadAgeModel();
          console.log("Age model loaded successfully:", loadedSession);
          setAgeSession(loadedSession);
          updateLoadState("age", { status: "loaded", progress: 100 });
        } catch (err) {
          console.error("Failed to load age model:", err);
          updateLoadState("age", {
            status: "error",
            progress: 100,
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const genderPromise = (async () => {
        try {
          const loadedSession = await loadGenderModel();
          console.log("Gender model loaded successfully:", loadedSession);
          setGenderSession(loadedSession);
          updateLoadState("gender", { status: "loaded", progress: 100 });
        } catch (err) {
          console.error("Failed to load gender model:", err);
          updateLoadState("gender", {
            status: "error",
            progress: 100,
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const expressionPromise = (async () => {
        try {
          const loadedSession = await loadExpressionModel();
          console.log("Expression model loaded successfully:", loadedSession);
          setExpressionSession(loadedSession);
          updateLoadState("expression", { status: "loaded", progress: 100 });
        } catch (err) {
          console.error("Failed to load expression model:", err);
          updateLoadState("expression", {
            status: "error",
            progress: 100,
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      const mediapipePromise = (async () => {
        try {
          const detector = await initializeFaceDetector();
          console.log("MediaPipe face detector ready:", detector);
          setFaceDetectorReady(true);
          updateLoadState("mediapipe", { status: "loaded", progress: 100 });
        } catch (err) {
          console.error("Failed to load MediaPipe face detector:", err);
          updateLoadState("mediapipe", {
            status: "error",
            progress: 100,
            error: err instanceof Error ? err.message : "Unknown error",
          });
        }
      })();

      await Promise.all([facePromise, agePromise, genderPromise, expressionPromise, mediapipePromise]);
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
    if (!videoRef.current || !session || !faceDetectorReady) return;

    const size = 224;

    try {
      // Initialize face detector and extract face frame
      await initializeFaceDetector();
      const faceData = await extractFaceFrame(videoRef.current, size);

      if (!faceData) {
        console.log("No face detected");
        setResult(null);
        setAgeResult(null);
        setGenderResult(null);
        setExpressionResult(null);
        return;
      }

      const { imageData } = faceData;
      console.log("Face detected and extracted using MediaPipe.");

      // Preprocess to NCHW float32 [1,3,224,224], normalized like in Python
      const inputData = preprocessImageData(imageData);
      renderPreprocessedToCanvas(inputData, size);

      // Run face detection
      const prediction = await classifyImage(session, inputData);
      setResult(prediction);

      // Run age classification if age model is loaded and face is detected
      if (ageSession && prediction.label === "Face") {
        const agePrediction = await classifyAge(ageSession, inputData);
        setAgeResult(agePrediction);
      } else {
        setAgeResult(null);
      }

      // Run gender classification if gender model is loaded and face is detected
      if (genderSession && prediction.label === "Face") {
        const genderPrediction = await classifyGender(genderSession, inputData);
        setGenderResult(genderPrediction);
      } else {
        setGenderResult(null);
      }

      // Run expression classification if expression model is loaded and face is detected
      if (expressionSession && prediction.label === "Face") {
        const expressionPrediction = await classifyExpression(expressionSession, inputData);
        setExpressionResult(expressionPrediction);
      } else {
        setExpressionResult(null);
      }
    } catch (err) {
      console.error("Error during face capture or classification:", err);
      setResult(null);
      setAgeResult(null);
      setGenderResult(null);
      setExpressionResult(null);
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
      <div style={{ marginBottom: 16, width: "100%", maxWidth: 480 }}>
        {Object.entries(loadState).map(([key, info]) => {
          const color = info.status === "loaded" ? "#16a34a" : info.status === "error" ? "#dc2626" : "#f59e0b";
          return (
            <div key={key} style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 14, marginBottom: 4 }}>
                <span>{info.label} Model</span>
                <span style={{ color }}>
                  {info.status === "loading"
                    ? "Loading"
                    : info.status === "loaded"
                    ? "Ready"
                    : info.status === "error"
                    ? "Error"
                    : "Idle"}
                </span>
              </div>
              <div style={{ background: "#e5e7eb", borderRadius: 6, height: 10, overflow: "hidden" }}>
                <div
                  style={{
                    width: `${info.progress}%`,
                    height: "100%",
                    background: color,
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
              {info.error && <div style={{ color: "#dc2626", fontSize: 12, marginTop: 4 }}>Error: {info.error}</div>}
            </div>
          );
        })}
      </div>
      <video ref={videoRef} playsInline muted autoPlay style={{ width: 320, height: 240, backgroundColor: "#ccc" }} />
      <div>
        <button onClick={startCamera} style={{ marginRight: 8 }}>
          Start Camera
        </button>
        <button
          onClick={handleCapture}
          disabled={!session || !ageSession || !genderSession || !expressionSession || !faceDetectorReady}
        >
          Capture & Classify
        </button>
      </div>
      <div style={{ marginTop: 16 }}>
        <h3>Preprocessed Input (what the model sees)</h3>
        <canvas id="debug-preprocessed" style={{ width: 224, height: 224, border: "1px solid #ccc" }} />
      </div>
      {result && (
        <div style={{ marginTop: 16 }}>
          <h2>Face Detection: {result.label}</h2>
          <p style={{ fontSize: 18 }}>
            Confidence: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
          </p>
          <div style={{ fontSize: 14, color: "#666" }}>
            <div>Not Face: {(result.probabilities.notFace * 100).toFixed(2)}%</div>
            <div>Face: {(result.probabilities.face * 100).toFixed(2)}%</div>
          </div>
        </div>
      )}
      {ageResult && (
        <div style={{ marginTop: 16, borderTop: "2px solid #333", paddingTop: 16 }}>
          <h2>Age Group: {ageResult.label}</h2>
          <p style={{ fontSize: 18 }}>
            Confidence: <strong>{(ageResult.confidence * 100).toFixed(2)}%</strong>
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
    </div>
  );
};

export default App;
