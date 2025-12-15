// src/preprocess.ts
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

let faceDetector: FaceDetector | null = null;

// Initialize MediaPipe FaceDetector
export async function initializeFaceDetector(): Promise<FaceDetector> {
  if (faceDetector) {
    return faceDetector;
  }

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
  );

  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
  });

  return faceDetector;
}

// Extract face region from video/image using MediaPipe
export async function extractFaceFrame(
  videoElement: HTMLVideoElement | HTMLCanvasElement,
  targetSize: number = 224
): Promise<{ imageData: ImageData; bbox: { x: number; y: number; width: number; height: number } | null } | null> {
  if (!faceDetector) {
    faceDetector = await initializeFaceDetector();
  }

  const width = (videoElement as HTMLVideoElement).videoWidth || videoElement.width;
  const height = (videoElement as HTMLVideoElement).videoHeight || videoElement.height;

  // Detect faces directly from video element
  const detectionResult = faceDetector.detectForVideo(videoElement as HTMLVideoElement, performance.now());

  if (!detectionResult.detections || detectionResult.detections.length === 0) {
    return null; // No face detected
  }

  // Get the first (largest) detected face
  const detection = detectionResult.detections[0];
  const boundingBox = detection.boundingBox;

  if (!boundingBox) return null;

  // Extract bounding box coordinates in pixels
  const bbox = {
    x: Math.max(0, boundingBox.originX),
    y: Math.max(0, boundingBox.originY),
    width: boundingBox.width,
    height: boundingBox.height,
  };

  console.log("Detected bbox:", bbox, "Video dimensions:", width, height);

  // Add padding to bounding box (10% padding)
  const padding = 0.1;
  const paddedBbox = {
    x: Math.max(0, bbox.x - bbox.width * padding),
    y: Math.max(0, bbox.y - bbox.height * padding),
    width: bbox.width * (1 + 2 * padding),
    height: bbox.height * (1 + 2 * padding),
  };

  // Ensure bbox stays within image bounds
  paddedBbox.width = Math.min(paddedBbox.width, width - paddedBbox.x);
  paddedBbox.height = Math.min(paddedBbox.height, height - paddedBbox.y);

  // Create canvas and draw the full video frame first
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(videoElement, 0, 0, width, height);

  // Create cropped canvas with target size
  const croppedCanvas = document.createElement("canvas");
  const croppedCtx = croppedCanvas.getContext("2d");
  if (!croppedCtx) return null;

  croppedCanvas.width = targetSize;
  croppedCanvas.height = targetSize;
  croppedCtx.drawImage(
    canvas,
    paddedBbox.x,
    paddedBbox.y,
    paddedBbox.width,
    paddedBbox.height,
    0,
    0,
    targetSize,
    targetSize
  );

  const imageData = croppedCtx.getImageData(0, 0, targetSize, targetSize);
  return { imageData, bbox: paddedBbox };
}

export function preprocessImageData(imageData: ImageData): Float32Array {
  const { data, width, height } = imageData; // RGBA
  const floatData = new Float32Array(1 * 3 * height * width);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  // NCHW
  const channelSize = width * height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const i = idx * 4; // RGBA index

      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;

      floatData[0 * channelSize + idx] = (r - mean[0]) / std[0];
      floatData[1 * channelSize + idx] = (g - mean[1]) / std[1];
      floatData[2 * channelSize + idx] = (b - mean[2]) / std[2];
    }
  }

  return floatData;
}
