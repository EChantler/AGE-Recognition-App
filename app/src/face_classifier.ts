// src/faceClassifier.ts
import { InferenceSession, Tensor } from "onnxruntime-web";

let sessionPromise: Promise<InferenceSession> | null = null;

export function loadModel() {
  if (!sessionPromise) {
    const base = import.meta.env.BASE_URL || "/";
    const modelUrl = `${base}models/face_binary.onnx`;
    const dataUrl = `${base}models/face_binary.onnx.data`;
    sessionPromise = InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      externalData: [
        {
          data: dataUrl,
          path: "face_binary.onnx.data",
        },
      ],
    });
  }
  return sessionPromise;
}

export async function classifyImage(session: InferenceSession, inputData: Float32Array) {
  // [1,3,224,224]
  let min = inputData[0],
    max = inputData[0],
    sum = 0;
  for (let i = 0; i < inputData.length; i++) {
    const val = inputData[i];
    if (val < min) min = val;
    if (val > max) max = val;
    sum += val;
  }

  console.log("Input data stats:", {
    min,
    max,
    mean: sum / inputData.length,
    first10: Array.from(inputData.slice(0, 10)),
  });

  const tensor = new Tensor("float32", inputData, [1, 3, 224, 224]);
  const outputs = await session.run({ input: tensor });
  const logits = outputs["logits"].data as Float32Array; // [1,2]

  console.log("Raw logits:", logits);

  const [logit0, logit1] = logits;

  // Apply softmax to get probabilities
  const exp0 = Math.exp(logit0);
  const exp1 = Math.exp(logit1);
  const sumExp = exp0 + exp1;
  const prob0 = exp0 / sumExp; // class 0 = Not Face
  const prob1 = exp1 / sumExp; // class 1 = Face

  // argmax
  // Choose class by highest probability to avoid mapping mistakes
  const predIdx = prob1 > prob0 ? 1 : 0;
  const labels = ["Not Face", "Face"];
  const confidence = Math.max(prob0, prob1);

  return {
    label: labels[predIdx],
    confidence: confidence,
    probabilities: {
      notFace: prob0,
      face: prob1,
    },
  };
}
