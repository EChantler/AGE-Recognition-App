// src/faceClassifier.ts
import { InferenceSession, Tensor } from "onnxruntime-web";

let faceSessionPromise: Promise<InferenceSession> | null = null;
let ageSessionPromise: Promise<InferenceSession> | null = null;

export function loadModel() {
  if (!faceSessionPromise) {
    const base = import.meta.env.BASE_URL || "/";
    const modelUrl = `${base}models/face_binary.onnx`;
    const dataUrl = `${base}models/face_binary.onnx.data`;
    faceSessionPromise = InferenceSession.create(modelUrl, {
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
  return faceSessionPromise;
}

export function loadAgeModel() {
  if (!ageSessionPromise) {
    const base = import.meta.env.BASE_URL || "/";
    const modelUrl = `${base}models/age.onnx`;
    const dataUrl = `${base}models/age.onnx.data`;
    ageSessionPromise = InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      externalData: [
        {
          data: dataUrl,
          path: "age.onnx.data",
        },
      ],
    });
  }
  return ageSessionPromise;
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

  console.log("Raw outputs keys:", Object.keys(outputs));
  console.log("Raw outputs:", outputs);

  // Get output - try both "output" and "logits" names
  let logits: Float32Array | undefined;
  if (outputs["output"]) {
    logits = outputs["output"].data as Float32Array;
  } else if (outputs["logits"]) {
    logits = outputs["logits"].data as Float32Array;
  }

  if (!logits) {
    throw new Error(`No output found. Available outputs: ${Object.keys(outputs).join(", ")}`);
  }

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

export async function classifyAge(session: InferenceSession, inputData: Float32Array) {
  const tensor = new Tensor("float32", inputData, [1, 3, 224, 224]);
  const outputs = await session.run({ input: tensor });

  console.log("Age model raw outputs:", outputs);

  // Get output logits
  let logits: Float32Array | undefined;
  if (outputs["output"]) {
    logits = outputs["output"].data as Float32Array;
  } else if (outputs["logits"]) {
    logits = outputs["logits"].data as Float32Array;
  }

  if (!logits || logits.length !== 5) {
    throw new Error(`Expected 5 age class logits. Got: ${logits?.length || 0}`);
  }

  console.log("Raw age logits:", Array.from(logits));

  // Apply softmax to get probabilities
  const expValues = Array.from(logits).map((l) => Math.exp(l));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  const probabilities = expValues.map((exp) => exp / sumExp);

  // Get prediction (argmax)
  const predIdx = probabilities.indexOf(Math.max(...probabilities));
  const ageLabels = ["18-20", "21-30", "31-40", "41-50", "51-60"];
  const confidence = probabilities[predIdx];

  return {
    label: ageLabels[predIdx],
    confidence: confidence,
    probabilities: {
      "18-20": probabilities[0],
      "21-30": probabilities[1],
      "31-40": probabilities[2],
      "41-50": probabilities[3],
      "51-60": probabilities[4],
    },
  };
}
