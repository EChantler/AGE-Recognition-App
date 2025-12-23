// src/faceClassifier.ts
import * as ort from "onnxruntime-web";
import { InferenceSession, Tensor } from "onnxruntime-web";

// Configure ONNX Runtime for CPU-only inference
ort.env.wasm.simd = true; // enable SIMD for better CPU performance
ort.env.wasm.numThreads = 1; // control threading (adjust as needed)

let faceSessionPromise: Promise<InferenceSession> | null = null;
let ageSessionPromise: Promise<InferenceSession> | null = null;
let genderSessionPromise: Promise<InferenceSession> | null = null;
let expressionSessionPromise: Promise<InferenceSession> | null = null;
let expressionEfficientSessionPromise: Promise<InferenceSession> | null = null;

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

export function loadGenderModel() {
  if (!genderSessionPromise) {
    const base = import.meta.env.BASE_URL || "/";
    const modelUrl = `${base}models/gender.onnx`;
    const dataUrl = `${base}models/gender.onnx.data`;
    genderSessionPromise = InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      externalData: [
        {
          data: dataUrl,
          path: "gender.onnx.data",
        },
      ],
    });
  }
  return genderSessionPromise;
}

export function loadExpressionModel() {
  if (!expressionSessionPromise) {
    const base = import.meta.env.BASE_URL || "/";
    const modelUrl = `${base}models/expression.onnx`;
    const dataUrl = `${base}models/expression.onnx.data`;
    expressionSessionPromise = InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      externalData: [
        {
          data: dataUrl,
          path: "expression.onnx.data",
        },
      ],
    });
  }
  return expressionSessionPromise;
}

export function loadExpressionEfficientModel() {
  if (!expressionEfficientSessionPromise) {
    const base = import.meta.env.BASE_URL || "/";
    const modelUrl = `${base}models/expression_efficient_net.onnx`;
    const dataUrl = `${base}models/expression_efficient_net.onnx.data`;
    expressionEfficientSessionPromise = InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      externalData: [
        {
          data: dataUrl,
          path: "expression_efficient_net.onnx.data",
        },
      ],
    });
  }
  return expressionEfficientSessionPromise;
}

export async function classifyImage(session: InferenceSession, inputData: Float32Array) {
  // [1,3,224,224]
  const startTime = performance.now();

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

  const endTime = performance.now();
  const duration = endTime - startTime;

  return {
    label: labels[predIdx],
    confidence: confidence,
    probabilities: {
      notFace: prob0,
      face: prob1,
    },
    duration,
  };
}

export async function classifyAge(session: InferenceSession, inputData: Float32Array) {
  const startTime = performance.now();

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

  if (!logits || logits.length !== 3) {
    throw new Error(`Expected 3 age class logits. Got: ${logits?.length || 0}`);
  }

  console.log("Raw age logits:", Array.from(logits));

  // Apply softmax to get probabilities
  const expValues = Array.from(logits).map((l) => Math.exp(l));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  const probabilities = expValues.map((exp) => exp / sumExp);

  // Get prediction (argmax)
  const predIdx = probabilities.indexOf(Math.max(...probabilities));
  const ageLabels = ["Young", "Middle", "Old"];
  const confidence = probabilities[predIdx];

  const endTime = performance.now();
  const duration = endTime - startTime;

  return {
    label: ageLabels[predIdx],
    confidence: confidence,
    probabilities: {
      YOUNG: probabilities[0],
      MIDDLE: probabilities[1],
      OLD: probabilities[2],
    },
    duration,
  };
}

export async function classifyGender(session: InferenceSession, inputData: Float32Array) {
  const startTime = performance.now();

  const tensor = new Tensor("float32", inputData, [1, 3, 224, 224]);
  const outputs = await session.run({ input: tensor });

  console.log("Gender model raw outputs:", outputs);

  // Get output logits
  let logits: Float32Array | undefined;
  if (outputs["output"]) {
    logits = outputs["output"].data as Float32Array;
  } else if (outputs["logits"]) {
    logits = outputs["logits"].data as Float32Array;
  }

  if (!logits || logits.length !== 2) {
    throw new Error(`Expected 2 gender class logits. Got: ${logits?.length || 0}`);
  }

  console.log("Raw gender logits:", Array.from(logits));

  // Apply softmax to get probabilities
  const expValues = Array.from(logits).map((l) => Math.exp(l));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  const probabilities = expValues.map((exp) => exp / sumExp);

  // Get prediction (argmax)
  const predIdx = probabilities.indexOf(Math.max(...probabilities));
  const genderLabels = ["Female", "Male"];
  const confidence = probabilities[predIdx];

  const endTime = performance.now();
  const duration = endTime - startTime;

  return {
    label: genderLabels[predIdx],
    confidence: confidence,
    probabilities: {
      female: probabilities[0],
      male: probabilities[1],
    },
    duration,
  };
}

export async function classifyExpression(session: InferenceSession, inputData: Float32Array) {
  const startTime = performance.now();

  const tensor = new Tensor("float32", inputData, [1, 3, 224, 224]);
  const outputs = await session.run({ input: tensor });

  console.log("Expression model raw outputs:", outputs);

  // Get output logits
  let logits: Float32Array | undefined;
  if (outputs["output"]) {
    logits = outputs["output"].data as Float32Array;
  } else if (outputs["logits"]) {
    logits = outputs["logits"].data as Float32Array;
  }

  if (!logits || logits.length !== 7) {
    throw new Error(`Expected 7 expression class logits. Got: ${logits?.length || 0}`);
  }

  console.log("Raw expression logits:", Array.from(logits));

  // Apply softmax to get probabilities
  const expValues = Array.from(logits).map((l) => Math.exp(l));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  const probabilities = expValues.map((exp) => exp / sumExp);

  // Get prediction (argmax)
  const predIdx = probabilities.indexOf(Math.max(...probabilities));
  const expressionLabels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];
  const confidence = probabilities[predIdx];

  const endTime = performance.now();
  const duration = endTime - startTime;

  return {
    label: expressionLabels[predIdx],
    confidence: confidence,
    probabilities: {
      angry: probabilities[0],
      disgust: probabilities[1],
      fear: probabilities[2],
      happy: probabilities[3],
      neutral: probabilities[4],
      sad: probabilities[5],
      surprise: probabilities[6],
    },
    duration,
  };
}

export async function classifyExpressionEfficient(session: InferenceSession, inputData: Float32Array) {
  // Same output structure as classifyExpression but uses the EfficientNet-based model.
  return classifyExpression(session, inputData);
}
