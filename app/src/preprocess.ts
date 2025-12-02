// src/preprocess.ts
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
