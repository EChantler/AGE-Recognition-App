import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import * as ort from 'onnxruntime-web'

// Disable multi-threading to avoid worker issues in production
ort.env.wasm.numThreads = 1
ort.env.wasm.proxy = false

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
