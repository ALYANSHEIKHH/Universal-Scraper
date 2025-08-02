// src/types/index.ts
export interface PredictionResult {
  label: string
  confidence: number
  // Add other expected fields here
}


export interface SummaryData {
  label: string
  confidence: number
  details?: Record<string, unknown> // optional object if server returns more
}