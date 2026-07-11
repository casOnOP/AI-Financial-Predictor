// backend/services/modelService.js
import axios from "axios";

const PYTHON_INFERENCE_URL = process.env.PYTHON_INFERENCE_URL || process.env.MODEL_API || "http://localhost:8001/predict";
const TIMEOUT_MS = parseInt(process.env.MODEL_TIMEOUT_MS || "30000", 10);

export async function predictBatch(rows) {
    if (!Array.isArray(rows)) {
        throw new Error("predictBatch expects an array of rows");
    }
    try {
        const resp = await axios.post(
            PYTHON_INFERENCE_URL,
            { rows },
            { timeout: TIMEOUT_MS, headers: { "Content-Type": "application/json" } }
        );
        if (resp.status === 200 && resp.data && Array.isArray(resp.data.predictions)) {
            return resp.data.predictions;
        }
        throw new Error(`Unexpected response from model service: ${resp.status}`);
    } catch (err) {
        // Log more details for debugging
        if (err.response) {
            console.error("Model service error response data:", JSON.stringify(err.response.data, null, 2));
            const serverDetail = err.response.data?.detail || err.response.data;
            throw new Error(`Model service request failed: ${err.response.status} ${err.response.statusText} — ${typeof serverDetail === "string" ? serverDetail : JSON.stringify(serverDetail)}`);
        } else {
            throw new Error(`Model service request failed: ${err.message}`);
        }
    }
}
