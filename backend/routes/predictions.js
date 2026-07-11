// backend/routes/predictions.js
import express from "express";
import { predictBatch } from "../services/modelService.js";

const router = express.Router();

/**
 * POST /api/predict
 * Body: { rows: [ { feature1: value1, ... }, ... ] }
 * Returns: { predictions: [number, ...] }
 */
router.post("/", async (req, res) => {
    try {
        const { rows } = req.body;
        if (!rows || !Array.isArray(rows) || rows.length === 0) {
            return res.status(400).json({ error: 'Request body must contain "rows": non-empty array' });
        }
        if (rows.length > 1000) {
            return res.status(400).json({ error: "Max 1000 rows per request" });
        }

        const predictions = await predictBatch(rows);
        return res.json({ predictions });
    } catch (err) {
        console.error("Prediction error:", err.message);
        return res.status(500).json({ error: err.message });
    }
});

// Health check for this route
router.get("/health", (req, res) => {
    res.json({ status: "ok", route: "/api/predict" });
});

export default router;
