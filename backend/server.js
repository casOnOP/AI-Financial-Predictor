// server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
// body-parser is optional; express.json works fine
// import bodyParser from "body-parser";

// Import Routes
import predictionRoutes from "./routes/predictions.js";
import analyticsRoutes from "./routes/analytics.js";
import campaignRoutes from "./routes/campaigns.js";

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// CORS: allow your frontend during development
app.use(
    cors({
        origin: ["http://localhost:3000", "http://127.0.0.1:3000"],
        credentials: true,
    })
);

// Middleware
app.use(express.json({ limit: "5mb" })); // parse JSON bodies

// Mount routes
app.use("/api/predict", predictionRoutes); // predictionRoutes defines '/' and '/health'
app.use("/api/analytics", analyticsRoutes);
app.use("/api/campaigns", campaignRoutes);

// Health Check
app.get("/health", (req, res) => {
    res.json({ status: "ok", service: "node-backend" });
});

// Default root
app.get("/", (req, res) => {
    res.json({ message: "Backend API is running successfully 🚀" });
});

// Start Server
app.listen(PORT, () => {
    console.log(`✅ Backend server live → http://localhost:${PORT}`);
});
