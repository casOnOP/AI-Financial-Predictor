// DistrictInfoCard.jsx
import { useEffect, useState, useRef } from "react";

/* Collapsible right-side drawer with numeric prediction display */
export default function DistrictInfoCard({ open, onClose, district: propDistrictName }) {
    const [districtName, setDistrictName] = useState(propDistrictName || null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [raw, setRaw] = useState(null); // entire response
    const mountedRef = useRef(true);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
        };
    }, []);

    // If parent passes district prop, prefer it
    useEffect(() => {
        if (propDistrictName) {
            setDistrictName(propDistrictName);
        }
    }, [propDistrictName]);

    // Also listen for global "district-selected" events dispatched by the map
    useEffect(() => {
        function onDistrictSelected(e) {
            const name = e?.detail?.name || e?.detail;
            if (name) setDistrictName(name);
        }
        window.addEventListener("district-selected", onDistrictSelected);
        return () => window.removeEventListener("district-selected", onDistrictSelected);
    }, []);

    useEffect(() => {
        if (!districtName) {
            setRaw(null);
            setError(null);
            setLoading(false);
            return;
        }

        let cancelled = false;
        (async () => {
            setLoading(true);
            setError(null);
            setRaw(null);

            try {
                const q = encodeURIComponent(districtName);
                const resp = await fetch(`/api/analytics/district?name=${q}`, {
                    headers: { "Accept": "application/json" },
                });

                const contentType = resp.headers.get("content-type") || "";
                if (!resp.ok) {
                    // attempt to read JSON or text error payload
                    let detail;
                    try {
                        detail = await resp.json();
                    } catch {
                        detail = await resp.text();
                    }
                    throw new Error(`Model/analytics service error: ${resp.status} ${resp.statusText} — ${JSON.stringify(detail)}`);
                }

                if (!contentType.includes("application/json")) {
                    // helpful debug message if backend returns HTML (common cause of EOF '<' error)
                    const txt = await resp.text();
                    throw new Error(`Unexpected non-JSON response from analytics endpoint: ${txt.slice(0, 512)}`);
                }

                const data = await resp.json();
                if (!cancelled && mountedRef.current) {
                    setRaw(data);
                }
            } catch (err) {
                if (!cancelled && mountedRef.current) {
                    console.error("DistrictInfoCard fetch error:", err);
                    setError(String(err.message || err));
                }
            } finally {
                if (!cancelled && mountedRef.current) setLoading(false);
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [districtName]);

    const predVal = raw?.prediction?.score ?? null;
    const details = raw?.details || raw?.district?.meta || {};

    return (
        <aside className={`side-drawer ${open ? "open" : ""}`}>
            <div className="drawer-header">
                <div className="drawer-title">District Insights</div>
                <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
            </div>

            <div className="drawer-section">
                <div className="kv">
                    <span className="k">District</span>
                    <span className="v">{raw?.district?.name || districtName || "—"}</span>
                </div>

                <div className="kv">
                    <span className="k">Predicted target</span>
                    <span className="v" style={{ fontWeight: 700 }}>
                        {loading ? "Loading…" : predVal != null ? Number(predVal).toLocaleString() : "—"}
                    </span>
                </div>

                <div className="kv">
                    <span className="k">Status</span>
                    <span className="v" style={{ color: error ? "#c53030" : undefined }}>
                        {error ? "Error" : loading ? "Running" : "OK"}
                    </span>
                </div>

                {error && (
                    <div style={{ marginTop: 8, color: "#c53030" }}>
                        <small>{error}</small>
                    </div>
                )}
            </div>

            <div className="divider" />

            <div className="drawer-section">
                <div className="section-title">Key Metrics (if available)</div>
                <ul className="metric-list">
                    <li>Literacy rate: {fmt(details.literacy_rate)}</li>
                    <li>Agriculture dependence: {fmt(details.agri_dependence)}</li>
                    <li>Bank branches: {fmt(details.bank_branches)}</li>
                    <li>Employment rate: {fmt(details.employment_rate)}</li>
                    <li>Avg income: {fmt(details.avg_income)}</li>
                </ul>
            </div>

            <div className="divider" />

            <div className="drawer-section">
                <label className="toggle">
                    <input type="checkbox" onChange={() => { /* future projection TODO */ }} />
                    <span>Future Projection (beta)</span>
                </label>
                <p className="hint">Toggle to view scheme under projected next-year indicators.</p>
            </div>

            <div className="legend">
                <div className="legend-title">Visualization Legend</div>
                <div className="legend-items">
                    <Badge color="#22c55e" label="Low" />
                    <Badge color="#f97316" label="Medium" />
                    <Badge color="#ef4444" label="High" />
                </div>
            </div>
        </aside>
    );
}

function Badge({ color, label }) {
    return (
        <span className="legend-badge">
            <i className="legend-swatch" style={{ backgroundColor: color }} />
            {label}
        </span>
    );
}

function fmt(v) {
    if (v == null) return "—";
    if (typeof v === "number") return v >= 0 && v <= 1 ? `${Math.round(v * 100)}%` : v.toLocaleString();
    return String(v);
}
