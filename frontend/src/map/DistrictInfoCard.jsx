import { useEffect, useState, useRef } from "react";

/* Collapsible right-side drawer with numeric prediction display
   Props:
     - open (boolean | undefined): if boolean, component is controlled. Otherwise internal.
     - onClose (function): callback to call when user clicks close (used when controlled).
     - district (string): district name (optional)
     - prediction (object): optional prediction object provided by parent; if present, the component
                            will use this instead of fetching from the analytics endpoint.
*/
export default function DistrictInfoCard({
    open: openProp,
    onClose,
    district: propDistrictName,
    prediction: propPrediction, // optional; prefer parent's provided prediction when available
}) {
    // controlled vs uncontrolled open state
    const [internalOpen, setInternalOpen] = useState(false);
    const open = typeof openProp === "boolean" ? openProp : internalOpen;

    // district name state (can come from prop or window events)
    const [districtName, setDistrictName] = useState(propDistrictName || null);

    // raw response / derived states (either from propPrediction or from local fetch)
    const [raw, setRaw] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // keep track of mounted to avoid setting state after unmount
    const mountedRef = useRef(true);
    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
        };
    }, []);

    // sync district name when prop changes
    useEffect(() => {
        if (propDistrictName) {
            setDistrictName(propDistrictName);
            // if parent didn't control open, open internally
            if (typeof openProp !== "boolean") {
                setInternalOpen(true);
            }
        }
    }, [propDistrictName, openProp]);

    // sync when parent supplies a prediction prop: adopt it and skip fetching
    useEffect(() => {
        if (propPrediction !== undefined && propPrediction !== null) {
            // parent provided prediction; use it as the canonical data
            setRaw(normalizePredictionShape(propPrediction));
            setError(null);
            setLoading(false);
        }
    }, [propPrediction]);

    // Listen for global district-selected events (backwards compat)
    useEffect(() => {
        function onDistrictSelected(e) {
            const name = e?.detail?.name || e?.detail;
            if (name) {
                setDistrictName(name);
                if (typeof openProp !== "boolean") setInternalOpen(true);
            }
        }
        window.addEventListener("district-selected", onDistrictSelected);
        return () => window.removeEventListener("district-selected", onDistrictSelected);
    }, [openProp]);

    // Listen for drawer-open events (backwards compat)
    useEffect(() => {
        function onDrawerOpen(e) {
            const shouldOpen = !!(e?.detail?.open);
            if (typeof openProp !== "boolean") setInternalOpen(shouldOpen);
        }
        window.addEventListener("drawer-open", onDrawerOpen);
        return () => window.removeEventListener("drawer-open", onDrawerOpen);
    }, [openProp]);

    // Fetch effect: only runs when (1) parent DID NOT provide propPrediction and (2) districtName present
    useEffect(() => {
        // if parent provided prediction, skip fetching (we already sync'd above)
        if (propPrediction !== undefined && propPrediction !== null) {
            return;
        }

        if (!districtName) {
            setRaw(null);
            setError(null);
            setLoading(false);
            return;
        }

        let cancelled = false;
        (async () => {
            if (!mountedRef.current) return;
            setLoading(true);
            setError(null);
            setRaw(null);

            try {
                const q = encodeURIComponent(districtName);
                const resp = await fetch(`/api/analytics/district?name=${q}`, {
                    headers: { Accept: "application/json" },
                });

                const contentType = resp.headers.get("content-type") || "";

                if (!resp.ok) {
                    // try to extract JSON error body then fallback to text
                    let detail;
                    try {
                        detail = await resp.json();
                    } catch {
                        try {
                            detail = await resp.text();
                        } catch {
                            detail = `HTTP ${resp.status} ${resp.statusText}`;
                        }
                    }
                    throw new Error(
                        `Model/analytics service error: ${resp.status} ${resp.statusText} — ${JSON.stringify(
                            detail
                        )}`
                    );
                }

                if (!contentType.includes("application/json")) {
                    const txt = await resp.text();
                    throw new Error(
                        `Unexpected non-JSON response from analytics endpoint: ${txt?.slice?.(0, 512) ?? txt}`
                    );
                }

                const data = await resp.json();
                if (!cancelled && mountedRef.current) {
                    setRaw(normalizePredictionShape(data));
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
    }, [districtName, propPrediction]);

    // Close handler: if parent controls open, call onClose; otherwise close internal
    function handleClose() {
        if (typeof openProp === "boolean") {
            if (onClose) onClose();
        } else {
            setInternalOpen(false);
        }
    }

    // derive prediction value and details in a robust way
    const predVal =
        // try the shape that IndiaMap uses: prediction.predicted_value
        raw?.predicted_value ??
        // or legacy: raw.prediction.score
        raw?.prediction?.score ??
        // or nested raw.raw.predictions[0] style
        (raw?.raw?.predictions && Array.isArray(raw.raw.predictions) ? raw.raw.predictions[0] : null) ??
        null;

    const details = raw?.details || raw?.district?.meta || raw?.district?.details || {};

    return (
        <aside className={`side-drawer ${open ? "open" : ""}`} aria-hidden={!open}>
            <div className="drawer-header">
                <div className="drawer-title">District Insights</div>
                <button className="close-btn" onClick={handleClose} aria-label="Close">
                    ×
                </button>
            </div>

            <div className="drawer-section">
                <div className="kv">
                    <span className="k">District</span>
                    <span className="v">{raw?.district?.name || districtName || "—"}</span>
                </div>

                <div className="kv">
                    <span className="k">Predicted target</span>
                    <span className="v" style={{ fontWeight: 700 }}>
                        {loading ? "Loading…" : predVal != null ? formatNumber(predVal) : "—"}
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
                    <input
                        type="checkbox"
                        onChange={() => {
                            /* future projection TODO */
                        }}
                    />
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

/* Small presentational helpers */
function Badge({ color, label }) {
    return (
        <span className="legend-badge" aria-hidden="true">
            <i className="legend-swatch" style={{ backgroundColor: color, display: "inline-block", width: 12, height: 12, marginRight: 6 }} />
            {label}
        </span>
    );
}

function fmt(v) {
    if (v == null) return "—";
    if (typeof v === "number") return v >= 0 && v <= 1 ? `${Math.round(v * 100)}%` : v.toLocaleString();
    // if numeric string, try to coerce
    const n = Number(v);
    if (!Number.isNaN(n)) return n >= 0 && n <= 1 ? `${Math.round(n * 100)}%` : n.toLocaleString();
    return String(v);
}

function formatNumber(v) {
    if (v == null) return "—";
    const n = Number(v);
    if (!Number.isNaN(n)) return n.toLocaleString();
    return String(v);
}

/* Normalize prediction shapes that may come from different backends:
   - If parent sends { predicted_value, raw, details } we keep it as-is
   - If backend returns { district, details, prediction } we produce a consistent object
*/
function normalizePredictionShape(input) {
    if (!input) return null;

    // If input already looks normalized
    if (input.predicted_value !== undefined || input.raw !== undefined || input.details !== undefined) {
        return input;
    }

    // If server returned { district, details, prediction }
    if (input.prediction || input.district || input.details) {
        const out = {
            raw: input,
            details: input.details ?? (input.district?.meta ?? null),
        };

        // Map common prediction shapes
        if (input.prediction && typeof input.prediction === "object") {
            if (input.prediction.score !== undefined) {
                out.predicted_value = input.prediction.score;
            } else if (input.prediction.raw?.predictions && Array.isArray(input.prediction.raw.predictions)) {
                out.predicted_value = input.prediction.raw.predictions[0];
            } else {
                // keep prediction object inside raw
                out.prediction = input.prediction;
            }
        } else if (input.predicted_value !== undefined) {
            out.predicted_value = input.predicted_value;
        }

        // attach district object if available
        if (input.district) out.district = input.district;

        return out;
    }

    // Fallback: wrap the input
    return { raw: input, predicted_value: input.predicted_value ?? null, details: input.details ?? null };
}
