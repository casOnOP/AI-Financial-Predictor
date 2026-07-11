// IndiaMap.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./mapStyles.css";
import DistrictInfoCard from "./DistrictInfoCard";

const INDIA_CENTER = [22.9734, 78.6569]; // approximate
const INDIA_ZOOM = 4.5;
const DISTRICT_ZOOM = 7; // optionally keep zoom-threshold behavior

// sanitize a state name into the same filename pattern your splitter used
function safeFilename(name = "") {
    return String(name)
        .replace(/[^0-9A-Za-z_\- ]/g, "") // remove strange chars
        .trim()
        .replace(/[-\s]+/g, "_");
}

export default function IndiaMap() {
    const [statesGeo, setStatesGeo] = useState(null);
    // holds currently-displayed district GeoJSON (could be per-state or full)
    const [currentDistricts, setCurrentDistricts] = useState(null);
    const [error, setError] = useState("");
    const [loadingDistricts, setLoadingDistricts] = useState(false);

    // drawer / selection states
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [selectedDistrictName, setSelectedDistrictName] = useState(null);
    const [prediction, setPrediction] = useState(null); // { loading, predicted_value, error, raw }

    // cache: { "Karnataka": geojson, "Maharashtra": geojson, ... }
    const districtsCacheRef = useRef({});

    const statesRef = useRef();
    const districtsRef = useRef();

    useEffect(() => {
        fetch("/map-layers/india_states.geojson")
            .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
            .then(setStatesGeo)
            .catch(() =>
                setError(
                    "India states GeoJSON not found. Place it at public/map-layers/india_states.geojson"
                )
            );
    }, []);

    const karnatakaStyle = useMemo(
        () => ({
            fillColor: "#f0f8ff",
            weight: 2,
            opacity: 1,
            color: "#1e90ff",
            fillOpacity: 0.5,
        }),
        []
    );

    const defaultStyle = useMemo(
        () => ({
            fillColor: "#e6e6e6",
            weight: 1,
            opacity: 1,
            color: "#999",
            fillOpacity: 0.4,
        }),
        []
    );

    const districtStyle = useMemo(
        () => ({
            fillColor: "#ffeedd",
            weight: 0.8,
            opacity: 1,
            color: "#cc5a00",
            fillOpacity: 0.35,
        }),
        []
    );

    function onEachState(feature, layer) {
        const name =
            feature.properties?.NAME_1 ||
            feature.properties?.STATE_NAME ||
            feature.properties?.ST_NM ||
            feature.properties?.name ||
            feature.properties?.shapeName ||
            "State";

        layer.on({
            mouseover: () => {
                layer.setStyle({ weight: 2, color: "#333", fillOpacity: 0.6 });
            },
            mouseout: () => {
                // keep visible highlight for Karnataka if you want — optional
                layer.setStyle(name === "Karnataka" ? karnatakaStyle : defaultStyle);
            },
            click: (e) => {
                // On click: zoom, then load that state's districts
                const map = layer._map;
                if (map) {
                    // fly to clicked point (zoom in)
                    map.flyTo(e.latlng, Math.max(7.5, DISTRICT_ZOOM), { duration: 0.7 });
                }
                loadStateDistricts(name);
            },
        });

        layer.bindTooltip(`<div class="map-preview">${name}</div>`, {
            sticky: true,
            opacity: 0.95,
            className: "scheme-tooltip",
        });

        layer.setStyle(name === "Karnataka" ? karnatakaStyle : defaultStyle);
    }

    function onEachDistrict(feature, layer) {
        const districtName =
            feature.properties?.NAME_2 ||
            feature.properties?.DISTRICT ||
            feature.properties?.name ||
            feature.properties?.shapeName ||
            "District";

        layer.on({
            mouseover: () => {
                layer.setStyle({ weight: 2, color: "#333", fillOpacity: 0.65 });
            },
            mouseout: () => {
                layer.setStyle(districtStyle);
            },
            click: (e) => {
                const map = layer._map;
                if (map) {
                    map.fitBounds(layer.getBounds(), { maxZoom: 12 });
                }

                // --- FIX: call the handler directly so the drawer/prediction flow runs ---
                handleDistrictClick(districtName, feature);
            },
        });

        layer.bindTooltip(`<div class="map-preview">${districtName}</div>`, {
            sticky: true,
            opacity: 0.95,
            className: "scheme-tooltip",
        });

        layer.setStyle(districtStyle);
    }

    // Try to fetch per-state districts; fallback to global if not present
    async function loadStateDistricts(stateName) {
        if (!stateName) return;
        if (districtsCacheRef.current[stateName]) {
            // use cached
            setCurrentDistricts(districtsCacheRef.current[stateName]);
            return;
        }

        setLoadingDistricts(true);
        const safe = safeFilename(stateName);
        const perStatePath = `/map-layers/split_by_state/${safe}_districts.geojson`;
        const globalPath = `/map-layers/india_districts.geojson`;

        try {
            // Try per-state file first
            let resp = await fetch(perStatePath);
            if (resp.ok) {
                const gj = await resp.json();
                districtsCacheRef.current[stateName] = gj;
                setCurrentDistricts(gj);
                setLoadingDistricts(false);
                return;
            }

            // if per-state returned 404 or error, fallback to slicing global file
            console.warn(`Per-state file not found: ${perStatePath}, response: ${resp.status}`);
        } catch (err) {
            console.warn("Error fetching per-state file:", err);
        }

        // Fallback: try global file and filter for the clicked state (client-side)
        try {
            const gresp = await fetch(globalPath);
            if (!gresp.ok) throw new Error("global not found");
            const ggj = await gresp.json();

            // filter features where NAME_1 (or similar) matches stateName
            const stateKeyCandidates = ["NAME_1", "STATE_NAME", "st_nm", "NAME1", "state_name", "shapeGroup"];
            const filtered = {
                type: "FeatureCollection",
                features: (ggj.features || []).filter((f) => {
                    const props = f.properties || {};
                    // rough match: any candidate equals (case-insensitive) the stateName or contains it
                    return stateKeyCandidates.some((k) => {
                        if (!props[k]) return false;
                        try {
                            return String(props[k]).toLowerCase() === String(stateName).toLowerCase() ||
                                String(props[k]).toLowerCase().includes(String(stateName).toLowerCase());
                        } catch {
                            return false;
                        }
                    });
                }),
            };

            // if filtered empty, keep original global as last resort
            const result = filtered.features.length ? filtered : ggj;
            districtsCacheRef.current[stateName] = result;
            setCurrentDistricts(result);
        } catch (err) {
            console.error("Failed to load global districts fallback:", err);
            // show a brief error toast via state (not to block user)
            setError("Could not load district boundaries for that state.");
        } finally {
            setLoadingDistricts(false);
        }
    }

    // UPDATED: call analytics endpoint (district metadata + prediction)
    async function handleDistrictClick(districtName, feature) {
        setSelectedDistrictName(districtName);
        setDrawerOpen(true);
        setPrediction({ loading: true });

        try {
            // Prefer name query — backend will also accept id if you prefer (id param)
            const url = `/api/analytics/district?name=${encodeURIComponent(districtName)}`;
            const resp = await fetch(url);
            if (!resp.ok) {
                const body = await resp.text().catch(() => null);
                throw new Error(`Analytics request failed: ${resp.status} ${resp.statusText}${body ? " — " + body : ""}`);
            }

            const data = await resp.json();
            // Expected: { district, details, prediction }
            const predRaw = data?.prediction;
            let predObj = null;
            if (predRaw) {
                if (predRaw.score !== undefined) {
                    predObj = { loading: false, predicted_value: predRaw.score, raw: predRaw };
                } else if (predRaw.error) {
                    predObj = { loading: false, predicted_value: null, error: predRaw.error, raw: predRaw };
                } else if (predRaw.raw?.predictions && Array.isArray(predRaw.raw.predictions)) {
                    predObj = { loading: false, predicted_value: predRaw.raw.predictions[0], raw: predRaw.raw };
                } else {
                    predObj = { loading: false, predicted_value: null, raw: predRaw };
                }
            } else {
                predObj = { loading: false, predicted_value: null, raw: null };
            }

            // Attach details to raw for DistrictInfoCard (it reads prediction.raw.details)
            if (!predObj.raw) predObj.raw = {};
            predObj.raw.details = data.details ?? (data.district?.meta ?? null);

            setPrediction(predObj);
        } catch (err) {
            console.error("Analytics fetch error:", err);
            setPrediction({ loading: false, predicted_value: null, error: String(err.message || err), raw: null });
        }
    }

    // Optionally, close drawer handler
    function closeDrawer() {
        setDrawerOpen(false);
        setSelectedDistrictName(null);
        setPrediction(null);
    }

    // Optionally, a map events watcher to load districts when zoom >= threshold
    function DistrictZoomWatcher() {
        const map = useMapEvents({
            zoomend: () => {
                const z = map.getZoom();
                if (z >= DISTRICT_ZOOM) {
                    // if no districts currently displayed, and user didn't click a state, just load full global (or skip)
                    if (!currentDistricts && !loadingDistricts) {
                        // load global (or choose to wait until state click). We keep this conservative and do nothing.
                        // Uncomment to auto-load global when zooming in:
                        // loadStateDistricts(""); // this will fetch global fallback
                    }
                } else {
                    // zoomed out: keep districts (or clear to free memory). This design keeps them visible until user navigates away.
                    // If you want to unload on zoom-out, uncomment:
                    // setCurrentDistricts(null);
                }
            },
        });
        return null;
    }

    return (
        <div className="map-page">
            <header className="dash-header">
                <h1 className="dash-title">AI Financial Recommender — India</h1>
                <div className="dash-sub">Click a state to load its districts (Karnataka no longer special-cased)</div>
            </header>

            <MapContainer
                center={INDIA_CENTER}
                zoom={INDIA_ZOOM}
                className="map-root"
                scrollWheelZoom={true}
                zoomSnap={0.25}
                zoomDelta={0.5}
                wheelPxPerZoomLevel={60}
            >
                <TileLayer
                    attribution="&copy; OpenStreetMap"
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                <DistrictZoomWatcher />

                {statesGeo ? (
                    <GeoJSON ref={statesRef} data={statesGeo} onEachFeature={onEachState} />
                ) : (
                    <div className="map-toast">{error || "Loading map…"}</div>
                )}

                {loadingDistricts && (
                    <div className="map-toast">Loading districts…</div>
                )}

                {currentDistricts ? (
                    <GeoJSON ref={districtsRef} data={currentDistricts} onEachFeature={onEachDistrict} />
                ) : null}
            </MapContainer>

            {/* Drawer (outside the MapContainer so it overlays) */}
            <DistrictInfoCard
                open={drawerOpen}
                onClose={closeDrawer}
                district={selectedDistrictName}
                prediction={prediction}
            />
        </div>
    );
}
