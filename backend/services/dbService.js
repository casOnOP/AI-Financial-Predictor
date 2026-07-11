// backend/services/dbService.js
import { readFile } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// where to find JSON metadata (adjust if your structure differs)
const DATA_DIR = process.env.BACKEND_DATA_DIR || path.join(__dirname, "..", "data");
const DISTRICTS_FILE = path.join(DATA_DIR, "districts.json");
const SCHEMES_FILE = path.join(DATA_DIR, "schemes.json");

/**
 * Expected structure:
 *  - districts.json: [ { "id": "KA-UDUPI", "state": "Karnataka", "name": "Udupi", "meta": { ... } }, ... ]
 *  - schemes.json: [ { "id": "S001", "title": "PM Kisan", "description": "...", "criteria": { ... } }, ... ]
 *
 * If you don't have these yet, create small sample files first (see README below).
 */

let _cache = {
    districts: null, // array
    districtIndexById: null, // map
    schemes: null,
    initialized: false,
};

async function _loadJson(filePath) {
    const raw = await readFile(filePath, { encoding: "utf-8" });
    return JSON.parse(raw);
}

async function _ensureLoaded() {
    if (_cache.initialized) return;

    try {
        const [districts, schemes] = await Promise.all([
            _loadJson(DISTRICTS_FILE).catch(() => []),
            _loadJson(SCHEMES_FILE).catch(() => []),
        ]);

        _cache.districts = Array.isArray(districts) ? districts : [];
        _cache.schemes = Array.isArray(schemes) ? schemes : [];

        _cache.districtIndexById = new Map();
        for (const d of _cache.districts) {
            if (d.id) _cache.districtIndexById.set(String(d.id).toLowerCase(), d);
        }
    } catch (err) {
        console.error("dbService: error loading data files:", err);
        _cache.districts = _cache.districts || [];
        _cache.schemes = _cache.schemes || [];
        _cache.districtIndexById = _cache.districtIndexById || new Map();
    }

    _cache.initialized = true;
}

/* Public API */

/**
 * Return all districts (shallow copy)
 */
export async function getAllDistricts() {
    await _ensureLoaded();
    return [..._cache.districts];
}

/**
 * Lookup district by id (case-insensitive)
 */
export async function getDistrictById(id) {
    if (!id) return null;
    await _ensureLoaded();
    return _cache.districtIndexById.get(String(id).toLowerCase()) ?? null;
}

/**
 * Fuzzy lookup by name: simple contains match (case-insensitive).
 * Returns the first match or null if none found. Use getDistrictsByName for all matches.
 */
export async function findDistrictByName(name) {
    if (!name) return null;
    await _ensureLoaded();
    const q = String(name).toLowerCase();
    // exact id match?
    if (_cache.districtIndexById.has(q)) return _cache.districtIndexById.get(q);

    // exact name match
    const exact = _cache.districts.find(
        (d) => String(d.name || "").toLowerCase() === q
    );
    if (exact) return exact;

    // contains match
    const contains = _cache.districts.find((d) =>
        String(d.name || "").toLowerCase().includes(q)
    );
    return contains || null;
}

/**
 * Return all districts matching name query (for UI autocomplete)
 */
export async function getDistrictsByNameQuery(name, limit = 20) {
    await _ensureLoaded();
    if (!name) return _cache.districts.slice(0, limit);
    const q = String(name).toLowerCase();
    return _cache.districts
        .filter((d) => String(d.name || "").toLowerCase().includes(q))
        .slice(0, limit);
}

/**
 * Get scheme metadata by id
 */
export async function getSchemeById(schemeId) {
    await _ensureLoaded();
    return _cache.schemes.find((s) => String(s.id).toLowerCase() === String(schemeId).toLowerCase()) || null;
}

/**
 * Simple helper to "enrich" a district object with a numeric model prediction
 * and a chosen recommended scheme. This function does NOT call your ML model;
 * it just maps a numeric score -> scheme id using simple thresholds.
 *
 * You can replace this logic with more advanced business rules.
 *
 * @param {Object} district - district metadata (as returned by getDistrictById)
 * @param {number} score - numeric model prediction (higher -> more eligible)
 * @returns {Object} enriched object { district, prediction: { score, schemeId, scheme, confidence } }
 */
export async function enrichDistrictWithPrediction(district, score) {
    await _ensureLoaded();
    const s = Number(score ?? NaN);
    if (!district || Number.isNaN(s)) {
        return { district, prediction: null };
    }

    // Example threshold mapping (tune to your model's meaning)
    // score < 0.2  => low  => scheme S_LOW
    // 0.2 <= s < 0.6 => medium => S_MED
    // s >= 0.6 => high => S_HIGH
    // If you have more complex scheme logic, implement it in another module and call it here.
    let schemeId = null;
    let confidence = Math.min(Math.max(s, 0), 1); // clamp 0..1 for UI
    if (s >= 0.6) schemeId = _cache.schemes.find((x) => x.tag === "high")?.id ?? _cache.schemes[0]?.id;
    else if (s >= 0.2) schemeId = _cache.schemes.find((x) => x.tag === "medium")?.id ?? _cache.schemes[0]?.id;
    else schemeId = _cache.schemes.find((x) => x.tag === "low")?.id ?? _cache.schemes[0]?.id;

    const scheme = schemeId ? await getSchemeById(schemeId) : null;

    return {
        district,
        prediction: {
            score: s,
            schemeId,
            scheme,
            confidence,
        },
    };
}

/**
 * Force reload of JSON files (useful during development)
 */
export async function reloadData() {
    _cache = {
        districts: null,
        districtIndexById: null,
        schemes: null,
        initialized: false,
    };
    await _ensureLoaded();
    return true;
}
