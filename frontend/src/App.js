// src/App.js
import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import IndiaMap from "./map/IndiaMap";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<IndiaMap />} />
        {/* If you later need a fallback route, add: 
            <Route path="*" element={<Navigate to="/" replace />} /> 
            (import Navigate from react-router-dom)
        */}
      </Routes>
    </Router>
  );
}

export default App;
