import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// #region agent log
(() => {
  const probe = document.createElement("div");
  probe.className = "flex hidden";
  probe.style.cssText = "position:absolute;left:-9999px;top:0";
  document.documentElement.appendChild(probe);
  const cs = getComputedStyle(probe);
  const bodyBg = getComputedStyle(document.body).backgroundImage || "";
  fetch("http://127.0.0.1:7253/ingest/b4fff296-3ffe-4632-a44f-abd9007d22be", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Debug-Session-Id": "0fd45e",
    },
    body: JSON.stringify({
      sessionId: "0fd45e",
      runId: "style-check",
      hypothesisId: "H1",
      location: "main.tsx:boot",
      message: "tailwind utility application check",
      data: {
        displayFlex: cs.display,
        tailwindLikelyOk: cs.display === "flex",
        bodyHasGradient: bodyBg.includes("gradient"),
      },
      timestamp: Date.now(),
    }),
  }).catch(() => {});
  probe.remove();
})();
// #endregion

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
