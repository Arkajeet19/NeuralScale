import { useState } from "react";
import UploadZone from "./components/UploadZone.jsx";
import ComparisonViewer from "./components/ComparisonViewer.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";

const API_URL = "http://localhost:8000";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("gan");

  const handleFile = async (file) => {
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upscale`, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Request failed");
      }
      const data = await res.json();
      setResult(data);
      setMode("gan");
    } catch (e) {
      setError(`${e.message}. Is the backend running at ${API_URL}?`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-base">
      <header className="border-b border-border px-6 py-4 flex items-baseline justify-between">
        <div>
          <h1 className="font-display font-semibold text-lg text-ink">NeuralScale</h1>
          <p className="font-mono text-[11px] text-muted mt-0.5">
            4x super-resolution &middot; EDSR-baseline + SRGAN fine-tune
          </p>
        </div>
        <a
          href="https://github.com/Arkajeet19/NeuralScale"
          target="_blank"
          rel="noreferrer"
          className="font-mono text-xs text-muted hover:text-teal transition-colors duration-150"
        >
          Source
        </a>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-6">
        <div className="flex flex-col gap-6">
          <UploadZone onFileSelected={handleFile} disabled={loading} />

          {loading && (
            <p className="font-mono text-xs text-teal">Running inference on three outputs...</p>
          )}
          {error && (
            <p className="font-mono text-xs text-amber">{error}</p>
          )}

          {result && (
            <ComparisonViewer
              images={{
                bicubic: result.bicubic_b64,
                plain: result.plain_b64,
                gan: result.gan_b64,
              }}
              mode={mode}
              onModeChange={setMode}
            />
          )}
        </div>

        <aside className="bg-panel border border-border rounded-sm p-4 h-fit">
          <h2 className="font-display text-xs text-muted uppercase tracking-wide mb-3">
            Readouts
          </h2>
          <MetricsPanel result={result} mode={mode} device={result?.device} />
        </aside>
      </main>
    </div>
  );
}
