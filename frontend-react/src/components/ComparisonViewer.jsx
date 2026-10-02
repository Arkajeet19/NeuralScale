import { useRef, useState } from "react";

const MODES = [
  { key: "bicubic", label: "Bicubic" },
  { key: "plain", label: "Model" },
  { key: "gan", label: "Model + GAN" },
];

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

export default function ComparisonViewer({ images, mode, onModeChange }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragState = useRef(null);
  const containerRef = useRef(null);

  const clampZoom = (z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = -e.deltaY * 0.002;
    setZoom((z) => clampZoom(z + delta));
  };

  const handlePointerDown = (e) => {
    if (zoom <= 1) return;
    dragState.current = { startX: e.clientX, startY: e.clientY, pan };
    containerRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!dragState.current) return;
    const dx = e.clientX - dragState.current.startX;
    const dy = e.clientY - dragState.current.startY;
    setPan({
      x: dragState.current.pan.x + dx,
      y: dragState.current.pan.y + dy,
    });
  };

  const handlePointerUp = (e) => {
    dragState.current = null;
    containerRef.current?.releasePointerCapture(e.pointerId);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const currentSrc = images[mode];

  return (
    <div className="flex flex-col gap-3">
      {/* mode switch */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 bg-panel border border-border rounded-sm p-1">
          {MODES.map((m) => (
            <button
              key={m.key}
              onClick={() => onModeChange(m.key)}
              className={`
                font-mono text-xs px-3 py-1.5 rounded-sm transition-colors duration-150
                ${mode === m.key
                  ? m.key === "gan"
                    ? "bg-amber text-base"
                    : "bg-teal text-base"
                  : "text-muted hover:text-ink"}
              `}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 font-mono text-xs text-muted">
          <span>{Math.round(zoom * 100)}%</span>
          <button
            onClick={resetView}
            className="border border-border rounded-sm px-2 py-1 hover:border-muted hover:text-ink transition-colors duration-150"
          >
            Reset view
          </button>
        </div>
      </div>

      {/* viewport */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        className="relative overflow-hidden rounded-sm border border-border bg-panel2 aspect-video select-none"
        style={{ cursor: zoom > 1 ? "grab" : "default" }}
      >
        <img
          src={`data:image/png;base64,${currentSrc}`}
          alt={mode}
          draggable={false}
          className="absolute inset-0 w-full h-full object-contain"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "center",
          }}
        />
      </div>
      <p className="font-mono text-[11px] text-muted">
        Scroll to zoom &middot; drag to pan &middot; switching modes keeps your zoom position
      </p>
    </div>
  );
}
