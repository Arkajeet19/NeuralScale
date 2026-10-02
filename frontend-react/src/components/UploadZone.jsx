import { useRef, useState } from "react";

export default function UploadZone({ onFileSelected, disabled }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelected(file);
  };

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`
        border rounded-sm px-8 py-10 text-center cursor-pointer
        transition-colors duration-150
        ${dragOver ? "border-teal bg-panel2" : "border-border bg-panel"}
        ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-muted"}
      `}
    >
      <p className="font-display text-ink text-sm">
        Drop a low-resolution image, or click to choose a file
      </p>
      <p className="font-mono text-muted text-xs mt-2">
        JPG / PNG &middot; auto-downscaled above 512px on the long edge
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelected(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}
