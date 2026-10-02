const MODE_LABELS = {
  bicubic: "Bicubic",
  plain: "Model (DIV2K)",
  gan: "Model + GAN (game-finetuned)",
};

function Row({ label, value }) {
  return (
    <div className="flex items-baseline justify-between py-2 border-b border-border last:border-0">
      <span className="font-display text-xs text-muted">{label}</span>
      <span className="font-mono text-sm text-ink">{value}</span>
    </div>
  );
}

export default function MetricsPanel({ result, mode, device }) {
  if (!result) {
    return (
      <div className="font-mono text-xs text-muted leading-relaxed">
        Upload an image to see readouts here -- input/output resolution,
        per-method inference time, and active comparison mode.
      </div>
    );
  }

  const { original_size, output_size, timing_ms } = result;

  return (
    <div>
      <Row label="Viewing" value={MODE_LABELS[mode]} />
      <Row label="Input" value={`${original_size[0]} x ${original_size[1]}`} />
      <Row label="Output" value={`${output_size[0]} x ${output_size[1]}`} />
      <Row label="Scale" value="4x" />
      <Row label="Inference" value={`${timing_ms[mode]} ms`} />
      <Row label="Device" value={device} />

      <div className="mt-4 pt-3 border-t border-border">
        <div className="flex items-center gap-2 text-[11px] font-mono text-muted mb-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-teal" />
          Trained on DIV2K
        </div>
        <div className="flex items-center gap-2 text-[11px] font-mono text-muted">
          <span className="inline-block w-2 h-2 rounded-full bg-amber" />
          GAN-finetuned on 180+ game screenshots
        </div>
      </div>
    </div>
  );
}
