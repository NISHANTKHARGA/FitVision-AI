import { useCallback, useRef, useState } from 'react';

/**
 * Before/after comparison slider.
 * `before` = person image (left), `after` = generated try-on (left→right reveal).
 */
export default function BeforeAfter({ beforeSrc, afterSrc, beforeLabel = 'Original', afterLabel = 'Try-On' }) {
  const ref = useRef(null);
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);

  const setFromClientX = useCallback((clientX) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(97, Math.max(3, x)));
  }, []);

  return (
    <div
      ref={ref}
      className="relative select-none overflow-hidden rounded-2xl border border-line bg-ink"
      onPointerDown={(e) => { dragging.current = true; ref.current?.setPointerCapture?.(e.pointerId); setFromClientX(e.clientX); }}
      onPointerMove={(e) => dragging.current && setFromClientX(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
    >
      <img src={afterSrc} alt={afterLabel} className="block h-full w-full object-contain" draggable={false} />

      {/* Before image clipped from the left */}
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={beforeSrc} alt={beforeLabel} className="block h-full w-full object-contain" draggable={false} />
      </div>

      {/* Labels */}
      <span className="absolute left-3 top-3 rounded-full bg-black/55 px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.14em] text-white backdrop-blur-sm">
        {beforeLabel}
      </span>
      <span className="absolute right-3 top-3 rounded-full bg-accent/90 px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.14em] text-white backdrop-blur-sm">
        {afterLabel}
      </span>

      {/* Divider + handle */}
      <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -ml-px w-0.5 bg-paper/90 shadow-[0_0_12px_rgba(0,0,0,0.45)]" />
        <div className="absolute top-1/2 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/50 bg-paper text-ink shadow-lg">
          <svg width="18" height="14" viewBox="0 0 18 14" fill="none">
            <path d="M5 1L1 7l4 6M13 1l4 6-4 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>

      {/* Accessible range */}
      <input
        type="range" min={3} max={97} value={pos} aria-label="Comparison slider"
        onChange={(e) => setPos(Number(e.target.value))}
        className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0"
      />
    </div>
  );
}
