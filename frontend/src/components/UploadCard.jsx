import { useCallback, useEffect, useRef, useState } from 'react';
import { UploadCloud, X, Image as ImageIcon } from 'lucide-react';

/**
 * Drag-and-drop image upload with preview.
 */
export default function UploadCard({ label, hint, file, previewUrl, onFile, onClear, aspect = 'aspect-[3/4]' }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const pick = useCallback((f) => {
    if (!f) return;
    if (!f.type.startsWith('image/')) return;
    onFile(f);
  }, [onFile]);

  const url = previewUrl || (file ? URL.createObjectURL(file) : null);
  useEffect(() => () => { if (previewUrl == null && url) URL.revokeObjectURL(url); }, [url, previewUrl]);

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="label !mb-0">{label}</span>
        {file && (
          <button onClick={onClear} className="inline-flex items-center gap-1 text-[0.72rem] font-medium text-muted transition-colors hover:text-accent">
            <X size={12} /> Remove
          </button>
        )}
      </div>

      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]); }}
        className={`group relative flex ${aspect} w-full cursor-pointer items-center justify-center overflow-hidden rounded-2xl border border-dashed transition-all duration-300 ${
          dragging ? 'border-accent bg-accent-soft/40 scale-[1.01]' : 'border-line bg-[#FFFDFB] hover:border-faint'
        }`}
      >
        {url ? (
          <>
            <img src={url} alt={label} className="absolute inset-0 h-full w-full object-contain" />
            <div className="absolute inset-0 flex items-end justify-end bg-gradient-to-t from-black/35 via-transparent to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100">
              <span className="m-3 inline-flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1.5 text-[0.72rem] font-medium text-white backdrop-blur">
                <ImageIcon size={12} /> Replace image
              </span>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-3 px-6 text-center">
            <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-line-soft text-muted transition-colors group-hover:bg-accent-soft group-hover:text-accent">
              <UploadCloud size={26} strokeWidth={1.5} />
            </span>
            <div>
              <p className="text-sm font-semibold text-ink">Drop an image, or browse</p>
              <p className="mt-1 max-w-[16rem] text-[0.72rem] leading-5 text-muted">{hint}</p>
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => pick(e.target.files?.[0])}
        />
      </div>
    </div>
  );
}
