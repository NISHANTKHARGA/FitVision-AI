import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Save, Loader2, CheckCircle2, Shirt, Trash2, Image as ImageIcon, AlertTriangle, History,
  Cpu, HardDrive, FileJson,
} from 'lucide-react';
import { getProfile, putProfile, getHistory, deleteHistory, ApiError, apiHealth } from '../lib/api';

const FIELDS = [
  { key: 'height_cm', label: 'Height', min: 120, max: 230 },
  { key: 'chest_cm', label: 'Chest circumference', min: 70, max: 160 },
  { key: 'waist_cm', label: 'Waist circumference', min: 55, max: 160 },
  { key: 'shoulder_cm', label: 'Shoulder width', min: 30, max: 60 },
  { key: 'hip_cm', label: 'Hip circumference', min: 70, max: 170 },
];

const FITS = [['slim', 'Slim'], ['regular', 'Regular'], ['oversized', 'Oversized']];

export default function Profile() {
  const [form, setForm] = useState({ height_cm: null, chest_cm: null, waist_cm: null, shoulder_cm: null, hip_cm: null, preferred_fit: 'regular' });
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [histLoading, setHistLoading] = useState(true);
  const [engine, setEngine] = useState(null);

  useEffect(() => {
    getProfile().then((p) => setForm((f) => ({ ...f, ...p }))).catch(() => {});
    getHistory().then((h) => setHistory(h.items || [])).catch(() => {}).finally(() => setHistLoading(false));
    apiHealth().then(setEngine).catch(() => {});
  }, []);

  const save = async () => {
    setSaving(true); setError(null);
    try {
      const payload = { ...form };
      for (const k of Object.keys(payload)) if (payload[k] === '' || payload[k] == null) payload[k] = null;
      await putProfile(payload);
      setSavedAt(new Date());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e.message);
    } finally {
      setSaving(false);
    }
  };

  const remove = async (id) => {
    if (!confirm('Delete this history item and its images from this machine?')) return;
    await deleteHistory(id);
    setHistory((h) => h.filter((x) => x.id !== id));
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <p className="eyebrow">Your Profile</p>
      <h1 className="mt-1.5 font-display text-3xl font-medium tracking-tight sm:text-4xl">Measurements, preferences, history.</h1>

      <div className="mt-8 grid gap-6 lg:grid-cols-5">
        {/* Measurements form */}
        <section className="card p-6 lg:col-span-2">
          <h2 className="flex items-center gap-2 font-display text-lg font-semibold"><Shirt size={18} className="text-accent" /> Body measurement profile</h2>
          <p className="mt-1 text-[0.78rem] leading-5 text-muted">Stored locally in SQLite on this machine. Used only to pre-fill the studio and power fit analysis. Never uploaded anywhere.</p>

          <div className="mt-5 grid grid-cols-2 gap-4">
            {FIELDS.map((f) => (
              <label key={f.key} className="block">
                <span className="label">{f.label}</span>
                <div className="relative">
                  <input
                    type="number" inputMode="decimal" min={f.min} max={f.max}
                    className="input pr-9"
                    value={form[f.key] ?? ''}
                    placeholder="—"
                    onChange={(e) => setForm({ ...form, [f.key]: e.target.value === '' ? null : Number(e.target.value) })}
                  />
                  <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-faint">cm</span>
                </div>
              </label>
            ))}
          </div>

          <div className="mt-4">
            <span className="label">Preferred fit</span>
            <div className="grid grid-cols-3 gap-2">
              {FITS.map(([v, l]) => (
                <button
                  key={v}
                  onClick={() => setForm({ ...form, preferred_fit: v })}
                  className={`rounded-xl border px-3 py-2.5 text-sm font-semibold capitalize transition-all ${form.preferred_fit === v ? 'border-ink bg-ink text-paper' : 'border-line bg-[#FFFDFB] text-muted hover:border-faint'}`}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>

          {error && <p className="mt-3 flex items-start gap-2 rounded-xl bg-red-50 px-3 py-2.5 text-[0.78rem] text-red-900"><AlertTriangle size={15} className="mt-0.5 shrink-0" />{error}</p>}

          <button onClick={save} disabled={saving} className="btn-primary mt-5 w-full justify-center">
            {saving ? <Loader2 size={17} className="animate-spin" /> : savedAt ? <CheckCircle2 size={17} /> : <Save size={17} />}
            {saving ? 'Saving…' : savedAt ? `Saved at ${savedAt.toLocaleTimeString()}` : 'Save profile'}
          </button>

          <p className="mt-3 text-[0.68rem] leading-5 text-faint">Photo-based measurement estimates from the studio are never written here automatically — only what you type and save.</p>
        </section>

        {/* Inference engine / bring-your-own-model */}
        <section className="card p-6 lg:col-span-2 lg:col-start-1">
          <h2 className="flex items-center gap-2 font-display text-lg font-semibold"><Cpu size={18} className="text-accent" /> Inference engine</h2>
          <p className="mt-1 text-[0.78rem] leading-5 text-muted">Which model renders your try-ons on this machine. Swap it for your own fine-tuned checkpoint with one environment variable — see <code className="rounded bg-line-soft px-1 font-mono text-[0.72rem]">docs/OWN_MODEL.md</code>.</p>
          <dl className="mt-4 space-y-3 text-[0.8rem]">
            <div className="flex items-start justify-between gap-3 border-b border-line-soft pb-3">
              <dt className="font-semibold text-muted">Model</dt>
              <dd className="text-right font-medium">{engine?.fashn?.label ?? '—'}</dd>
            </div>
            <div className="flex items-start justify-between gap-3 border-b border-line-soft pb-3">
              <dt className="flex items-center gap-1.5 font-semibold text-muted"><HardDrive size={13} /> Weights</dt>
              <dd className="text-right font-mono text-[0.72rem] leading-5">{engine?.fashn?.weights_size_mb ? `${(engine.fashn.weights_size_mb / 1024).toFixed(2)} GB` : 'not found'}<span className="block text-faint">{engine?.fashn?.weights_dir}</span></dd>
            </div>
            <div className="flex items-start justify-between gap-3">
              <dt className="font-semibold text-muted">Status</dt>
              <dd className="text-right">{engine?.fashn?.loaded ? <span className="inline-flex items-center gap-1.5 font-semibold text-sage"><CheckCircle2 size={13} /> Loaded &amp; ready</span> : <span className="inline-flex items-center gap-1.5 text-muted"><Loader2 size={13} /> Loads on first render / at startup</span>}</dd>
            </div>
          </dl>
          <div className="mt-4 rounded-xl bg-[#F3F0E9] px-4 py-3 text-[0.7rem] leading-5 text-muted">
            <p className="flex items-center gap-1.5 font-bold text-ink"><FileJson size={13} /> To use your own trained model</p>
            <p className="mt-1">1 · Train/LoRA-fine-tune an FASHN-architecture checkpoint on your dataset (docs/OWN_MODEL.md).</p>
            <p>2 · Verify it: <code className="font-mono">python backend/scripts/verify_custom_weights.py --weights-dir D:\path --smoke</code></p>
            <p>3 · Restart the backend with <code className="font-mono">$env:FASHN_WEIGHTS=&quot;D:\path&quot;</code> — done. Reports keep working unchanged.</p>
          </div>
        </section>

        {/* History */}
        <section className="lg:col-span-3">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 font-display text-lg font-semibold"><History size={18} className="text-accent" /> Try-on history</h2>
            <span className="text-[0.72rem] text-muted">{history.length} saved {history.length === 1 ? 'result' : 'results'}</span>
          </div>

          {histLoading ? (
            <div className="mt-6 flex items-center justify-center py-16 text-muted"><Loader2 className="animate-spin" size={22} /></div>
          ) : history.length === 0 ? (
            <div className="card mt-4 flex flex-col items-center px-6 py-14 text-center">
              <ImageIcon size={34} className="text-faint" />
              <h3 className="mt-3 font-display text-lg font-semibold">Nothing saved yet</h3>
              <p className="mt-1 max-w-sm text-[0.82rem] leading-6 text-muted">Unsaved try-ons expire after 24 hours. Tick “Save to history” in the studio — or tap “Save to history” on a result — to keep it here.</p>
              <Link to="/studio" className="btn-primary mt-5">Open the Try-On Studio</Link>
            </div>
          ) : (
            <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
              {history.map((item) => (
                <figure key={item.id} className="card group overflow-hidden">
                  <div className="relative aspect-[2/3] w-full overflow-hidden bg-[#EDEAE2]">
                    <img src={item.image_url} alt="Saved try-on result" className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.04]" loading="lazy" />
                    <button onClick={() => remove(item.id)} aria-label="Delete" className="absolute right-2 top-2 rounded-full bg-black/55 p-2 text-white opacity-0 backdrop-blur transition-opacity group-hover:opacity-100 hover:bg-red-600">
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <figcaption className="px-3 py-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[0.72rem] font-bold capitalize">{item.category === 'tops' ? 'Upper body' : 'Lower body'} · {item.fit}</span>
                      {item.has_measurements && <span className="rounded-full bg-line-soft px-2 py-0.5 text-[0.58rem] font-bold uppercase tracking-wider text-muted">measured</span>}
                    </div>
                    <p className="mt-0.5 text-[0.65rem] text-faint">{item.created_at ? new Date(item.created_at).toLocaleString() : ''}</p>
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
