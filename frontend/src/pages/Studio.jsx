import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Shirt, Layers, Sparkles, Ruler, ChevronDown, ChevronUp, FlaskConical,
  Info, AlertTriangle, CheckCircle2, Loader2, Wand2, ArrowRight, BadgeCheck, Timer,
} from 'lucide-react';
import UploadCard from '../components/UploadCard';
import {
  analyzePerson, analyzeGarment, generateTryOn, getProfile, ApiError,
} from '../lib/api';

const PRESETS = [
  { id: 'draft', label: 'Draft', steps: 4, guidance: 1.0, sub: '~10 s · g1.0 · fastest, softest' },
  { id: 'fast', label: 'Fast', steps: 8, guidance: 1.2, sub: '~15 s · g1.2 · speed over polish' },
  { id: 'balanced', label: 'Balanced', steps: 12, guidance: 1.5, sub: '~20 s · g1.5 · solid default' },
  { id: 'best', label: 'Best', steps: 20, guidance: 1.5, sub: '~35–45 s · g1.5 · max quality', recommended: true },
];

const FITS = [
  { id: 'slim', label: 'Slim Fit', desc: 'Close to the body, tapered' },
  { id: 'regular', label: 'Regular Fit', desc: 'Classic, balanced ease' },
  { id: 'oversized', label: 'Oversized', desc: 'Relaxed, dropped shoulders' },
];

const CATS = [
  { id: 'tops', label: 'Upper Body', desc: 'T-shirts, shirts, jackets, knitwear' },
  { id: 'bottoms', label: 'Lower Body', desc: 'Trousers, shorts, skirts, jeans' },
];

function Pill({ ok, children }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.68rem] font-semibold ${ok ? 'bg-sage-soft text-sage' : 'bg-amber-100 text-amber-900'}`}>
      {ok ? <CheckCircle2 size={11} /> : <AlertTriangle size={11} />}
      {children}
    </span>
  );
}

function Field({ label, value, onChange, suffix = 'cm', placeholder = '—', min, max }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <div className="relative">
        <input
          type="number"
          inputMode="decimal"
          className="input pr-9"
          value={value ?? ''}
          min={min} max={max}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
        />
        <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-faint">{suffix}</span>
      </div>
    </label>
  );
}

export default function Studio() {
  const navigate = useNavigate();

  const [personFile, setPersonFile] = useState(null);
  const [personUrl, setPersonUrl] = useState(null);
  const [personAnalysis, setPersonAnalysis] = useState(null);
  const [personLoading, setPersonLoading] = useState(false);

  const [garmentFile, setGarmentFile] = useState(null);
  const [garmentUrl, setGarmentUrl] = useState(null);
  const [garmentAnalysis, setGarmentAnalysis] = useState(null);
  const [garmentLoading, setGarmentLoading] = useState(false);

  const [category, setCategory] = useState('tops');
  const [garmentPhotoType, setGarmentPhotoType] = useState('model');
  const [fit, setFit] = useState('regular');
  const [steps, setSteps] = useState(20);
  const [guidance, setGuidance] = useState(1.5);
  const [activePreset, setActivePreset] = useState('best');
  const [segmentationFree, setSegmentationFree] = useState(true);
  const [seed, setSeed] = useState(42);
  const [saveHistory, setSaveHistory] = useState(false);

  const [body, setBody] = useState({ height_cm: null, chest_cm: null, waist_cm: null, shoulder_cm: null, hip_cm: null });
  const [garmentM, setGarmentM] = useState({ chest_width_cm: null, shoulder_cm: null, length_cm: null, sleeve_cm: null, waist_width_cm: null, hip_width_cm: null });
  const [showBody, setShowBody] = useState(true);
  const [showGarment, setShowGarment] = useState(true);
  const [estimating, setEstimating] = useState(false);

  const [sampleLoading, setSampleLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);

  // Prefill measurements from saved profile
  useEffect(() => {
    getProfile().then((p) => {
      setBody((b) => ({
        ...b,
        height_cm: p.height_cm ?? b.height_cm,
        chest_cm: p.chest_cm ?? b.chest_cm,
        waist_cm: p.waist_cm ?? b.waist_cm,
        shoulder_cm: p.shoulder_cm ?? b.shoulder_cm,
        hip_cm: p.hip_cm ?? b.hip_cm,
      }));
      if (p.preferred_fit) setFit(p.preferred_fit);
    }).catch(() => {});
  }, []);

  const handlePerson = useCallback(async (f) => {
    setPersonFile(f);
    setPersonAnalysis(null);
    const url = URL.createObjectURL(f);
    setPersonUrl(url);
    setPersonLoading(true);
    try {
      const res = await analyzePerson(f);
      setPersonAnalysis(res);
    } catch (e) {
      setPersonAnalysis({ detected: false, error: e.message, quality: null });
    } finally {
      setPersonLoading(false);
    }
  }, []);

  const handleGarment = useCallback(async (f) => {
    setGarmentFile(f);
    setGarmentAnalysis(null);
    const url = URL.createObjectURL(f);
    setGarmentUrl(url);
    setGarmentLoading(true);
    try {
      const res = await analyzeGarment(f);
      setGarmentAnalysis(res);
    } catch (e) {
      setGarmentAnalysis({ error: e.message, dominant_colors: [] });
    } finally {
      setGarmentLoading(false);
    }
  }, []);

  const loadSamples = async () => {
    setSampleLoading(true);
    setError(null);
    try {
      const [pRes, gRes] = await Promise.all([fetch('/samples/sample-person.webp'), fetch('/samples/sample-garment.webp')]);
      const pBlob = await pRes.blob();
      const gBlob = await gRes.blob();
      const p = new File([pBlob], 'sample-person.webp', { type: 'image/webp' });
      const g = new File([gBlob], 'sample-garment.webp', { type: 'image/webp' });
      handlePerson(p);
      handleGarment(g);
    } catch (e) {
      setError('Could not load the bundled sample photos.');
    } finally {
      setSampleLoading(false);
    }
  };

  const estimateFromPhoto = async () => {
    if (!personFile || !body.height_cm) {
      setError('Enter your height first — the estimate is scaled from your known height.');
      return;
    }
    setEstimating(true);
    setError(null);
    try {
      const res = await analyzePerson(personFile, body.height_cm);
      setPersonAnalysis(res);
      if (res.estimates) {
        setBody((b) => ({
          ...b,
          shoulder_cm: res.estimates.shoulder_cm ?? b.shoulder_cm,
          hip_cm: Math.round(res.estimates.hip_circumference_cm) ?? b.hip_cm,
          chest_cm: Math.round(res.estimates.chest_circumference_cm) ?? b.chest_cm,
        }));
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setEstimating(false);
    }
  };

  const canGenerate = personFile && garmentFile && !generating;

  const doGenerate = async () => {
    if (!canGenerate) return;
    setGenerating(true);
    setError(null);
    try {
      const { job_id } = await generateTryOn({
        person: personFile, garment: garmentFile, category, fit,
        steps, seed, garmentPhotoType, saveHistory, guidance, segmentationFree,
        body: Object.fromEntries(Object.entries(body).filter(([, v]) => v != null)),
        garmentM: Object.fromEntries(Object.entries(garmentM).filter(([, v]) => v != null)),
      });
      navigate(`/results/${job_id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : `Generation failed: ${e.message}`);
      setGenerating(false);
    }
  };

  const estSeconds = useMemo(() => Math.max(10, Math.round(steps * (guidance >= 1.2 ? 1.7 : 1.6))), [steps, guidance]);
  const topColors = garmentAnalysis?.dominant_colors?.slice(0, 5) ?? [];

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:py-12">
      {/* Page header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Try-On Studio</p>
          <h1 className="mt-1.5 font-display text-3xl font-medium tracking-tight sm:text-4xl">Dress the mannequin — <em className="text-accent">you</em>.</h1>
          <p className="mt-2 max-w-lg text-sm leading-6 text-muted">Upload a photo of yourself and a garment, pick a fit, optionally add measurements, then generate a locally-rendered try-on with a full honesty report.</p>
        </div>
        <button onClick={loadSamples} disabled={sampleLoading} className="btn-ghost">
          {sampleLoading ? <Loader2 className="animate-spin" size={16} /> : <FlaskConical size={16} />}
          Try with sample photos
        </button>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-5">
        {/* ----------------------------- LEFT: uploads & options ----------------------------- */}
        <div className="space-y-6 lg:col-span-3">
          <div className="grid gap-5 sm:grid-cols-2">
            <div>
              <UploadCard
                label="1 · Your photo"
                hint="Clear, front-facing, good light. Full body or upper body. No sunglasses or faces-covered shots."
                file={personFile}
                previewUrl={personUrl}
                onFile={handlePerson}
                onClear={() => { setPersonFile(null); setPersonUrl(null); setPersonAnalysis(null); }}
              />
              {/* Person analysis strip */}
              <div className="mt-2 min-h-[2rem]">
                {personLoading && <p className="inline-flex items-center gap-2 text-xs text-muted"><Loader2 size={12} className="animate-spin" /> Analysing pose & quality…</p>}
                {personAnalysis && !personLoading && (
                  <div className="flex flex-wrap gap-1.5">
                    {personAnalysis.detected ? <Pill ok>Person detected</Pill> : <Pill ok={false}>No pose found</Pill>}
                    {personAnalysis.quality && <Pill ok={personAnalysis.quality.sharp}>{personAnalysis.quality.sharp ? 'Sharp' : 'Soft'}</Pill>}
                    {personAnalysis.quality && <Pill ok={personAnalysis.quality.well_lit}>Lighting</Pill>}
                    {personAnalysis.quality && <Pill ok={personAnalysis.quality.resolution_ok}>Resolution</Pill>}
                    {personAnalysis.error && <p className="w-full text-[0.7rem] text-amber-800">{personAnalysis.error}</p>}
                  </div>
                )}
              </div>
            </div>

            <div>
              <UploadCard
                label="2 · The garment"
                hint="Product shot or worn-model photo. Plain background is ideal. One garment per image."
                aspect="aspect-[3/4]"
                file={garmentFile}
                previewUrl={garmentAnalysis?.garment_png_b64 ? `data:image/png;base64,${garmentAnalysis.garment_png_b64}` : garmentUrl}
                onFile={handleGarment}
                onClear={() => { setGarmentFile(null); setGarmentUrl(null); setGarmentAnalysis(null); }}
              />
              <div className="mt-2 min-h-[2rem]">
                {garmentLoading && <p className="inline-flex items-center gap-2 text-xs text-muted"><Loader2 size={12} className="animate-spin" /> Removing background & sampling colours…</p>}
                {garmentAnalysis && !garmentLoading && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    {topColors.map((c) => (
                      <span key={c.hex + c.rgb} title={`${c.hex} · ${Math.round(c.share * 100)}%`} className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface py-0.5 pl-0.5 pr-2 text-[0.68rem] font-mono text-muted">
                        <span className="h-3.5 w-3.5 rounded-full border border-black/10" style={{ background: c.hex }} />
                        {c.hex}
                      </span>
                    ))}
                    {garmentAnalysis.error && <p className="w-full text-[0.7rem] text-amber-800">{garmentAnalysis.error}</p>}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Category + prompt type */}
          <div className="card p-5">
            <div className="flex items-center gap-2">
              <Layers size={16} className="text-accent" />
              <h2 className="font-display text-base font-semibold">Garment category</h2>
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              {CATS.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setCategory(c.id)}
                  className={`rounded-xl border p-4 text-left transition-all duration-200 ${category === c.id ? 'border-ink bg-ink text-paper shadow-card' : 'border-line bg-[#FFFDFB] hover:border-faint'}`}
                >
                  <span className="block text-sm font-bold">{c.label}</span>
                  <span className={`mt-1 block text-[0.72rem] leading-5 ${category === c.id ? 'text-[#C9C4B8]' : 'text-muted'}`}>{c.desc}</span>
                </button>
              ))}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-4 text-[0.8rem]">
              <span className="font-semibold text-muted">Garment photo type:</span>
              {[['model', 'Worn by a person'], ['flat-lay', 'Flat-lay / product']].map(([v, l]) => (
                <label key={v} className="inline-flex cursor-pointer items-center gap-2">
                  <input type="radio" name="photoType" className="accent-[#B4432F]" checked={garmentPhotoType === v} onChange={() => setGarmentPhotoType(v)} />
                  <span>{l}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Fit preference */}
          <div className="card p-5">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Shirt size={16} className="text-accent" />
                <h2 className="font-display text-base font-semibold">Fit preference</h2>
              </div>
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-[0.65rem] font-semibold text-amber-900">
                <Info size={11} /> Previews are a visual approximation
              </span>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 sm:gap-3">
              {FITS.map((f) => (
                <button
                  key={f.id}
                  onClick={() => setFit(f.id)}
                  className={`rounded-xl border px-3 py-4 transition-all duration-200 ${fit === f.id ? 'border-accent bg-accent-soft shadow-card ring-1 ring-accent/40' : 'border-line bg-[#FFFDFB] hover:border-faint'}`}
                >
                  <span className="block text-sm font-bold sm:text-base">{f.label}</span>
                  <span className="mt-1 hidden text-[0.68rem] leading-4 text-muted sm:block">{f.desc}</span>
                </button>
              ))}
            </div>
            <p className="mt-3 text-[0.7rem] leading-5 text-muted">
              Slim / regular / oversized previews are rendered by a documented torso-anchored non-rigid warp applied to the AI result
              (the try-on model has no native fit control). They illustrate silhouette differences — they do not predict fabric drape or physical fit.
            </p>
          </div>

          {/* Generation controls */}
          <div className="card p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <Timer size={16} className="text-accent" />
                <h2 className="font-display text-base font-semibold">Render settings</h2>
              </div>
              <span className="font-mono text-[0.7rem] text-muted">≈ {estSeconds} s render{estSeconds < 30 ? ' — under 30 s' : ''} · first run after restart adds ~45 s model load</span>
            </div>
            <div className="mt-4">
              <label className="label">Speed preset — {steps} steps · guidance {guidance}</label>
              <div className="grid grid-cols-4 gap-1.5">
                {PRESETS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => { setActivePreset(p.id); setSteps(p.steps); setGuidance(p.guidance); }}
                    className={`rounded-xl border px-2 py-2 text-center transition-all ${activePreset === p.id ? 'border-ink bg-ink text-paper' : 'border-line bg-[#FFFDFB] hover:border-faint'}`}
                  >
                    <span className="block text-[0.75rem] font-bold">{p.label}</span>
                    <span className={`block text-[0.6rem] ${activePreset === p.id ? 'text-[#C9C4B8]' : 'text-muted'}`}>{p.sub}</span>
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-[0.65rem] leading-4 text-faint">Draft (4 steps, g1.0) is the fastest and softest — print and face detail drop most. Balanced (12, g1.5) is the model&apos;s recommended setting. Best (20, g1.5) is the default here: close to the 30-step reference used internally. Times assume a free RTX 4060-class GPU.</p>
            </div>
            <div className="mt-3">
              <label className="label" htmlFor="seed">Seed <button onClick={() => setSeed(Math.floor(Math.random() * 100000))} className="ml-1 text-accent hover:underline">randomize</button></label>
              <input id="seed" type="number" className="input" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
              <p className="mt-1 text-[0.65rem] text-faint">Same seed + same photos = reproducible output.</p>
            </div>
            <label className="mt-4 flex cursor-pointer items-start gap-2.5 rounded-xl border border-line bg-[#FFFDFB] p-3 text-[0.78rem] leading-5 text-muted">
              <input type="checkbox" className="mt-0.5 accent-[#B4432F]" checked={saveHistory} onChange={(e) => setSaveHistory(e.target.checked)} />
              <span><strong className="text-ink">Save to history</strong> — keep this result and its photos in your local history. Off by default: unsaved results live as temporary files and auto-expire after 24 h. Your photos never leave this machine.</span>
            </label>
            <div className="mt-3 rounded-xl border border-line bg-[#FFFDFB] p-3">
              <span className="label !mb-1.5">Render mode</span>
              <div className="grid grid-cols-2 gap-2">
                {[
                  ['maskless', 'Maskless (default)', 'Re-renders the whole frame. Strongest consistency between face, hands, garment and background.'],
                  ['masked', 'Masked (keep pixels)', 'Outside the garment mask the original pixels stay 1:1 — max body fidelity; best when outfit silhouette differs a lot.'],
                ].map(([mode, label, desc]) => {
                  const isMaskless = mode === 'maskless';
                  const active = isMaskless ? segmentationFree : !segmentationFree;
                  return (
                    <button
                      key={mode}
                      onClick={() => setSegmentationFree(isMaskless)}
                      className={`rounded-lg border p-3 text-left transition-all ${active ? 'border-accent bg-accent-soft/60 ring-1 ring-accent/40' : 'border-line hover:border-faint'}`}
                    >
                      <span className="block text-[0.75rem] font-bold">{label}</span>
                      <span className="mt-1 block text-[0.66rem] leading-4 text-muted">{desc}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* ----------------------------- RIGHT: measurements + generate ----------------------------- */}
        <div className="space-y-6 lg:col-span-2">
          <div className="card overflow-hidden">
            <button onClick={() => setShowBody((v) => !v)} className="flex w-full items-center justify-between px-5 py-4 text-left">
              <span className="flex items-center gap-2">
                <Ruler size={16} className="text-accent" />
                <span className="font-display text-base font-semibold">Your measurements <span className="text-xs font-sans font-medium text-muted">optional</span></span>
              </span>
              {showBody ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {showBody && (
              <div className="border-t border-line-soft px-5 py-4">
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Height" value={body.height_cm} onChange={(v) => setBody({ ...body, height_cm: v })} min={120} max={230} />
                  <Field label="Chest circ." value={body.chest_cm} onChange={(v) => setBody({ ...body, chest_cm: v })} min={70} max={160} />
                  <Field label="Waist circ." value={body.waist_cm} onChange={(v) => setBody({ ...body, waist_cm: v })} min={55} max={160} />
                  <Field label="Shoulder width" value={body.shoulder_cm} onChange={(v) => setBody({ ...body, shoulder_cm: v })} min={30} max={60} />
                  <Field label="Hip circ." value={body.hip_cm} onChange={(v) => setBody({ ...body, hip_cm: v })} min={70} max={170} />
                </div>
                <button onClick={estimateFromPhoto} disabled={estimating || !personFile} className="mt-3 inline-flex items-center gap-2 text-[0.75rem] font-semibold text-accent hover:underline disabled:cursor-not-allowed disabled:text-faint">
                  {estimating ? <Loader2 size={13} className="animate-spin" /> : <Wand2 size={13} />}
                  Estimate from my photo (rough, needs height + full body)
                </button>
                {personAnalysis?.estimates && (
                  <p className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-[0.68rem] leading-5 text-amber-900">{personAnalysis.estimates.accuracy_note}</p>
                )}
              </div>
            )}
          </div>

          <div className="card overflow-hidden">
            <button onClick={() => setShowGarment((v) => !v)} className="flex w-full items-center justify-between px-5 py-4 text-left">
              <span className="flex items-center gap-2">
                <Shirt size={16} className="text-accent" />
                <span className="font-display text-base font-semibold">Garment measurements <span className="text-xs font-sans font-medium text-muted">optional</span></span>
              </span>
              {showGarment ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {showGarment && (
              <div className="border-t border-line-soft px-5 py-4">
                <p className="mb-3 text-[0.7rem] leading-5 text-muted">Flat, half-width style (as on a spec sheet). Never invented from the photo — enter what you know.</p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Chest width (½)" value={garmentM.chest_width_cm} onChange={(v) => setGarmentM({ ...garmentM, chest_width_cm: v })} min={20} max={70} />
                  <Field label="Shoulder width" value={garmentM.shoulder_cm} onChange={(v) => setGarmentM({ ...garmentM, shoulder_cm: v })} min={25} max={70} />
                  <Field label="Body length" value={garmentM.length_cm} onChange={(v) => setGarmentM({ ...garmentM, length_cm: v })} min={30} max={110} />
                  <Field label="Sleeve length" value={garmentM.sleeve_cm} onChange={(v) => setGarmentM({ ...garmentM, sleeve_cm: v })} min={10} max={70} />
                  <Field label="Waist width (½)" value={garmentM.waist_width_cm} onChange={(v) => setGarmentM({ ...garmentM, waist_width_cm: v })} min={20} max={70} />
                  <Field label="Hip width (½)" value={garmentM.hip_width_cm} onChange={(v) => setGarmentM({ ...garmentM, hip_width_cm: v })} min={20} max={70} />
                </div>
              </div>
            )}
          </div>

          {/* Generate panel */}
          <div className="overflow-hidden rounded-2xl bg-ink p-6 text-paper shadow-lift">
            <p className="flex items-center gap-2 text-[0.68rem] font-bold uppercase tracking-[0.2em] text-[#E8C9BC]"><Sparkles size={12} /> Generate</p>
            <h2 className="mt-2 font-display text-2xl font-medium leading-snug">Ready when you are.</h2>
            <ul className="mt-3 space-y-1.5 text-[0.78rem] leading-5 text-[#B9B3A5]">
              <li className="flex gap-2"><BadgeCheck size={14} className="mt-0.5 shrink-0 text-sage-soft" /> Identity & background preserved as much as the model allows</li>
              <li className="flex gap-2"><BadgeCheck size={14} className="mt-0.5 shrink-0 text-sage-soft" /> Colour · pattern · logo preservation measured after render</li>
              <li className="flex gap-2"><BadgeCheck size={14} className="mt-0.5 shrink-0 text-sage-soft" /> Slim / Regular / Oversized comparison generated with your result</li>
            </ul>
            {error && (
              <p className="mt-4 flex items-start gap-2 rounded-xl border border-red-400/30 bg-red-500/10 px-3 py-2.5 text-[0.78rem] leading-5 text-red-200">
                <AlertTriangle size={15} className="mt-0.5 shrink-0" /> {error}
              </p>
            )}
            <button onClick={doGenerate} disabled={!canGenerate} className="btn-primary mt-5 w-full justify-center !bg-paper !py-3.5 !text-[0.95rem] !text-ink hover:!bg-accent-soft">
              {generating ? (
                <><Loader2 size={18} className="animate-spin" /> Sending to the GPU…</>
              ) : (
                <><Wand2 size={18} /> Generate Try-On <ArrowRight size={18} /></>
              )}
            </button>
            <p className="mt-3 text-center text-[0.68rem] text-[#8F887A]">
              Real local diffusion inference — seconds on a free GPU, not a wait-screen theatre.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
