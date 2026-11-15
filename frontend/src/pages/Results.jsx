import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  Download, BookmarkCheck, BookmarkPlus, ArrowLeft, Loader2, CheckCircle2,
  AlertTriangle, HelpCircle, ChevronDown, ChevronUp, Shirt, Sparkles,
  RotateCcw, ShieldCheck, ScanEye, Ruler, Info, Trash2,
} from 'lucide-react';
import BeforeAfter from '../components/BeforeAfter';
import { getJob, resultImageUrl, API, saveToHistory } from '../lib/api';

const STAGES = [
  { id: 'queued', label: 'Queued' },
  { id: 'loading', label: 'Loading model' },
  { id: 'preprocess', label: 'Pose & garment analysis' },
  { id: 'generating', label: 'Rendering try-on (diffusion)' },
  { id: 'fit-variants', label: 'Fit previews' },
  { id: 'analysis', label: 'Preservation & fit analysis' },
  { id: 'done', label: 'Done' },
];

function StatusIcon({ status, size = 16 }) {
  if (status === 'pass' || status === 'preserved') return <CheckCircle2 size={size} className="shrink-0 text-sage" />;
  if (status === 'warning' || status === 'potentially_altered') return <AlertTriangle size={size} className="shrink-0 text-amber-600" />;
  return <HelpCircle size={size} className="shrink-0 text-faint" />;
}

function verdictStyle(status) {
  switch (status) {
    case 'tight': return 'bg-red-100 text-red-900';
    case 'ideal': return 'bg-sage-soft text-sage';
    case 'relaxed': return 'bg-amber-100 text-amber-900';
    case 'loose': return 'bg-sky-100 text-sky-900';
    default: return 'bg-line-soft text-muted';
  }
}

function fmtElapsed(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function ProgressScreen({ job, elapsed }) {
  const stageIdx = Math.max(0, STAGES.findIndex((s) => s.id === job?.stage));
  return (
    <div className="mx-auto max-w-3xl px-4 py-16 text-center sm:px-6">
      <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-3xl bg-ink">
        <Shirt size={34} className="animate-pulse-soft text-paper" strokeWidth={1.5} />
      </div>
      <h1 className="mt-6 font-display text-3xl font-medium tracking-tight">Rendering your try-on…</h1>
      <p className="mt-2 font-mono text-5xl font-medium tabular-nums text-accent">{fmtElapsed(elapsed)}</p>
      <p className="mt-3 text-sm text-muted">Local diffusion inference on your GPU — this is the real thing, not a progress bar theatre. Keep this tab open.</p>

      <ol className="mx-auto mt-8 max-w-md space-y-2 text-left">
        {STAGES.slice(0, -1).map((s, i) => {
          const state = i < stageIdx ? 'done' : i === stageIdx ? 'now' : 'todo';
          return (
            <li key={s.id} className={`flex items-center gap-3 rounded-xl border px-4 py-2.5 text-sm transition-colors ${
              state === 'now' ? 'border-accent/40 bg-accent-soft/50 font-semibold text-ink'
              : state === 'done' ? 'border-transparent text-sage'
              : 'border-transparent text-faint'
            }`}>
              {state === 'done' ? <CheckCircle2 size={16} /> : state === 'now' ? <Loader2 size={16} className="animate-spin text-accent" /> : <span className="h-4 w-4 rounded-full border border-current opacity-40" />}
              {s.label}
            </li>
          );
        })}
      </ol>
      <p className="mt-6 text-xs text-faint">{job?.message}</p>
    </div>
  );
}

export default function Results() {
  const { jobId } = useParams();
  const [job, setJob] = useState(null);
  const [notFound, setNotFound] = useState(false);
  const [activeFit, setActiveFit] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [saved, setSaved] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [showMethod, setShowMethod] = useState(false);
  const startRef = useRef(Date.now());

  useEffect(() => {
    let alive = true;
    const tick = () => alive && setElapsed((Date.now() - startRef.current) / 1000);
    const timer = setInterval(tick, 1000);
    const poll = async () => {
      try {
        const j = await getJob(jobId);
        if (!alive) return;
        setJob(j);
        if (j?.status === 'done') {
          setActiveFit((cur) => cur || j.result?.fit || 'regular');
          setSaved(Boolean(j.result?.saved_to_history));
        }
        if (j?.status === 'queued' || j?.status === 'running') {
          setTimeout(poll, 4000);
        }
      } catch (e) {
        if (alive) setNotFound(true);
      }
    };
    poll();
    return () => { alive = false; clearInterval(timer); };
  }, [jobId]);

  const result = job?.result ?? null;
  const status = job?.status;

  const personSrc = `${API}/api/results/${jobId}/source/person`;
  const garmentSrc = `${API}/api/results/${jobId}/source/garment`;
  const fitImage = (f) => resultImageUrl(jobId, f);

  const dlName = useMemo(() => `fitvision-tryon-${result?.fit ?? 'result'}.png`, [result?.fit]);

  if (notFound) {
    return (
      <div className="mx-auto max-w-lg px-4 py-24 text-center">
        <AlertTriangle size={40} className="mx-auto text-faint" />
        <h1 className="mt-4 font-display text-2xl">Result not found</h1>
        <p className="mt-2 text-sm text-muted">This result either never existed or its unsaved files expired. Generate a new try-on — it only takes a few minutes.</p>
        <Link to="/studio" className="btn-primary mt-6"><ArrowLeft size={16} /> Back to Studio</Link>
      </div>
    );
  }

  if (!result) {
    if (job?.saved_in_db) {
      return (
        <div className="mx-auto max-w-lg px-4 py-24 text-center">
          <BookmarkCheck size={40} className="mx-auto text-sage" />
          <h1 className="mt-4 font-display text-2xl">This one is in your history</h1>
          <p className="mt-2 text-sm text-muted">This saved result was generated in an earlier session, so its full analysis report is no longer in memory. The image is kept in your local history.</p>
          <Link to="/profile" className="btn-primary mt-6">Open Profile & History</Link>
        </div>
      );
    }
    if (status === 'error') {
      return (
        <div className="mx-auto max-w-xl px-4 py-20 text-center">
          <AlertTriangle size={40} className="mx-auto text-amber-600" />
          <h1 className="mt-4 font-display text-2xl">The render failed — nothing to show, and we won&apos;t fake it.</h1>
          <p className="mt-3 rounded-xl bg-red-50 px-4 py-3 font-mono text-[0.75rem] leading-6 text-red-900">{job?.error}</p>
          <div className="mt-4 text-left text-[0.82rem] leading-6 text-muted">
            <p className="font-semibold text-ink">Common causes</p>
            <ul className="mt-1 list-disc space-y-1 pl-5">
              <li>GPU out of memory — close other GPU apps and retry with Draft (8 steps).</li>
              <li>First-run model download failed — check internet for the one-time ~2 GB weight fetch.</li>
              <li>A person/garment image the model can&apos;t parse — use a clear, front-facing photo.</li>
            </ul>
          </div>
          <Link to="/studio" className="btn-primary mt-6"><RotateCcw size={16} /> Try again</Link>
        </div>
      );
    }
    return <ProgressScreen job={job} elapsed={elapsed} />;
  }

  const mainFit = activeFit ?? result.fit;
  const preservation = result.preservation;
  const reliability = result.reliability;
  const fitAnalysis = result.fit_analysis;
  const recs = result.recommendations;

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:py-10">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link to="/studio" className="inline-flex items-center gap-1.5 text-[0.78rem] font-semibold text-muted hover:text-ink"><ArrowLeft size={14} /> New try-on</Link>
          <h1 className="mt-2 font-display text-3xl font-medium tracking-tight sm:text-4xl">Your fit, visualised.</h1>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[0.72rem] text-muted">
            <span className="rounded-full border border-line bg-surface px-2.5 py-1 font-medium capitalize">{result.category === 'tops' ? 'Upper body' : 'Lower body'} · {result.fit} fit selected</span>
            <span className="rounded-full border border-line bg-surface px-2.5 py-1 font-medium">{result.steps} steps · g{result.guidance_scale ?? 1.5} · seed {result.seed}</span>
            <span className="rounded-full border border-line bg-surface px-2.5 py-1 font-medium">Local FASHN VTON v1.5</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <a href={fitImage(mainFit)} download={dlName} className="btn-ghost"><Download size={16} /> Download image</a>
          {!saved ? (
            <button onClick={async () => { await saveToHistory(jobId); setSaved(true); }} className="btn-ghost"><BookmarkPlus size={16} /> Save to history</button>
          ) : (
            <span className="inline-flex items-center gap-2 rounded-full border border-sage/40 bg-sage-soft px-4 py-2 text-[0.8rem] font-semibold text-sage"><BookmarkCheck size={16} /> Saved to history</span>
          )}
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        {/* ------------------------- LEFT: comparison + variants ------------------------- */}
        <div className="lg:col-span-3">
          <BeforeAfter beforeSrc={personSrc} afterSrc={fitImage(mainFit)} beforeLabel="Before" afterLabel={`${mainFit} fit`} />

          <div className="mt-4">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-lg font-semibold">Fit preference comparison</h2>
              <span className="inline-flex items-center gap-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.14em] text-faint"><Info size={11} /> visual approximation</span>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3">
              {['slim', 'regular', 'oversized'].map((f) => (
                <button
                  key={f}
                  onClick={() => setActiveFit(f)}
                  className={`group overflow-hidden rounded-2xl border text-left transition-all duration-300 ${mainFit === f ? 'border-accent ring-2 ring-accent/35 shadow-lift' : 'border-line hover:border-faint'}`}
                >
                  <div className="aspect-[2/3] w-full overflow-hidden bg-[#EDEAE2]">
                    <img src={fitImage(f)} alt={`${f} fit preview`} className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]" loading="lazy" />
                  </div>
                  <span className={`block px-3 py-2 text-center text-[0.72rem] font-bold uppercase tracking-[0.16em] ${mainFit === f ? 'bg-accent text-paper' : 'bg-surface text-muted'}`}>{f}</span>
                </button>
              ))}
            </div>
            <p className="mt-2 text-[0.68rem] leading-5 text-faint">{result.fit_variants_note}</p>
          </div>
        </div>

        {/* ------------------------------ RIGHT: reports ------------------------------ */}
        <div className="space-y-5 lg:col-span-2">

          {/* Fit analysis */}
          <section className="card p-5">
            <div className="flex items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 font-display text-lg font-semibold"><Ruler size={17} className="text-accent" /> Measurement-based fit analysis</h2>
              <span className="rounded-full bg-line-soft px-2.5 py-1 text-[0.65rem] font-bold uppercase tracking-wider text-muted">{fitAnalysis?.verdicts?.length ? 'Measured' : 'No data'}</span>
            </div>
            {fitAnalysis?.verdicts?.length ? (
              <>
                <div className="mt-3 divide-y divide-line-soft">
                  {fitAnalysis.verdicts.map((v) => (
                    <div key={v.area} className="grid grid-cols-[auto_1fr_auto] items-center gap-3 py-2.5">
                      <span className="w-16 text-[0.8rem] font-bold">{v.area}</span>
                      <span className="text-[0.7rem] leading-5 text-muted">{v.note}</span>
                      <span className={`rounded-full px-2.5 py-1 text-[0.62rem] font-bold uppercase tracking-wider ${verdictStyle(v.status)}`}>{v.status}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 rounded-xl bg-sage-soft px-4 py-3 text-[0.82rem] leading-6 text-sage">
                  <strong>Overall — </strong>{fitAnalysis.overall}
                  <p className="mt-1 text-[0.75rem]">Suggested fit for this garment: <strong className="capitalize">{fitAnalysis.recommended_fit}</strong></p>
                </div>
                <button onClick={() => setShowRules((v) => !v)} className="mt-2 inline-flex items-center gap-1.5 text-[0.72rem] font-semibold text-muted hover:text-ink">
                  {showRules ? <ChevronUp size={13} /> : <ChevronDown size={13} />} {showRules ? 'Hide' : 'Show'} the sizing rules
                </button>
                {showRules && (
                  <pre className="mt-2 max-h-56 overflow-auto rounded-xl bg-ink p-3 font-mono text-[0.62rem] leading-5 text-[#C9C4B8]">{JSON.stringify(fitAnalysis.rules_used, null, 2)}</pre>
                )}
                <p className="mt-2 text-[0.68rem] leading-5 text-faint">{fitAnalysis.disclaimer}</p>
              </>
            ) : (
              <p className="mt-3 rounded-xl bg-amber-50 px-4 py-3 text-[0.8rem] leading-6 text-amber-900">{fitAnalysis?.overall}</p>
            )}
          </section>

          {/* Garment preservation */}
          <section className="card p-5">
            <div className="flex items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 font-display text-lg font-semibold"><ScanEye size={17} className="text-accent" /> Garment detail preservation</h2>
              <span className="rounded-full bg-line-soft px-2.5 py-1 text-[0.65rem] font-bold uppercase tracking-wider text-muted">CV checks</span>
            </div>
            <p className="mt-1.5 text-[0.75rem] leading-5 text-muted">{preservation?.summary}</p>
            <ul className="mt-3 divide-y divide-line-soft">
              {preservation?.items?.map((item) => (
                <li key={item.aspect} className="flex items-start gap-2.5 py-2.5">
                  <StatusIcon status={item.status} size={17} />
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[0.8rem] font-bold">{item.aspect}</span>
                      {item.measured && <span className="rounded bg-line-soft px-1.5 py-0.5 font-mono text-[0.58rem] uppercase text-muted">measured</span>}
                    </div>
                    <p className="mt-0.5 text-[0.7rem] leading-5 text-muted">{item.detail}</p>
                  </div>
                </li>
              ))}
            </ul>
            <button onClick={() => setShowMethod((v) => !v)} className="mt-1 inline-flex items-center gap-1.5 text-[0.72rem] font-semibold text-muted hover:text-ink">
              {showMethod ? <ChevronUp size={13} /> : <ChevronDown size={13} />} How these are measured
            </button>
            {showMethod && (
              <ul className="mt-2 space-y-1.5 rounded-xl bg-[#F3F0E9] p-3 text-[0.68rem] leading-5 text-muted">
                {preservation?.method_notes?.map((n, i) => <li key={i}>· {n}</li>)}
              </ul>
            )}
          </section>

          {/* Reliability */}
          <section className="card p-5">
            <div className="flex items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 font-display text-lg font-semibold"><ShieldCheck size={17} className="text-accent" /> Reliability report</h2>
              <span className="rounded-full bg-line-soft px-2.5 py-1 text-[0.65rem] font-bold uppercase tracking-wider text-muted">No vanity scores</span>
            </div>
            <ul className="mt-3 divide-y divide-line-soft">
              {reliability?.items?.map((item) => (
                <li key={item.check} className="flex items-start gap-2.5 py-2.5">
                  <StatusIcon status={item.status} size={17} />
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[0.8rem] font-bold">{item.check}</span>
                      {item.measured && <span className="rounded bg-line-soft px-1.5 py-0.5 font-mono text-[0.58rem] uppercase text-muted">measured</span>}
                    </div>
                    <p className="mt-0.5 text-[0.7rem] leading-5 text-muted">{item.detail}</p>
                  </div>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-[0.68rem] leading-5 text-faint">{reliability?.notes?.join(' ')}</p>
          </section>
        </div>
      </div>

      {/* --------------------- Recommendations (secondary) --------------------- */}
      {recs && (
        <section className="mt-6 card flex flex-wrap items-start gap-6 p-6">
          <div className="min-w-0 flex-1">
            <h2 className="flex items-center gap-2 font-display text-lg font-semibold"><Sparkles size={17} className="text-accent" /> {recs.title}</h2>
            <ul className="mt-2 space-y-1.5 text-[0.8rem] leading-6 text-muted">
              {recs.suggestions.map((s, i) => <li key={i}>· {s}</li>)}
            </ul>
            <p className="mt-2 text-[0.68rem] text-faint">{recs.note}</p>
          </div>
          <div className="flex shrink-0 flex-col items-center gap-2">
            <span className="label !mb-0">Palette</span>
            <div className="flex gap-1.5">
              {recs.colors.map((c) => (
                <div key={c} className="flex flex-col items-center gap-1">
                  <span className="h-10 w-10 rounded-xl border border-black/10 shadow-sm" style={{ background: c }} />
                  <span className="font-mono text-[0.55rem] text-muted">{c}</span>
                </div>
              ))}
            </div>
            <span className="mt-1 text-[0.6rem] text-faint">Garment dominant: {result.garment?.dominant_colors?.[0]?.hex ?? '—'}</span>
          </div>
        </section>
      )}

      {/* Attribution + delete */}
      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-5 text-[0.72rem] text-muted">
        <p>{result.attribution} Garment fit estimates are not size advice; check the brand&apos;s chart. Faces are re-rendered by the model and can drift slightly.</p>
        <button onClick={async () => { if (confirm('Delete this result and its files from this machine?')) { await fetch(`${API}/api/history/${jobId}`, { method: 'DELETE' }); window.location.href = '/studio'; } }} className="inline-flex items-center gap-1.5 font-semibold text-muted transition-colors hover:text-accent">
          <Trash2 size={13} /> Delete this result
        </button>
      </div>
    </div>
  );
}
