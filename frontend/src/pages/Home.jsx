import { Link } from 'react-router-dom';
import {
  ArrowRight, Sparkles, Ruler, ScanEye, ShieldCheck, Upload, Wand2, BadgeCheck,
  SlidersHorizontal, ChevronRight, Cpu,
} from 'lucide-react';

function SampleStrip() {
  return (
    <div className="relative mx-auto w-full max-w-md lg:max-w-none">
      {/* Person photo */}
      <figure className="absolute left-0 top-8 z-10 w-40 rotate-[-4deg] rounded-2xl border border-line bg-surface p-1.5 shadow-lift sm:w-48 lg:w-52">
        <img src="/samples/sample-person.webp" alt="Person photograph" className="aspect-[3/4.2] w-full rounded-xl object-cover" />
        <figcaption className="flex items-center justify-between px-2 py-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted">
          <span>You</span><span className="text-sage">Input</span>
        </figcaption>
      </figure>

      {/* Garment photo */}
      <figure className="absolute right-0 top-0 z-10 w-32 rotate-[5deg] rounded-2xl border border-line bg-surface p-1.5 shadow-lift sm:w-40 lg:w-44">
        <img src="/samples/sample-garment.webp" alt="Garment photograph" className="aspect-square w-full rounded-xl object-contain bg-[#F0EDE6]" />
        <figcaption className="flex items-center justify-between px-2 py-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted">
          <span>Garment</span><span className="text-sage">Input</span>
        </figcaption>
      </figure>

      {/* Result photo */}
      <figure className="relative z-0 mx-auto mt-28 w-52 rounded-2xl border border-ink/10 bg-ink p-1.5 shadow-lift sm:w-64 lg:mt-32 lg:w-72">
        <img src="/samples/sample-result.png" alt="Generated virtual try-on result" className="aspect-[2/3] w-full rounded-xl object-cover" />
        <figcaption className="flex items-center justify-between px-2 py-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-[#B9B3A5]">
          <span>Output</span><span className="inline-flex items-center gap-1 text-[#E8C9BC]"><Sparkles size={10} /> AI try-on</span>
        </figcaption>
      </figure>

      <div className="pointer-events-none absolute -right-3 bottom-16 z-20 hidden animate-fade-up rounded-2xl border border-line bg-surface/95 px-4 py-3 shadow-lift backdrop-blur sm:block" style={{ animationDelay: '0.5s' }}>
        <p className="text-[0.62rem] font-bold uppercase tracking-[0.16em] text-muted">Preservation check</p>
        <div className="mt-1 flex items-center gap-1.5 text-[0.78rem] font-semibold text-sage">
          <BadgeCheck size={14} /> Colour · Logo · Pattern
        </div>
      </div>

      <div className="pointer-events-none absolute -left-2 bottom-40 z-20 hidden animate-fade-up rounded-2xl border border-line bg-surface/95 px-4 py-3 shadow-lift backdrop-blur sm:block" style={{ animationDelay: '0.75s' }}>
        <p className="text-[0.62rem] font-bold uppercase tracking-[0.16em] text-muted">Fit previews</p>
        <p className="mt-1 text-[0.78rem] font-semibold text-ink">Slim · Regular · Oversized</p>
      </div>
    </div>
  );
}

const steps = [
  { icon: Upload, n: '01', title: 'Upload two photos', text: 'A clear photo of you and a product shot of the garment. Front-facing works best.' },
  { icon: Wand2, n: '02', title: 'AI generates the try-on', text: 'FASHN VTON v1.5 renders a photorealistic result locally on your GPU — no cloud, no API fees.' },
  { icon: BadgeCheck, n: '03', title: 'Read the fit report', text: 'Preservation checks, measurement-based fit analysis and honest reliability notes — before you buy.' },
];

const differentiators = [
  {
    icon: SlidersHorizontal,
    title: 'Fit Preference Comparison',
    text: 'Slim, regular and oversized previews side by side — rendered by a documented torso-anchored warp on top of the AI result, clearly labelled as a visual approximation.',
    tag: 'Signature feature',
  },
  {
    icon: ScanEye,
    title: 'Garment Detail Preservation',
    text: 'Colour histograms, dominant-colour ΔE, edge-density + ORB feature matching, CLIP embeddings and OCR token checks compare the source garment against what the model actually produced.',
    tag: 'Signature feature',
  },
  {
    icon: Ruler,
    title: 'Measurement-Based Fit Analysis',
    text: 'Your body and garment measurements run through transparent, configurable sizing rules — chest, shoulder, length, sleeve, waist. You see the rules, not a mystery score.',
    tag: 'Signature feature',
  },
];

export default function Home() {
  return (
    <div>
      {/* ------------------------------ HERO ------------------------------ */}
      <section className="relative overflow-hidden">
        <div aria-hidden className="pointer-events-none absolute -right-40 -top-40 h-[30rem] w-[30rem] rounded-full bg-[radial-gradient(circle,rgba(180,67,47,0.10),transparent_70%)]" />
        <div aria-hidden className="pointer-events-none absolute -left-40 top-40 h-[26rem] w-[26rem] rounded-full bg-[radial-gradient(circle,rgba(85,96,63,0.10),transparent_70%)]" />

        <div className="mx-auto grid max-w-6xl items-center gap-12 px-4 pb-16 pt-12 sm:px-6 lg:grid-cols-2 lg:gap-8 lg:pb-24 lg:pt-20">
          <div className="max-w-xl">
            <p className="eyebrow animate-fade-up">AI Virtual Try-On · Fit Intelligence</p>
            <h1 className="mt-4 animate-fade-up font-display text-[2.6rem] font-medium leading-[1.05] tracking-tight text-ink sm:text-6xl" style={{ animationDelay: '0.08s' }}>
              Your Style.<br />Your Fit.<br />
              <em className="text-accent">Before You Buy.</em>
            </h1>
            <p className="mt-6 animate-fade-up text-lg leading-7 text-muted" style={{ animationDelay: '0.16s' }}>
              Experience intelligent virtual try-on with AI-powered garment analysis and fit insights.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3 animate-fade-up" style={{ animationDelay: '0.24s' }}>
              <Link to="/studio" className="btn-primary text-base">
                Start Virtual Try-On <ArrowRight size={18} />
              </Link>
              <a href="#how" className="btn-ghost">See how it works</a>
            </div>
            <dl className="mt-10 grid max-w-md grid-cols-3 gap-4 animate-fade-up" style={{ animationDelay: '0.32s' }}>
              {[
                ['100%', 'Local inference'],
                ['Apache-2.0', 'Open model license'],
                ['0', 'Photos stored by default'],
              ].map(([n, t]) => (
                <div key={t} className="border-l border-line pl-3">
                  <dt className="font-display text-xl font-semibold text-ink">{n}</dt>
                  <dd className="mt-0.5 text-[0.68rem] leading-4 text-muted">{t}</dd>
                </div>
              ))}
            </dl>
          </div>

          <SampleStrip />
        </div>
      </section>

      {/* --------------------------- HOW IT WORKS -------------------------- */}
      <section id="how" className="border-y border-line bg-[#F3F0E9]">
        <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6 lg:py-20">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="eyebrow">The workflow</p>
              <h2 className="mt-2 font-display text-3xl font-medium tracking-tight sm:text-4xl">Three steps to knowing.</h2>
            </div>
            <p className="max-w-xs text-sm leading-6 text-muted">Real diffusion-model inference running on your machine — nothing ships to a paid API.</p>
          </div>
          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {steps.map((s) => (
              <div key={s.n} className="card group p-6 transition-shadow hover:shadow-lift">
                <div className="flex items-center justify-between">
                  <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent-soft text-accent transition-colors group-hover:bg-accent group-hover:text-paper">
                    <s.icon size={20} strokeWidth={1.8} />
                  </span>
                  <span className="font-mono text-xs text-faint">{s.n}</span>
                </div>
                <h3 className="mt-4 font-display text-lg font-semibold">{s.title}</h3>
                <p className="mt-1.5 text-sm leading-6 text-muted">{s.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------- DIFFERENTIATORS ------------------------- */}
      <section className="mx-auto max-w-6xl px-4 py-14 sm:px-6 lg:py-20">
        <p className="eyebrow">Beyond ordinary try-on</p>
        <h2 className="mt-2 max-w-2xl font-display text-3xl font-medium tracking-tight sm:text-4xl">
          Not just <em className="text-accent">how it looks</em> — whether it fits, and whether the design survived.
        </h2>

        <div className="mt-10 divide-y divide-line border-y border-line">
          {differentiators.map((d) => (
            <div key={d.title} className="grid gap-4 py-6 sm:grid-cols-[auto_1fr_auto] sm:items-center sm:gap-8">
              <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-ink text-paper">
                <d.icon size={22} strokeWidth={1.6} />
              </span>
              <div>
                <h3 className="font-display text-xl font-semibold leading-snug">{d.title}</h3>
                <p className="mt-1 max-w-2xl text-sm leading-6 text-muted">{d.text}</p>
              </div>
              <span className="hidden rounded-full border border-accent/30 bg-accent-soft px-3 py-1 text-[0.68rem] font-bold uppercase tracking-[0.14em] text-accent sm:inline-block">{d.tag}</span>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col items-start justify-between gap-6 rounded-2xl bg-ink p-6 text-paper sm:flex-row sm:items-center sm:p-8">
          <div>
            <p className="flex items-center gap-2 text-[0.7rem] font-bold uppercase tracking-[0.2em] text-[#E8C9BC]"><Cpu size={12} /> Runs on your hardware</p>
            <h3 className="mt-2 font-display text-2xl font-medium leading-snug">RTX 4060 · 8 GB VRAM · ≈ 5–10 minutes per render</h3>
            <p className="mt-1 max-w-xl text-sm leading-6 text-[#B9B3A5]">FASHN VTON v1.5 in bfloat16, 20–50 sampling steps, no internet required after download. First run downloads ~2 GB of weights.</p>
          </div>
          <Link to="/studio" className="btn-primary shrink-0 !bg-paper !text-ink hover:!bg-accent-soft">
            Open the Studio <ChevronRight size={18} />
          </Link>
        </div>
      </section>

      {/* --------------------------- TRUST STRIP --------------------------- */}
      <section className="mx-auto max-w-6xl px-4 pb-16 sm:px-6">
        <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-[0.78rem] text-muted">
          <span className="inline-flex items-center gap-2"><ShieldCheck size={15} className="text-sage" /> Photos never stored unless you click “Save to history”</span>
          <span className="inline-flex items-center gap-2"><ScanEye size={15} className="text-sage" /> Measured metrics over vanity scores</span>
          <span className="inline-flex items-center gap-2"><Ruler size={15} className="text-sage" /> Transparent, editable sizing rules</span>
        </div>
      </section>
    </div>
  );
}
