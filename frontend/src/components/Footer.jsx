import { Link } from 'react-router-dom';
import { Shirt, Code, ShieldCheck, Cpu } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-line bg-ink text-[#C9C4B8]">
      <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
        <div className="grid gap-10 md:grid-cols-4">
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 text-paper">
              <Shirt size={18} />
              <span className="font-display text-lg font-semibold">FitVision AI</span>
            </div>
            <p className="mt-3 max-w-sm text-sm leading-6">
              Intelligent 2D virtual try-on with fit-preference simulation, garment preservation
              analysis and measurement-based fit insights — running locally, not in the cloud.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-[0.7rem]">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-white/15 px-3 py-1">
                <Cpu size={12} /> Local inference · FASHN VTON v1.5 (Apache-2.0)
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-white/15 px-3 py-1">
                <ShieldCheck size={12} /> Uploads not stored unless you save them
              </span>
            </div>
          </div>

          <div>
            <p className="label !text-[#8F887A]">Product</p>
            <ul className="mt-2 space-y-2 text-sm">
              <li><Link to="/studio" className="hover:text-paper">Virtual Try-On Studio</Link></li>
              <li><Link to="/profile" className="hover:text-paper">Profile & History</Link></li>
            </ul>
          </div>

          <div>
            <p className="label !text-[#8F887A]">Honest limitations</p>
            <ul className="mt-2 space-y-1.5 text-[0.78rem] leading-5">
              <li>· Try-on images are diffusion renders, not physics simulations.</li>
              <li>· Fit previews are visual approximations, not physical predictions.</li>
              <li>· Photo-based measurements are rough estimates only.</li>
              <li>· Preservation checks are decision-support, not proof of fidelity.</li>
            </ul>
          </div>
        </div>
        <div className="mt-10 flex flex-wrap items-center justify-between gap-4 border-t border-white/10 pt-6 text-[0.75rem] text-[#8F887A]">
          <span>© {new Date().getFullYear()} FitVision AI · MVP built for local execution (RTX-class GPU, 8 GB VRAM).</span>
          <a href="https://github.com/fashn-AI/fashn-vton-1.5" target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 hover:text-paper">
            <Code size={14} /> Model: fashn-AI/fashn-vton-1.5
          </a>
        </div>
      </div>
    </footer>
  );
}
