import { Link, NavLink, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Shirt, Menu, X } from 'lucide-react';

const links = [
  { to: '/', label: 'Home' },
  { to: '/studio', label: 'Try-On Studio' },
  { to: '/profile', label: 'Profile & History' },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo(0, 0); }, [location.pathname]);

  return (
    <header className="sticky top-0 z-40 border-b border-line/70 bg-paper/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-ink text-paper transition-colors group-hover:bg-accent-deep">
            <Shirt size={18} strokeWidth={1.8} />
          </span>
          <span className="font-display text-[1.35rem] font-semibold tracking-tight">
            FitVision<span className="text-accent">&nbsp;AI</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `rounded-full px-4 py-2 text-[0.83rem] font-medium transition-colors ${
                  isActive ? 'bg-ink text-paper' : 'text-muted hover:text-ink'
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
          <Link to="/studio" className="btn-primary ml-2 !px-5 !py-2 text-[0.83rem]">
            Start Try-On
          </Link>
        </nav>

        <button className="p-2 md:hidden" onClick={() => setOpen((v) => !v)} aria-label="Toggle menu">
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {open && (
        <div className="border-t border-line bg-paper px-4 py-4 md:hidden">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `block rounded-xl px-4 py-3 text-sm font-medium ${isActive ? 'bg-ink text-paper' : 'text-ink'}`
              }
            >
              {l.label}
            </NavLink>
          ))}
          <Link to="/studio" className="btn-primary mt-2 w-full justify-center">Start Virtual Try-On</Link>
        </div>
      )}
    </header>
  );
}
