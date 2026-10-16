import React from 'react';
import { Header } from '@/components/Header';
import { HomePage } from '@/pages/HomePage';

export const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#060a12] text-slate-200 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-300">
      <Header />
      <main className="flex-1">
        <HomePage />
      </main>
      <footer className="border-t border-white/[0.06] bg-[#070c17] py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Argus Prompt Optimization Middleware &copy; {new Date().getFullYear()}</span>
          <span className="font-mono text-[11px] text-slate-400">Phase 1 Foundation &bull; Dark Navy / Cyan Palette</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
