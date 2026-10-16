import React from 'react';
import { StatusBadge } from './StatusBadge';
import { Shield, GitBranch, Terminal } from 'lucide-react';

export interface HeaderProps {
  apiStatus?: 'online' | 'offline' | 'idle';
}

export const Header: React.FC<HeaderProps> = ({ apiStatus = 'idle' }) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.08] bg-[#070c17]/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-100 tracking-tight text-lg">Argus</span>
              <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/20">
                Phase 1
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">Intelligent Prompt Optimization Middleware</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-400 font-mono bg-[#0c1424] px-3 py-1.5 rounded-lg border border-white/5">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span>FastAPI + Vite</span>
          </div>

          <StatusBadge
            status={apiStatus}
            label={apiStatus === 'online' ? 'API Online' : apiStatus === 'offline' ? 'API Offline' : 'API Standby'}
          />

          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-white/5 transition-colors"
            title="Repository"
          >
            <GitBranch className="w-4 h-4" />
          </a>
        </div>
      </div>
    </header>
  );
};
