import React from 'react';
import { OptimizationSession } from '@/types';
import {
  History,
  Plus,
  Trash2,
  X,
  Clock,
  Cpu,
  TrendingDown,
  Layers,
  ChevronRight,
  Shield,
} from 'lucide-react';

export interface SidebarHistoryProps {
  isOpen: boolean;
  onClose: () => void;
  sessions: OptimizationSession[];
  currentSessionId?: string | null;
  onSelectSession: (session: OptimizationSession) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
  onClearAll: () => void;
}

export const SidebarHistory: React.FC<SidebarHistoryProps> = ({
  isOpen,
  onClose,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onClearAll,
}) => {
  // Format relative time helper
  const formatTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      const now = new Date();
      const diffSec = Math.floor((now.getTime() - date.getTime()) / 1000);
      if (diffSec < 60) return 'Just now';
      if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
      if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
      return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    } catch {
      return 'Recent';
    }
  };

  return (
    <>
      {/* Backdrop overlay for mobile & focus */}
      {isOpen && (
        <div
          aria-hidden="true"
          onClick={onClose}
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm transition-opacity duration-300"
        />
      )}

      {/* Drawer Panel */}
      <aside
        className={`fixed top-0 left-0 z-50 h-full w-80 sm:w-96 bg-[#070c17]/95 border-r border-white/[0.08] backdrop-blur-2xl shadow-[10px_0_40px_rgba(0,0,0,0.8)] flex flex-col justify-between transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Drawer Header */}
        <div className="p-5 border-b border-white/[0.07] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <History className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-white tracking-tight">Agentic History</h2>
                <p className="text-[11px] text-slate-400">Previous prompt optimizations</p>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-all cursor-pointer"
              title="Close sidebar"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* New Session Button */}
          <button
            type="button"
            onClick={() => {
              onNewSession();
              onClose();
            }}
            className="w-full py-2.5 px-3.5 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-500/20 to-violet-500/20 hover:from-cyan-500/30 hover:to-violet-500/30 text-cyan-200 border border-cyan-500/30 transition-all cursor-pointer shadow-sm"
          >
            <Plus className="w-4 h-4 text-cyan-400" />
            <span>New Optimization Session</span>
          </button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          {sessions.length === 0 ? (
            <div className="h-48 flex flex-col items-center justify-center text-center p-4 text-slate-500 space-y-2">
              <Clock className="w-8 h-8 opacity-30 text-cyan-400" />
              <p className="text-xs font-medium text-slate-400">No sessions recorded yet</p>
              <p className="text-[11px] text-slate-500 max-w-[200px]">
                Optimized prompts will automatically be saved to browser storage.
              </p>
            </div>
          ) : (
            sessions.map((session) => {
              const isSelected = session.id === currentSessionId;
              const reduction = session.result?.tokenReductionPercent ?? 0;

              return (
                <div
                  key={session.id}
                  onClick={() => {
                    onSelectSession(session);
                    onClose();
                  }}
                  className={`group relative p-3 rounded-xl border transition-all cursor-pointer select-none ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                      : 'bg-[#0a101d] border-white/5 hover:bg-[#0f172a] hover:border-white/10'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="text-xs font-semibold text-slate-200 line-clamp-1 group-hover:text-cyan-200 transition-colors">
                      {session.title || 'Untitled Optimization'}
                    </h3>

                    {/* Delete button */}
                    <button
                      type="button"
                      onClick={(e) => onDeleteSession(session.id, e)}
                      className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 transition-opacity cursor-pointer shrink-0"
                      title="Delete session"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <p className="text-[11px] text-slate-400 line-clamp-2 mt-1 font-mono">
                    {session.rawPrompt}
                  </p>

                  <div className="mt-3 flex items-center justify-between text-[10px] font-mono text-slate-400 pt-2 border-t border-white/[0.04]">
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-400">{formatTime(session.timestamp)}</span>
                      <span className="text-slate-600">&bull;</span>
                      <span className="text-cyan-400">{session.targetModel}</span>
                      {session.comparisonResult && (
                        <span className="px-1.5 py-0.2 rounded bg-violet-950/80 text-violet-300 border border-violet-500/30 text-[9px]">
                          A/B Tested
                        </span>
                      )}
                    </div>

                    {reduction > 0 ? (
                      <span className="font-semibold text-emerald-400 flex items-center gap-0.5">
                        <TrendingDown className="w-3 h-3" />
                        -{reduction.toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-slate-500">0%</span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Drawer Footer */}
        <div className="p-4 border-t border-white/[0.07] bg-[#060a14] space-y-2">
          {sessions.length > 0 && (
            <button
              type="button"
              onClick={onClearAll}
              className="w-full py-2 px-3 rounded-lg text-xs font-medium text-slate-400 hover:text-red-300 hover:bg-red-950/20 border border-white/5 hover:border-red-500/20 transition-all cursor-pointer flex items-center justify-center gap-1.5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear Session History</span>
            </button>
          )}

          <div className="flex items-center justify-center gap-1.5 text-[10px] font-mono text-slate-400 pt-1">
            <Shield className="w-3 h-3 text-cyan-400" />
            <span>Browser-local history &bull; Private to this machine</span>
          </div>
        </div>
      </aside>
    </>
  );
};
