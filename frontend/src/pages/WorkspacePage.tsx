import React, { useState } from 'react';
import { GlowingRing } from '@/components/GlowingRing';
import { PromptInputPanel } from '@/components/PromptInputPanel';
import { ResultFlipCard } from '@/components/ResultFlipCard';
import { DualComparisonView } from '@/components/DualComparisonView';
import { SidebarHistory } from '@/components/SidebarHistory';
import { apiService } from '@/services/api';
import {
  OptimizationMode,
  TargetModel,
  OptimizationResult,
  OptimizationStatus,
  OptimizationSession,
  DualExecutionResult,
  OllamaStatus,
} from '@/types';
import {
  Sparkles,
  Zap,
  RotateCcw,
  Sliders,
  ChevronDown,
  ChevronUp,
  Cpu,
  Shield,
  Layers,
  ArrowRight,
  Split,
  FileText,
} from 'lucide-react';

export interface WorkspacePageProps {
  ollamaStatus?: OllamaStatus | null;
  isSidebarOpen: boolean;
  onCloseSidebar: () => void;
  sessions: OptimizationSession[];
  onSessionsChange: (sessions: OptimizationSession[]) => void;
  onToggleSidebar?: () => void;
}

export const WorkspacePage: React.FC<WorkspacePageProps> = ({
  ollamaStatus,
  isSidebarOpen,
  onCloseSidebar,
  sessions,
  onSessionsChange,
  onToggleSidebar,
}) => {
  const [rawPrompt, setRawPrompt] = useState<string>('');
  const [mode, setMode] = useState<OptimizationMode>('balanced');
  const [targetModel, setTargetModel] = useState<TargetModel>('gemini-2.0-flash');
  const [status, setStatus] = useState<OptimizationStatus>('idle');
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [dualResult, setDualResult] = useState<DualExecutionResult | null>(null);
  const [activeView, setActiveView] = useState<'single' | 'dual'>('dual');
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isInputExpanded, setIsInputExpanded] = useState<boolean>(true);

  // Execute Optimization Only
  const handleOptimize = async () => {
    const trimmed = rawPrompt.trim();
    if (!trimmed) {
      setErrorMessage('Please enter or select a prompt before optimizing.');
      setStatus('error');
      return;
    }

    setErrorMessage(null);
    setStatus('optimizing');

    try {
      const optimizationResult = await apiService.optimizePrompt({
        prompt: trimmed,
        mode,
        target_model: targetModel,
      });

      const title =
        trimmed.split('\n')[0].replace(/^[#\s*]+/, '').trim().substring(0, 38) +
        (trimmed.length > 38 ? '...' : '');

      const newSession: OptimizationSession = {
        id: `sess_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
        timestamp: new Date().toISOString(),
        title: title || 'Optimization Session',
        rawPrompt: trimmed,
        mode,
        targetModel,
        result: optimizationResult,
        comparisonResult: null,
      };

      const updatedSessions = [newSession, ...sessions.slice(0, 49)];
      onSessionsChange(updatedSessions);
      try {
        localStorage.setItem('argus_optimization_sessions', JSON.stringify(updatedSessions));
      } catch (err) {
        console.warn('Unable to persist session to localStorage', err);
      }

      setCurrentSessionId(newSession.id);
      setResult(optimizationResult);
      setDualResult(null);
      setActiveView('single');
      setStatus('success');
      setIsInputExpanded(false);
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : 'An unexpected error occurred during prompt optimization'
      );
      setStatus('error');
    }
  };

  // Execute Dual Prompt A/B Comparison
  const handleCompare = async () => {
    const trimmed = rawPrompt.trim();
    if (!trimmed) {
      setErrorMessage('Please enter or select a prompt before running A/B comparison.');
      setStatus('error');
      return;
    }

    setErrorMessage(null);
    setIsComparing(true);
    setStatus('optimizing');

    try {
      const comparisonResult = await apiService.comparePrompt({
        prompt: trimmed,
        mode,
        target_model: targetModel,
      });

      const title =
        trimmed.split('\n')[0].replace(/^[#\s*]+/, '').trim().substring(0, 38) +
        (trimmed.length > 38 ? '...' : '');

      const newSession: OptimizationSession = {
        id: `sess_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
        timestamp: new Date().toISOString(),
        title: title || 'A/B Dual Execution Session',
        rawPrompt: trimmed,
        mode,
        targetModel,
        result: comparisonResult.optimization,
        comparisonResult: comparisonResult,
      };

      const updatedSessions = [newSession, ...sessions.slice(0, 49)];
      onSessionsChange(updatedSessions);
      try {
        localStorage.setItem('argus_optimization_sessions', JSON.stringify(updatedSessions));
      } catch (err) {
        console.warn('Unable to persist session to localStorage', err);
      }

      setCurrentSessionId(newSession.id);
      setResult(comparisonResult.optimization);
      setDualResult(comparisonResult);
      setActiveView('dual');
      setStatus('success');
      setIsInputExpanded(false);
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : 'An unexpected error occurred during dual A/B comparison'
      );
      setStatus('error');
    } finally {
      setIsComparing(false);
    }
  };

  // Select previous session from history
  const handleSelectSession = (session: OptimizationSession) => {
    setRawPrompt(session.rawPrompt);
    setMode(session.mode);
    setTargetModel(session.targetModel);
    setResult(session.result);
    setDualResult(session.comparisonResult || null);
    setActiveView(session.comparisonResult ? 'dual' : 'single');
    setCurrentSessionId(session.id);
    setStatus('success');
    setIsInputExpanded(false);
    setErrorMessage(null);
  };

  // Start new clean session
  const handleNewSession = () => {
    setRawPrompt('');
    setResult(null);
    setDualResult(null);
    setCurrentSessionId(null);
    setStatus('idle');
    setIsInputExpanded(true);
    setErrorMessage(null);
  };

  // Delete individual session
  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const filtered = sessions.filter((s) => s.id !== id);
    onSessionsChange(filtered);
    try {
      localStorage.setItem('argus_optimization_sessions', JSON.stringify(filtered));
    } catch {
      // Ignored
    }
    if (currentSessionId === id) {
      setCurrentSessionId(null);
    }
  };

  // Clear all history
  const handleClearAllSessions = () => {
    if (window.confirm('Are you sure you want to clear all optimization history?')) {
      onSessionsChange([]);
      try {
        localStorage.removeItem('argus_optimization_sessions');
      } catch {
        // Ignored
      }
      setCurrentSessionId(null);
    }
  };

  // Update optimized prompt text
  const handleOptimizedPromptChange = (updated: string) => {
    if (result) {
      const updatedResult = {
        ...result,
        optimizedPrompt: updated,
      };
      setResult(updatedResult);

      if (currentSessionId) {
        const updatedSessions = sessions.map((s) =>
          s.id === currentSessionId ? { ...s, result: updatedResult } : s
        );
        onSessionsChange(updatedSessions);
        try {
          localStorage.setItem('argus_optimization_sessions', JSON.stringify(updatedSessions));
        } catch {
          // Ignored
        }
      }
    }
  };

  const isSuccess = status === 'success' && result !== null;

  return (
    <div className="relative min-h-[calc(100vh-4rem)] flex flex-col justify-between overflow-hidden px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
      {/* 1. Atmospheric Glowing Celestial Ring Backdrop */}
      <GlowingRing status={status} minimized={isSuccess && !isInputExpanded} />

      {/* 2. Left Slide-out Agentic History Drawer */}
      <SidebarHistory
        isOpen={isSidebarOpen}
        onClose={onCloseSidebar}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        onClearAll={handleClearAllSessions}
      />

      {/* 3. Main Stage Content */}
      <div className="relative z-10 max-w-6xl mx-auto w-full space-y-8">
        {/* Top Argus Identity Header */}
        <header className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Argus Prompt Optimization &amp; A/B Comparison</span>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-white">
            Understand. <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-violet-400">Optimize.</span> Execute.
          </h1>

          <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Neural prompt compression and dual A/B execution engine. Compare original versus optimized
            prompts on the same frontier model with explainable quality and cost telemetry.
          </p>
        </header>

        {/* Success Collapsible Input Bar: When result is visible */}
        {isSuccess && (
          <div className="w-full max-w-5xl mx-auto">
            <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl backdrop-blur-xl bg-[#090f1e]/80 border border-white/[0.08] shadow-md">
              <div className="flex items-center gap-2.5 text-xs text-slate-300">
                <span className="font-mono text-cyan-400 font-semibold">Source Input:</span>
                <span className="line-clamp-1 max-w-[200px] sm:max-w-md text-slate-400 font-mono text-[11px]">
                  {rawPrompt}
                </span>
              </div>

              <div className="flex items-center gap-2">
                {/* View switcher between Dual A/B and 3D Flip Card */}
                {dualResult && (
                  <div className="flex items-center bg-[#070b16] rounded-lg p-0.5 border border-white/5 text-[11px] font-mono">
                    <button
                      type="button"
                      onClick={() => setActiveView('dual')}
                      className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1 cursor-pointer ${
                        activeView === 'dual'
                          ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      <Split className="w-3 h-3" />
                      <span>A/B Comparison</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveView('single')}
                      className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1 cursor-pointer ${
                        activeView === 'single'
                          ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      <FileText className="w-3 h-3" />
                      <span>Prompt Card</span>
                    </button>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() => setIsInputExpanded(!isInputExpanded)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-[#0f172a] hover:bg-white/10 border border-white/5 transition-all cursor-pointer"
                >
                  {isInputExpanded ? (
                    <>
                      <ChevronUp className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Collapse Input</span>
                    </>
                  ) : (
                    <>
                      <ChevronDown className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Edit Input Prompt</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={handleNewSession}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold text-cyan-300 hover:text-white bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-500/30 transition-all cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>New Prompt</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Floating Multiline Prompt Input Panel */}
        {(!isSuccess || isInputExpanded) && (
          <PromptInputPanel
            rawPrompt={rawPrompt}
            onPromptChange={(val) => {
              setRawPrompt(val);
              if (errorMessage) setErrorMessage(null);
            }}
            mode={mode}
            onModeChange={setMode}
            targetModel={targetModel}
            onTargetModelChange={setTargetModel}
            onOptimize={handleOptimize}
            onCompare={handleCompare}
            onClear={() => {
              setRawPrompt('');
              setErrorMessage(null);
            }}
            isLoading={status === 'optimizing' && !isComparing}
            isComparing={isComparing}
            ollamaStatus={ollamaStatus}
            errorMessage={errorMessage}
          />
        )}

        {/* Presentation Stage: Either Dual Comparison View or 3D Flip Card */}
        {isSuccess && activeView === 'dual' && dualResult && (
          <DualComparisonView
            result={dualResult}
            onResetWorkspace={handleNewSession}
          />
        )}

        {isSuccess && (activeView === 'single' || !dualResult) && result && (
          <ResultFlipCard
            result={result}
            onOptimizedPromptChange={handleOptimizedPromptChange}
            onResetWorkspace={handleNewSession}
          />
        )}
      </div>

      {/* Bottom Subtle Workspace Attribution */}
      <div className="relative z-10 text-center pt-8 text-[11px] font-mono text-slate-400 flex items-center justify-center gap-4">
        <span>Argus Core &bull; A/B Comparison Engine</span>
        <span className="text-slate-700">&bull;</span>
        <span>Dual Prompt Execution</span>
        <span className="text-slate-700">&bull;</span>
        <span>Explainable 9-Dimension Telemetry</span>
      </div>
    </div>
  );
};
