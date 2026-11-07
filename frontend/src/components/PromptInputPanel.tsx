import React, { useRef, useEffect } from 'react';
import {
  OptimizationMode,
  TargetModel,
  ExamplePrompt,
  CREDENTIALED_MODELS,
  OllamaStatus,
} from '@/types';
import { EXAMPLE_PROMPTS, estimateTokens } from '@/services/mockOptimizer';
import {
  Sparkles,
  Zap,
  Trash2,
  Sliders,
  Cpu,
  Layers,
  FileText,
  CornerDownLeft,
  Info,
  Split,
  Scale,
} from 'lucide-react';

export interface PromptInputPanelProps {
  rawPrompt: string;
  onPromptChange: (value: string) => void;
  mode: OptimizationMode;
  onModeChange: (mode: OptimizationMode) => void;
  targetModel: TargetModel;
  onTargetModelChange: (model: TargetModel) => void;
  onOptimize: () => void;
  onCompare?: () => void;
  onClear: () => void;
  isLoading: boolean;
  isComparing?: boolean;
  ollamaStatus?: OllamaStatus | null;
  errorMessage?: string | null;
}

export const PromptInputPanel: React.FC<PromptInputPanelProps> = ({
  rawPrompt,
  onPromptChange,
  mode,
  onModeChange,
  targetModel,
  onTargetModelChange,
  onOptimize,
  onCompare,
  onClear,
  isLoading,
  isComparing = false,
  ollamaStatus,
  errorMessage,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const estimation = estimateTokens(rawPrompt);
  const isEmpty = rawPrompt.trim().length === 0;

  // Keydown shortcut listener for Ctrl+Enter or Cmd+Enter
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isEmpty && !isLoading) {
        onOptimize();
      }
    }
  };

  const handleSelectExample = (example: ExamplePrompt) => {
    onPromptChange(example.content);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const selectedModel =
    CREDENTIALED_MODELS.find((m) => m.id === targetModel) || CREDENTIALED_MODELS[0];

  return (
    <div className="relative z-10 w-full max-w-4xl mx-auto transition-all duration-700 ease-out animate-float-subtle">
      {/* Floating Glassmorphic Container */}
      <div className="rounded-2xl backdrop-blur-2xl bg-[#090f1e]/85 border border-white/[0.09] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.85)] p-5 sm:p-7 space-y-6">
        {/* Top Header & Sample Loader Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/[0.06]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-mono font-semibold uppercase tracking-wider text-cyan-300">
                Raw Prompt Input
              </span>
              <p className="text-[11px] text-slate-400">
                Input instructions to compress tokens while preserving intent
              </p>
            </div>
          </div>

          {/* Quick-loader sample pills */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-slate-400 font-mono flex items-center gap-1 mr-1">
              <Sparkles className="w-3 h-3 text-cyan-400" /> Samples:
            </span>
            {EXAMPLE_PROMPTS.map((ex) => (
              <button
                key={ex.id}
                type="button"
                onClick={() => handleSelectExample(ex)}
                disabled={isLoading}
                title={ex.description}
                className="text-[11px] font-medium px-2.5 py-1 rounded-md bg-[#0f172a] hover:bg-cyan-950/80 hover:text-cyan-300 hover:border-cyan-500/40 text-slate-300 border border-white/5 transition-all cursor-pointer disabled:opacity-50"
              >
                {ex.title.split(' ')[0]}
              </button>
            ))}

            {!isEmpty && (
              <button
                type="button"
                onClick={onClear}
                disabled={isLoading}
                title="Clear input"
                className="text-[11px] p-1.5 rounded-md text-slate-400 hover:text-red-400 hover:bg-red-950/30 transition-all cursor-pointer ml-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Textarea Input Editor */}
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={rawPrompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={7}
            placeholder="Enter or paste your raw prompt here to compress tokens and preserve critical constraints... (or click a sample prompt above)"
            className="w-full p-4 rounded-xl bg-[#060a14]/90 border border-white/10 text-slate-100 placeholder-slate-500 font-mono text-xs sm:text-sm leading-relaxed resize-y min-h-[170px] focus:outline-none focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 transition-all selection:bg-cyan-500/30"
          />

          {/* Bottom Editor Status Bar: Live Characters & Token Estimate */}
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-slate-400 px-1">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1 text-slate-300">
                <span className="text-slate-500 font-sans">Chars:</span>{' '}
                <strong className="text-white">{estimation.charCount.toLocaleString()}</strong>
              </span>
              <span className="text-slate-600">&bull;</span>
              <span className="flex items-center gap-1 text-cyan-300" title="Based on standard ~4 characters per token heuristic">
                <span className="text-slate-500 font-sans">Estimated:</span>{' '}
                <strong className="text-cyan-200">~{estimation.estimatedTokens.toLocaleString()} tokens</strong>
              </span>
              <span className="hidden md:inline-flex items-center gap-1 text-[10px] text-slate-500">
                <Info className="w-3 h-3 text-slate-500" />
                (~4 chars/tok heuristic estimate)
              </span>
            </div>

            <div className="text-[11px] text-slate-500 hidden sm:flex items-center gap-1">
              <span>Press</span>
              <kbd className="px-1.5 py-0.5 rounded bg-[#0f172a] text-cyan-300 border border-white/10 font-mono text-[10px]">
                Ctrl+Enter
              </kbd>
              <span>to optimize</span>
            </div>
          </div>
        </div>

        {/* Configuration Controls Grid: Profile & Model */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 pt-2 border-t border-white/[0.06] items-end">
          {/* Profile Selector (7 cols) */}
          <div className="md:col-span-7 space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                <span>Optimization Profile</span>
              </label>
              <span className="text-[10px] font-mono text-slate-400 capitalize">
                {mode === 'balanced' && 'High Preservation & Density'}
                {mode === 'concise' && 'Aggressive Token Pruning'}
                {mode === 'structure' && 'Standardized Schemas'}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-1.5 p-1 rounded-xl bg-[#060a14] border border-white/5">
              {(['balanced', 'concise', 'structure'] as OptimizationMode[]).map((m) => {
                const isActive = mode === m;
                return (
                  <button
                    key={m}
                    type="button"
                    onClick={() => onModeChange(m)}
                    disabled={isLoading}
                    className={`py-2 px-2.5 rounded-lg text-xs font-medium transition-all text-center capitalize cursor-pointer ${
                      isActive
                        ? 'bg-gradient-to-r from-cyan-500/20 to-violet-500/20 text-cyan-200 border border-cyan-500/40 shadow-sm font-semibold'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent'
                    }`}
                  >
                    {m}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Target Model Selector (5 cols) */}
          <div className="md:col-span-5 space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="credentialed-target-model" className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-violet-400" />
                <span>Target Model</span>
              </label>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-500/20">
                {selectedModel.provider}
              </span>
            </div>

            <select
              id="credentialed-target-model"
              value={targetModel}
              onChange={(e) => onTargetModelChange(e.target.value as TargetModel)}
              disabled={isLoading}
              className="w-full p-2.5 rounded-xl bg-[#060a14] border border-white/10 text-slate-200 text-xs font-medium focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 transition-all cursor-pointer"
            >
              {CREDENTIALED_MODELS.map((m) => (
                <option key={m.id} value={m.id} className="bg-[#0b1324] text-slate-200">
                  {m.name} &bull; ${m.inputPricePerMillionUsd.toFixed(2)}/1M ({m.contextWindow})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Error message banner if present */}
        {errorMessage && (
          <div className="p-3 rounded-xl bg-red-950/40 border border-red-500/30 text-red-200 text-xs flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Execute Action Button Bar: Single Optimize & A/B Dual Execution */}
        <div className="pt-2 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Option 1: Optimize Prompt Only */}
          <button
            type="button"
            onClick={onOptimize}
            disabled={isEmpty || isLoading || isComparing}
            className={`py-3.5 px-4 rounded-xl font-semibold text-xs sm:text-sm transition-all duration-300 flex items-center justify-center gap-2 shadow-md cursor-pointer ${
              isEmpty || isLoading || isComparing
                ? 'bg-slate-800/60 text-slate-500 border border-white/5 cursor-not-allowed'
                : 'bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-200 hover:text-white border border-cyan-500/40 hover:border-cyan-400 transform active:scale-[0.99]'
            }`}
          >
            {isLoading ? (
              <>
                <Zap className="w-4 h-4 text-cyan-200 animate-spin" />
                <span>Optimizing Prompt...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 text-cyan-300" />
                <span>Optimize Prompt Only</span>
              </>
            )}
          </button>

          {/* Option 2: Full A/B Dual Execution & Comparison */}
          <button
            type="button"
            onClick={onCompare || onOptimize}
            disabled={isEmpty || isLoading || isComparing}
            className={`py-3.5 px-4 rounded-xl font-semibold text-xs sm:text-sm transition-all duration-300 flex items-center justify-center gap-2 shadow-lg cursor-pointer ${
              isEmpty || isLoading || isComparing
                ? 'bg-slate-800/60 text-slate-500 border border-white/5 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 via-blue-600 to-violet-600 hover:from-cyan-400 hover:via-blue-500 hover:to-violet-500 text-white border border-cyan-400/40 shadow-[0_0_25px_rgba(6,182,212,0.35)] hover:shadow-[0_0_35px_rgba(139,92,246,0.5)] transform active:scale-[0.99]'
            }`}
          >
            {isComparing ? (
              <>
                <Split className="w-4 h-4 text-white animate-spin" />
                <span>Running A/B Dual Execution...</span>
              </>
            ) : (
              <>
                <Split className="w-4 h-4 text-white" />
                <span>Run A/B Dual Execution</span>
                <span className="hidden sm:inline-block text-[10px] font-mono opacity-80 ml-1 px-1.5 py-0.5 rounded bg-black/20 border border-white/10">
                  Ctrl+Enter
                </span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
