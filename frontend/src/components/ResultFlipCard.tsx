import React, { useState } from 'react';
import {
  OptimizationResult,
  TargetModel,
  CREDENTIALED_MODELS,
} from '@/types';
import {
  Copy,
  Check,
  RotateCw,
  RotateCcw,
  ShieldCheck,
  Cpu,
  Zap,
  TrendingDown,
  DollarSign,
  Clock,
  Edit3,
  Eye,
  CheckCircle2,
  AlertCircle,
  FileCode,
  Tag,
  BarChart2,
} from 'lucide-react';

export interface ResultFlipCardProps {
  result: OptimizationResult;
  onOptimizedPromptChange?: (updated: string) => void;
  onResetWorkspace?: () => void;
}

export const ResultFlipCard: React.FC<ResultFlipCardProps> = ({
  result,
  onOptimizedPromptChange,
  onResetWorkspace,
}) => {
  const [isFlipped, setIsFlipped] = useState<boolean>(false);
  const [isCopied, setIsCopied] = useState<boolean>(false);
  const [isEditing, setIsEditing] = useState<boolean>(false);

  // Target model pricing & details
  const modelOption =
    CREDENTIALED_MODELS.find((m) => m.id === result.targetModel) ||
    CREDENTIALED_MODELS.find((m) => m.id === 'gemini-2.0-flash') ||
    CREDENTIALED_MODELS[0];

  const ratePerMillion = modelOption.inputPricePerMillionUsd;

  // Token Accounting with zero-denominator safety
  const tOrig = result.originalMetrics?.estimatedTokens || 0;
  const tOpt = result.optimizedMetrics?.estimatedTokens || 0;
  const tSaved = tOrig - tOpt;
  const tokenReductionPercent =
    tOrig > 0 ? ((tOrig - tOpt) / tOrig) * 100 : 0.0;

  // Character Footprint
  const cOrig = result.originalMetrics?.charCount || 0;
  const cOpt = result.optimizedMetrics?.charCount || 0;
  const cDelta = cOpt - cOrig;
  const charDeltaPercent = cOrig > 0 ? (cDelta / cOrig) * 100 : 0.0;

  // Cost Projections (rate per 1M tokens)
  const costOrig = (tOrig * ratePerMillion) / 1_000_000;
  const costOpt = (tOpt * ratePerMillion) / 1_000_000;
  const costSaved = Math.max(0, costOrig - costOpt);
  const costSavingsPercent =
    costOrig > 0 ? ((costOrig - costOpt) / costOrig) * 100 : 0.0;

  // Latency & Throughput
  const latencyMs = result.executionTimeMs || 0;
  const throughputTokensPerSec =
    latencyMs > 0 && tOpt > 0
      ? Math.round(tOpt / (latencyMs / 1000))
      : 0;

  // Copy to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.optimizedPrompt);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2200);
    } catch {
      // Fallback
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2200);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto py-2 transition-all duration-700 ease-out">
      {/* 3D Perspective Container */}
      <div className="perspective-1200 w-full min-h-[580px]">
        {/* Flipping Inner Container */}
        <div
          className={`relative w-full min-h-[580px] rounded-2xl transition-transform duration-700 transform-style-3d ${
            isFlipped ? 'rotate-y-180' : ''
          }`}
        >
          {/* ===================================================================
              FRONT FACE: Optimized Prompt & Action Controls
              =================================================================== */}
          <div
            className={`w-full min-h-[580px] rounded-2xl p-6 sm:p-8 flex flex-col justify-between backdrop-blur-2xl bg-[#090f1e]/90 border border-white/[0.12] shadow-[0_25px_70px_-15px_rgba(0,0,0,0.85)] backface-hidden ${
              isFlipped ? 'pointer-events-none' : ''
            }`}
          >
            {/* Front Header */}
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-white/[0.08]">
                {/* Status & Strategy Badges */}
                <div className="flex flex-wrap items-center gap-2">
                  <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-emerald-950/80 text-emerald-300 border border-emerald-500/30">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Constraints Verified</span>
                  </div>

                  <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-cyan-950/70 text-cyan-300 border border-cyan-500/30 uppercase tracking-wider">
                    <Tag className="w-3 h-3 text-cyan-400" />
                    <span>{result.strategy || 'Compress'}</span>
                  </div>

                  <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-violet-950/70 text-violet-300 border border-violet-500/30 capitalize">
                    <span>{result.mode} Profile</span>
                  </div>

                  {tokenReductionPercent > 0 && (
                    <div className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-mono font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-400/30">
                      <TrendingDown className="w-3.5 h-3.5 text-cyan-400" />
                      <span>-{tokenReductionPercent.toFixed(1)}% Tokens</span>
                    </div>
                  )}
                </div>

                {/* Header Action Controls */}
                <div className="flex items-center gap-2">
                  {/* Edit Toggle */}
                  <button
                    type="button"
                    onClick={() => setIsEditing(!isEditing)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-[#0f172a] hover:bg-[#1e293b] border border-white/5 transition-all cursor-pointer"
                    title={isEditing ? 'Preview formatted text' : 'Edit prompt'}
                  >
                    {isEditing ? (
                      <>
                        <Eye className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Preview</span>
                      </>
                    ) : (
                      <>
                        <Edit3 className="w-3.5 h-3.5 text-slate-400" />
                        <span>Edit</span>
                      </>
                    )}
                  </button>

                  {/* Copy Button with Feedback */}
                  <button
                    type="button"
                    onClick={handleCopy}
                    className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer shadow-md ${
                      isCopied
                        ? 'bg-emerald-600 text-white border border-emerald-400/50'
                        : 'bg-cyan-500/20 text-cyan-200 border border-cyan-500/40 hover:bg-cyan-500/30 hover:border-cyan-400/60'
                    }`}
                  >
                    {isCopied ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span>Copied to Clipboard!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy Prompt</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Title Banner */}
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-mono text-cyan-300 uppercase tracking-wider flex items-center gap-1.5">
                  <FileCode className="w-3.5 h-3.5" />
                  Optimized Prompt Formulation
                </span>
                <span className="font-mono text-slate-500">
                  {cOpt.toLocaleString()} chars &bull; ~{tOpt.toLocaleString()} tokens
                </span>
              </div>
            </div>

            {/* Prompt Body: Formatted View vs. Editable View */}
            <div className="my-4 flex-1">
              {isEditing ? (
                <textarea
                  value={result.optimizedPrompt}
                  onChange={(e) =>
                    onOptimizedPromptChange &&
                    onOptimizedPromptChange(e.target.value)
                  }
                  rows={10}
                  className="w-full h-full min-h-[260px] p-4 rounded-xl bg-[#060914]/90 border border-cyan-500/40 text-slate-100 font-mono text-xs sm:text-sm leading-relaxed resize-y focus:outline-none focus:ring-1 focus:ring-cyan-500/40 selection:bg-cyan-500/30"
                  placeholder="Optimized prompt text..."
                />
              ) : (
                <div className="relative group rounded-xl bg-[#050811]/90 border border-white/[0.07] p-4 sm:p-5 overflow-y-auto max-h-[340px] select-text">
                  <pre className="font-mono text-xs sm:text-sm leading-relaxed text-slate-100 whitespace-pre-wrap break-words">
                    {result.optimizedPrompt}
                  </pre>
                </div>
              )}

              {/* Preserved Constraints Tags */}
              {result.preservedConstraints &&
                result.preservedConstraints.length > 0 && (
                  <div className="mt-3 flex flex-wrap items-center gap-1.5">
                    <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1 mr-1">
                      <ShieldCheck className="w-3 h-3 text-emerald-400" /> Preserved:
                    </span>
                    {result.preservedConstraints.map((constraint, idx) => (
                      <span
                        key={idx}
                        className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/50 text-emerald-300 border border-emerald-500/20"
                      >
                        {constraint}
                      </span>
                    ))}
                  </div>
                )}
            </div>

            {/* Front Footer: Model Info & Prominent 3D Flip Trigger */}
            <div className="pt-4 border-t border-white/[0.08] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                <span className="px-2 py-1 rounded bg-[#0f172a] border border-white/5 text-slate-300">
                  Target: <strong className="text-white">{modelOption.name}</strong>
                </span>
                <span className="text-slate-600">&bull;</span>
                <span className="text-slate-400 flex items-center gap-1">
                  <Clock className="w-3 h-3 text-cyan-400" /> {latencyMs}ms
                </span>
                {onResetWorkspace && (
                  <>
                    <span className="text-slate-600">&bull;</span>
                    <button
                      type="button"
                      onClick={onResetWorkspace}
                      className="text-cyan-400 hover:text-cyan-200 underline cursor-pointer text-xs"
                    >
                      New Prompt
                    </button>
                  </>
                )}
              </div>

              {/* 3D Flip Trigger Button */}
              <button
                type="button"
                onClick={() => setIsFlipped(true)}
                className="group inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-xs transition-all duration-300 bg-gradient-to-r from-violet-600/30 to-cyan-600/30 hover:from-violet-600/50 hover:to-cyan-600/50 text-cyan-200 hover:text-white border border-cyan-400/40 hover:border-cyan-300 shadow-[0_0_20px_rgba(139,92,246,0.25)] hover:shadow-[0_0_25px_rgba(6,182,212,0.4)] cursor-pointer"
              >
                <BarChart2 className="w-4 h-4 text-cyan-300 group-hover:scale-110 transition-transform" />
                <span>Technical & Cost Audit</span>
                <RotateCw className="w-3.5 h-3.5 text-violet-300 group-hover:rotate-180 transition-transform duration-500" />
              </button>
            </div>
          </div>

          {/* ===================================================================
              REVERSE FACE: Technical & Mathematical Telemetry Audit
              =================================================================== */}
          <div
            className={`absolute inset-0 w-full min-h-[580px] rounded-2xl p-6 sm:p-8 flex flex-col justify-between backdrop-blur-2xl bg-[#080d1b]/95 border border-cyan-500/25 shadow-[0_25px_70px_-15px_rgba(0,0,0,0.9)] backface-hidden rotate-y-180 ${
              !isFlipped ? 'pointer-events-none' : ''
            }`}
          >
            {/* Reverse Header */}
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-violet-950/80 border border-violet-500/30 flex items-center justify-center text-violet-400">
                    <BarChart2 className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                      Technical & Mathematical Telemetry Audit
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30">
                        P11
                      </span>
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      Deterministic token accounting, character variance, and economic projections
                    </p>
                  </div>
                </div>

                {/* Flip Back Button */}
                <button
                  type="button"
                  onClick={() => setIsFlipped(false)}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-cyan-200 bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-500/40 hover:border-cyan-400 transition-all cursor-pointer shadow-sm"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
                  <span>← Back to Prompt</span>
                </button>
              </div>

              {/* Reverse Content Grid: 4 Metric Quadrants */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-5">
                {/* 1. Token Accounting ($T_{orig}, T_{opt}, T_{saved}, \%_{reduction}$) */}
                <div className="p-4 rounded-xl bg-[#060a14]/90 border border-white/[0.07] space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-semibold text-cyan-300 flex items-center gap-1.5">
                      <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                      1. Token Accounting
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">~4 chars / token</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2.5 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[11px] text-slate-400">Original (T_orig)</div>
                      <div className="text-base font-bold text-slate-200 mt-0.5">
                        {tOrig.toLocaleString()}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[11px] text-slate-400">Optimized (T_opt)</div>
                      <div className="text-base font-bold text-cyan-300 mt-0.5">
                        {tOpt.toLocaleString()}
                      </div>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-cyan-950/40 border border-cyan-500/20 flex items-center justify-between text-xs font-mono">
                    <div>
                      <div className="text-[10px] uppercase text-cyan-300/80">Tokens Saved (T_saved)</div>
                      <div className="text-sm font-bold text-cyan-200">
                        {tSaved > 0 ? `+${tSaved.toLocaleString()}` : `${tSaved.toLocaleString()}`}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] uppercase text-cyan-300/80">Reduction (%_red)</div>
                      <div className="text-sm font-bold text-emerald-400">
                        {tokenReductionPercent.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* 2. Character Footprint & Variance */}
                <div className="p-4 rounded-xl bg-[#060a14]/90 border border-white/[0.07] space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-semibold text-violet-300 flex items-center gap-1.5">
                      <FileCode className="w-3.5 h-3.5 text-violet-400" />
                      2. Character Footprint
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">Byte variance</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2.5 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[11px] text-slate-400">Raw Input Chars</div>
                      <div className="text-base font-bold text-slate-200 mt-0.5">
                        {cOrig.toLocaleString()}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[11px] text-slate-400">Optimized Chars</div>
                      <div className="text-base font-bold text-violet-300 mt-0.5">
                        {cOpt.toLocaleString()}
                      </div>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-violet-950/40 border border-violet-500/20 flex items-center justify-between text-xs font-mono">
                    <div>
                      <div className="text-[10px] uppercase text-violet-300/80">Delta (Δ chars)</div>
                      <div className="text-sm font-bold text-violet-200">
                        {cDelta >= 0 ? `+${cDelta.toLocaleString()}` : `${cDelta.toLocaleString()}`}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] uppercase text-violet-300/80">Variance</div>
                      <div className="text-sm font-bold text-violet-300">
                        {charDeltaPercent.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* 3. Cost Projection & Economic Impact */}
                <div className="p-4 rounded-xl bg-[#060a14]/90 border border-white/[0.07] space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-semibold text-emerald-300 flex items-center gap-1.5">
                      <DollarSign className="w-3.5 h-3.5 text-emerald-400" />
                      3. Cost Projections ({modelOption.name})
                    </span>
                    <span className="text-[10px] font-mono text-emerald-400">
                      ${ratePerMillion.toFixed(2)} / 1M
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                    <div className="p-2 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[10px] text-slate-400">Original Cost</div>
                      <div className="text-xs font-bold text-slate-200 mt-0.5">
                        ${costOrig.toFixed(6)}
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[10px] text-slate-400">Optimized Cost</div>
                      <div className="text-xs font-bold text-cyan-300 mt-0.5">
                        ${costOpt.toFixed(6)}
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-emerald-950/50 border border-emerald-500/20">
                      <div className="text-[10px] text-emerald-300">Net Saved</div>
                      <div className="text-xs font-bold text-emerald-400 mt-0.5">
                        ${costSaved.toFixed(6)}
                      </div>
                    </div>
                  </div>

                  <p className="text-[10px] text-slate-500 leading-tight">
                    * Heuristic projection based on input token rates. Output costs and prompt caching not included.
                  </p>
                </div>

                {/* 4. Inference Latency & Router Details */}
                <div className="p-4 rounded-xl bg-[#060a14]/90 border border-white/[0.07] space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-semibold text-amber-300 flex items-center gap-1.5">
                      <Clock className="w-3.5 h-3.5 text-amber-400" />
                      4. Latency & Performance
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">
                      Strategy: {result.strategy || 'Compress'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2.5 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[11px] text-slate-400">Inference Latency</div>
                      <div className="text-base font-bold text-amber-300 mt-0.5">
                        {latencyMs} ms
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-[#0b1220] border border-white/5">
                      <div className="text-[11px] text-slate-400">Throughput</div>
                      <div className="text-base font-bold text-slate-200 mt-0.5">
                        {throughputTokensPerSec > 0 ? `~${throughputTokensPerSec} tok/s` : 'N/A'}
                      </div>
                    </div>
                  </div>

                  <div className="text-[11px] font-mono text-slate-400 flex items-center justify-between px-1">
                    <span>Task: <strong className="text-slate-200 capitalize">{result.taskCategory || 'General'}</strong></span>
                    <span>Provider: <strong className="text-cyan-300">{modelOption.provider}</strong></span>
                  </div>
                </div>
              </div>
            </div>

            {/* Reverse Footer */}
            <div className="pt-4 border-t border-white/[0.08] flex items-center justify-between">
              <span className="text-[11px] font-mono text-slate-500">
                Mathematical integrity verified &bull; Zero denominator safety active
              </span>

              <button
                type="button"
                onClick={() => setIsFlipped(false)}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-gradient-to-r from-cyan-600 to-violet-600 hover:from-cyan-500 hover:to-violet-500 transition-all cursor-pointer shadow-md"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Return to Optimized Prompt</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
