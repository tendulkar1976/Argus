import React, { useState } from 'react';
import {
  DualExecutionResult,
  ComparisonWinner,
  OverallVerdict,
} from '@/types';
import {
  Sparkles,
  Zap,
  RotateCcw,
  Copy,
  Check,
  TrendingDown,
  DollarSign,
  Clock,
  Cpu,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Scale,
  Split,
  FileText,
  BarChart3,
  Award,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

export interface DualComparisonViewProps {
  result: DualExecutionResult;
  onResetWorkspace?: () => void;
}

export const DualComparisonView: React.FC<DualComparisonViewProps> = ({
  result,
  onResetWorkspace,
}) => {
  const [activeTab, setActiveTab] = useState<'responses' | 'scorecard' | 'transformations' | 'metrics'>('responses');
  const [copiedOriginal, setCopiedOriginal] = useState<boolean>(false);
  const [copiedOptimized, setCopiedOptimized] = useState<boolean>(false);
  const [showOriginalPrompt, setShowOriginalPrompt] = useState<boolean>(false);
  const [showOptimizedPrompt, setShowOptimizedPrompt] = useState<boolean>(false);

  const {
    target_model,
    target_provider,
    is_equivalent_conditions,
    equivalence_notes,
    optimization,
    transformations,
    original_execution,
    optimized_execution,
    comparison,
    performance,
  } = result;

  const handleCopyOriginal = async () => {
    try {
      await navigator.clipboard.writeText(original_execution.response_text);
      setCopiedOriginal(true);
      setTimeout(() => setCopiedOriginal(false), 2000);
    } catch {
      // Ignored
    }
  };

  const handleCopyOptimized = async () => {
    try {
      await navigator.clipboard.writeText(optimized_execution.response_text);
      setCopiedOptimized(true);
      setTimeout(() => setCopiedOptimized(false), 2000);
    } catch {
      // Ignored
    }
  };

  const getVerdictBadge = (verdict: OverallVerdict = 'similar') => {
    switch (verdict) {
      case 'optimized_better':
        return {
          label: 'Optimized Prompt Outperformed',
          color: 'bg-emerald-950/80 text-emerald-300 border-emerald-500/40',
          icon: <Award className="w-4 h-4 text-emerald-400" />,
        };
      case 'original_better':
        return {
          label: 'Original Prompt Outperformed',
          color: 'bg-blue-950/80 text-blue-300 border-blue-500/40',
          icon: <Award className="w-4 h-4 text-blue-400" />,
        };
      case 'similar':
        return {
          label: 'Outputs Equivalent in Quality',
          color: 'bg-slate-900/80 text-slate-300 border-slate-600/40',
          icon: <Scale className="w-4 h-4 text-slate-400" />,
        };
      case 'inconclusive':
      default:
        return {
          label: 'Evaluation Inconclusive',
          color: 'bg-amber-950/80 text-amber-300 border-amber-500/40',
          icon: <HelpCircle className="w-4 h-4 text-amber-400" />,
        };
    }
  };

  const getWinnerBadge = (winner: ComparisonWinner) => {
    switch (winner) {
      case 'optimized':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-500/30">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            Optimized
          </span>
        );
      case 'original':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-blue-950/80 text-blue-300 border border-blue-500/30">
            <CheckCircle2 className="w-3 h-3 text-blue-400" />
            Original
          </span>
        );
      case 'equivalent':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-mono text-slate-300 bg-slate-900 border border-white/10">
            <Scale className="w-3 h-3 text-slate-400" />
            Equivalent
          </span>
        );
      case 'inconclusive':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-mono text-amber-300 bg-amber-950/60 border border-amber-500/30">
            <HelpCircle className="w-3 h-3 text-amber-400" />
            Inconclusive
          </span>
        );
    }
  };

  const verdict = comparison?.overall_verdict || 'similar';
  const verdictInfo = getVerdictBadge(verdict);

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 animate-float-subtle">
      {/* 1. Master Verdict & Summary Card */}
      <div className="rounded-2xl backdrop-blur-2xl bg-[#090f1e]/90 border border-white/[0.12] p-5 sm:p-7 shadow-[0_20px_50px_rgba(0,0,0,0.8)] space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-white/[0.08]">
          <div className="flex flex-wrap items-center gap-2.5">
            <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${verdictInfo.color}`}>
              {verdictInfo.icon}
              <span>{verdictInfo.label}</span>
            </div>

            <span className="text-xs font-mono text-slate-400 bg-[#0f172a] px-2.5 py-1 rounded-lg border border-white/5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Model: <strong className="text-white">{target_model}</strong> ({target_provider})</span>
            </span>

            {is_equivalent_conditions ? (
              <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/50 px-2.5 py-1 rounded-lg border border-emerald-500/20 flex items-center gap-1">
                <Check className="w-3 h-3" /> Identical A/B Conditions
              </span>
            ) : (
              <span className="text-[11px] font-mono text-amber-400 bg-amber-950/50 px-2.5 py-1 rounded-lg border border-amber-500/20 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> Differing Conditions
              </span>
            )}
          </div>

          {onResetWorkspace && (
            <button
              type="button"
              onClick={onResetWorkspace}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-cyan-300 hover:text-white bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-500/30 transition-all cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>New Prompt</span>
            </button>
          )}
        </div>

        {/* Verdict Rationale */}
        {comparison?.summary_rationale && (
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-sans">
            {comparison.summary_rationale}
          </p>
        )}

        {/* Quick KPI Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
          <div className="p-3 rounded-xl bg-[#060a14] border border-white/5">
            <span className="text-[10px] font-mono uppercase text-slate-400">Prompt Tokens</span>
            <div className="text-sm sm:text-base font-bold text-cyan-300 mt-0.5 flex items-center gap-1">
              <span>{performance.prompt_tokens_optimized}</span>
              <span className="text-xs text-slate-500 font-normal">from {performance.prompt_tokens_original}</span>
              <span className="text-xs text-emerald-400 font-mono font-semibold ml-1">
                (-{performance.prompt_token_reduction_percent}%)
              </span>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-[#060a14] border border-white/5">
            <span className="text-[10px] font-mono uppercase text-slate-400">Output Response</span>
            <div className="text-sm sm:text-base font-bold text-slate-200 mt-0.5">
              <span>{performance.response_tokens_optimized ?? 'N/A'} tok</span>
              <span className="text-xs text-slate-500 font-normal ml-1">
                (Orig: {performance.response_tokens_original ?? 'N/A'})
              </span>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-[#060a14] border border-white/5">
            <span className="text-[10px] font-mono uppercase text-slate-400">Cost Savings</span>
            <div className="text-sm sm:text-base font-bold text-emerald-400 mt-0.5">
              {performance.cost_saved_usd !== null && performance.cost_saved_usd !== undefined
                ? `$${performance.cost_saved_usd.toFixed(6)}`
                : 'Free / N/A'}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-[#060a14] border border-white/5">
            <span className="text-[10px] font-mono uppercase text-slate-400">Latency Variance</span>
            <div className="text-sm sm:text-base font-bold text-amber-300 mt-0.5">
              <span>{performance.latency_optimized_ms}ms</span>
              <span className="text-xs text-slate-500 font-normal ml-1">
                ({performance.latency_difference_ms >= 0 ? `+${performance.latency_difference_ms}` : performance.latency_difference_ms}ms)
              </span>
            </div>
          </div>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex flex-wrap items-center gap-1 pt-3 border-t border-white/[0.08]">
          {[
            { id: 'responses', label: 'A/B Model Responses', icon: <Split className="w-3.5 h-3.5" /> },
            { id: 'scorecard', label: 'Quality Scorecard (9 Dimensions)', icon: <Award className="w-3.5 h-3.5" /> },
            { id: 'transformations', label: 'What Argus Changed', icon: <FileText className="w-3.5 h-3.5" /> },
            { id: 'metrics', label: 'Performance & Resource Audit', icon: <BarChart3 className="w-3.5 h-3.5" /> },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                activeTab === tab.id
                  ? 'bg-cyan-500/20 text-cyan-200 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ===================================================================
          TAB 1: Side-by-Side Model Responses (Sections A & B)
          =================================================================== */}
      {activeTab === 'responses' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          {/* Execution A: Original Prompt & Output */}
          <div className="rounded-2xl backdrop-blur-xl bg-[#080d1a]/90 border border-white/[0.08] p-5 sm:p-6 space-y-4 shadow-lg">
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-slate-400" />
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300">
                  Execution A &bull; Original Prompt
                </h3>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowOriginalPrompt(!showOriginalPrompt)}
                  className="text-[11px] font-mono text-slate-400 hover:text-slate-200 flex items-center gap-1"
                >
                  <span>Prompt</span>
                  {showOriginalPrompt ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>

                <button
                  type="button"
                  onClick={handleCopyOriginal}
                  className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-white/5 transition-all cursor-pointer"
                  title="Copy Original Response"
                >
                  {copiedOriginal ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>

            {/* Collapsible Prompt Preview */}
            {showOriginalPrompt && (
              <div className="p-3 rounded-xl bg-[#050811] border border-white/5 text-xs font-mono text-slate-300 whitespace-pre-wrap max-h-[160px] overflow-y-auto">
                {original_execution.prompt}
              </div>
            )}

            {/* Response Output Body */}
            <div className="p-4 rounded-xl bg-[#050811]/90 border border-white/5 min-h-[300px] max-h-[460px] overflow-y-auto select-text font-mono text-xs leading-relaxed text-slate-100">
              {original_execution.metrics.status === 'error' ? (
                <div className="p-3 rounded-lg bg-red-950/40 border border-red-500/30 text-red-200 text-xs">
                  <strong>Execution Failed:</strong> {original_execution.metrics.error_message || 'Provider connection error'}
                </div>
              ) : (
                <pre className="whitespace-pre-wrap break-words">{original_execution.response_text}</pre>
              )}
            </div>

            {/* Metrics Footer */}
            <div className="pt-2 flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400 border-t border-white/[0.06]">
              <span>In: {original_execution.metrics.input_tokens ?? 'N/A'} tok &bull; Out: {original_execution.metrics.output_tokens ?? 'N/A'} tok</span>
              <span>{original_execution.metrics.latency_ms}ms &bull; ${original_execution.metrics.cost_usd?.toFixed(6) ?? '0.00'}</span>
            </div>
          </div>

          {/* Execution B: Optimized Prompt & Output */}
          <div className="rounded-2xl backdrop-blur-xl bg-[#090f20]/95 border border-cyan-500/25 p-5 sm:p-6 space-y-4 shadow-lg shadow-cyan-950/20">
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-300">
                  Execution B &bull; Argus Optimized Prompt
                </h3>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowOptimizedPrompt(!showOptimizedPrompt)}
                  className="text-[11px] font-mono text-cyan-400 hover:text-cyan-200 flex items-center gap-1"
                >
                  <span>Prompt</span>
                  {showOptimizedPrompt ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>

                <button
                  type="button"
                  onClick={handleCopyOptimized}
                  className="p-1.5 rounded-md text-cyan-400 hover:text-white hover:bg-cyan-950/50 transition-all cursor-pointer"
                  title="Copy Optimized Response"
                >
                  {copiedOptimized ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>

            {/* Collapsible Prompt Preview */}
            {showOptimizedPrompt && (
              <div className="p-3 rounded-xl bg-[#050811] border border-cyan-500/20 text-xs font-mono text-cyan-200 whitespace-pre-wrap max-h-[160px] overflow-y-auto">
                {optimized_execution.prompt}
              </div>
            )}

            {/* Response Output Body */}
            <div className="p-4 rounded-xl bg-[#050811]/90 border border-cyan-500/20 min-h-[300px] max-h-[460px] overflow-y-auto select-text font-mono text-xs leading-relaxed text-slate-100">
              {optimized_execution.metrics.status === 'error' ? (
                <div className="p-3 rounded-lg bg-red-950/40 border border-red-500/30 text-red-200 text-xs">
                  <strong>Execution Failed:</strong> {optimized_execution.metrics.error_message || 'Provider connection error'}
                </div>
              ) : (
                <pre className="whitespace-pre-wrap break-words">{optimized_execution.response_text}</pre>
              )}
            </div>

            {/* Metrics Footer */}
            <div className="pt-2 flex flex-wrap items-center justify-between text-[11px] font-mono text-cyan-300 border-t border-white/[0.06]">
              <span>In: {optimized_execution.metrics.input_tokens ?? 'N/A'} tok &bull; Out: {optimized_execution.metrics.output_tokens ?? 'N/A'} tok</span>
              <span>{optimized_execution.metrics.latency_ms}ms &bull; ${optimized_execution.metrics.cost_usd?.toFixed(6) ?? '0.00'}</span>
            </div>
          </div>
        </div>
      )}

      {/* ===================================================================
          TAB 2: Quality Scorecard (9 Dimensions - Section D)
          =================================================================== */}
      {activeTab === 'scorecard' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Strengths & Weaknesses Comparison */}
            <div className="p-5 rounded-2xl bg-[#080d1a] border border-white/5 space-y-3">
              <h4 className="text-xs font-mono font-bold text-slate-300 uppercase">Original Response Profile</h4>
              {comparison?.original_strengths && comparison.original_strengths.length > 0 && (
                <div>
                  <span className="text-[10px] font-mono text-emerald-400 uppercase">Strengths</span>
                  <ul className="list-disc list-inside text-xs text-slate-300 mt-1 space-y-1">
                    {comparison.original_strengths.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}
              {comparison?.original_weaknesses && comparison.original_weaknesses.length > 0 && (
                <div>
                  <span className="text-[10px] font-mono text-amber-400 uppercase">Observations / Weaknesses</span>
                  <ul className="list-disc list-inside text-xs text-slate-400 mt-1 space-y-1">
                    {comparison.original_weaknesses.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="p-5 rounded-2xl bg-[#080d1a] border border-cyan-500/20 space-y-3">
              <h4 className="text-xs font-mono font-bold text-cyan-300 uppercase">Optimized Response Profile</h4>
              {comparison?.optimized_strengths && comparison.optimized_strengths.length > 0 && (
                <div>
                  <span className="text-[10px] font-mono text-emerald-400 uppercase">Strengths</span>
                  <ul className="list-disc list-inside text-xs text-slate-200 mt-1 space-y-1">
                    {comparison.optimized_strengths.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}
              {comparison?.optimized_weaknesses && comparison.optimized_weaknesses.length > 0 && (
                <div>
                  <span className="text-[10px] font-mono text-amber-400 uppercase">Observations / Trade-offs</span>
                  <ul className="list-disc list-inside text-xs text-slate-400 mt-1 space-y-1">
                    {comparison.optimized_weaknesses.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* 9 Dimensions Breakdown */}
          <div className="space-y-3">
            <h4 className="text-xs font-mono font-bold text-slate-400 uppercase px-1">
              Dimensional Evaluation Breakdown
            </h4>

            {comparison?.dimensions.map((dim, idx) => (
              <div
                key={idx}
                className="p-4 sm:p-5 rounded-xl bg-[#080d1a] border border-white/5 space-y-3 hover:border-white/10 transition-colors"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-cyan-950 text-cyan-400 text-xs font-mono flex items-center justify-center border border-cyan-500/30">
                      {idx + 1}
                    </span>
                    <h5 className="text-xs font-bold text-slate-200">{dim.dimension_label}</h5>
                  </div>
                  <div>{getWinnerBadge(dim.winner)}</div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="p-2.5 rounded-lg bg-[#050811] border border-white/5 text-slate-300">
                    <span className="text-[10px] font-mono text-slate-500 block uppercase">Original Assessment</span>
                    {dim.original_assessment}
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#050811] border border-white/5 text-cyan-200">
                    <span className="text-[10px] font-mono text-cyan-500 block uppercase">Optimized Assessment</span>
                    {dim.optimized_assessment}
                  </div>
                </div>

                <p className="text-xs text-slate-400 font-sans leading-relaxed border-t border-white/[0.04] pt-2">
                  <strong className="text-slate-300">Rationale:</strong> {dim.explanation}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ===================================================================
          TAB 3: What Argus Changed (Section C)
          =================================================================== */}
      {activeTab === 'transformations' && (
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-[#080d1a] border border-white/5 text-xs text-slate-300 flex items-center justify-between">
            <span>Optimization Strategy Applied: <strong className="text-cyan-300 uppercase font-mono">{optimization.strategy}</strong></span>
            <span>Task Category: <strong className="text-violet-300 capitalize font-mono">{optimization.taskCategory}</strong></span>
          </div>

          <div className="space-y-3">
            {transformations.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-xs">
                No transformations recorded. Prompt was preserved in full.
              </div>
            ) : (
              transformations.map((t, idx) => (
                <div
                  key={idx}
                  className="p-5 rounded-xl bg-[#080d1a] border border-white/5 space-y-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-mono px-2.5 py-0.5 rounded bg-cyan-950/70 text-cyan-300 border border-cyan-500/20 uppercase">
                      {t.category.replace(/_/g, ' ')}
                    </span>
                    {t.intent_preserved && (
                      <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                        <ShieldCheck className="w-3 h-3" /> Intent Preserved
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
                    <div className="p-3 rounded-lg bg-[#050811] border border-red-500/10 text-red-200/90">
                      <span className="text-[10px] text-slate-500 uppercase block font-sans">Original Segment</span>
                      {t.original_segment}
                    </div>

                    <div className="p-3 rounded-lg bg-[#050811] border border-emerald-500/10 text-emerald-200/90">
                      <span className="text-[10px] text-slate-500 uppercase block font-sans">Optimized Segment</span>
                      {t.optimized_segment}
                    </div>
                  </div>

                  <div className="text-xs text-slate-300 space-y-1 font-sans border-t border-white/[0.04] pt-2">
                    <p><strong className="text-slate-400 font-mono text-[11px]">Reason:</strong> {t.reason}</p>
                    <p><strong className="text-slate-400 font-mono text-[11px]">Expected Benefit:</strong> {t.expected_benefit}</p>
                    {t.potential_tradeoff_or_risk && (
                      <p className="text-amber-300/90"><strong className="text-amber-400 font-mono text-[11px]">Trade-off / Risk:</strong> {t.potential_tradeoff_or_risk}</p>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* ===================================================================
          TAB 4: Performance & Resource Audit (Section E)
          =================================================================== */}
      {activeTab === 'metrics' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 1. Token Accounting Comparison */}
            <div className="p-5 rounded-2xl bg-[#080d1a] border border-white/5 space-y-3">
              <span className="text-xs font-mono font-bold text-cyan-300 uppercase flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                1. Token Footprint Comparison
              </span>

              <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded-lg bg-[#050811] border border-white/5">
                  <div className="text-[10px] text-slate-400">Prompt A (Orig)</div>
                  <div className="text-sm font-bold text-slate-200 mt-0.5">{performance.prompt_tokens_original}</div>
                </div>
                <div className="p-2.5 rounded-lg bg-[#050811] border border-white/5">
                  <div className="text-[10px] text-cyan-400">Prompt B (Opt)</div>
                  <div className="text-sm font-bold text-cyan-300 mt-0.5">{performance.prompt_tokens_optimized}</div>
                </div>
                <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/20">
                  <div className="text-[10px] text-emerald-300">Pruned</div>
                  <div className="text-sm font-bold text-emerald-400 mt-0.5">-{performance.prompt_token_reduction_percent}%</div>
                </div>
              </div>

              <div className="text-[11px] font-mono text-slate-400 pt-1">
                Total Output Generated: <strong className="text-white">{performance.response_tokens_optimized ?? 'N/A'} tokens</strong> vs Original <strong className="text-slate-300">{performance.response_tokens_original ?? 'N/A'} tokens</strong>
              </div>
            </div>

            {/* 2. Economic Cost Comparison */}
            <div className="p-5 rounded-2xl bg-[#080d1a] border border-white/5 space-y-3">
              <span className="text-xs font-mono font-bold text-emerald-300 uppercase flex items-center gap-1.5">
                <DollarSign className="w-3.5 h-3.5 text-emerald-400" />
                2. Financial & Cost Projections
              </span>

              <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded-lg bg-[#050811] border border-white/5">
                  <div className="text-[10px] text-slate-400">Cost Execution A</div>
                  <div className="text-xs font-bold text-slate-200 mt-0.5">
                    ${performance.cost_original_usd?.toFixed(6) ?? '0.000000'}
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-[#050811] border border-white/5">
                  <div className="text-[10px] text-cyan-400">Cost Execution B</div>
                  <div className="text-xs font-bold text-cyan-300 mt-0.5">
                    ${performance.cost_optimized_usd?.toFixed(6) ?? '0.000000'}
                  </div>
                </div>
                <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/20">
                  <div className="text-[10px] text-emerald-300">Net Savings</div>
                  <div className="text-xs font-bold text-emerald-400 mt-0.5">
                    ${performance.cost_saved_usd?.toFixed(6) ?? '0.000000'}
                  </div>
                </div>
              </div>

              <p className="text-[10px] text-slate-500 leading-tight">
                * Cost derived using verified provider input/output pricing per 1M tokens.
              </p>
            </div>
          </div>

          {/* Evaluator Overhead Disclosure */}
          {comparison?.evaluator_metrics && (
            <div className="p-3.5 rounded-xl bg-[#080d1a] border border-white/5 text-[11px] font-mono text-slate-400 flex flex-wrap items-center justify-between gap-2">
              <span>Evaluator LLM Overhead: {comparison.evaluator_metrics.latency_ms}ms &bull; {comparison.evaluator_metrics.total_tokens ?? 'N/A'} total tokens</span>
              <span className="text-slate-500">Evaluator cost tracked separately (${comparison.evaluator_metrics.cost_usd?.toFixed(6) ?? '0.00'})</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
