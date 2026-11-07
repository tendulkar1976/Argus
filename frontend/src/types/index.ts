/**
 * Argus Frontend Types
 * Domain and API contracts
 */

export type ServiceStatus = 'healthy' | 'degraded' | 'unhealthy' | 'checking';

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error';
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
  message?: string;
}

export interface ApiError {
  detail: string;
  code?: string;
  timestamp?: string;
}

export interface SystemMetrics {
  totalOptimizations: number;
  averageTokenReductionPercent: number;
  estimatedCostSavingsUsd: number;
  activeModel: string;
}

/* =========================================================================
   Phase 2, 3 & 3.1: Prompt Optimization Workspace Contracts
   ========================================================================= */

export type OptimizationMode = 'balanced' | 'concise' | 'structure';

export type OptimizationStrategy = 'compress' | 'clarify' | 'structure' | 'enrich' | 'preserve';

export type TaskCategory =
  | 'coding'
  | 'data_analysis'
  | 'writing'
  | 'summarization'
  | 'research'
  | 'extraction'
  | 'planning'
  | 'general';

export type ValidationStatus = 'passed' | 'warning' | 'fallback';

export type OptimizationOutcome = 'validated_rewrite' | 'unchanged_original' | 'fallback';

export type LengthResult = 'compressed' | 'expanded' | 'unchanged';

export type TargetModel =
  | 'gemini-2.0-flash'
  | 'gemini-1.5-pro'
  | 'deepseek-chat'
  | 'deepseek-reasoner'
  | 'qwen2.5:1.5b'
  | string;

export interface ModelOption {
  id: TargetModel;
  name: string;
  provider: string;
  contextWindow: string;
  inputPricePerMillionUsd: number;
  description: string;
  isIllustrative?: boolean;
}

export const CREDENTIALED_MODELS: ModelOption[] = [
  {
    id: 'gemini-2.0-flash',
    name: 'Gemini 2.0 Flash',
    provider: 'Google Gemini',
    contextWindow: '1.05M',
    inputPricePerMillionUsd: 0.10,
    description: 'High-speed multimodal frontier model with ultra-low latency',
  },
  {
    id: 'gemini-1.5-pro',
    name: 'Gemini 1.5 Pro',
    provider: 'Google Gemini',
    contextWindow: '2.09M',
    inputPricePerMillionUsd: 1.25,
    description: 'Massive 2M context window with advanced multi-step reasoning',
  },
  {
    id: 'deepseek-chat',
    name: 'DeepSeek Chat (V3)',
    provider: 'DeepSeek',
    contextWindow: '64k',
    inputPricePerMillionUsd: 0.14,
    description: 'High throughput, highly cost-effective general LLM',
  },
  {
    id: 'deepseek-reasoner',
    name: 'DeepSeek Reasoner (R1)',
    provider: 'DeepSeek',
    contextWindow: '64k',
    inputPricePerMillionUsd: 0.55,
    description: 'Deep chain-of-thought mathematical and analytical reasoning',
  },
  {
    id: 'qwen2.5:1.5b',
    name: 'Qwen 2.5 (1.5B Local)',
    provider: 'Ollama (Local)',
    contextWindow: '32k',
    inputPricePerMillionUsd: 0.0,
    description: 'Local edge inference running entirely offline at zero cost',
  },
];

export interface OptimizationSession {
  id: string;
  timestamp: string;
  title: string;
  rawPrompt: string;
  mode: OptimizationMode;
  targetModel: TargetModel;
  result: OptimizationResult;
  comparisonResult?: DualExecutionResult | null;
}

/* =========================================================================
   Phase 11.1: Dual Execution & Explainable Comparison Contracts
   ========================================================================= */

export type ComparisonWinner = 'original' | 'optimized' | 'equivalent' | 'inconclusive';

export type OverallVerdict = 'optimized_better' | 'original_better' | 'similar' | 'inconclusive';

export type TransformationCategory =
  | 'redundancy_removal'
  | 'instruction_clarification'
  | 'constraint_preservation'
  | 'context_restructuring'
  | 'ambiguity_reduction'
  | 'output_format_specification'
  | 'unnecessary_wording_removal'
  | 'general_optimization';

export interface ExecutionMetricDetails {
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  cached_tokens?: number | null;
  latency_ms: number;
  cost_usd?: number | null;
  cost_status: string;
  provider: string;
  model: string;
  status: 'success' | 'error' | 'timeout';
  error_message?: string | null;
}

export interface SingleExecutionResult {
  prompt: string;
  response_text: string;
  metrics: ExecutionMetricDetails;
  fallback_used?: boolean;
  fallback_from?: string | null;
}

export interface TransformationRecord {
  original_segment: string;
  optimized_segment: string;
  category: TransformationCategory;
  reason: string;
  expected_benefit: string;
  intent_preserved: boolean;
  potential_tradeoff_or_risk?: string | null;
}

export interface DimensionEvaluation {
  dimension: string;
  dimension_label: string;
  original_assessment: string;
  optimized_assessment: string;
  difference: string;
  winner: ComparisonWinner;
  explanation: string;
}

export interface ResponseComparison {
  dimensions: DimensionEvaluation[];
  overall_verdict: OverallVerdict;
  summary_rationale: string;
  original_strengths: string[];
  original_weaknesses: string[];
  optimized_strengths: string[];
  optimized_weaknesses: string[];
  evaluator_metrics?: ExecutionMetricDetails | null;
}

export interface ComparisonPerformanceMetrics {
  prompt_tokens_original: number;
  prompt_tokens_optimized: number;
  prompt_tokens_saved: number;
  prompt_token_reduction_percent: number;

  response_tokens_original?: number | null;
  response_tokens_optimized?: number | null;
  response_token_difference?: number | null;

  total_tokens_original?: number | null;
  total_tokens_optimized?: number | null;
  total_token_difference?: number | null;

  cost_original_usd?: number | null;
  cost_optimized_usd?: number | null;
  cost_saved_usd?: number | null;
  cost_savings_percent?: number | null;

  latency_original_ms: number;
  latency_optimized_ms: number;
  latency_difference_ms: number;
}

export interface DualExecutionRequestPayload {
  prompt: string;
  mode?: OptimizationMode;
  target_model?: TargetModel;
  system_prompt?: string | null;
  temperature?: number;
  max_tokens?: number | null;
  run_comparison?: boolean;
}

export interface DualExecutionResult {
  request_id: string;
  timestamp: string;
  target_model: string;
  target_provider: string;
  is_equivalent_conditions: boolean;
  equivalence_notes?: string | null;
  optimization: OptimizationResult;
  transformations: TransformationRecord[];
  original_execution: SingleExecutionResult;
  optimized_execution: SingleExecutionResult;
  comparison?: ResponseComparison | null;
  performance: ComparisonPerformanceMetrics;
}

export type OptimizationStatus = 'idle' | 'optimizing' | 'success' | 'error';

export interface TokenEstimation {
  charCount: number;
  estimatedTokens: number;
}

export interface OptimizeRequestPayload {
  prompt: string;
  mode: OptimizationMode;
  target_model: TargetModel;
}

export interface OptimizationResult {
  originalPrompt: string;
  optimizedPrompt: string;
  mode: OptimizationMode;
  targetModel: TargetModel;
  
  // Phase 3.1 & 3.2 Task-Aware Metadata
  taskCategory: string;
  strategy: OptimizationStrategy;
  appliedStrategy?: string;
  outcome?: OptimizationOutcome;
  lengthResult?: LengthResult;
  validationStatus: ValidationStatus;
  isPreserved: boolean;
  primaryAction?: string;
  primaryDeliverable?: string;
  explicitConstraints: string[];
  requiredOperations?: string[];

  // Token & Length Metrics (Heuristic)
  originalMetrics: TokenEstimation;
  optimizedMetrics: TokenEstimation;
  tokenReductionPercent: number;
  tokenChangePercent: number;
  tokenSavings: number;
  charDelta: number;
  preservedConstraints: string[];
  executionTimeMs: number;
  model: string;
  timestamp: string;
  isSimulated?: boolean;
}

export interface ExamplePrompt {
  id: string;
  title: string;
  category: string;
  description: string;
  content: string;
}

export interface OllamaStatus {
  connected: boolean;
  base_url: string;
  model: string;
  model_installed: boolean;
  available_models: string[];
  error?: string | null;
}
