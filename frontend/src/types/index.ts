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
