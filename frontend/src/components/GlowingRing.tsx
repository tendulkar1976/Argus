import React from 'react';

export interface GlowingRingProps {
  status?: 'idle' | 'optimizing' | 'success' | 'error';
  minimized?: boolean;
}

export const GlowingRing: React.FC<GlowingRingProps> = ({
  status = 'idle',
  minimized = false,
}) => {
  const isOptimizing = status === 'optimizing';
  const isSuccess = status === 'success' || minimized;

  return (
    <div
      aria-hidden="true"
      className={`pointer-events-none fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-0 transition-all duration-1000 ease-out select-none ${
        isSuccess
          ? 'scale-75 -translate-y-[62%] opacity-35'
          : isOptimizing
          ? 'scale-105 opacity-95'
          : 'scale-100 opacity-80'
      }`}
    >
      <div className="relative w-[650px] h-[650px] sm:w-[850px] sm:h-[850px] md:w-[1000px] md:h-[1000px] flex items-center justify-center">
        {/* Deep ambient radial backdrop glow */}
        <div
          className={`absolute inset-0 rounded-full blur-[120px] transition-all duration-1000 ${
            isOptimizing
              ? 'bg-gradient-to-tr from-cyan-600/30 via-violet-600/40 to-blue-500/30'
              : 'bg-gradient-to-tr from-cyan-900/15 via-violet-900/25 to-blue-900/20'
          }`}
        />

        {/* Slow rotating SVG Arc/Ring */}
        <svg
          className="absolute inset-0 w-full h-full animate-ring-spin-slow"
          viewBox="0 0 1000 1000"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient id="ringGradientPrimary" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.85" />
              <stop offset="35%" stopColor="#3b82f6" stopOpacity="0.75" />
              <stop offset="70%" stopColor="#8b5cf6" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#d946ef" stopOpacity="0.4" />
            </linearGradient>

            <linearGradient id="ringGradientSecondary" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.6" />
              <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.3" />
            </linearGradient>

            <filter id="bloomGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="18" result="blur1" />
              <feGaussianBlur stdDeviation="35" result="blur2" />
              <feMerge>
                <feMergeNode in="blur2" />
                <feMergeNode in="blur1" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <filter id="celestialMist" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="60" />
            </filter>
          </defs>

          {/* Deep celestial mist layer */}
          <circle
            cx="500"
            cy="500"
            r="380"
            stroke="url(#ringGradientPrimary)"
            strokeWidth="32"
            strokeDasharray="200 40 180 80 400 120"
            strokeLinecap="round"
            filter="url(#celestialMist)"
            opacity={isOptimizing ? 0.6 : 0.35}
          />

          {/* Primary Luminous Outer Arc */}
          <circle
            cx="500"
            cy="500"
            r="420"
            stroke="url(#ringGradientPrimary)"
            strokeWidth="2.5"
            strokeDasharray="300 120 450 90 200 60"
            strokeLinecap="round"
            filter="url(#bloomGlow)"
            className="animate-ring-pulse"
          />

          {/* Concentric Secondary Ring */}
          <circle
            cx="500"
            cy="500"
            r="380"
            stroke="url(#ringGradientSecondary)"
            strokeWidth="1.5"
            strokeDasharray="140 160 380 100 240 140"
            strokeLinecap="round"
            opacity="0.7"
          />

          {/* Precision Orbital Markers */}
          <circle cx="500" cy="80" r="4" fill="#22d3ee" filter="url(#bloomGlow)" />
          <circle cx="920" cy="500" r="3" fill="#8b5cf6" filter="url(#bloomGlow)" />
          <circle cx="500" cy="920" r="4.5" fill="#3b82f6" filter="url(#bloomGlow)" />
          <circle cx="80" cy="500" r="3" fill="#06b6d4" filter="url(#bloomGlow)" />

          {/* Subtle Inner Accent Ring */}
          <circle
            cx="500"
            cy="500"
            r="340"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth="1"
            strokeDasharray="12 24"
          />
        </svg>
      </div>
    </div>
  );
};
