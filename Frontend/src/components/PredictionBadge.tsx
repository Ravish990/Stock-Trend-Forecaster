import React from 'react';
import { ArrowUpRight, ArrowDownRight, MinusCircle } from 'lucide-react';

interface PredictionBadgeProps {
  direction?: 'UP' | 'DOWN' | string | null;
  confidence?: string | null;
  size?: 'sm' | 'md' | 'lg';
  showConfidence?: boolean;
}

export const PredictionBadge: React.FC<PredictionBadgeProps> = ({
  direction,
  confidence,
  size = 'md',
  showConfidence = false,
}) => {
  if (!direction) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-slate-800/80 px-2.5 py-0.5 text-xs font-medium text-slate-400 border border-slate-700">
        <MinusCircle className="h-3 w-3" />
        Unavailable
      </span>
    );
  }

  const isUp = direction.toUpperCase() === 'UP';
  const isHighConfidence = confidence?.toLowerCase().includes('high');

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs',
    lg: 'px-3 py-1.5 text-sm',
  }[size];

  return (
    <div className="inline-flex items-center gap-1.5">
      <span
        className={`inline-flex items-center gap-1 font-semibold rounded-full border transition-colors ${sizeClasses} ${
          isUp
            ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
            : 'bg-rose-500/15 border-rose-500/30 text-rose-400'
        }`}
      >
        {isUp ? (
          <ArrowUpRight className="h-3.5 w-3.5 stroke-[2.5]" />
        ) : (
          <ArrowDownRight className="h-3.5 w-3.5 stroke-[2.5]" />
        )}
        <span>{direction}</span>
      </span>

      {showConfidence && confidence && (
        <span
          className={`rounded-md px-2 py-0.5 text-[11px] font-medium border ${
            isHighConfidence
              ? isUp
                ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300'
                : 'bg-rose-950/60 border-rose-500/40 text-rose-300'
              : 'bg-slate-800/60 border-slate-700/60 text-slate-300'
          }`}
        >
          {isHighConfidence ? 'High Conf' : 'Low Conf'}
        </span>
      )}
    </div>
  );
};

