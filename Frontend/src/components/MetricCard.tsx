import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subvalue?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subvalue,
  icon: Icon,
  trend,
  highlight = false,
}) => {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border p-5 transition-all duration-200 hover:border-slate-700 ${
        highlight
          ? 'bg-gradient-to-br from-slate-900 to-emerald-950/20 border-emerald-500/30'
          : 'bg-slate-900/90 border-slate-800/80'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</span>
        {Icon && (
          <div
            className={`flex h-8 w-8 items-center justify-center rounded-lg border ${
              highlight
                ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400'
                : 'bg-slate-800/80 border-slate-700/60 text-slate-400'
            }`}
          >
            <Icon className="h-4 w-4" />
          </div>
        )}
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-2xl font-bold tracking-tight text-white">{value}</span>
        {subvalue && (
          <span
            className={`text-xs font-semibold ${
              trend === 'up'
                ? 'text-emerald-400'
                : trend === 'down'
                ? 'text-rose-400'
                : 'text-slate-400'
            }`}
          >
            {subvalue}
          </span>
        )}
      </div>
    </div>
  );
};

