import React from 'react';
import { PredictionSnapshot } from '../types/prediction';
import { PredictionBadge } from './PredictionBadge';
import {
  Calendar,
  Clock,
  Cpu,
  Layers,
  Percent,
  Sparkles,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';

interface ForecastCardProps {
  prediction: PredictionSnapshot | null;
  loading?: boolean;
}

export const ForecastCard: React.FC<ForecastCardProps> = ({
  prediction,
  loading = false,
}) => {
  if (loading) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-6 animate-pulse">
        <div className="h-6 w-48 bg-slate-800 rounded mb-4" />
        <div className="grid grid-cols-2 gap-4">
          <div className="h-20 bg-slate-800/60 rounded-xl" />
          <div className="h-20 bg-slate-800/60 rounded-xl" />
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-800 bg-slate-900/40 p-8 text-center text-slate-400">
        <AlertCircle className="mx-auto h-8 w-8 text-amber-400 mb-2" />
        <div className="text-base font-semibold text-slate-300">No Model Forecast Available</div>
        <div className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
          This ticker does not have sufficient chronological observations to meet the model's 10-step sequence requirement.
        </div>
      </div>
    );
  }

  const {
    ticker,
    company_name,
    sector,
    industry,
    latest_close,
    latest_data_date,
    predicted_price_5d,
    predicted_return_pct,
    probability_up,
    probability_down,
    predicted_direction,
    confidence_label,
    prediction_date,
    forecast_horizon,
    model_name,
  } = prediction;

  const isUp = predicted_direction === 'UP';
  const probUp = probability_up ?? 50.0;
  const probDown = probability_down ?? 50.0;
  const retPct = predicted_return_pct ?? 0.0;

  return (
    <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 via-slate-900/95 to-slate-950 p-6 shadow-xl">
      {/* Background glow accent */}
      <div
        className={`absolute -top-24 -right-24 h-48 w-48 rounded-full blur-3xl opacity-20 pointer-events-none ${
          isUp ? 'bg-emerald-500' : 'bg-rose-500'
        }`}
      />

      {/* Header Info */}
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <span className="text-2xl font-black tracking-tight text-white">{ticker}</span>
            <PredictionBadge
              direction={predicted_direction}
              confidence={confidence_label}
              size="md"
              showConfidence
            />
          </div>
          <div className="text-sm font-medium text-slate-300 mt-0.5">
            {company_name || 'Company Name Unavailable'}
          </div>
          {(sector || industry) && (
            <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
              {sector && <span>{sector}</span>}
              {sector && industry && <span>•</span>}
              {industry && <span>{industry}</span>}
            </div>
          )}
        </div>

        <div className="flex flex-col items-end text-right">
          <div className="text-xs uppercase tracking-wider text-slate-400">Latest Actual Close</div>
          <div className="text-2xl font-bold text-white">
            ${latest_close != null ? latest_close.toFixed(2) : '—'}
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-1 mt-0.5">
            <Calendar className="h-3 w-3" />
            <span>As of {latest_data_date || 'N/A'}</span>
          </div>
        </div>
      </div>

      {/* Main Forecast Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 my-6">
        {/* Forecasted Price */}
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
          <div className="text-xs text-slate-400 font-medium">Predicted Close (5 Trading Days)</div>
          <div className="mt-1 text-2xl font-black text-sky-300">
            ${predicted_price_5d != null ? predicted_price_5d.toFixed(2) : '—'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Target observation {forecast_horizon} trading sessions out
          </div>
        </div>

        {/* Expected Return */}
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
          <div className="text-xs text-slate-400 font-medium">Expected 5-Day Return</div>
          <div
            className={`mt-1 text-2xl font-black ${
              retPct >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {retPct >= 0 ? `+${retPct.toFixed(2)}%` : `${retPct.toFixed(2)}%`}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Exponential conversion from log-return head
          </div>
        </div>

        {/* Confidence Tier */}
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 sm:col-span-2 lg:col-span-1">
          <div className="text-xs text-slate-400 font-medium">Model Classification</div>
          <div className="mt-1 text-lg font-bold text-slate-100">
            {confidence_label || 'Unclassified'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Directional threshold: {isUp ? '> 50.0%' : '≤ 50.0%'}
          </div>
        </div>
      </div>

      {/* Probability Gauge Bar */}
      <div className="rounded-xl border border-slate-800/80 bg-slate-950/40 p-4">
        <div className="flex items-center justify-between text-xs font-semibold mb-2">
          <span className="text-emerald-400 flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Prob UP: {probUp.toFixed(1)}%
          </span>
          <span className="text-rose-400 flex items-center gap-1.5">
            Prob DOWN: {probDown.toFixed(1)}%
            <span className="h-2 w-2 rounded-full bg-rose-400" />
          </span>
        </div>

        {/* Dual Progress Bar */}
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800 flex">
          <div
            className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all duration-500"
            style={{ width: `${probUp}%` }}
          />
          <div
            className="h-full bg-gradient-to-r from-rose-400 to-rose-600 transition-all duration-500"
            style={{ width: `${probDown}%` }}
          />
        </div>

        <div className="mt-2 text-center text-[11px] text-slate-500">
          Estimated via sigmoid of calibrated directional head
        </div>
      </div>

      {/* Model Metadata & Generation Timestamp */}
      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-slate-800/60 pt-4 text-[11px] text-slate-500">
        <div className="flex items-center gap-2">
          <Cpu className="h-3.5 w-3.5 text-slate-400" />
          <span>Model: <code className="text-slate-300">{model_name || 'LSTM-V2-h5'}</code></span>
        </div>
        <div className="flex items-center gap-2">
          <Layers className="h-3.5 w-3.5 text-slate-400" />
          <span>Horizon: <strong className="text-slate-300">5 Trading Days</strong></span>
        </div>
        {prediction_date && (
          <div className="flex items-center gap-2">
            <Clock className="h-3.5 w-3.5 text-slate-400" />
            <span>Generated: {new Date(prediction_date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} UTC</span>
          </div>
        )}
      </div>

      {/* Mandatory Disclaimer */}
      <div className="mt-3 text-[10px] text-slate-500 bg-slate-950/60 rounded-lg p-2.5 border border-slate-800/40">
        <strong>Notice:</strong> Forecasts are machine learning statistical approximations generated for research and visualization purposes only. Model confidence is not certainty and does not constitute financial, investment, or trading advice.
      </div>
    </div>
  );
};

