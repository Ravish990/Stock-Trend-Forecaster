import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { LeaderboardResponse, PredictionSnapshot } from '../types/prediction';
import { PredictionBadge } from '../components/PredictionBadge';
import {
  Trophy,
  Sparkles,
  TrendingUp,
  TrendingDown,
  ShieldCheck,
  ShieldAlert,
  Info,
  ExternalLink,
} from 'lucide-react';

interface LeaderboardPageProps {
  onSelectStock: (ticker: string) => void;
}

export const LeaderboardPage: React.FC<LeaderboardPageProps> = ({ onSelectStock }) => {
  const [activeCategory, setActiveCategory] = useState('highest_probability_up');
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const categories = [
    { id: 'highest_probability_up', label: 'Highest Prob UP', icon: Sparkles },
    { id: 'highest_expected_return', label: 'Highest Expected Return', icon: TrendingUp },
    { id: 'highest_probability_down', label: 'Highest Prob DOWN', icon: TrendingDown },
    { id: 'lowest_expected_return', label: 'Lowest Expected Return', icon: TrendingDown },
    { id: 'high_confidence_up', label: 'High Confidence UP (>60%)', icon: ShieldCheck },
    { id: 'high_confidence_down', label: 'High Confidence DOWN (<40%)', icon: ShieldAlert },
  ];

  const fetchLeaderboard = async (cat: string) => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getLeaderboard(cat, 30);
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch leaderboard data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchLeaderboard(activeCategory);
  }, [activeCategory]);

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="border-b border-slate-800/80 pb-5">
        <div className="flex items-center gap-2.5">
          <Trophy className="h-6 w-6 text-amber-400" />
          <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
            Market Forecast Leaderboards
          </h1>
        </div>
        <p className="mt-1 text-xs sm:text-sm text-slate-400">
          Rankings of all supported stocks categorized by upward probability, projected magnitude, and confidence conviction.
        </p>
      </div>

      {/* Critical Mathematical Distiction Notice */}
      <div className="flex items-start gap-3 rounded-2xl border border-sky-500/30 bg-sky-500/10 p-4 text-xs text-sky-200">
        <Info className="h-5 w-5 text-sky-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-sky-300">Quantitative Distinction:</strong> The probability of going UP represents the model's confidence in a positive directional move over 5 sessions, whereas expected percentage return reflects the regression head's magnitude estimate. High probability does not necessarily mean high expected percentage return, and volatile stocks may show large expected returns with lower directional probability.
        </div>
      </div>

      {/* Tab Selector */}
      <div className="flex flex-wrap gap-2 border-b border-slate-800 pb-2">
        {categories.map((c) => {
          const Icon = c.icon;
          const isActive = activeCategory === c.id;
          return (
            <button
              key={c.id}
              onClick={() => setActiveCategory(c.id)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition-all ${
                isActive
                  ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 shadow-sm'
                  : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Icon className={`h-4 w-4 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`} />
              <span>{c.label}</span>
            </button>
          );
        })}
      </div>

      {/* Category Description */}
      {data && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-xs text-slate-300 flex items-center justify-between">
          <div>
            <strong className="text-white">{data.title}:</strong> {data.description}
          </div>
          <span className="text-slate-500 text-[11px]">Top {data.items.length} Tickers</span>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* Leaderboard Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-md">
        <table className="w-full text-left text-xs sm:text-sm">
          <thead className="border-b border-slate-800 bg-slate-950/80 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <tr>
              <th className="py-3.5 pl-4 pr-3 text-center w-12">Rank</th>
              <th className="px-3 py-3.5">Ticker</th>
              <th className="px-3 py-3.5">Company Name</th>
              <th className="px-3 py-3.5 text-right">Latest Close</th>
              <th className="px-3 py-3.5 text-right">Predicted 5D Price</th>
              <th className="px-3 py-3.5 text-right">Expected Return</th>
              <th className="px-3 py-3.5 text-center">Probability UP</th>
              <th className="px-3 py-3.5 text-center">Direction</th>
              <th className="hidden lg:table-cell px-3 py-3.5 text-left">Confidence</th>
              <th className="py-3.5 pl-2 pr-4 text-center">Action</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800/60">
            {loading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={`lb-skel-${i}`} className="animate-pulse">
                  <td className="py-3.5 pl-4 pr-3 text-center"><div className="h-4 w-6 mx-auto rounded bg-slate-800" /></td>
                  <td className="px-3 py-3.5"><div className="h-4 w-12 rounded bg-slate-800" /></td>
                  <td className="px-3 py-3.5"><div className="h-4 w-36 rounded bg-slate-800" /></td>
                  <td className="px-3 py-3.5 text-right"><div className="h-4 w-16 ml-auto rounded bg-slate-800" /></td>
                  <td className="px-3 py-3.5 text-right"><div className="h-4 w-16 ml-auto rounded bg-slate-800" /></td>
                  <td className="px-3 py-3.5 text-right"><div className="h-4 w-14 ml-auto rounded bg-slate-800" /></td>
                  <td className="px-3 py-3.5 text-center"><div className="h-4 w-12 mx-auto rounded bg-slate-800" /></td>
                  <td className="px-3 py-3.5 text-center"><div className="h-5 w-16 mx-auto rounded-full bg-slate-800" /></td>
                  <td className="hidden lg:table-cell px-3 py-3.5"><div className="h-4 w-28 rounded bg-slate-800" /></td>
                  <td className="py-3.5 pl-2 pr-4 text-center"><div className="h-4 w-6 mx-auto rounded bg-slate-800" /></td>
                </tr>
              ))
            ) : !data || data.items.length === 0 ? (
              <tr>
                <td colSpan={10} className="py-12 text-center text-slate-500">
                  No stocks currently meet this category threshold.
                </td>
              </tr>
            ) : (
              data.items.map((item, index) => {
                const retPct = item.predicted_return_pct ?? 0;
                return (
                  <tr
                    key={item.ticker}
                    onClick={() => onSelectStock(item.ticker)}
                    className="cursor-pointer transition-colors hover:bg-slate-800/40"
                  >
                    <td className="py-3.5 pl-4 pr-3 text-center font-mono font-bold text-slate-400">
                      {index === 0 ? (
                        <span className="flex items-center justify-center h-6 w-6 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/40 mx-auto text-xs">
                          1
                        </span>
                      ) : index === 1 ? (
                        <span className="flex items-center justify-center h-6 w-6 rounded-full bg-slate-400/20 text-slate-200 border border-slate-400/40 mx-auto text-xs">
                          2
                        </span>
                      ) : index === 2 ? (
                        <span className="flex items-center justify-center h-6 w-6 rounded-full bg-amber-700/20 text-amber-500 border border-amber-600/40 mx-auto text-xs">
                          3
                        </span>
                      ) : (
                        <span>#{index + 1}</span>
                      )}
                    </td>

                    <td className="px-3 py-3.5 font-bold text-white">
                      <span className="rounded bg-slate-800/60 px-2 py-0.5 text-xs text-emerald-400 font-mono border border-slate-700/50">
                        {item.ticker}
                      </span>
                    </td>

                    <td className="px-3 py-3.5 font-medium text-slate-200">
                      <div className="max-w-[200px] truncate">{item.company_name || 'Unavailable'}</div>
                    </td>

                    <td className="px-3 py-3.5 text-right font-semibold text-slate-200">
                      ${item.latest_close?.toFixed(2)}
                    </td>

                    <td className="px-3 py-3.5 text-right font-semibold text-sky-300">
                      ${item.predicted_price_5d?.toFixed(2)}
                    </td>

                    <td className="px-3 py-3.5 text-right font-bold">
                      <span className={retPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        {retPct >= 0 ? `+${retPct.toFixed(2)}%` : `${retPct.toFixed(2)}%`}
                      </span>
                    </td>

                    <td className="px-3 py-3.5 text-center">
                      <div className="inline-flex items-center gap-1.5">
                        <span className="font-semibold text-slate-100">
                          {item.probability_up?.toFixed(1)}%
                        </span>
                        <div className="w-12 h-1.5 rounded-full bg-slate-800 overflow-hidden hidden sm:block">
                          <div
                            className="h-full bg-emerald-400"
                            style={{ width: `${item.probability_up ?? 50}%` }}
                          />
                        </div>
                      </div>
                    </td>

                    <td className="px-3 py-3.5 text-center">
                      <PredictionBadge direction={item.predicted_direction} size="sm" />
                    </td>

                    <td className="hidden lg:table-cell px-3 py-3.5 text-xs text-slate-300">
                      <span
                        className={`rounded px-2 py-0.5 text-[11px] font-medium border ${
                          item.confidence_label?.includes('High')
                            ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
                            : 'bg-slate-800/60 border-slate-700/40 text-slate-400'
                        }`}
                      >
                        {item.confidence_label}
                      </span>
                    </td>

                    <td className="py-3.5 pl-2 pr-4 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectStock(item.ticker);
                        }}
                        className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-emerald-400 transition-colors"
                        title="View Details"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

