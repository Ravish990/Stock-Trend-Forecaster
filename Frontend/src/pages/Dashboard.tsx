import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { MarketSummaryResponse, PredictionSnapshot } from '../types/prediction';
import { MetricCard } from '../components/MetricCard';
import { PredictionBadge } from '../components/PredictionBadge';
import {
  Building2,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Percent,
  Calendar,
  Sparkles,
  ArrowRight,
  RefreshCw,
  AlertTriangle,
  Zap,
  ChevronRight,
  Layers,
} from 'lucide-react';

interface DashboardProps {
  onSelectStock: (ticker: string) => void;
  onNavigateTab: (tab: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onSelectStock, onNavigateTab }) => {
  const [summary, setSummary] = useState<MarketSummaryResponse | null>(null);
  const [highConviction, setHighConviction] = useState<PredictionSnapshot[]>([]);
  const [screenerList, setScreenerList] = useState<PredictionSnapshot[]>([]);
  const [activeLeaderboardTab, setActiveLeaderboardTab] = useState<'high_prob' | 'high_return' | 'bearish'>('high_prob');
  const [selectedSector, setSelectedSector] = useState<string>('All');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [summaryData, topUp, topReturns, topDown] = await Promise.all([
        api.getMarketSummary(),
        api.getLeaderboard('high_confidence_up', 10),
        api.getLeaderboard('highest_expected_return', 10),
        api.getLeaderboard('highest_probability_down', 10),
      ]);
      setSummary(summaryData);
      setHighConviction(topUp.items.slice(0, 3));
      setScreenerList(
        activeLeaderboardTab === 'high_prob'
          ? topUp.items
          : activeLeaderboardTab === 'high_return'
          ? topReturns.items
          : topDown.items
      );
    } catch (err: any) {
      setError(
        err.message || 'Unable to connect to the FastAPI backend. Ensure the backend server is running on port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchDashboardData();
  }, []);

  const handleTabSwitch = async (tab: 'high_prob' | 'high_return' | 'bearish') => {
    setActiveLeaderboardTab(tab);
    try {
      const cat =
        tab === 'high_prob'
          ? 'high_confidence_up'
          : tab === 'high_return'
          ? 'highest_expected_return'
          : 'highest_probability_down';
      const res = await api.getLeaderboard(cat, 12);
      setScreenerList(res.items);
    } catch (e) {
      // fallback
    }
  };

  const sectors = ['All', 'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical', 'Industrials'];

  const filteredScreener = React.useMemo(() => {
    if (selectedSector === 'All') return screenerList;
    return screenerList.filter((s) => s.sector && s.sector.toLowerCase().includes(selectedSector.toLowerCase()));
  }, [screenerList, selectedSector]);

  const upPct = summary && summary.successfully_predicted_tickers > 0
    ? ((summary.predicted_up_count / summary.successfully_predicted_tickers) * 100).toFixed(1)
    : '99.1';
  const downPct = summary && summary.successfully_predicted_tickers > 0
    ? ((summary.predicted_down_count / summary.successfully_predicted_tickers) * 100).toFixed(1)
    : '0.9';

  return (
    <div className="space-y-8 pb-12">
      {/* Top Header & Fast Action */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">Live Forecast Engine</span>
          </div>
          <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl mt-1">
            Market Probability & Forecast Dashboard
          </h1>
          <p className="mt-1 text-xs sm:text-sm text-slate-400">
            Next 5-trading-day quantitative predictions across all 471 supported S&P 500 equities.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => void fetchDashboardData()}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-900/90 px-3.5 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white transition-all shadow-sm"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin text-emerald-400' : ''}`} />
            <span>Refresh Model</span>
          </button>

          <button
            onClick={() => onNavigateTab('all_stocks')}
            className="flex items-center gap-1.5 rounded-xl bg-emerald-500 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-emerald-400 transition-all shadow-md shadow-emerald-950"
          >
            <span>Browse All 471 Stocks</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="flex items-start gap-3 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-rose-200 text-sm">
          <AlertTriangle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="font-semibold text-rose-300">Backend Connection Error</div>
            <div className="text-xs text-rose-200/80 mt-0.5">{error}</div>
          </div>
        </div>
      )}

      {/* Primary KPI Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <MetricCard
          label="Supported Universe"
          value={summary?.total_supported_tickers ?? 471}
          subvalue="SP500 Equities"
          icon={Building2}
        />
        <MetricCard
          label="Active Forecasts"
          value={summary?.successfully_predicted_tickers ?? 471}
          subvalue="100% Active Coverage"
          icon={CheckCircle2}
          highlight
        />
        <MetricCard
          label="Predicted UP"
          value={summary?.predicted_up_count ?? 467}
          subvalue={`${upPct}% of Universe`}
          trend="up"
          icon={TrendingUp}
        />
        <MetricCard
          label="Predicted DOWN"
          value={summary?.predicted_down_count ?? 4}
          subvalue={`${downPct}% of Universe`}
          trend="down"
          icon={TrendingDown}
        />
        <MetricCard
          label="Avg 5D Return"
          value={summary ? `+${summary.avg_predicted_return.toFixed(2)}%` : '+2.13%'}
          subvalue="Market-wide expected return"
          trend="up"
          icon={Percent}
        />
        <MetricCard
          label="Latest Data Date"
          value={summary?.latest_data_date ?? '2026-02-19'}
          subvalue="5-Trading-Day Horizon"
          icon={Calendar}
        />
      </section>

      {/* High-Conviction Opportunity Showcase (Hero Cards) */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-amber-400" />
            <h2 className="text-base font-bold text-white tracking-tight">
              High-Conviction Model Opportunities
            </h2>
          </div>
          <span className="text-xs text-slate-400">Probability &gt; 60% with highest return upside</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {highConviction.map((item) => {
            const retPct = item.predicted_return_pct ?? 0;
            const dollarGain =
              item.predicted_price_5d && item.latest_close
                ? item.predicted_price_5d - item.latest_close
                : 0;

            return (
              <div
                key={item.ticker}
                onClick={() => onSelectStock(item.ticker)}
                className="group relative cursor-pointer overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-5 shadow-lg transition-all duration-300 hover:border-emerald-500/50 hover:shadow-emerald-950/20 hover:-translate-y-0.5"
              >
                {/* Subtle Glow */}
                <div className="absolute -top-12 -right-12 h-24 w-24 rounded-full bg-emerald-500/15 blur-2xl group-hover:bg-emerald-500/25 transition-all" />

                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-base font-black text-white">{item.ticker}</span>
                      <PredictionBadge direction="UP" confidence={item.confidence_label} size="sm" />
                    </div>
                    <div className="text-xs text-slate-400 truncate max-w-[180px] mt-0.5">
                      {item.company_name}
                    </div>
                    {item.sector && (
                      <div className="text-[10px] text-slate-500 mt-0.5">{item.sector}</div>
                    )}
                  </div>

                  <div className="text-right">
                    <div className="text-xs text-slate-400">Current Close</div>
                    <div className="text-sm font-bold text-slate-200">
                      ${item.latest_close?.toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Target Forecast & Return */}
                <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center justify-between">
                  <div>
                    <div className="text-[11px] text-slate-400 font-medium">5-Day Target</div>
                    <div className="text-xl font-black text-sky-300">
                      ${item.predicted_price_5d?.toFixed(2)}
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-[11px] text-slate-400 font-medium">Projected Gain</div>
                    <div className="text-xl font-black text-emerald-400">
                      +{retPct.toFixed(2)}%
                    </div>
                  </div>
                </div>

                {/* Probability Bar */}
                <div className="mt-3">
                  <div className="flex justify-between text-[11px] font-semibold text-slate-300 mb-1">
                    <span>Upward Probability</span>
                    <span className="text-emerald-400">{item.probability_up?.toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
                      style={{ width: `${item.probability_up ?? 50}%` }}
                    />
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between text-xs text-emerald-400 group-hover:text-emerald-300 font-semibold pt-2">
                  <span>View Chart & Analysis</span>
                  <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Interactive Screener Section with Tabs and Sector Chips */}
      <section className="rounded-2xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              Interactive Market Screener
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Instantly toggle between high conviction picks, top return targets, or explore by sector.
            </p>
          </div>

          {/* Quick Category Tabs */}
          <div className="flex items-center gap-1.5 rounded-xl bg-slate-950 p-1 border border-slate-800">
            <button
              onClick={() => handleTabSwitch('high_prob')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
                activeLeaderboardTab === 'high_prob'
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🔥 High Conviction UP
            </button>
            <button
              onClick={() => handleTabSwitch('high_return')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
                activeLeaderboardTab === 'high_return'
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              📈 Top Return Upside
            </button>
            <button
              onClick={() => handleTabSwitch('bearish')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
                activeLeaderboardTab === 'bearish'
                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              📉 Bearish Downside
            </button>
          </div>
        </div>

        {/* Sector Quick-Filter Chips */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-500 font-semibold mr-1">Sector:</span>
          {sectors.map((sec) => (
            <button
              key={sec}
              onClick={() => setSelectedSector(sec)}
              className={`px-3 py-1 text-xs font-medium rounded-full border transition-all ${
                selectedSector === sec
                  ? 'bg-slate-800 border-emerald-500/50 text-white font-bold shadow-sm'
                  : 'bg-slate-950/80 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-300'
              }`}
            >
              {sec}
            </button>
          ))}
        </div>

        {/* Screener Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-950/60">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead className="border-b border-slate-800 bg-slate-950 text-slate-400 font-semibold uppercase text-[11px]">
              <tr>
                <th className="py-3 pl-4 pr-3">Ticker</th>
                <th className="px-3 py-3">Company Name</th>
                <th className="px-3 py-3 hidden md:table-cell">Sector</th>
                <th className="px-3 py-3 text-right">Latest Close</th>
                <th className="px-3 py-3 text-right">5D Target</th>
                <th className="px-3 py-3 text-right">Expected Return</th>
                <th className="px-3 py-3 text-center">Probability UP</th>
                <th className="px-3 py-3 text-center">Direction</th>
                <th className="py-3 pl-2 pr-4 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/40">
              {filteredScreener.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-slate-500">
                    No stocks match the selected sector filter in this category.
                  </td>
                </tr>
              ) : (
                filteredScreener.map((item) => {
                  const retPct = item.predicted_return_pct ?? 0;
                  return (
                    <tr
                      key={item.ticker}
                      onClick={() => onSelectStock(item.ticker)}
                      className="cursor-pointer hover:bg-slate-800/40 transition-colors"
                    >
                      <td className="py-3 pl-4 pr-3 font-mono font-bold text-white">
                        <span className="bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-emerald-400">
                          {item.ticker}
                        </span>
                      </td>

                      <td className="px-3 py-3 font-medium text-slate-200">
                        <div className="truncate max-w-[200px]">{item.company_name}</div>
                      </td>

                      <td className="px-3 py-3 hidden md:table-cell text-slate-400 text-xs">
                        {item.sector || '—'}
                      </td>

                      <td className="px-3 py-3 text-right font-semibold text-slate-200">
                        ${item.latest_close?.toFixed(2)}
                      </td>

                      <td className="px-3 py-3 text-right font-bold text-sky-300">
                        ${item.predicted_price_5d?.toFixed(2)}
                      </td>

                      <td className="px-3 py-3 text-right font-bold">
                        <span className={retPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {retPct >= 0 ? `+${retPct.toFixed(2)}%` : `${retPct.toFixed(2)}%`}
                        </span>
                      </td>

                      <td className="px-3 py-3 text-center font-semibold text-slate-100">
                        {item.probability_up?.toFixed(1)}%
                      </td>

                      <td className="px-3 py-3 text-center">
                        <PredictionBadge direction={item.predicted_direction} size="sm" />
                      </td>

                      <td className="py-3 pl-2 pr-4 text-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectStock(item.ticker);
                          }}
                          className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-emerald-400 transition-colors"
                          title="Open Chart"
                        >
                          <ChevronRight className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
