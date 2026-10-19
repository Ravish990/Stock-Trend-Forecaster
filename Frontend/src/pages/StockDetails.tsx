import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { StockDetailResponse, StockHistoryResponse, StockMetadata } from '../types/stock';
import { ForecastCard } from '../components/ForecastCard';
import { StockPriceChart } from '../components/StockPriceChart';
import {
  ArrowLeft,
  Building2,
  Calendar,
  Layers,
  Search,
  RefreshCw,
  AlertTriangle,
  ChevronRight,
  TrendingUp,
} from 'lucide-react';

interface StockDetailsProps {
  ticker: string;
  onBack: () => void;
  onSelectStock: (ticker: string) => void;
}

export const StockDetails: React.FC<StockDetailsProps> = ({
  ticker,
  onBack,
  onSelectStock,
}) => {
  const [details, setDetails] = useState<StockDetailResponse | null>(null);
  const [historyData, setHistoryData] = useState<StockHistoryResponse | null>(null);
  const [peers, setPeers] = useState<StockMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTicker, setSearchTicker] = useState('');

  const loadStock = async (t: string) => {
    try {
      setLoading(true);
      setError(null);
      const [detailRes, histRes] = await Promise.all([
        api.getStockDetails(t),
        api.getStockHistory(t, 90),
      ]);
      setDetails(detailRes);
      setHistoryData(histRes);

      // Load sector peers if sector available
      if (detailRes.sector) {
        try {
          const peerRes = await api.getStocks({ sector: detailRes.sector, page_size: 5 });
          setPeers(peerRes.items.filter((p) => p.ticker !== t).slice(0, 4));
        } catch {
          setPeers([]);
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || `Failed to fetch data for ticker ${t}.`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadStock(ticker);
  }, [ticker]);

  const handleSearchSwitch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchTicker.trim()) return;
    onSelectStock(searchTicker.trim().toUpperCase());
    setSearchTicker('');
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Top Bar: Back button, Ticker Jump, and Refresh */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Back to Explorer</span>
          </button>

          <span className="text-slate-600">/</span>

          <div className="text-xs text-slate-400">
            Stock Analysis for <strong className="font-mono text-emerald-400">{ticker}</strong>
            {details?.company_name && (
              <span className="text-slate-300 ml-1.5">({details.company_name})</span>
            )}
          </div>
        </div>

        {/* Quick Ticker Switcher */}
        <div className="flex items-center gap-2">
          <form onSubmit={handleSearchSwitch} className="relative">
            <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-500 pointer-events-none" />
            <input
              type="text"
              placeholder="Switch stock (e.g. MSFT)..."
              value={searchTicker}
              onChange={(e) => setSearchTicker(e.target.value)}
              className="w-48 rounded-xl border border-slate-800 bg-slate-900 py-1.5 pl-8 pr-2.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
            />
          </form>

          <button
            onClick={() => void loadStock(ticker)}
            disabled={loading}
            className="rounded-xl border border-slate-800 bg-slate-900 p-2 text-slate-400 hover:text-white transition-colors"
            title="Refresh Stock Data"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin text-emerald-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="flex items-start gap-3 rounded-2xl border border-rose-500/40 bg-rose-500/10 p-4 text-xs text-rose-200">
          <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-bold text-rose-300">Ticker Analysis Notice</div>
            <div className="mt-0.5">{error}</div>
          </div>
        </div>
      )}

      {/* Forecast Showcase Card */}
      <ForecastCard
        prediction={details?.prediction ?? null}
        loading={loading}
      />

      {/* Historical Stock Price & 5-Day Horizon Chart */}
      {historyData && (
        <StockPriceChart
          ticker={ticker}
          history={historyData.history}
          forecastPoint={historyData.forecast_point}
          latestClose={historyData.latest_close}
        />
      )}

      {/* Peer Stocks in Same Sector */}
      {peers.length > 0 && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-white">
                Similar Sector Peers ({details?.sector})
              </h3>
            </div>
            <span className="text-xs text-slate-500">Click to compare forecast</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {peers.map((p) => (
              <div
                key={p.ticker}
                onClick={() => onSelectStock(p.ticker)}
                className="cursor-pointer rounded-xl border border-slate-800/80 bg-slate-950/60 p-3 hover:border-emerald-500/40 hover:bg-slate-900/80 transition-all flex items-center justify-between"
              >
                <div>
                  <div className="font-mono font-bold text-white text-xs">{p.ticker}</div>
                  <div className="text-[11px] text-slate-400 truncate max-w-[130px]">{p.company_name}</div>
                  <div className="text-[11px] font-semibold text-slate-300 mt-0.5">
                    ${p.latest_close?.toFixed(2)}
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-xs font-bold text-emerald-400">
                    {p.probability_up != null ? `${p.probability_up.toFixed(1)}% UP` : '—'}
                  </div>
                  <div className="text-[10px] text-sky-300">
                    {p.predicted_return_pct != null
                      ? `${p.predicted_return_pct >= 0 ? '+' : ''}${p.predicted_return_pct.toFixed(2)}%`
                      : '—'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Historical Observations Table */}
      {details && details.history_summary.length > 0 && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-4">
            <div>
              <h3 className="text-sm font-bold text-white">Recent Historical Sessions</h3>
              <p className="text-xs text-slate-400 mt-0.5">Last 15 observed market closes from the dataset</p>
            </div>
            <span className="text-xs text-slate-500">Sorted chronologically</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400 font-semibold uppercase text-[11px]">
                <tr>
                  <th className="py-2.5 pl-3 pr-2">Date</th>
                  <th className="px-2 py-2.5 text-right">Close Price</th>
                  <th className="px-2 py-2.5 text-right hidden sm:table-cell">Open</th>
                  <th className="px-2 py-2.5 text-right hidden sm:table-cell">High</th>
                  <th className="px-2 py-2.5 text-right hidden sm:table-cell">Low</th>
                  <th className="py-2.5 pl-2 pr-3 text-right">Volume</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40">
                {details.history_summary.slice().reverse().map((row, idx) => (
                  <tr key={`recent-${idx}`} className="hover:bg-slate-800/30">
                    <td className="py-2 pl-3 pr-2 font-mono text-slate-300">{row.date}</td>
                    <td className="px-2 py-2 text-right font-bold text-white">${row.close.toFixed(2)}</td>
                    <td className="px-2 py-2 text-right text-slate-400 hidden sm:table-cell">
                      {row.open != null ? `$${row.open.toFixed(2)}` : '—'}
                    </td>
                    <td className="px-2 py-2 text-right text-slate-400 hidden sm:table-cell">
                      {row.high != null ? `$${row.high.toFixed(2)}` : '—'}
                    </td>
                    <td className="px-2 py-2 text-right text-slate-400 hidden sm:table-cell">
                      {row.low != null ? `$${row.low.toFixed(2)}` : '—'}
                    </td>
                    <td className="py-2 pl-2 pr-3 text-right text-slate-400 font-mono">
                      {row.volume != null ? row.volume.toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
