import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../services/api';
import { PredictionSnapshot } from '../types/prediction';
import { PredictionBadge } from '../components/PredictionBadge';
import { Pagination } from '../components/Pagination';
import {
  LineChart,
  Search,
  Filter,
  RotateCcw,
  ArrowUpDown,
  ArrowDown,
  ArrowUp,
  ExternalLink,
} from 'lucide-react';

interface MarketPredictionsProps {
  onSelectStock: (ticker: string) => void;
}

export const MarketPredictions: React.FC<MarketPredictionsProps> = ({ onSelectStock }) => {
  const [predictions, setPredictions] = useState<PredictionSnapshot[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [search, setSearch] = useState('');
  const [direction, setDirection] = useState('');
  const [confidence, setConfidence] = useState('');
  const [minProbability, setMinProbability] = useState<number | ''>('');
  const [minReturn, setMinReturn] = useState<number | ''>('');
  const [sortBy, setSortBy] = useState('probability_up');
  const [sortOrder, setSortOrder] = useState('desc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);

  const fetchPredictions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getPredictions({
        search: search.trim() || undefined,
        direction: direction || undefined,
        confidence: confidence || undefined,
        min_probability: minProbability !== '' ? Number(minProbability) : undefined,
        min_return: minReturn !== '' ? Number(minReturn) : undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
        page,
        page_size: pageSize,
      });
      setPredictions(res.items);
      setTotal(res.total);
      setTotalPages(res.total_pages);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch predictions.');
    } finally {
      setLoading(false);
    }
  }, [search, direction, confidence, minProbability, minReturn, sortBy, sortOrder, page, pageSize]);

  useEffect(() => {
    void fetchPredictions();
  }, [fetchPredictions]);

  const handleSort = (col: string) => {
    if (sortBy === col) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(col);
      setSortOrder(['probability_up', 'predicted_return_pct', 'latest_close', 'predicted_price_5d'].includes(col) ? 'desc' : 'asc');
    }
    setPage(1);
  };

  const resetFilters = () => {
    setSearch('');
    setDirection('');
    setConfidence('');
    setMinProbability('');
    setMinReturn('');
    setSortBy('probability_up');
    setSortOrder('desc');
    setPage(1);
  };

  const renderSortIcon = (column: string) => {
    if (sortBy !== column) {
      return <ArrowUpDown className="h-3 w-3 text-slate-600 opacity-60 group-hover:opacity-100" />;
    }
    return sortOrder === 'desc' ? (
      <ArrowDown className="h-3 w-3 text-emerald-400" />
    ) : (
      <ArrowUp className="h-3 w-3 text-emerald-400" />
    );
  };

  return (
    <div className="space-y-6 pb-12">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <LineChart className="h-6 w-6 text-emerald-400" />
            <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
              Market Predictions Screener
            </h1>
          </div>
          <p className="mt-1 text-xs sm:text-sm text-slate-400">
            Multi-factor screening across all 471 valid forecasts with configurable probability and return thresholds.
          </p>
        </div>

        <div className="text-xs text-slate-400 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl">
          Matching: <strong className="text-white">{total}</strong> forecasts
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-4 shadow-sm space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          {/* Search */}
          <div className="relative sm:col-span-2">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400 pointer-events-none" />
            <input
              type="text"
              placeholder="Search ticker or company..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 py-2 pl-9 pr-3 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Direction */}
          <div>
            <select
              value={direction}
              onChange={(e) => {
                setDirection(e.target.value);
                setPage(1);
              }}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500/50"
            >
              <option value="">Direction: All</option>
              <option value="UP">UP Direction</option>
              <option value="DOWN">DOWN Direction</option>
            </select>
          </div>

          {/* Confidence */}
          <div>
            <select
              value={confidence}
              onChange={(e) => {
                setConfidence(e.target.value);
                setPage(1);
              }}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500/50"
            >
              <option value="">Confidence: All</option>
              <option value="High">High Confidence</option>
              <option value="Low">Low Confidence</option>
            </select>
          </div>

          {/* Min Probability */}
          <div>
            <input
              type="number"
              placeholder="Min Prob % (e.g. 60)"
              value={minProbability}
              onChange={(e) => {
                setMinProbability(e.target.value === '' ? '' : Number(e.target.value));
                setPage(1);
              }}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Reset */}
          <div>
            <button
              onClick={resetFilters}
              className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-slate-800 bg-slate-800/80 px-3 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset</span>
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* Predictions Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-md">
        <table className="w-full text-left text-xs sm:text-sm">
          <thead className="border-b border-slate-800 bg-slate-950/80 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <tr>
              <th
                onClick={() => handleSort('ticker')}
                className="cursor-pointer py-3.5 pl-4 pr-3 hover:text-white group select-none"
              >
                <div className="flex items-center gap-1.5">
                  <span>Ticker</span>
                  {renderSortIcon('ticker')}
                </div>
              </th>

              <th
                onClick={() => handleSort('company_name')}
                className="cursor-pointer px-3 py-3.5 hover:text-white group select-none"
              >
                <div className="flex items-center gap-1.5">
                  <span>Company</span>
                  {renderSortIcon('company_name')}
                </div>
              </th>

              <th
                onClick={() => handleSort('latest_close')}
                className="cursor-pointer px-3 py-3.5 text-right hover:text-white group select-none"
              >
                <div className="flex items-center justify-end gap-1.5">
                  <span>Latest Close</span>
                  {renderSortIcon('latest_close')}
                </div>
              </th>

              <th
                onClick={() => handleSort('predicted_price_5d')}
                className="cursor-pointer px-3 py-3.5 text-right hover:text-white group select-none"
              >
                <div className="flex items-center justify-end gap-1.5">
                  <span>5D Predicted</span>
                  {renderSortIcon('predicted_price_5d')}
                </div>
              </th>

              <th
                onClick={() => handleSort('predicted_return_pct')}
                className="cursor-pointer px-3 py-3.5 text-right hover:text-white group select-none"
              >
                <div className="flex items-center justify-end gap-1.5">
                  <span>Return %</span>
                  {renderSortIcon('predicted_return_pct')}
                </div>
              </th>

              <th
                onClick={() => handleSort('probability_up')}
                className="cursor-pointer px-3 py-3.5 text-center hover:text-white group select-none"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>Probability UP</span>
                  {renderSortIcon('probability_up')}
                </div>
              </th>

              <th className="px-3 py-3.5 text-center">Direction</th>

              <th className="hidden lg:table-cell px-3 py-3.5 text-left">Confidence</th>

              <th className="py-3.5 pl-2 pr-4 text-center">Action</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800/60">
            {loading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={`p-skel-${i}`} className="animate-pulse">
                  <td className="py-3.5 pl-4 pr-3"><div className="h-4 w-12 rounded bg-slate-800" /></td>
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
            ) : predictions.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-12 text-center text-slate-500">
                  No predictions found matching your filter criteria.
                </td>
              </tr>
            ) : (
              predictions.map((item) => {
                const retPct = item.predicted_return_pct ?? 0;
                return (
                  <tr
                    key={item.ticker}
                    onClick={() => onSelectStock(item.ticker)}
                    className="cursor-pointer transition-colors hover:bg-slate-800/40"
                  >
                    <td className="py-3.5 pl-4 pr-3 font-bold text-white">
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

      <Pagination
        currentPage={page}
        totalPages={totalPages}
        pageSize={pageSize}
        totalItems={total}
        onPageChange={(p) => setPage(p)}
        onPageSizeChange={(s) => {
          setPageSize(s);
          setPage(1);
        }}
      />
    </div>
  );
};

