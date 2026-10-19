import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../services/api';
import { StockMetadata } from '../types/stock';
import { StockTable } from '../components/StockTable';
import { Pagination } from '../components/Pagination';
import { Search, RotateCcw, Building2, Sparkles, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

interface AllStocksProps {
  onSelectStock: (ticker: string) => void;
}

export const AllStocks: React.FC<AllStocksProps> = ({ onSelectStock }) => {
  const [stocks, setStocks] = useState<StockMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter & Pagination state
  const [search, setSearch] = useState('');
  const [sector, setSector] = useState('');
  const [direction, setDirection] = useState('');
  const [confidence, setConfidence] = useState('');
  const [quickFilter, setQuickFilter] = useState<'all' | 'high_up' | 'up' | 'down'>('all');
  const [sortBy, setSortBy] = useState('ticker');
  const [sortOrder, setSortOrder] = useState('asc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);

  const fetchStocks = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      let dirParam = direction || undefined;
      let confParam = confidence || undefined;

      if (quickFilter === 'high_up') {
        dirParam = 'UP';
        confParam = 'High';
      } else if (quickFilter === 'up') {
        dirParam = 'UP';
      } else if (quickFilter === 'down') {
        dirParam = 'DOWN';
      }

      const res = await api.getStocks({
        search: search.trim() || undefined,
        sector: sector || undefined,
        direction: dirParam,
        confidence: confParam,
        page,
        page_size: pageSize,
        sort_by: sortBy,
        sort_order: sortOrder,
      });

      setStocks(res.items);
      setTotal(res.total);
      setTotalPages(res.total_pages);
    } catch (err: any) {
      setError(err.message || 'Failed to load stock directory.');
    } finally {
      setLoading(false);
    }
  }, [search, sector, direction, confidence, quickFilter, page, pageSize, sortBy, sortOrder]);

  useEffect(() => {
    void fetchStocks();
  }, [fetchStocks]);

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      if (['probability_up', 'predicted_return_pct', 'latest_close', 'predicted_price_5d'].includes(column)) {
        setSortOrder('desc');
      } else {
        setSortOrder('asc');
      }
    }
    setPage(1);
  };

  const handleQuickFilter = (qf: 'all' | 'high_up' | 'up' | 'down') => {
    setQuickFilter(qf);
    if (qf === 'all') {
      setDirection('');
      setConfidence('');
      setSearch('');
    }
    setPage(1);
  };

  const resetFilters = () => {
    setSearch('');
    setSector('');
    setDirection('');
    setConfidence('');
    setQuickFilter('all');
    setSortBy('ticker');
    setSortOrder('asc');
    setPage(1);
  };

  const sectorOptions = [
    'Technology',
    'Healthcare',
    'Financial Services',
    'Consumer Cyclical',
    'Industrials',
    'Communication Services',
    'Consumer Defensive',
    'Energy',
    'Utilities',
    'Real Estate',
    'Basic Materials',
  ];

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <Building2 className="h-6 w-6 text-emerald-400" />
            <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
              All 471 Companies Directory
            </h1>
          </div>
          <p className="mt-1 text-xs sm:text-sm text-slate-400">
            Search, filter, and inspect verified metadata and machine learning forecasts across all supported S&P 500 tickers.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-900 border border-slate-800 px-3.5 py-1.5 rounded-xl shadow-sm">
          <span>Total Mapped Universe: <strong className="text-emerald-400">471 Tickers</strong></span>
          <span className="text-slate-600">•</span>
          <span>Showing: <strong className="text-white">{stocks.length}</strong> of {total}</span>
        </div>
      </div>

      {/* Quick Filter Bar */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => handleQuickFilter('all')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
            quickFilter === 'all'
              ? 'bg-slate-100 text-slate-950 shadow-md font-extrabold'
              : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-white'
          }`}
        >
          <span>All 471 Companies</span>
        </button>

        <button
          onClick={() => handleQuickFilter('high_up')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
            quickFilter === 'high_up'
              ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-950 font-extrabold'
              : 'bg-slate-900 border border-slate-800 text-emerald-400 hover:border-emerald-500/40'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" />
          <span>🚀 High Conviction UP (&gt;60%)</span>
        </button>

        <button
          onClick={() => handleQuickFilter('up')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
            quickFilter === 'up'
              ? 'bg-emerald-500/30 border border-emerald-500 text-emerald-300 font-extrabold'
              : 'bg-slate-900 border border-slate-800 text-slate-300 hover:text-white'
          }`}
        >
          <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
          <span>Bullish (UP)</span>
        </button>

        <button
          onClick={() => handleQuickFilter('down')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
            quickFilter === 'down'
              ? 'bg-rose-500/30 border border-rose-500 text-rose-300 font-extrabold'
              : 'bg-slate-900 border border-slate-800 text-slate-300 hover:text-white'
          }`}
        >
          <TrendingDown className="h-3.5 w-3.5 text-rose-400" />
          <span>Bearish (DOWN)</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-4 shadow-sm space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {/* Search Input */}
          <div className="relative sm:col-span-2">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400 pointer-events-none" />
            <input
              type="text"
              placeholder="Filter by ticker (AAPL) or company name..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setQuickFilter('all');
                setPage(1);
              }}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 py-2 pl-9 pr-3 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Sector Selector */}
          <div>
            <select
              value={sector}
              onChange={(e) => {
                setSector(e.target.value);
                setPage(1);
              }}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500/50"
            >
              <option value="">All Sectors (11)</option>
              {sectorOptions.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Direction Filter */}
          <div>
            <select
              value={direction}
              onChange={(e) => {
                setDirection(e.target.value);
                setQuickFilter('all');
                setPage(1);
              }}
              className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500/50"
            >
              <option value="">Direction: All</option>
              <option value="UP">UP Direction</option>
              <option value="DOWN">DOWN Direction</option>
            </select>
          </div>

          {/* Reset Filters */}
          <div className="flex items-center gap-2">
            <button
              onClick={resetFilters}
              className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-slate-800 bg-slate-800/80 px-3 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset Filters</span>
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* Data Table */}
      <StockTable
        stocks={stocks}
        loading={loading}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onSort={handleSort}
        onSelectStock={onSelectStock}
      />

      {/* Pagination */}
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
