import React, { useState, useEffect, useRef } from 'react';
import {
  TrendingUp,
  LayoutDashboard,
  Building2,
  LineChart,
  Trophy,
  Activity,
  CheckCircle2,
  Search,
  X,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react';
import { api } from '../services/api';
import { StockMetadata } from '../types/stock';

interface NavbarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  onSelectTicker: (ticker: string) => void;
  latestDataDate?: string;
  activePredictionsCount?: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onTabChange,
  onSelectTicker,
  latestDataDate = '2026-02-19',
  activePredictionsCount = 471,
}) => {
  const [searchValue, setSearchValue] = useState('');
  const [suggestions, setSuggestions] = useState<StockMetadata[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Debounced search for suggestions
  useEffect(() => {
    const timer = setTimeout(async () => {
      const q = searchValue.trim();
      if (q.length >= 1) {
        try {
          setIsSearching(true);
          const res = await api.getStocks({ search: q, page_size: 6 });
          setSuggestions(res.items);
          setShowDropdown(true);
        } catch {
          setSuggestions([]);
        } finally {
          setIsSearching(false);
        }
      } else {
        setSuggestions([]);
        setShowDropdown(false);
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [searchValue]);

  // Click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (ticker: string) => {
    onSelectTicker(ticker);
    setSearchValue('');
    setShowDropdown(false);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchValue.trim()) return;
    if (suggestions.length > 0) {
      handleSelect(suggestions[0].ticker);
    } else {
      handleSelect(searchValue.trim().toUpperCase());
    }
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'all_stocks', label: 'All Stocks (471)', icon: Building2 },
    { id: 'predictions', label: 'Market Predictions', icon: LineChart },
    { id: 'leaderboard', label: 'Leaderboard', icon: Trophy },
    { id: 'performance', label: 'Model Performance', icon: Activity },
    { id: 'validation', label: 'Validation Audit', icon: CheckCircle2 },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800/80 bg-slate-950/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        {/* Brand */}
        <div
          onClick={() => onTabChange('dashboard')}
          className="flex cursor-pointer items-center gap-3 transition-opacity hover:opacity-90 shrink-0"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500/25 to-teal-500/10 border border-emerald-500/40 text-emerald-400 shadow-sm shadow-emerald-950">
            <TrendingUp className="h-5 w-5" />
          </div>
          <div>
            <div className="text-base font-bold tracking-tight text-white sm:text-lg">
              Stock Trend Forecaster
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-medium text-slate-300">5-Day Forecast Engine</span>
              <span className="text-slate-600">•</span>
              <span className="hidden sm:inline text-slate-400">{activePredictionsCount} Stocks Active</span>
            </div>
          </div>
        </div>

        {/* Global Live Search Bar */}
        <div ref={dropdownRef} className="relative hidden md:block max-w-xs w-full mx-4">
          <form onSubmit={handleSearchSubmit} className="relative flex items-center">
            <Search className="absolute left-3 h-4 w-4 text-slate-400 pointer-events-none" />
            <input
              type="text"
              placeholder="Search ticker (AAPL) or company..."
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              onFocus={() => {
                if (suggestions.length > 0) setShowDropdown(true);
              }}
              className="w-full pl-9 pr-8 py-1.5 text-xs bg-slate-900 border border-slate-800 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition-all shadow-inner"
            />
            {searchValue && (
              <button
                type="button"
                onClick={() => setSearchValue('')}
                className="absolute right-2.5 text-slate-500 hover:text-slate-300"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </form>

          {/* Auto-suggest Live Dropdown */}
          {showDropdown && suggestions.length > 0 && (
            <div className="absolute left-0 right-0 top-full mt-1.5 rounded-xl border border-slate-800 bg-slate-900/98 shadow-2xl backdrop-blur-xl overflow-hidden z-50">
              <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider font-semibold text-slate-500 border-b border-slate-800/80 bg-slate-950/60">
                Matching Companies ({suggestions.length})
              </div>
              <div className="divide-y divide-slate-800/50">
                {suggestions.map((item) => (
                  <div
                    key={item.ticker}
                    onClick={() => handleSelect(item.ticker)}
                    className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-slate-800/70 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-emerald-400 text-xs bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                        {item.ticker}
                      </span>
                      <div className="text-xs text-slate-200 truncate max-w-[140px]">
                        {item.company_name}
                      </div>
                    </div>

                    <div className="flex items-center gap-2 text-right">
                      {item.latest_close != null && (
                        <span className="text-xs font-semibold text-white">
                          ${item.latest_close.toFixed(2)}
                        </span>
                      )}
                      {item.predicted_direction && (
                        <span
                          className={`flex items-center text-[10px] font-bold px-1.5 py-0.5 rounded ${
                            item.predicted_direction === 'UP'
                              ? 'bg-emerald-500/20 text-emerald-300'
                              : 'bg-rose-500/20 text-rose-300'
                          }`}
                        >
                          {item.predicted_direction === 'UP' ? (
                            <ArrowUpRight className="h-3 w-3 mr-0.5" />
                          ) : (
                            <ArrowDownRight className="h-3 w-3 mr-0.5" />
                          )}
                          {item.predicted_direction}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
                  isActive
                    ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-sm shadow-emerald-950/40'
                    : 'text-slate-400 hover:text-white hover:bg-slate-900/80 border border-transparent'
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                <span className="hidden lg:inline">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
