import React from 'react';
import { StockMetadata } from '../types/stock';
import { PredictionBadge } from './PredictionBadge';
import { ArrowUpDown, ArrowUp, ArrowDown, ExternalLink } from 'lucide-react';

interface StockTableProps {
  stocks: StockMetadata[];
  loading?: boolean;
  sortBy: string;
  sortOrder: string;
  onSort: (column: string) => void;
  onSelectStock: (ticker: string) => void;
}

export const StockTable: React.FC<StockTableProps> = ({
  stocks,
  loading = false,
  sortBy,
  sortOrder,
  onSort,
  onSelectStock,
}) => {
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
    <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-md">
      <table className="w-full text-left text-xs sm:text-sm">
        <thead className="border-b border-slate-800 bg-slate-950/80 text-xs font-semibold uppercase tracking-wider text-slate-400">
          <tr>
            <th
              onClick={() => onSort('ticker')}
              className="cursor-pointer py-3.5 pl-4 pr-3 hover:text-white group select-none"
            >
              <div className="flex items-center gap-1.5">
                <span>Ticker</span>
                {renderSortIcon('ticker')}
              </div>
            </th>

            <th
              onClick={() => onSort('company_name')}
              className="cursor-pointer px-3 py-3.5 hover:text-white group select-none"
            >
              <div className="flex items-center gap-1.5">
                <span>Company</span>
                {renderSortIcon('company_name')}
              </div>
            </th>

            <th
              onClick={() => onSort('sector')}
              className="hidden md:table-cell cursor-pointer px-3 py-3.5 hover:text-white group select-none"
            >
              <div className="flex items-center gap-1.5">
                <span>Sector</span>
                {renderSortIcon('sector')}
              </div>
            </th>

            <th
              onClick={() => onSort('latest_close')}
              className="cursor-pointer px-3 py-3.5 text-right hover:text-white group select-none"
            >
              <div className="flex items-center justify-end gap-1.5">
                <span>Latest Close</span>
                {renderSortIcon('latest_close')}
              </div>
            </th>

            <th
              onClick={() => onSort('predicted_price_5d')}
              className="cursor-pointer px-3 py-3.5 text-right hover:text-white group select-none"
            >
              <div className="flex items-center justify-end gap-1.5">
                <span>5D Predicted</span>
                {renderSortIcon('predicted_price_5d')}
              </div>
            </th>

            <th
              onClick={() => onSort('predicted_return_pct')}
              className="cursor-pointer px-3 py-3.5 text-right hover:text-white group select-none"
            >
              <div className="flex items-center justify-end gap-1.5">
                <span>Exp Return</span>
                {renderSortIcon('predicted_return_pct')}
              </div>
            </th>

            <th
              onClick={() => onSort('probability_up')}
              className="cursor-pointer px-3 py-3.5 text-center hover:text-white group select-none"
            >
              <div className="flex items-center justify-center gap-1.5">
                <span>Prob UP</span>
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
              <tr key={`skeleton-${i}`} className="animate-pulse">
                <td className="py-3.5 pl-4 pr-3"><div className="h-4 w-12 rounded bg-slate-800" /></td>
                <td className="px-3 py-3.5"><div className="h-4 w-36 rounded bg-slate-800" /></td>
                <td className="hidden md:table-cell px-3 py-3.5"><div className="h-4 w-24 rounded bg-slate-800" /></td>
                <td className="px-3 py-3.5 text-right"><div className="h-4 w-16 ml-auto rounded bg-slate-800" /></td>
                <td className="px-3 py-3.5 text-right"><div className="h-4 w-16 ml-auto rounded bg-slate-800" /></td>
                <td className="px-3 py-3.5 text-right"><div className="h-4 w-14 ml-auto rounded bg-slate-800" /></td>
                <td className="px-3 py-3.5 text-center"><div className="h-4 w-12 mx-auto rounded bg-slate-800" /></td>
                <td className="px-3 py-3.5 text-center"><div className="h-5 w-16 mx-auto rounded-full bg-slate-800" /></td>
                <td className="hidden lg:table-cell px-3 py-3.5"><div className="h-4 w-28 rounded bg-slate-800" /></td>
                <td className="py-3.5 pl-2 pr-4 text-center"><div className="h-4 w-6 mx-auto rounded bg-slate-800" /></td>
              </tr>
            ))
          ) : stocks.length === 0 ? (
            <tr>
              <td colSpan={10} className="py-12 text-center text-slate-500">
                No matching stocks found for the current filter criteria.
              </td>
            </tr>
          ) : (
            stocks.map((stock) => {
              const retPct = stock.predicted_return_pct;
              return (
                <tr
                  key={stock.ticker}
                  onClick={() => onSelectStock(stock.ticker)}
                  className="cursor-pointer transition-colors hover:bg-slate-800/40"
                >
                  {/* Ticker */}
                  <td className="py-3.5 pl-4 pr-3 font-bold text-white">
                    <span className="rounded bg-slate-800/60 px-2 py-0.5 text-xs text-emerald-400 font-mono border border-slate-700/50">
                      {stock.ticker}
                    </span>
                  </td>

                  {/* Company Name */}
                  <td className="px-3 py-3.5 font-medium text-slate-200">
                    <div className="max-w-[200px] truncate" title={stock.company_name ?? ''}>
                      {stock.company_name ?? (
                        <span className="text-slate-500 italic">Name Unavailable</span>
                      )}
                    </div>
                  </td>

                  {/* Sector */}
                  <td className="hidden md:table-cell px-3 py-3.5 text-xs text-slate-400">
                    <div className="max-w-[150px] truncate" title={stock.sector ?? ''}>
                      {stock.sector ?? '—'}
                    </div>
                  </td>

                  {/* Latest Close */}
                  <td className="px-3 py-3.5 text-right font-semibold text-slate-200">
                    ${stock.latest_close != null ? stock.latest_close.toFixed(2) : '—'}
                  </td>

                  {/* Predicted 5D Price */}
                  <td className="px-3 py-3.5 text-right font-semibold text-sky-300">
                    {stock.predicted_price_5d != null
                      ? `$${stock.predicted_price_5d.toFixed(2)}`
                      : <span className="text-slate-600 font-normal">N/A</span>}
                  </td>

                  {/* Expected Return */}
                  <td className="px-3 py-3.5 text-right font-bold">
                    {retPct != null ? (
                      <span className={retPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        {retPct >= 0 ? `+${retPct.toFixed(2)}%` : `${retPct.toFixed(2)}%`}
                      </span>
                    ) : (
                      <span className="text-slate-600 font-normal">N/A</span>
                    )}
                  </td>

                  {/* Probability UP */}
                  <td className="px-3 py-3.5 text-center">
                    {stock.probability_up != null ? (
                      <div className="inline-flex items-center gap-1.5">
                        <span className="font-semibold text-slate-100">
                          {stock.probability_up.toFixed(1)}%
                        </span>
                        <div className="w-12 h-1.5 rounded-full bg-slate-800 overflow-hidden hidden sm:block">
                          <div
                            className="h-full bg-emerald-400"
                            style={{ width: `${stock.probability_up}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      <span className="text-slate-600">N/A</span>
                    )}
                  </td>

                  {/* Direction */}
                  <td className="px-3 py-3.5 text-center">
                    <PredictionBadge
                      direction={stock.predicted_direction}
                      confidence={stock.confidence_label}
                      size="sm"
                    />
                  </td>

                  {/* Confidence */}
                  <td className="hidden lg:table-cell px-3 py-3.5 text-xs text-slate-300">
                    {stock.confidence_label ? (
                      <span
                        className={`rounded px-2 py-0.5 text-[11px] font-medium border ${
                          stock.confidence_label.includes('High')
                            ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
                            : 'bg-slate-800/60 border-slate-700/40 text-slate-400'
                        }`}
                      >
                        {stock.confidence_label}
                      </span>
                    ) : (
                      <span className="text-slate-600">Insufficient Data</span>
                    )}
                  </td>

                  {/* Action Link */}
                  <td className="py-3.5 pl-2 pr-4 text-center">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectStock(stock.ticker);
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
  );
};

