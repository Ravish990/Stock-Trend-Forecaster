import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceDot,
  CartesianGrid,
} from 'recharts';
import { StockHistoryItem } from '../types/stock';
import { Calendar, TrendingUp } from 'lucide-react';

interface StockPriceChartProps {
  ticker: string;
  history: StockHistoryItem[];
  forecastPoint?: StockHistoryItem | null;
  latestClose: number;
}

export const StockPriceChart: React.FC<StockPriceChartProps> = ({
  ticker,
  history,
  forecastPoint,
  latestClose,
}) => {
  const [range, setRange] = useState<'30D' | '60D' | '90D' | 'ALL'>('60D');

  // Filter history based on range
  const filteredHistory = React.useMemo(() => {
    if (!history || history.length === 0) return [];
    if (range === '30D') return history.slice(-30);
    if (range === '60D') return history.slice(-60);
    if (range === '90D') return history.slice(-90);
    return history;
  }, [history, range]);

  // Combine historical data with the future 5-day forecast point
  const chartData = React.useMemo(() => {
    const data = filteredHistory.map((item) => ({
      date: item.date,
      close: item.close,
      predicted_price: item.predicted_price ?? null,
      forecast_close: null as number | null,
      is_forecast: false,
    }));

    if (forecastPoint && forecastPoint.close != null) {
      // Connect latest historical close to the future forecast close with a dotted target
      if (data.length > 0) {
        data[data.length - 1].forecast_close = data[data.length - 1].close;
      }
      data.push({
        date: `${forecastPoint.date} (5D)`,
        close: null as any,
        predicted_price: null,
        forecast_close: forecastPoint.close,
        is_forecast: true,
      });
    }

    return data;
  }, [filteredHistory, forecastPoint]);

  const minPrice = React.useMemo(() => {
    const prices = filteredHistory.map((d) => d.close);
    if (forecastPoint?.close) prices.push(forecastPoint.close);
    const min = Math.min(...prices);
    return Math.floor(min * 0.95);
  }, [filteredHistory, forecastPoint]);

  const maxPrice = React.useMemo(() => {
    const prices = filteredHistory.map((d) => d.close);
    if (forecastPoint?.close) prices.push(forecastPoint.close);
    const max = Math.max(...prices);
    return Math.ceil(max * 1.05);
  }, [filteredHistory, forecastPoint]);

  const hasHistoricalPredictions = filteredHistory.some((h) => h.predicted_price != null);

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-bold text-white tracking-tight">
              {ticker} Price Trajectory & 5-Day Horizon
            </h3>
            {forecastPoint && (
              <span className="rounded-md bg-sky-500/15 border border-sky-500/30 px-2 py-0.5 text-[11px] font-semibold text-sky-300">
                Forecast Target: ${forecastPoint.close.toFixed(2)}
              </span>
            )}
          </div>
          <div className="text-xs text-slate-400 mt-0.5 flex items-center gap-1.5">
            <Calendar className="h-3.5 w-3.5 text-slate-500" />
            <span>Actual historical closing prices vs isolated 5-trading-day model prediction</span>
          </div>
        </div>

        {/* Date Range Controls */}
        <div className="flex items-center gap-1 rounded-xl bg-slate-950 p-1 border border-slate-800">
          {(['30D', '60D', '90D', 'ALL'] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                range === r
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />

            <XAxis
              dataKey="date"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
              minTickGap={25}
            />

            <YAxis
              domain={[minPrice, maxPrice]}
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
              tickFormatter={(v) => `$${v}`}
            />

            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0]?.payload;
                return (
                  <div className="rounded-xl border border-slate-700 bg-slate-950/95 p-3 text-xs shadow-xl backdrop-blur-md">
                    <div className="font-semibold text-slate-300 border-b border-slate-800 pb-1 mb-1.5">
                      {label}
                    </div>
                    {d.close != null && (
                      <div className="flex items-center justify-between gap-4 py-0.5">
                        <span className="text-slate-400">Actual Close:</span>
                        <span className="font-bold text-emerald-400">${Number(d.close).toFixed(2)}</span>
                      </div>
                    )}
                    {d.forecast_close != null && d.is_forecast && (
                      <div className="flex items-center justify-between gap-4 py-0.5">
                        <span className="text-sky-300 font-medium">5-Day Forecast:</span>
                        <span className="font-bold text-sky-300">
                          ${Number(d.forecast_close).toFixed(2)}
                        </span>
                      </div>
                    )}
                    {d.predicted_price != null && (
                      <div className="flex items-center justify-between gap-4 py-0.5">
                        <span className="text-amber-300">Historical Model Est:</span>
                        <span className="font-bold text-amber-300">
                          ${Number(d.predicted_price).toFixed(2)}
                        </span>
                      </div>
                    )}
                  </div>
                );
              }}
            />

            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ paddingBottom: '10px', fontSize: '11px' }}
            />

            {/* Actual Historical Price Line & Area */}
            <Area
              type="monotone"
              dataKey="close"
              name="Actual Close"
              stroke="#10b981"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#priceGradient)"
              connectNulls={false}
            />

            {/* Historical Model Forecast (if available, e.g. for AAPL) */}
            {hasHistoricalPredictions && (
              <Line
                type="monotone"
                dataKey="predicted_price"
                name="Historical Forecast"
                stroke="#f59e0b"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
                connectNulls={false}
              />
            )}

            {/* 5-Day Forward Forecast Line */}
            {forecastPoint && (
              <Line
                type="linear"
                dataKey="forecast_close"
                name="5-Day Forecast Projection"
                stroke="#38bdf8"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ r: 5, fill: '#38bdf8', strokeWidth: 2, stroke: '#0284c7' }}
                connectNulls={true}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 text-[11px] text-slate-500 flex items-center justify-between">
        <span>* Dashed blue line projects from the latest historical close directly to the 5-trading-day estimate.</span>
        <span>Latest Close: ${latestClose.toFixed(2)}</span>
      </div>
    </div>
  );
};

