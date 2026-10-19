import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { AllStocks } from './pages/AllStocks';
import { MarketPredictions } from './pages/MarketPredictions';
import { LeaderboardPage } from './pages/LeaderboardPage';
import { ModelPerformance } from './pages/ModelPerformance';
import { ValidationReport } from './pages/ValidationReport';
import { StockDetails } from './pages/StockDetails';

export default function App() {
  const [currentTab, setCurrentTab] = useState<string>('dashboard');
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [prevTab, setPrevTab] = useState<string>('dashboard');

  const handleSelectStock = (ticker: string) => {
    setSelectedTicker(ticker);
    setPrevTab(currentTab === 'stock_details' ? prevTab : currentTab);
    setCurrentTab('stock_details');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleBackFromDetails = () => {
    setCurrentTab(prevTab || 'all_stocks');
  };

  const handleTabChange = (tab: string) => {
    setCurrentTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Top Header / Navigation */}
      <Navbar
        currentTab={currentTab}
        onTabChange={handleTabChange}
        onSelectTicker={handleSelectStock}
        latestDataDate="2026-02-19"
        activePredictionsCount={471}
      />

      {/* Main Content Area */}
      <main className="flex-1 mx-auto max-w-7xl w-full px-4 sm:px-6 pt-6 sm:pt-8">
        {currentTab === 'dashboard' && (
          <Dashboard
            onSelectStock={handleSelectStock}
            onNavigateTab={handleTabChange}
          />
        )}

        {currentTab === 'all_stocks' && (
          <AllStocks onSelectStock={handleSelectStock} />
        )}

        {currentTab === 'predictions' && (
          <MarketPredictions onSelectStock={handleSelectStock} />
        )}

        {currentTab === 'leaderboard' && (
          <LeaderboardPage onSelectStock={handleSelectStock} />
        )}

        {currentTab === 'performance' && (
          <ModelPerformance />
        )}

        {currentTab === 'validation' && (
          <ValidationReport />
        )}

        {currentTab === 'stock_details' && selectedTicker && (
          <StockDetails
            ticker={selectedTicker}
            onBack={handleBackFromDetails}
            onSelectStock={handleSelectStock}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/90 py-6 text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">Stock Trend Forecaster</span>
            <span>•</span>
            <span>Multi-Ticker PyTorch LSTM 5-Day Horizon Engine</span>
          </div>

          <div className="text-slate-500 text-center sm:text-right">
            <span>Research & Visualization Only • Not Financial or Trading Advice</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
