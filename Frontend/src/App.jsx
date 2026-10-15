import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [tickers, setTickers] = useState([]);
  const [ticker, setTicker] = useState("");
  const [date, setDate] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get(`${API}/tickers`).then((r) => setTickers(r.data)).catch(() => {});
  }, []);

  async function handlePredict(e) {
    e.preventDefault();
    if (!ticker) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const { data } = await axios.post(`${API}/predict`, {
        ticker: ticker.toUpperCase(),
        date: date || "latest",
      });
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  const isUp = result?.direction === "Up";

  return (
    <div className="app">
      <header>
        <h1>📈 Stock Trend Forecaster</h1>
        <p className="subtitle">5-day ahead direction &amp; price prediction</p>
      </header>

      <form onSubmit={handlePredict} className="form">
        <div className="inputs">
          <input
            list="ticker-list"
            placeholder="Ticker (e.g. AAPL)"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            className="input"
            required
          />
          <datalist id="ticker-list">
            {tickers.map((t) => <option key={t} value={t} />)}
          </datalist>

          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="input"
            title="Leave blank to use latest available date"
          />
        </div>
        <button type="submit" disabled={loading} className="btn">
          {loading ? "Predicting…" : "Predict"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className={`card ${isUp ? "up" : "down"}`}>
          <div className="card-header">
            <span className="ticker-label">{result.ticker}</span>
            <span className={`direction-badge ${isUp ? "up" : "down"}`}>
              {isUp ? "▲ UP" : "▼ DOWN"}
            </span>
          </div>

          {result.warning && <div className="warning">{result.warning}</div>}

          <div className="dates">
            <span>As of: <strong>{result.as_of_date}</strong></span>
            <span>→ Target ({result.horizon_days}d): <strong>{result.target_date}</strong></span>
          </div>

          <div className="metrics">
            <Metric label="Current Price" value={`$${result.current_price.toFixed(2)}`} />
            <Metric label="Predicted Price" value={`$${result.predicted_price.toFixed(2)}`} highlight />
            <Metric
              label="Expected Return"
              value={`${result.predicted_log_return > 0 ? "+" : ""}${result.predicted_log_return.toFixed(3)}%`}
              color={result.predicted_log_return >= 0 ? "green" : "red"}
            />
            <Metric
              label="Confidence"
              value={`${isUp ? result.prob_up : (100 - result.prob_up).toFixed(2)}%`}
              sub={`P(Up) = ${result.prob_up}%`}
            />
          </div>

          <div className="confidence-bar-wrap">
            <div className="confidence-bar" style={{ width: `${result.prob_up}%` }} />
            <span className="bar-label">{result.prob_up}% Up</span>
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, highlight, color, sub }) {
  return (
    <div className={`metric ${highlight ? "highlight" : ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={color ? { color } : {}}>
        {value}
      </span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  );
}
