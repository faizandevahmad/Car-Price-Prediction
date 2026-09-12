import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  api,
  type CarFeatures,
  type FeedbackResponse,
  type HistoryItem,
  type ModelInfo,
  type PredictionResponse,
} from "./api";

/** Format PKR for display using lac / crore (UI only; values stay numeric underneath). */
function money(n: number): string {
  if (!Number.isFinite(n)) return "—";
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);

  const trim = (value: number) => {
    const fixed = value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
    return fixed.replace(/\.0+$/, "").replace(/(\.\d)0$/, "$1");
  };

  if (abs >= 10_000_000) {
    return `${sign}Rs ${trim(abs / 10_000_000)} crore`;
  }
  if (abs >= 100_000) {
    return `${sign}Rs ${trim(abs / 100_000)} lac`;
  }
  return `${sign}Rs ${Math.round(abs).toLocaleString("en-PK")}`;
}

/** Exact PKR with Pakistani-style grouping, e.g. Rs 35,45,313 */
function moneyExact(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return `Rs ${Math.round(n).toLocaleString("en-IN")}`;
}

const defaultForm = (info?: ModelInfo | null): CarFeatures => ({
  make: info?.makes?.[0] ?? "Toyota",
  model: info?.models_by_make?.[info?.makes?.[0] ?? "Toyota"]?.[0] ?? "Camry",
  year: 2019,
  mileage: 45000,
  condition: "good",
  fuel_type: "gasoline",
  transmission: "automatic",
  body_type: "sedan",
  accident_history: false,
  owners: 1,
});

export default function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [form, setForm] = useState<CarFeatures>(defaultForm());
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [actualPrice, setActualPrice] = useState("");
  const [notes, setNotes] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [tuneResult, setTuneResult] = useState<FeedbackResponse | null>(null);
  const [loading, setLoading] = useState<"boot" | "predict" | "tune" | "train" | null>("boot");
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState(false);

  const models = useMemo(
    () => info?.models_by_make?.[form.make] ?? [],
    [info, form.make],
  );

  async function refreshMeta() {
    const [modelInfo, hist] = await Promise.all([api.info(), api.history(10)]);
    setInfo(modelInfo);
    setHistory(hist.items);
    setHistoryTotal(hist.total);
    setOnline(true);
    return modelInfo;
  }

  useEffect(() => {
    (async () => {
      try {
        const modelInfo = await refreshMeta();
        setForm((prev) => {
          const next = defaultForm(modelInfo);
          return { ...next, ...prev, make: prev.make || next.make, model: prev.model || next.model };
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Cannot reach FarePulse API. Start the backend on port 8000.");
        setOnline(false);
      } finally {
        setLoading(null);
      }
    })();
  }, []);

  function updateField<K extends keyof CarFeatures>(key: K, value: CarFeatures[K]) {
    setForm((prev) => {
      const next = { ...prev, [key]: value };
      if (key === "make" && info) {
        const list = info.models_by_make[value as string] ?? [];
        next.model = list[0] ?? "";
      }
      return next;
    });
    setTuneResult(null);
  }

  async function onPredict(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading("predict");
    setTuneResult(null);
    try {
      const result = await api.predict(form);
      setPrediction(result);
      setActualPrice(String(Math.round(result.predicted_price)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(null);
    }
  }

  async function onFineTune(e: FormEvent) {
    e.preventDefault();
    const price = Number(actualPrice);
    if (!Number.isFinite(price) || price < 139_000) {
      setError("Enter a valid actual sale price (at least Rs 139,000).");
      return;
    }
    setError(null);
    setLoading("tune");
    try {
      const result = await api.feedback(form, price, notes.trim() || undefined);
      setTuneResult(result);
      const modelInfo = await refreshMeta();
      setInfo(modelInfo);
      // Refresh prediction with updated model
      const nextPred = await api.predict(form);
      setPrediction(nextPred);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Fine-tune failed");
    } finally {
      setLoading(null);
    }
  }

  async function onRetrain() {
    setError(null);
    setLoading("train");
    try {
      await api.train(false);
      await refreshMeta();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retrain failed");
    } finally {
      setLoading(null);
    }
  }

  const importance = prediction
    ? Object.entries(prediction.feature_importance).slice(0, 6)
    : [];

  return (
    <div className="page">
      <div className="road-glow" aria-hidden />
      <header className="topbar">
        <div className="brand-block">
          <p className="brand">FarePulse</p>
          <p className="tagline">Live used-car market pricing that learns from every sale.</p>
        </div>
        <div className="status-chip" data-online={online}>
          <span className="dot" />
          {online ? `Model v${info?.model_version ?? "—"} · ${info?.samples_used?.toLocaleString() ?? "—"} samples` : "API offline"}
        </div>
      </header>

      <main className="layout">
        <section className="hero-panel">
          <h1>Price it with confidence.</h1>
          <p>
            Enter the car details, get a market estimate, then teach FarePulse with the real
            sale price so the next prediction gets sharper.
          </p>
          <div className="metric-row">
            <div>
              <span className="label">MAE</span>
              <strong>{info?.metrics?.mae != null ? money(info.metrics.mae) : "—"}</strong>
            </div>
            <div>
              <span className="label">R²</span>
              <strong>{info?.metrics?.r2 != null ? info.metrics.r2.toFixed(3) : "—"}</strong>
            </div>
            <div>
              <span className="label">Feedbacks</span>
              <strong>{historyTotal}</strong>
            </div>
          </div>
        </section>

        <section className="form-panel">
          <div className="panel-head">
            <h2>Vehicle profile</h2>
            <button type="button" className="ghost" onClick={onRetrain} disabled={loading !== null}>
              {loading === "train" ? "Training…" : "Retrain model"}
            </button>
          </div>

          <form className="grid-form" onSubmit={onPredict}>
            <label>
              Make
              <select value={form.make} onChange={(e) => updateField("make", e.target.value)} disabled={!info}>
                {(info?.makes ?? []).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </label>
            <label>
              Model
              <select value={form.model} onChange={(e) => updateField("model", e.target.value)} disabled={!info}>
                {models.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </label>
            <label>
              Year
              <input
                type="number"
                min={1995}
                max={2026}
                value={form.year}
                onChange={(e) => updateField("year", Number(e.target.value))}
              />
            </label>
            <label>
              Mileage
              <input
                type="number"
                min={0}
                max={400000}
                step={500}
                value={form.mileage}
                onChange={(e) => updateField("mileage", Number(e.target.value))}
              />
            </label>
            <label>
              Condition
              <select value={form.condition} onChange={(e) => updateField("condition", e.target.value)}>
                {(info?.conditions ?? ["excellent", "good", "fair", "poor"]).map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <label>
              Fuel
              <select value={form.fuel_type} onChange={(e) => updateField("fuel_type", e.target.value)}>
                {(info?.fuel_types ?? []).map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <label>
              Transmission
              <select value={form.transmission} onChange={(e) => updateField("transmission", e.target.value)}>
                {(info?.transmissions ?? []).map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <label>
              Body
              <select value={form.body_type} onChange={(e) => updateField("body_type", e.target.value)}>
                {(info?.body_types ?? []).map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
            <label>
              Previous owners
              <input
                type="number"
                min={1}
                max={8}
                value={form.owners}
                onChange={(e) => updateField("owners", Number(e.target.value))}
              />
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={form.accident_history}
                onChange={(e) => updateField("accident_history", e.target.checked)}
              />
              Accident history
            </label>

            <button className="primary span-2" type="submit" disabled={loading !== null || !online}>
              {loading === "predict" ? "Estimating…" : loading === "boot" ? "Loading…" : "Get market estimate"}
            </button>
          </form>
        </section>

        <section className="result-panel" aria-live="polite">
          <h2>Estimate</h2>
          {!prediction ? (
            <p className="empty">Fill in the vehicle and run an estimate to see price, range, and drivers.</p>
          ) : (
            <>
              <p className="price">{money(prediction.predicted_price)}</p>
              <p className="price-exact">{moneyExact(prediction.predicted_price)}</p>
              <p className="range">
                Likely range {money(prediction.price_low)} – {money(prediction.price_high)}
              </p>
              <div className="confidence">
                <div className="bar">
                  <span style={{ width: `${prediction.confidence * 100}%` }} />
                </div>
                <small>{Math.round(prediction.confidence * 100)}% confidence · model v{prediction.model_version}</small>
              </div>
              <p className="advice">{prediction.advice}</p>
              <div className="importance">
                <h3>What moved the price</h3>
                <ul>
                  {importance.map(([name, value]) => (
                    <li key={name}>
                      <span>{name.replaceAll("_", " ")}</span>
                      <div className="mini-bar">
                        <i style={{ width: `${value * 100}%` }} />
                      </div>
                      <em>{Math.round(value * 100)}%</em>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </section>

        <section className="tune-panel">
          <h2>Fine-tune with a real sale</h2>
          <p className="help">
            Sold or bought this car? Submit the actual price. FarePulse appends it to the training
            set and re-trains automatically.
          </p>
          <form className="tune-form" onSubmit={onFineTune}>
            <label>
              Actual price (PKR)
              <input
                type="number"
                min={139000}
                step={10000}
                value={actualPrice}
                onChange={(e) => setActualPrice(e.target.value)}
                placeholder="e.g. 5143000"
                required
              />
            </label>
            <label>
              Notes (optional)
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Private sale, dealer trade-in…"
              />
            </label>
            <button className="accent" type="submit" disabled={loading !== null || !online}>
              {loading === "tune" ? "Learning…" : "Submit & fine-tune"}
            </button>
          </form>
          {tuneResult && (
            <div className="tune-toast">
              <strong>{tuneResult.message}</strong>
              <p>
                Now on model v{tuneResult.model_version} with {tuneResult.samples_used.toLocaleString()} samples.
                MAE {tuneResult.previous_metrics.mae != null ? money(tuneResult.previous_metrics.mae) : "—"} →{" "}
                {tuneResult.metrics.mae != null ? money(tuneResult.metrics.mae) : "—"}.
              </p>
            </div>
          )}
        </section>

        <section className="history-panel">
          <h2>Recent feedback</h2>
          {history.length === 0 ? (
            <p className="empty">No sale feedback yet — be the first to teach the model.</p>
          ) : (
            <ul className="history-list">
              {history.map((item) => (
                <li key={item.id}>
                  <div>
                    <strong>
                      {item.year} {item.make} {item.model}
                    </strong>
                    <small>{new Date(item.timestamp).toLocaleString()}</small>
                  </div>
                  <div className="hist-prices">
                    <span>
                      Pred {item.predicted_price != null ? money(item.predicted_price) : "—"}
                    </span>
                    <span className="actual">Actual {money(item.actual_price)}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>

      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <footer className="footer">
        FarePulse · Gradient boosting on synthetic market data + your live sale feedback
      </footer>
    </div>
  );
}
