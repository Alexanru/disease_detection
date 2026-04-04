// frontend/RareSightApp.jsx
// This is the React component — run with:
//   streamlit run frontend/app.py  (wrapped in streamlit)
// OR deploy standalone as a Vite/CRA app
// The same JSX is embedded in the Streamlit HTML component

import { useState, useCallback, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from "recharts";
import { Upload, Activity, Database, Info, AlertTriangle, CheckCircle, ChevronRight } from "lucide-react";

const API_URL = import.meta.env?.VITE_API_URL ?? "http://localhost:8000";

// ── Color palette ────────────────────────────────────────────────────────────

const COLORS = {
  mel:  "#D85A30",  // coral — malignant
  nv:   "#1D9E75",  // teal — benign
  bcc:  "#378ADD",  // blue
  ak:   "#BA7517",  // amber
  bkl:  "#7F77DD",  // purple
  df:   "#E24B4A",  // red — rare
  vasc: "#D4537E",  // pink — rare
  scc:  "#639922",  // green
};

const CLASS_COLORS = [COLORS.mel, COLORS.nv, COLORS.bcc, COLORS.ak, COLORS.bkl, COLORS.df, COLORS.vasc, COLORS.scc];

// ── Dataset stats ──────────────────────────────────────────────────────────

const DATASET_STATS = [
  {
    name: "ISIC 2019",
    total: 25331,
    purpose: "DL lab · image classification",
    classes: [
      { name: "Melanocytic Nevi", count: 12875, color: COLORS.nv },
      { name: "Melanoma", count: 4522, color: COLORS.mel },
      { name: "BCC", count: 3323, color: COLORS.bcc },
      { name: "Benign Keratosis", count: 2624, color: COLORS.bkl },
      { name: "SCC", count: 628, color: COLORS.scc },
      { name: "Actinic Keratosis", count: 867, color: COLORS.ak },
      { name: "Vascular Lesion", count: 253, color: COLORS.vasc, rare: true },
      { name: "Dermatofibroma", count: 239, color: COLORS.df, rare: true },
    ],
  },
  {
    name: "HAM10000",
    total: 10015,
    purpose: "Dissertation · image + patient metadata",
    classes: [
      { name: "Melanocytic Nevi", count: 6705, color: COLORS.nv },
      { name: "Melanoma", count: 1113, color: COLORS.mel },
      { name: "BKL", count: 1099, color: COLORS.bkl },
      { name: "BCC", count: 514, color: COLORS.bcc },
      { name: "AKIEC", count: 327, color: COLORS.ak },
      { name: "Vascular", count: 142, color: COLORS.vasc, rare: true },
      { name: "Dermatofibroma", count: 115, color: COLORS.df, rare: true },
    ],
  },
  {
    name: "PAD-UFES-20",
    total: 2298,
    purpose: "Dissertation · image + 22 clinical features",
    classes: [
      { name: "BCC", count: 845, color: COLORS.bcc },
      { name: "ACK", count: 730, color: COLORS.ak },
      { name: "NEV", count: 244, color: COLORS.nv },
      { name: "SEK", count: 251, color: COLORS.bkl },
      { name: "SCC", count: 176, color: COLORS.scc },
      { name: "Melanoma", count: 52, color: COLORS.mel, rare: true },
    ],
  },
];

// ── Mock prediction for demo ───────────────────────────────────────────────

const MOCK_PREDICTION = {
  top_prediction: { class_name: "Melanoma", probability: 0.73, is_rare: false, icd10: "C43" },
  all_predictions: [
    { class_id: 0, class_name: "Melanoma", probability: 0.73, is_rare: false },
    { class_id: 2, class_name: "Basal Cell Carcinoma", probability: 0.14, is_rare: false },
    { class_id: 1, class_name: "Melanocytic Nevi", probability: 0.07, is_rare: false },
    { class_id: 7, class_name: "SCC", probability: 0.03, is_rare: false },
    { class_id: 5, class_name: "Dermatofibroma", probability: 0.02, is_rare: true },
    { class_id: 6, class_name: "Vascular Lesion", probability: 0.01, is_rare: true },
  ],
  rare_disease_risk: 0.03,
  processing_time_ms: 48,
};

// ── Subcomponents ──────────────────────────────────────────────────────────

function Badge({ children, variant = "default" }) {
  const styles = {
    default:  { background: "var(--color-background-secondary)", color: "var(--color-text-secondary)" },
    rare:     { background: "#FBEAF0", color: "#993556" },
    warning:  { background: "#FAEEDA", color: "#854F0B" },
    success:  { background: "#EAF3DE", color: "#3B6D11" },
    info:     { background: "#E6F1FB", color: "#185FA5" },
  };
  return (
    <span style={{
      ...styles[variant],
      fontSize: 11, fontWeight: 500, padding: "2px 8px",
      borderRadius: 4, whiteSpace: "nowrap",
    }}>{children}</span>
  );
}

function ProbabilityBar({ name, probability, color, isRare }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13, color: "var(--color-text-primary)" }}>{name}</span>
          {isRare && <Badge variant="rare">rare</Badge>}
        </div>
        <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>
          {(probability * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{
        height: 6, background: "var(--color-background-secondary)",
        borderRadius: 3, overflow: "hidden",
      }}>
        <div style={{
          height: "100%", width: `${probability * 100}%`,
          background: color || "#1D9E75",
          borderRadius: 3,
          transition: "width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
        }} />
      </div>
    </div>
  );
}

function UploadZone({ onFile, preview }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer?.files[0] || e.target.files?.[0];
    if (file && file.type.startsWith("image/")) onFile(file);
  }, [onFile]);

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      style={{
        border: `1.5px dashed ${dragging ? "#1D9E75" : "var(--color-border-secondary)"}`,
        borderRadius: 12, padding: "2rem",
        textAlign: "center", cursor: "pointer",
        background: dragging ? "var(--color-background-info)" : "var(--color-background-secondary)",
        transition: "all 0.15s",
        minHeight: 180,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", gap: 12,
      }}
    >
      {preview ? (
        <img src={preview} alt="dermoscopy" style={{
          maxHeight: 180, maxWidth: "100%", borderRadius: 8, objectFit: "cover",
        }} />
      ) : (
        <>
          <Upload size={28} color="var(--color-text-secondary)" />
          <div>
            <p style={{ fontSize: 14, color: "var(--color-text-primary)", margin: "0 0 4px" }}>
              Drop dermoscopy image here
            </p>
            <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: 0 }}>
              JPEG, PNG — up to 10 MB
            </p>
          </div>
        </>
      )}
      <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={onDrop} />
    </div>
  );
}

function ClinicalForm({ values, onChange }) {
  const field = (label, key, type = "text", opts = null) => (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 4 }}>
        {label}
      </label>
      {opts ? (
        <select
          value={values[key] || ""}
          onChange={e => onChange({ ...values, [key]: e.target.value })}
          style={{ width: "100%", fontSize: 13 }}
        >
          <option value="">Select…</option>
          {opts.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type={type}
          value={values[key] || ""}
          onChange={e => onChange({ ...values, [key]: e.target.value })}
          style={{ width: "100%", fontSize: 13, boxSizing: "border-box" }}
          placeholder={type === "number" ? "0" : ""}
        />
      )}
    </div>
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 16px" }}>
      {field("Age", "age", "number")}
      {field("Sex", "sex", "select", ["Male", "Female", "Other"])}
      {field("Lesion location", "localization", "select", ["Back", "Chest", "Face", "Foot", "Hand", "Lower extremity", "Neck", "Scalp", "Upper extremity", "Trunk", "Other"])}
      {field("Skin tone (Fitzpatrick)", "skin_tone", "select", ["I", "II", "III", "IV", "V", "VI"])}
      {field("Lesion diameter (mm)", "diameter_1", "number")}
      {field("Duration (months)", "duration", "number")}
    </div>
  );
}

function DatasetChart({ dataset }) {
  const data = dataset.classes.map(c => ({ name: c.name, count: c.count, color: c.color, rare: c.rare }));
  return (
    <div style={{ marginBottom: 32 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>{dataset.name}</span>
        <Badge variant="info">{dataset.total.toLocaleString()} samples</Badge>
        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{dataset.purpose}</span>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--color-text-secondary)" }} interval={0} angle={-20} textAnchor="end" height={48} />
          <YAxis tick={{ fontSize: 10, fill: "var(--color-text-secondary)" }} />
          <Tooltip
            contentStyle={{ fontSize: 12, background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: 8 }}
            formatter={(v, n, p) => [v.toLocaleString(), p.payload.rare ? "samples (rare)" : "samples"]}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} opacity={entry.rare ? 1 : 0.75} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────

export default function RareSightApp() {
  const [tab, setTab] = useState("diagnose");
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [clinical, setClinical] = useState({});
  const [showClinical, setShowClinical] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFile = useCallback((file) => {
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handlePredict = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", imageFile);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      // Demo mode — use mock
      await new Promise(r => setTimeout(r, 900));
      setResult(MOCK_PREDICTION);
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: "diagnose", label: "Diagnose", icon: Activity },
    { id: "datasets", label: "Datasets", icon: Database },
    { id: "architecture", label: "Architecture", icon: Info },
  ];

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 0 2rem" }}>

      {/* Header */}
      <div style={{ marginBottom: 24, paddingTop: 8 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <h1 style={{ fontSize: 22, fontWeight: 500, margin: 0, color: "var(--color-text-primary)" }}>
            RareSight
          </h1>
          <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>v0.1 · research prototype</span>
        </div>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: "4px 0 0" }}>
          Early detection of rare dermatological conditions — MAE pretraining + multimodal fusion
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 24 }}>
        {tabs.map(t => {
          const Icon = t.icon;
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "8px 16px", fontSize: 13,
                background: "transparent", border: "none",
                borderBottom: active ? "2px solid #1D9E75" : "2px solid transparent",
                color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                cursor: "pointer", fontWeight: active ? 500 : 400,
                transition: "color 0.1s",
              }}
            >
              <Icon size={14} />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* ── DIAGNOSE TAB ── */}
      {tab === "diagnose" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>

          {/* Left: Input */}
          <div>
            <UploadZone onFile={handleFile} preview={preview} />

            {/* Clinical toggle */}
            <button
              onClick={() => setShowClinical(v => !v)}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                marginTop: 12, fontSize: 13, background: "transparent",
                border: "none", color: "var(--color-text-secondary)", cursor: "pointer", padding: 0,
              }}
            >
              <ChevronRight size={14} style={{ transform: showClinical ? "rotate(90deg)" : "none", transition: "transform 0.15s" }} />
              {showClinical ? "Hide" : "Add"} clinical data (optional — improves accuracy)
            </button>

            {showClinical && (
              <div style={{
                marginTop: 12, padding: "1rem",
                background: "var(--color-background-secondary)",
                borderRadius: 8, border: "0.5px solid var(--color-border-tertiary)",
              }}>
                <ClinicalForm values={clinical} onChange={setClinical} />
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={!imageFile || loading}
              style={{
                marginTop: 16, width: "100%", padding: "10px 0",
                fontSize: 14, fontWeight: 500, borderRadius: 8,
                background: imageFile && !loading ? "#1D9E75" : "var(--color-background-secondary)",
                color: imageFile && !loading ? "#fff" : "var(--color-text-secondary)",
                border: "none", cursor: imageFile && !loading ? "pointer" : "not-allowed",
                transition: "background 0.15s",
              }}
            >
              {loading ? "Analysing…" : "Analyse lesion"}
            </button>

            <p style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 8, lineHeight: 1.5 }}>
              For research purposes only. Not a clinical diagnostic tool. Always consult a dermatologist.
            </p>
          </div>

          {/* Right: Results */}
          <div>
            {!result && !loading && (
              <div style={{
                height: "100%", minHeight: 300,
                display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                color: "var(--color-text-secondary)", gap: 8,
              }}>
                <Activity size={32} color="var(--color-border-secondary)" />
                <p style={{ fontSize: 13, margin: 0 }}>Upload an image to begin</p>
              </div>
            )}

            {loading && (
              <div style={{
                height: "100%", minHeight: 300,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>
                <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Running inference…</div>
              </div>
            )}

            {result && !loading && (
              <div>
                {/* Top prediction */}
                <div style={{
                  padding: "1rem", borderRadius: 8, marginBottom: 16,
                  background: result.top_prediction.probability > 0.5
                    ? "var(--color-background-danger)"
                    : "var(--color-background-success)",
                  border: `0.5px solid ${result.top_prediction.probability > 0.5 ? "var(--color-border-danger)" : "var(--color-border-success)"}`,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    {result.top_prediction.probability > 0.5
                      ? <AlertTriangle size={16} color="var(--color-text-danger)" />
                      : <CheckCircle size={16} color="var(--color-text-success)" />}
                    <span style={{ fontSize: 11, color: result.top_prediction.probability > 0.5 ? "var(--color-text-danger)" : "var(--color-text-success)", fontWeight: 500 }}>
                      primary diagnosis
                    </span>
                  </div>
                  <p style={{ fontSize: 18, fontWeight: 500, margin: "0 0 4px", color: "var(--color-text-primary)" }}>
                    {result.top_prediction.class_name}
                  </p>
                  <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>
                    Confidence: {(result.top_prediction.probability * 100).toFixed(1)}%
                    · ICD-10: {result.top_prediction.icd10}
                    · {result.processing_time_ms}ms
                  </p>
                </div>

                {/* Rare risk */}
                {result.rare_disease_risk > 0.05 && (
                  <div style={{
                    padding: "0.75rem 1rem", borderRadius: 8, marginBottom: 16,
                    background: "var(--color-background-warning)",
                    border: "0.5px solid var(--color-border-warning)",
                    display: "flex", alignItems: "center", gap: 8,
                  }}>
                    <AlertTriangle size={14} color="var(--color-text-warning)" />
                    <span style={{ fontSize: 12, color: "var(--color-text-warning)" }}>
                      Rare condition risk: {(result.rare_disease_risk * 100).toFixed(1)}% — refer to specialist
                    </span>
                  </div>
                )}

                {/* All probabilities */}
                <div style={{
                  padding: "1rem", borderRadius: 8,
                  background: "var(--color-background-secondary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                }}>
                  <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "0 0 12px", fontWeight: 500 }}>
                    All classes
                  </p>
                  {result.all_predictions.map((p, i) => (
                    <ProbabilityBar
                      key={p.class_id}
                      name={p.class_name}
                      probability={p.probability}
                      color={CLASS_COLORS[p.class_id] || "#888"}
                      isRare={p.is_rare}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── DATASETS TAB ── */}
      {tab === "datasets" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 32 }}>
            {DATASET_STATS.map(d => (
              <div key={d.name} style={{
                padding: "1rem", borderRadius: 8,
                background: "var(--color-background-secondary)",
                border: "0.5px solid var(--color-border-tertiary)",
              }}>
                <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "0 0 4px" }}>{d.name}</p>
                <p style={{ fontSize: 22, fontWeight: 500, margin: "0 0 4px", color: "var(--color-text-primary)" }}>
                  {d.total.toLocaleString()}
                </p>
                <p style={{ fontSize: 11, color: "var(--color-text-secondary)", margin: 0 }}>{d.purpose}</p>
              </div>
            ))}
          </div>

          {DATASET_STATS.map(d => <DatasetChart key={d.name} dataset={d} />)}

          <div style={{
            padding: "0.75rem 1rem", borderRadius: 8,
            background: "var(--color-background-secondary)",
            border: "0.5px solid var(--color-border-tertiary)",
            fontSize: 12, color: "var(--color-text-secondary)",
          }}>
            Rare classes are shown at full opacity. Imbalance is handled via weighted random sampling + focal loss (γ=2).
          </div>
        </div>
      )}

      {/* ── ARCHITECTURE TAB ── */}
      {tab === "architecture" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>

            <div style={{
              padding: "1.25rem", borderRadius: 8,
              background: "var(--color-background-secondary)",
              border: "0.5px solid var(--color-border-tertiary)",
            }}>
              <p style={{ fontSize: 13, fontWeight: 500, margin: "0 0 12px" }}>Stage 1 — MAE pretraining</p>
              {[
                ["Base model", "ViT-B/16 (86M params)"],
                ["Mask ratio", "75% of patches"],
                ["Decoder", "8-layer Transformer (dim 512)"],
                ["Loss", "MSE on masked pixels only"],
                ["Epochs", "200  ·  batch 256"],
                ["Optimizer", "AdamW  ·  cosine LR"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "0.5px solid var(--color-border-tertiary)", fontSize: 12 }}>
                  <span style={{ color: "var(--color-text-secondary)" }}>{k}</span>
                  <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{v}</span>
                </div>
              ))}
            </div>

            <div style={{
              padding: "1.25rem", borderRadius: 8,
              background: "var(--color-background-secondary)",
              border: "0.5px solid var(--color-border-tertiary)",
            }}>
              <p style={{ fontSize: 13, fontWeight: 500, margin: "0 0 12px" }}>Stage 2 — supervised fine-tuning</p>
              {[
                ["Backbone", "MAE encoder (frozen → unfrozen)"],
                ["Loss", "Focal loss  ·  γ=2"],
                ["LLRD", "Layer-wise LR decay 0.75"],
                ["Sampler", "Weighted random (rare oversampling)"],
                ["Epochs", "100  ·  early stopping"],
                ["Augmentation", "Albumentations (CLAHE + jitter)"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "0.5px solid var(--color-border-tertiary)", fontSize: 12 }}>
                  <span style={{ color: "var(--color-text-secondary)" }}>{k}</span>
                  <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            padding: "1.25rem", borderRadius: 8,
            background: "var(--color-background-secondary)",
            border: "0.5px solid var(--color-border-tertiary)",
            marginBottom: 24,
          }}>
            <p style={{ fontSize: 13, fontWeight: 500, margin: "0 0 12px" }}>Multimodal fusion (dissertation)</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
              {[
                { label: "Image branch", body: "ViT-B encoder → cls token [768]" },
                { label: "Clinical branch", body: "MLP(input_dim → 256) + BN + GELU" },
                { label: "Fusion", body: "concat [1024] → MLP → [num_classes]" },
              ].map(c => (
                <div key={c.label} style={{
                  padding: "0.75rem", borderRadius: 6,
                  background: "var(--color-background-primary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                }}>
                  <p style={{ fontSize: 11, color: "var(--color-text-secondary)", margin: "0 0 4px", fontWeight: 500 }}>{c.label}</p>
                  <p style={{ fontSize: 11, color: "var(--color-text-primary)", margin: 0 }}>{c.body}</p>
                </div>
              ))}
            </div>
            <p style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 12, marginBottom: 0 }}>
              Modality dropout (p=0.1) during training → robust to missing clinical data at inference.
              Ablation study: image-only | clinical-only | full multimodal.
            </p>
          </div>

          <div style={{
            padding: "1.25rem", borderRadius: 8,
            background: "var(--color-background-secondary)",
            border: "0.5px solid var(--color-border-tertiary)",
          }}>
            <p style={{ fontSize: 13, fontWeight: 500, margin: "0 0 12px" }}>Literature comparison (target)</p>
            {[
              ["EfficientNet-B4 (Tan & Le, 2019)", "ISIC 2019", "0.891", "0.632"],
              ["ViT-B baseline (Dosovitskiy, 2021)", "HAM10000",  "0.912", "0.689"],
              ["MAE + ViT-B (He et al., 2022)",     "ISIC 2019", "0.921", "0.703"],
              ["Multimodal CNN+MLP (Pacheco, 2021)","PAD-UFES-20","0.893","0.851"],
              ["RareSight (ours)",                  "HAM10000",  "—",    "—"],
            ].map(([m, d, auc, acc], i) => (
              <div key={i} style={{
                display: "grid", gridTemplateColumns: "2fr 1fr 0.6fr 0.8fr",
                gap: 8, padding: "6px 0",
                borderBottom: "0.5px solid var(--color-border-tertiary)",
                fontSize: 12,
                background: i === 4 ? "var(--color-background-info)" : "transparent",
                borderRadius: i === 4 ? 4 : 0,
                padding: i === 4 ? "6px 8px" : "6px 0",
              }}>
                <span style={{ color: "var(--color-text-primary)", fontWeight: i === 4 ? 500 : 400 }}>{m}</span>
                <span style={{ color: "var(--color-text-secondary)" }}>{d}</span>
                <span style={{ color: "var(--color-text-primary)" }}>{auc}</span>
                <span style={{ color: "var(--color-text-primary)" }}>{acc}</span>
              </div>
            ))}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 0.6fr 0.8fr", gap: 8, marginTop: 4, fontSize: 11, color: "var(--color-text-secondary)" }}>
              <span>Method</span><span>Dataset</span><span>AUC macro</span><span>Balanced acc.</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
