import { Fragment } from "react";
import type {
  EndpointResult,
  PredictResponse,
  RetroResponse,
  RetroRoute,
  SimilarHit,
} from "../api/client";

type Tab = "predictions" | "similar" | "retro";

type Props = {
  tab: Tab;
  onTab: (t: Tab) => void;
  predict: PredictResponse | null;
  similar: SimilarHit[];
  retro: RetroResponse | null;
  loadingSimilar?: boolean;
  loadingRetro?: boolean;
};

function ProbBar({ value }: { value: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const color = value >= 0.7 ? "bg-rose-500" : value >= 0.4 ? "bg-amber-500" : "bg-brand-500";
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-brand-50">
      <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function MolFigure({
  svg,
  title,
  subtitle,
}: {
  svg: string;
  title: string;
  subtitle?: string;
}) {
  return (
    <figure className="min-w-[140px] flex-1 text-center font-sans">
      <div
        className="mol-svg mx-auto flex min-h-[120px] items-center justify-center"
        dangerouslySetInnerHTML={{ __html: svg || "" }}
      />
      <figcaption className="mt-1 space-y-0.5 px-1">
        <div className="font-sans text-xs font-semibold leading-snug text-brand-800">{title}</div>
        {subtitle ? (
          <div className="truncate font-mono text-[10px] text-brand-800/55" title={subtitle}>
            {subtitle}
          </div>
        ) : null}
      </figcaption>
    </figure>
  );
}

function EndpointList({ items, title }: { items: EndpointResult[]; title: string }) {
  return (
    <div>
      <h4 className="mb-3 font-display text-lg text-brand-900">{title}</h4>
      <div className="space-y-3">
        {items.map((e) => (
          <div key={e.target} className="rounded-xl border border-brand-100 bg-white/80 p-3">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="font-semibold text-brand-900">{e.target}</span>
              <span
                className={`rounded-md px-2 py-0.5 text-xs font-semibold ${
                  e.label ? "bg-rose-100 text-rose-700" : "bg-brand-50 text-brand-700"
                }`}
              >
                {e.label ? "active" : "inactive"} · {(e.probability * 100).toFixed(1)}%
              </span>
            </div>
            <ProbBar value={e.probability} />
            <p className="mt-1 text-xs text-brand-800/60">
              {e.model_name}
              {e.fallback ? " (fallback)" : ""} · ROC-AUC {e.roc_auc.toFixed(3)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Drop A+B duplicates of B+A (order-invariant precursor set). */
function dedupeRetroRoutes(routes: RetroRoute[]): RetroRoute[] {
  const seen = new Set<string>();
  const out: RetroRoute[] = [];
  for (const r of routes) {
    const key = [...r.precursors].map((s) => s.trim()).sort().join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(r);
  }
  return out;
}

export function ResultTabs({
  tab,
  onTab,
  predict,
  similar,
  retro,
  loadingSimilar,
  loadingRetro,
}: Props) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "predictions", label: "Predictions" },
    { id: "similar", label: "Similar" },
    { id: "retro", label: "Retrosynthesis" },
  ];

  const retroRoutes = retro?.routes ? dedupeRetroRoutes(retro.routes) : [];

  return (
    <section className="mt-10">
      <div className="mb-6 flex flex-wrap gap-2 border-b border-brand-100 pb-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onTab(t.id)}
            className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
              tab === t.id
                ? "bg-brand-600 text-white"
                : "bg-white text-brand-800 hover:bg-brand-50"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "predictions" && predict && (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-4">
            <div className="rounded-2xl border border-brand-100 bg-white/90 p-5 shadow-soft">
              <h3 className="font-display text-xl text-brand-900">QSAR</h3>
              <p className="mt-1 text-sm text-brand-800/70">
                Сводка по 12 Tox21-эндпоинтам (топовые модели из research).
              </p>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-brand-800/50">Risk</dt>
                  <dd className="text-lg font-semibold capitalize">{predict.qsar.risk_level}</dd>
                </div>
                <div>
                  <dt className="text-brand-800/50">Max P</dt>
                  <dd className="text-lg font-semibold">
                    {(predict.qsar.max_probability * 100).toFixed(1)}%
                  </dd>
                </div>
                <div>
                  <dt className="text-brand-800/50">Mean P</dt>
                  <dd className="font-semibold">
                    {(predict.qsar.mean_probability * 100).toFixed(1)}%
                  </dd>
                </div>
                <div>
                  <dt className="text-brand-800/50">Active</dt>
                  <dd className="font-semibold">
                    {predict.qsar.n_active_endpoints}/{predict.qsar.n_endpoints}
                  </dd>
                </div>
              </dl>
            </div>

            <div className="rounded-2xl border border-brand-100 bg-white/90 p-5 shadow-soft">
              <h3 className="font-display text-xl text-brand-900">ADMET</h3>
              <p className="mt-1 text-sm text-brand-800/70">
                Physchem + правила Lipinski / Veber (без отдельных ADMET ML-моделей).
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                {Object.entries(predict.admet.physchem).map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-brand-50/80 px-2 py-2">
                    <div className="text-xs text-brand-800/50">{k}</div>
                    <div className="font-semibold">{Number(v).toFixed(2)}</div>
                  </div>
                ))}
              </div>
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                <span
                  className={`rounded-md px-2 py-1 font-semibold ${
                    predict.admet.lipinski.lipinski_pass
                      ? "bg-brand-100 text-brand-800"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  Lipinski {predict.admet.lipinski.lipinski_pass ? "pass" : "fail"} (
                  {predict.admet.lipinski.lipinski_violations} viol.)
                </span>
                <span
                  className={`rounded-md px-2 py-1 font-semibold ${
                    predict.admet.lipinski.veber_pass
                      ? "bg-brand-100 text-brand-800"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  Veber {predict.admet.lipinski.veber_pass ? "pass" : "fail"}
                </span>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <EndpointList items={predict.nr} title="Nuclear Receptor (NR)" />
            <EndpointList items={predict.sr} title="Stress Response (SR)" />
          </div>
        </div>
      )}

      {tab === "similar" && (
        <div>
          {loadingSimilar && <p className="text-brand-700">Ищем похожие молекулы…</p>}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {similar.map((hit, idx) => (
              <article
                key={hit.smiles}
                className="rounded-2xl border border-brand-100 bg-white/90 p-4 shadow-soft"
              >
                <MolFigure
                  svg={hit.svg}
                  title={hit.name || hit.formula || hit.mol_id || `Аналог ${idx + 1}`}
                  subtitle={hit.formula && hit.name !== hit.formula ? `${hit.formula} · ${hit.smiles}` : hit.smiles}
                />
                <p className="mt-2 text-center text-sm font-semibold text-brand-700">
                  Tanimoto {(hit.tanimoto * 100).toFixed(1)}%
                </p>
              </article>
            ))}
          </div>
          {!loadingSimilar && similar.length === 0 && (
            <p className="text-brand-800/60">Нет результатов — сначала запустите поиск.</p>
          )}
        </div>
      )}

      {tab === "retro" && (
        <div className="space-y-4">
          {loadingRetro && <p className="text-brand-700">Строим ретросинтез…</p>}
          {retroRoutes.map((route, i) => (
            <article
              key={`${route.reaction}-${i}`}
              className="rounded-2xl border border-brand-100 bg-white/90 p-5 shadow-soft"
            >
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="rounded-md bg-brand-100 px-2 py-0.5 text-xs font-semibold text-brand-800">
                  {route.depth} stage{route.depth > 1 ? "s" : ""}
                </span>
                <h4 className="font-semibold text-brand-900">{route.reaction}</h4>
              </div>
              <p className="mb-3 text-sm text-brand-800/70">{route.description}</p>
              {route.intermediate && (
                <p className="mb-2 text-xs text-brand-800/60">
                  <span className="font-semibold">
                    {route.intermediate_label || "Интермедиат"}:
                  </span>{" "}
                  <span className="font-mono">{route.intermediate}</span>
                </p>
              )}
              <div className="flex flex-wrap items-center justify-center gap-2 md:gap-4">
                {route.precursors.map((smi, j) => (
                  <Fragment key={smi + j}>
                    {j > 0 && (
                      <span
                        className="select-none px-1 font-display text-5xl font-bold leading-none text-brand-600 md:text-6xl"
                        aria-hidden
                      >
                        +
                      </span>
                    )}
                    <MolFigure
                      svg={route.precursor_svgs[j] || ""}
                      title={route.precursor_labels?.[j] || smi}
                      subtitle={smi}
                    />
                  </Fragment>
                ))}
                <span
                  className="select-none px-2 font-display text-5xl font-bold leading-none text-brand-600 md:text-6xl"
                  aria-hidden
                >
                  →
                </span>
                <MolFigure
                  svg={retro?.svg || ""}
                  title={route.product_label || retro?.smiles || "Продукт"}
                  subtitle={retro?.smiles}
                />
              </div>
            </article>
          ))}
          {!loadingRetro && retroRoutes.length === 0 && (
            <p className="text-brand-800/60">
              Маршруты не найдены для доступных SMARTS-правил (1–2 стадии).
            </p>
          )}
        </div>
      )}
    </section>
  );
}
