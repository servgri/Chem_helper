import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  api,
  type PredictResponse,
  type RetroResponse,
  type SimilarHit,
} from "../api/client";
import { MoleculeEditor } from "../components/MoleculeEditor";
import { PipelineAnimation } from "../components/PipelineAnimation";
import { ResultTabs } from "../components/ResultTabs";
import { SmilesForm } from "../components/SmilesForm";

type Tab = "predictions" | "similar" | "retro";

export default function Home() {
  const [smiles, setSmiles] = useState("CCO");
  const [molName, setMolName] = useState("");
  const [molFormula, setMolFormula] = useState("");
  const [molblock, setMolblock] = useState("");
  const [svg, setSvg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(0);
  const [predict, setPredict] = useState<PredictResponse | null>(null);
  const [similar, setSimilar] = useState<SimilarHit[]>([]);
  const [retro, setRetro] = useState<RetroResponse | null>(null);
  const [tab, setTab] = useState<Tab>("predictions");
  const [loadingSimilar, setLoadingSimilar] = useState(false);
  const [loadingRetro, setLoadingRetro] = useState(false);

  const syncLock = useRef(false);
  const parseTimer = useRef<number | null>(null);
  const smilesFromEditorRef = useRef(false);

  const applyParse = useCallback(async (smi: string) => {
    if (!smi.trim()) return;
    try {
      syncLock.current = true;
      const parsed = await api.parseSmiles(smi.trim());
      setSmiles(parsed.canonical_smiles || parsed.smiles);
      setMolName(parsed.name || parsed.formula || "");
      setMolFormula(parsed.formula || "");
      setMolblock(parsed.molblock);
      setSvg(parsed.svg);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Parse error");
    } finally {
      window.setTimeout(() => {
        syncLock.current = false;
      }, 250);
    }
  }, []);

  useEffect(() => {
    applyParse("CCO");
  }, [applyParse]);

  const onSmilesChange = (v: string) => {
    setSmiles(v);
    if (smilesFromEditorRef.current) {
      smilesFromEditorRef.current = false;
      return;
    }
    if (parseTimer.current) window.clearTimeout(parseTimer.current);
    parseTimer.current = window.setTimeout(() => {
      if (!syncLock.current) applyParse(v);
    }, 450);
  };

  const onStructureChange = async (payload: { smiles: string; molfile: string }) => {
    if (syncLock.current) return;
    try {
      syncLock.current = true;
      if (payload.smiles) {
        const parsed = await api.parseSmiles(payload.smiles);
        smilesFromEditorRef.current = true;
        setSmiles(parsed.canonical_smiles || parsed.smiles);
        setMolName(parsed.name || parsed.formula || "");
        setMolFormula(parsed.formula || "");
        setMolblock(parsed.molblock || payload.molfile);
        setSvg(parsed.svg);
        setError(null);
      } else if (payload.molfile) {
        const parsed = await api.fromMolfile(payload.molfile);
        smilesFromEditorRef.current = true;
        setSmiles(parsed.canonical_smiles || parsed.smiles);
        setMolName(parsed.name || parsed.formula || "");
        setMolFormula(parsed.formula || "");
        setMolblock(parsed.molblock);
        setSvg(parsed.svg);
        setError(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Structure sync error");
    } finally {
      window.setTimeout(() => {
        syncLock.current = false;
      }, 250);
    }
  };

  const runSearch = async () => {
    if (!smiles.trim()) return;
    setLoading(true);
    setError(null);
    setPipelineStep(0);
    setPredict(null);
    setSimilar([]);
    setRetro(null);
    setTab("predictions");

    try {
      setPipelineStep(0);
      await applyParse(smiles);

      setPipelineStep(1);
      await new Promise((r) => window.setTimeout(r, 180));

      setPipelineStep(2);
      const pred = await api.predict(smiles.trim());
      setPredict(pred);
      setSvg(pred.svg);

      setPipelineStep(3);
      await new Promise((r) => window.setTimeout(r, 120));

      setPipelineStep(4);
      setLoadingSimilar(true);
      setLoadingRetro(true);
      const [sim, rt] = await Promise.all([
        api.similar(pred.smiles).catch(() => null),
        api.retrosynthesis(pred.smiles, 2).catch(() => null),
      ]);
      if (sim) setSimilar(sim.similar || sim.results || []);
      if (rt) setRetro(rt);

      setPipelineStep(5);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
      setPipelineStep(0);
    } finally {
      setLoading(false);
      setLoadingSimilar(false);
      setLoadingRetro(false);
    }
  };

  return (
    <div className="bg-grid min-h-screen font-sans">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-4 py-6">
        <div className="flex items-baseline gap-2">
          <span className="font-display text-2xl font-bold tracking-tight text-brand-700">
            ToxMol
          </span>
          <span className="font-sans text-sm font-medium text-brand-800/50">AI</span>
        </div>
        <span className="hidden font-sans text-sm text-brand-800/60 sm:inline">
          Tox21 QSAR · ADMET · Retrosynthesis
        </span>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-16 font-sans">
        <section className="mb-6">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center md:text-left"
          >
            <p className="mb-2 font-sans text-sm font-semibold uppercase tracking-[0.2em] text-brand-600">
              Tox21 QSAR workspace
            </p>
            <h1 className="font-display text-4xl font-bold leading-tight text-brand-950 sm:text-5xl">
              ToxMol
            </h1>
            <p className="mt-3 max-w-2xl font-sans text-sm text-brand-800/70 sm:text-base">
              Оценка токсичности по SMILES: 12 эндпоинтов Tox21, physchem-профиль,
              похожие структуры и rule-based ретросинтез.
            </p>
          </motion.div>
        </section>

        <section className="grid w-full grid-cols-1 items-stretch gap-3 md:grid-cols-2">
          <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-brand-100/80 bg-white/80 px-3.5 pb-[10px] pt-3 shadow-soft backdrop-blur">
            <h2 className="mb-2 text-center font-display text-lg text-brand-900 md:text-left">
              Поиск
            </h2>
            <div className="flex min-h-0 flex-1 flex-col">
              <SmilesForm
                value={smiles}
                onChange={onSmilesChange}
                onSubmit={runSearch}
                loading={loading}
              />
              {error && (
                <p className="mt-2 rounded-lg bg-rose-50 px-2.5 py-1.5 font-sans text-xs text-rose-700">
                  {error}
                </p>
              )}
            </div>
          </div>

          <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-brand-100/80 bg-white/80 p-3.5 shadow-soft backdrop-blur">
            <h2 className="mb-0.5 font-display text-lg text-brand-900">Превью формулы</h2>
            <p className="mb-2 font-sans text-[11px] text-brand-800/55">
              Обновляется из SMILES и из редактора
            </p>
            <figure className="flex min-h-0 flex-1 flex-col text-center">
              <div
                className="mol-svg flex min-h-[100px] flex-1 items-center justify-center overflow-hidden rounded-lg border border-brand-100/60 bg-white p-2"
                dangerouslySetInnerHTML={{
                  __html: svg || '<p class="text-brand-700/40 font-sans text-sm">Нет структуры</p>',
                }}
              />
              <figcaption className="mt-1.5 shrink-0 space-y-0.5 font-sans">
                <div className="text-xs font-semibold text-brand-800">
                  {molName || "Целевая молекула"}
                </div>
                {molFormula && molFormula !== molName ? (
                  <div className="text-[11px] text-brand-800/60">{molFormula}</div>
                ) : null}
                <div className="break-all font-mono text-[10px] text-brand-800/70">
                  {smiles || "—"}
                </div>
              </figcaption>
            </figure>
          </div>
        </section>

        <section className="mt-8 flex justify-center">
          <PipelineAnimation activeIndex={pipelineStep} running={loading} />
        </section>

        <section className="mt-4">
          <MoleculeEditor molblock={molblock} onStructureChange={onStructureChange} />
        </section>

        {(predict || loading) && (
          <ResultTabs
            tab={tab}
            onTab={setTab}
            predict={predict}
            similar={similar}
            retro={retro}
            loadingSimilar={loadingSimilar}
            loadingRetro={loadingRetro}
          />
        )}
      </main>
    </div>
  );
}
