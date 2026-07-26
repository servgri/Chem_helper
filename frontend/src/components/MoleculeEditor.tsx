import { useCallback, useEffect, useRef, useState } from "react";
import "ketcher-react/dist/index.css";

type Props = {
  molblock: string;
  onStructureChange: (payload: { smiles: string; molfile: string }) => void;
};

/**
 * Interactive Ketcher sketcher with automatic SMILES sync.
 */
export function MoleculeEditor({ molblock, onStructureChange }: Props) {
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [Editor, setEditor] = useState<any>(null);
  const [provider, setProvider] = useState<any>(null);
  const ketcherRef = useRef<any>(null);
  const applyingRef = useRef(false);
  const lastMolRef = useRef("");
  const lastSmilesRef = useRef("");
  const originRef = useRef<"external" | "editor">("external");
  const onStructureChangeRef = useRef(onStructureChange);
  onStructureChangeRef.current = onStructureChange;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const reactMod = await import("ketcher-react");
        const standMod = await import("ketcher-standalone");
        if (cancelled) return;
        const EditorComp = (reactMod as { Editor?: unknown }).Editor;
        const ProviderCtor = (standMod as { StandaloneStructServiceProvider?: new () => unknown })
          .StandaloneStructServiceProvider;
        if (!EditorComp || !ProviderCtor) {
          throw new Error("Ketcher exports missing");
        }
        setEditor(() => EditorComp);
        setProvider(new ProviderCtor());
        setReady(true);
      } catch (err) {
        console.warn("Ketcher failed to load", err);
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // SMILES → formula: push molblock from parent into Ketcher
  useEffect(() => {
    const k = ketcherRef.current;
    if (!k || !molblock) return;
    if (originRef.current === "editor") {
      originRef.current = "external";
      lastMolRef.current = molblock;
      return;
    }
    if (molblock === lastMolRef.current) return;
    applyingRef.current = true;
    lastMolRef.current = molblock;
    k.setMolecule(molblock)
      .catch(() => undefined)
      .finally(() => {
        applyingRef.current = false;
      });
  }, [molblock]);

  const pushFromEditor = useCallback(async () => {
    const k = ketcherRef.current;
    if (!k || applyingRef.current) return;
    setBusy(true);
    try {
      let smiles = "";
      let mf = "";
      try {
        smiles = String((await k.getSmiles?.()) || "").trim();
      } catch {
        smiles = "";
      }
      try {
        mf = String((await k.getMolfile?.()) || "");
      } catch {
        mf = "";
      }

      if (!smiles && !mf) return;
      if (smiles && smiles === lastSmilesRef.current) return;
      if (!smiles && mf && mf === lastMolRef.current) return;

      // Prefer SMILES path; fall back to molfile if smiles empty but structure exists
      if (!smiles && mf) {
        const hasAtoms =
          mf.includes("V2000") || mf.includes("V3000") || /\n\s*\d+\s+\d+\s+/.test(mf);
        if (!hasAtoms) return;
      }

      lastSmilesRef.current = smiles;
      if (mf) lastMolRef.current = mf;
      originRef.current = "editor";
      onStructureChangeRef.current({ smiles, molfile: mf });
    } catch (err) {
      console.warn("sync from editor failed", err);
    } finally {
      setBusy(false);
    }
  }, []);

  if (failed) {
    return (
      <div className="ketcher-wrap flex items-center justify-center p-6 text-sm text-brand-800/70">
        Редактор недоступен — используйте поле SMILES
      </div>
    );
  }

  if (!ready || !Editor || !provider) {
    return (
      <div className="ketcher-wrap flex items-center justify-center text-sm text-brand-800/60">
        Загрузка редактора структуры…
      </div>
    );
  }

  const Ed = Editor;
  return (
    <div className="ketcher-wrap flex flex-col">
      <div className="flex items-center justify-between gap-3 border-b border-brand-100 px-4 py-2">
        <div>
          <p className="text-sm font-medium text-brand-800">Редактор молекулы</p>
        </div>
        <button
          type="button"
          onClick={() => pushFromEditor()}
          disabled={busy}
          className="shrink-0 rounded-xl bg-brand-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-700 disabled:opacity-50"
        >
          {busy ? "Синхронизация…" : "Обновить SMILES"}
        </button>
      </div>
      <div className="ketcher-canvas relative flex-1">
        <Ed
          staticResourcesUrl={import.meta.env.BASE_URL || "/"}
          structServiceProvider={provider}
          errorHandler={(e: string) => console.warn(e)}
          onInit={(ketcher: any) => {
            ketcherRef.current = ketcher;
            (window as any).ketcher = ketcher;
            if (molblock) {
              applyingRef.current = true;
              lastMolRef.current = molblock;
              ketcher
                .setMolecule(molblock)
                .catch(() => undefined)
                .finally(() => {
                  applyingRef.current = false;
                });
            }
            // Subscribe to structure changes when API available
            try {
              const editor = ketcher.editor;
              if (editor?.subscribe) {
                editor.subscribe("change", () => {
                  window.setTimeout(() => pushFromEditor(), 300);
                });
              }
            } catch {
              /* ignore */
            }
            if ((ketcher as any).__toxmolTimer) {
              window.clearInterval((ketcher as any).__toxmolTimer);
            }
            (ketcher as any).__toxmolTimer = window.setInterval(() => {
              if (applyingRef.current) return;
              pushFromEditor();
            }, 1200);
          }}
        />
      </div>
    </div>
  );
}
