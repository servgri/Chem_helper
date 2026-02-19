export type EndpointResult = {
  target: string;
  family: string;
  probability: number;
  label: number;
  threshold: number;
  model_name: string;
  roc_auc: number;
  artifact?: string;
  fallback?: boolean;
};

export type PredictResponse = {
  smiles: string;
  svg: string;
  qsar: {
    mean_probability: number;
    max_probability: number;
    n_active_endpoints: number;
    n_endpoints: number;
    risk_level: string;
    active_targets: string[];
  };
  admet: {
    physchem: Record<string, number>;
    lipinski: {
      rules: Record<string, boolean>;
      lipinski_pass: boolean;
      veber_pass: boolean;
      lipinski_violations: number;
    };
    toxicity_profile: {
      mean_probability: number;
      max_probability: number;
      n_active: number;
    };
  };
  nr: EndpointResult[];
  sr: EndpointResult[];
  job_id?: number;
  errors?: Record<string, string>;
};

export type SimilarHit = {
  smiles: string;
  mol_id?: string | null;
  name?: string;
  formula?: string;
  tanimoto: number;
  svg: string;
};

export type SimilarResponse = {
  smiles: string;
  similar: SimilarHit[];
  results?: SimilarHit[];
  top_n: number;
};

export type RetroRoute = {
  reaction: string;
  description: string;
  precursors: string[];
  precursor_svgs: string[];
  precursor_labels?: string[];
  depth: number;
  product?: string;
  product_label?: string;
  intermediate?: string;
  intermediate_label?: string;
  steps?: Array<{
    depth: number;
    reaction: string;
    description: string;
    from_smiles: string;
    precursors: string[];
    precursor_svgs: string[];
  }>;
};

export type RetroResponse = {
  smiles: string;
  svg: string;
  max_depth: number;
  routes: RetroRoute[];
  n_routes: number;
};

export type ParseResponse = {
  smiles: string;
  canonical_smiles: string;
  name?: string;
  formula?: string;
  molblock: string;
  svg: string;
  physchem: Record<string, number>;
  admet?: PredictResponse["admet"];
  valid: boolean;
};

const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error((data as { detail?: string }).detail || `HTTP ${res.status}`);
  }
  return data as T;
}

export const api = {
  parseSmiles: (smiles: string) => post<ParseResponse>("/api/molecule/parse/", { smiles }),
  fromMolfile: (molfile: string) =>
    post<ParseResponse>("/api/molecule/from-molfile/", { molfile }),
  predict: (smiles: string) => post<PredictResponse>("/api/predict/", { smiles }),
  similar: (smiles: string, top_n = 12) =>
    post<SimilarResponse>("/api/similar/", { smiles, top_n }),
  retrosynthesis: (smiles: string, max_depth = 2) =>
    post<RetroResponse>("/api/retrosynthesis/", { smiles, max_depth }),
  health: async () => {
    const res = await fetch(`${API_URL}/api/health/`);
    return res.json();
  },
};

export { API_URL };
