type Props = {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  loading?: boolean;
};

export function SmilesForm({ value, onChange, onSubmit, loading }: Props) {
  return (
    <form
      className="flex h-full w-full max-w-full flex-1 flex-col gap-2 font-sans"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <label htmlFor="smiles" className="font-sans text-xs font-semibold text-brand-800">
        SMILES
      </label>
      <textarea
        id="smiles"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="CCO или c1ccccc1O"
        rows={1}
        className="box-border h-10 w-full max-w-full shrink-0 resize-none rounded-lg border border-brand-200/80 bg-white/90 px-2.5 py-2 font-mono text-xs outline-none ring-brand-500/30 transition focus:ring-2"
        spellCheck={false}
        autoComplete="off"
      />
      <div className="mt-auto w-full max-w-full">
        <button
          type="submit"
          disabled={loading || !value.trim()}
          className="box-border w-full max-w-full rounded-lg bg-brand-600 px-3 py-2 font-sans text-sm font-semibold text-white shadow-soft transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Считаем…" : "Поиск"}
        </button>
      </div>
    </form>
  );
}
