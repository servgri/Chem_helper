import { motion } from "framer-motion";

const STEPS = [
  { id: "input", label: "Input" },
  { id: "fp", label: "Fingerprint" },
  { id: "qsar", label: "QSAR / ADMET" },
  { id: "tox", label: "NR / SR" },
  { id: "out", label: "Results" },
] as const;

type Props = {
  activeIndex: number;
  running: boolean;
};

export function PipelineAnimation({ activeIndex, running }: Props) {
  const total = STEPS.length;
  const current = Math.max(0, Math.min(activeIndex, total - 1));
  const doneAll = activeIndex >= total;

  return (
    <div className="mx-auto w-full max-w-3xl py-4 font-sans">
      <div className="flex items-center justify-center gap-1">
        {STEPS.map((step, i) => {
          const filled =
            activeIndex >= total ? true : i < activeIndex || (running && i === activeIndex);
          const isCurrent = running && i === activeIndex && activeIndex < total;
          const isDone = activeIndex >= total || i < activeIndex;
          const connectorFilled = activeIndex >= total ? true : i < activeIndex;

          return (
            <div key={step.id} className="flex items-center gap-1">
              <div className="flex w-[4.75rem] flex-col items-center gap-2 sm:w-[5.5rem]">
                <motion.div
                  initial={false}
                  animate={{
                    backgroundColor: filled ? "#059669" : "#ffffff",
                    borderColor: filled ? "#047857" : "rgba(5,150,105,0.28)",
                    scale: isCurrent ? 1.06 : 1,
                  }}
                  transition={{ duration: 0.35, ease: "easeOut" }}
                  className="relative flex h-11 w-11 items-center justify-center rounded-full border-2 font-sans text-sm font-semibold"
                  style={{ color: filled ? "#fff" : "#047857" }}
                >
                  {isDone ? <span aria-hidden>✓</span> : i + 1}
                  {isCurrent && (
                    <span className="absolute inset-0 rounded-full ring-4 ring-brand-400/35" />
                  )}
                </motion.div>
                <span
                  className={`text-center font-sans text-[11px] font-semibold leading-tight sm:text-xs ${
                    filled ? "text-brand-700" : "text-brand-800/45"
                  }`}
                >
                  {step.label}
                </span>
              </div>

              {i < STEPS.length - 1 && (
                <div className="relative mb-6 h-1.5 w-8 overflow-hidden rounded-full bg-brand-100 sm:w-12 md:w-16">
                  <motion.div
                    className="absolute inset-y-0 left-0 rounded-full bg-brand-400"
                    initial={false}
                    animate={{
                      width: connectorFilled ? "100%" : isCurrent ? "45%" : "0%",
                    }}
                    transition={{ duration: 0.45, ease: "easeOut" }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-center font-sans text-xs text-brand-800/50">
        {running
          ? `Этап ${Math.min(current + 1, total)} из ${total}: ${STEPS[Math.min(current, total - 1)].label}`
          : doneAll
            ? "Пайплайн завершён"
            : activeIndex > 0
              ? `Пройдено: ${activeIndex} / ${total}`
              : "Ожидание запуска"}
      </p>
    </div>
  );
}
