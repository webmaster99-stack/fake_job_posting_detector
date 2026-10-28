"use client";

import { useEffect, useRef, useState } from "react";
import { predict, type PredictionResult } from "@/lib/api";
import { EMPTY_FORM, type PostingForm as Form, toPosting } from "@/lib/posting";
import { useServerStatus } from "@/lib/use-server-status";
import { type Mode, PostingForm } from "./posting-form";
import { Result } from "./result";
import { ServerStatusBanner } from "./server-status";

type Checked = { result: PredictionResult; posting: Form };

/** In paste mode only the title and the pasted text are sent. */
const submitted = (form: Form, mode: Mode): Form =>
  mode === "paste"
    ? {
        ...EMPTY_FORM,
        title: form.title,
        description: form.description,
        telecommuting: form.telecommuting,
        has_company_logo: form.has_company_logo,
        has_questions: form.has_questions,
      }
    : form;

export function Detector() {
  const { status, retry } = useServerStatus();
  const [form, setForm] = useState<Form>(EMPTY_FORM);
  const [mode, setMode] = useState<Mode>("fields");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checked, setChecked] = useState<Checked | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = resultRef.current;
    if (!checked || !el) return;
    // focus() alone scrolls a tall result somewhere in its middle; start at the verdict.
    el.focus({ preventScroll: true });
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el.scrollIntoView({ block: "start", behavior: reduce ? "auto" : "smooth" });
  }, [checked]);

  async function check() {
    const posting = submitted(form, mode);
    setSubmitting(true);
    setError(null);
    try {
      const result = await predict(toPosting(posting));
      setChecked({ result, posting });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const label = submitting
    ? status === "ready"
      ? "Checking…"
      : "Waiting for the server…"
    : "Check this posting";

  return (
    <div className="grid gap-x-16 gap-y-14 lg:grid-cols-[minmax(0,34rem)_minmax(0,1fr)]">
      <section aria-label="Posting" className="flex flex-col gap-6">
        <ServerStatusBanner status={status} onRetry={retry} />
        <PostingForm
          form={form}
          mode={mode}
          onChange={setForm}
          onModeChange={setMode}
          onSubmit={check}
          submitting={submitting}
          disabled={status === "unreachable"}
          submitLabel={label}
        />
        {error && (
          <p role="alert" className="text-fraud">
            {error}
          </p>
        )}
      </section>

      <div
        ref={resultRef}
        tabIndex={-1}
        aria-live="polite"
        className="scroll-mt-6 outline-none"
      >
        {checked ? (
          <Result result={checked.result} posting={checked.posting} />
        ) : (
          <div className="flex flex-col gap-3 border-l-4 border-highlight pl-5">
            <p className="font-display text-2xl font-bold">The verdict appears here.</p>
            <p className="max-w-[48ch] leading-relaxed text-slate">
              Fill in the ad as it&apos;s written, then check it. You&apos;ll see how likely it
              is to be a scam and which words and details made the difference.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
