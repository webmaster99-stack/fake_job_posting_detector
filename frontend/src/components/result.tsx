"use client";

import { Fragment, useState } from "react";
import { type PredictionResult, sendFeedback, type Signal } from "@/lib/api";
import { LONG_TEXT_FIELDS, type PostingForm } from "@/lib/posting";
import { highlight, label } from "@/lib/signals";

const SECTION_NAMES: Record<string, string> = {
  title: "Title",
  company_profile: "About the company",
  description: "Description",
  requirements: "Requirements",
  benefits: "Benefits",
};

// A calibrated probability is never exactly 0 or 1, so don't round it to one.
const percent = (p: number) =>
  p < 0.01 ? "under 1%" : p > 0.99 ? "over 99%" : `${Math.round(p * 100)}%`;

export function Result({ result, posting }: { result: PredictionResult; posting: PostingForm }) {
  const { fraud, probability, threshold, explanation } = result;
  const towardFraud = explanation.filter((s) => s.contribution > 0);
  const towardReal = explanation.filter((s) => s.contribution < 0);
  const sections = (["title", ...LONG_TEXT_FIELDS] as const)
    .filter((f) => posting[f].trim())
    .map((f) => ({ field: f, segments: highlight(posting[f], explanation) }));
  const hasMarks = sections.some((s) => s.segments.some((seg) => seg.mark));

  return (
    <article aria-labelledby="verdict" className="flex flex-col gap-10">
      <header className="flex flex-col gap-6">
        <h2
          id="verdict"
          data-fraud={fraud}
          className="verdict font-display text-[2.6rem] leading-[0.95] font-extrabold sm:text-6xl"
        >
          {fraud ? "Looks fraudulent" : "Looks genuine"}
        </h2>
        <p className="max-w-[60ch] text-lg leading-relaxed">
          {fraud
            ? "This posting resembles known scam ads. Don't send money, ID documents or bank details, and check the company through an independent source before you apply."
            : "Nothing here strongly resembles known scam ads. That isn't a guarantee: stay cautious about requests for money or personal documents."}
        </p>
        <Meter probability={probability} threshold={threshold} fraud={fraud} />
      </header>

      {(towardFraud.length > 0 || towardReal.length > 0) && (
        <section aria-labelledby="why" className="flex flex-col gap-4">
          <h3 id="why" className="text-xl font-bold">
            What the model noticed
          </h3>
          <div className="grid gap-6 sm:grid-cols-2">
            <SignalList title="Pointing to a scam" signals={towardFraud} tone="fraud" />
            <SignalList title="Pointing to a real job" signals={towardReal} tone="real" />
          </div>
        </section>
      )}

      {hasMarks && (
        <section aria-labelledby="marked" className="flex flex-col gap-4">
          <h3 id="marked" className="text-xl font-bold">
            Your posting, marked up
          </h3>
          <div className="paper flex flex-col gap-5">
            {sections.map(({ field, segments }) => (
              <div key={field}>
                <h4 className="mb-1 text-sm font-semibold text-slate">{SECTION_NAMES[field]}</h4>
                <p className="max-w-[72ch] leading-relaxed [overflow-wrap:anywhere] whitespace-pre-wrap">
                  {segments.map((seg, i) =>
                    seg.mark ? (
                      <mark key={i} data-tone={seg.mark}>
                        {seg.text}
                      </mark>
                    ) : (
                      <Fragment key={i}>{seg.text}</Fragment>
                    ),
                  )}
                </p>
              </div>
            ))}
          </div>
          <p className="flex flex-wrap gap-x-5 gap-y-1 text-sm text-slate">
            <span>
              <mark data-tone="fraud">Yellow</mark> words raised the score
            </span>
            <span>
              <mark data-tone="real">Underlined</mark> words lowered it
            </span>
          </p>
        </section>
      )}

      <Feedback key={result.prediction_id} predictionId={result.prediction_id} />

      <p className="text-sm text-slate">
        Checked by model {result.model_name}
        {result.model_version ? `, version ${result.model_version}` : ""}.
      </p>
    </article>
  );
}

function Meter({
  probability,
  threshold,
  fraud,
}: {
  probability: number;
  threshold: number;
  fraud: boolean;
}) {
  return (
    <figure className="flex max-w-xl flex-col gap-2">
      <figcaption className="flex items-baseline justify-between gap-4">
        <span className="font-semibold">Chance of fraud</span>
        <span className="font-display text-3xl font-bold tabular-nums">
          {percent(probability)}
        </span>
      </figcaption>
      <div
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(probability * 100)}
        aria-label="Chance of fraud"
        className="relative h-3 rounded-full bg-track"
      >
        <div
          className={`meter-fill h-full rounded-full ${fraud ? "bg-fraud" : "bg-real"}`}
          style={{ width: `max(${probability * 100}%, 0.75rem)` }}
        />
        <div
          aria-hidden
          className="absolute -top-1.5 -bottom-1.5 w-0.5 bg-ink"
          style={{ left: `${threshold * 100}%` }}
        />
      </div>
      <p className="text-sm text-slate">
        The line marks {percent(threshold)}: above it, the posting is flagged.
      </p>
    </figure>
  );
}

function SignalList({
  title,
  signals,
  tone,
}: {
  title: string;
  signals: Signal[];
  tone: "fraud" | "real";
}) {
  const largest = Math.max(...signals.map((s) => Math.abs(s.contribution)), 1e-9);
  return (
    <div>
      <h4 className="mb-3 font-semibold">{title}</h4>
      {signals.length === 0 ? (
        <p className="text-sm text-slate">Nothing notable.</p>
      ) : (
        <ul className="flex flex-col gap-2.5">
          {signals.map((s) => (
            <li key={`${s.kind}:${s.feature}`} className="flex flex-col gap-1">
              <span className="text-[0.95rem]">{label(s)}</span>
              <span
                aria-hidden
                className={`h-1 rounded-full ${tone === "fraud" ? "bg-fraud" : "bg-real"}`}
                style={{ width: `${(Math.abs(s.contribution) / largest) * 100}%` }}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

type FeedbackState =
  | { step: "ask" }
  | { step: "sending" }
  | { step: "sent"; correct: boolean; commented: boolean }
  | { step: "error"; message: string };

function Feedback({ predictionId }: { predictionId: string | null }) {
  const [state, setState] = useState<FeedbackState>({ step: "ask" });
  const [comment, setComment] = useState("");

  if (predictionId === null) {
    return (
      <p className="text-sm text-slate">
        Feedback isn&apos;t available for this check because it couldn&apos;t be saved.
      </p>
    );
  }

  async function send(correct: boolean, withComment = false) {
    setState({ step: "sending" });
    try {
      await sendFeedback(predictionId!, correct, withComment ? comment.trim() : undefined);
      setState({ step: "sent", correct, commented: withComment });
    } catch (e) {
      setState({ step: "error", message: (e as Error).message });
    }
  }

  return (
    <section aria-labelledby="feedback" className="flex flex-col gap-3 border-t border-rule pt-6">
      <h3 id="feedback" className="font-semibold">
        Was this verdict right?
      </h3>
      {state.step === "sent" ? (
        <div className="flex flex-col gap-3" aria-live="polite">
          <p>
            {state.commented
              ? "Comment sent. Thanks."
              : "Thanks. Your answer helps check the model against real postings."}
          </p>
          {!state.commented && (
            <form
              className="flex flex-col gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (comment.trim()) send(state.correct, true);
              }}
            >
              <label htmlFor="feedback-comment" className="text-sm font-semibold">
                Anything to add? (optional)
              </label>
              <textarea
                id="feedback-comment"
                className="control max-w-xl"
                rows={2}
                maxLength={2000}
                value={comment}
                onChange={(e) => setComment(e.target.value)}
              />
              <button type="submit" className="secondary self-start" disabled={!comment.trim()}>
                Send comment
              </button>
            </form>
          )}
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            className="secondary"
            disabled={state.step === "sending"}
            onClick={() => send(true)}
          >
            <span aria-hidden>👍</span> Yes
          </button>
          <button
            type="button"
            className="secondary"
            disabled={state.step === "sending"}
            onClick={() => send(false)}
          >
            <span aria-hidden>👎</span> No
          </button>
          {state.step === "error" && (
            <p role="alert" className="text-sm text-fraud">
              {state.message}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
