"use client";

import { useId } from "react";
import {
  CHOICES,
  EXAMPLES,
  type FlagField,
  type PostingForm as Form,
  type TextField,
} from "@/lib/posting";

export type Mode = "fields" | "paste";

type Props = {
  form: Form;
  mode: Mode;
  onChange: (form: Form) => void;
  onModeChange: (mode: Mode) => void;
  onSubmit: () => void;
  submitting: boolean;
  disabled: boolean;
  submitLabel: string;
};

const LONG = 20_000;
const SHORT = 300;

export function PostingForm(props: Props) {
  const { form, mode, onChange, onModeChange, onSubmit, submitting, disabled, submitLabel } =
    props;
  const set = (key: TextField | FlagField, value: string | boolean) =>
    onChange({ ...form, [key]: value });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="flex flex-col gap-8"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div role="radiogroup" aria-label="How to enter the posting" className="segmented">
          <button
            type="button"
            role="radio"
            aria-checked={mode === "fields"}
            onClick={() => onModeChange("fields")}
          >
            Section by section
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={mode === "paste"}
            onClick={() => onModeChange("paste")}
          >
            Paste the whole posting
          </button>
        </div>
        <div className="flex gap-3 text-sm">
          {EXAMPLES.map((example) => (
            <button
              key={example.name}
              type="button"
              className="link"
              onClick={() => {
                onModeChange("fields");
                onChange(example.form);
              }}
            >
              {example.name}
            </button>
          ))}
        </div>
      </div>

      <Input
        label="Job title"
        value={form.title}
        onChange={(v) => set("title", v)}
        maxLength={SHORT}
        required
      />

      {mode === "paste" ? (
        <TextArea
          label="Posting text"
          hint="Paste everything the ad says. Sections the model can't tell apart count as missing, so separate fields give a more reliable verdict."
          value={form.description}
          onChange={(v) => set("description", v)}
          rows={14}
        />
      ) : (
        <>
          <TextArea
            label="About the company"
            value={form.company_profile}
            onChange={(v) => set("company_profile", v)}
          />
          <TextArea
            label="Job description"
            value={form.description}
            onChange={(v) => set("description", v)}
            rows={7}
          />
          <TextArea
            label="Requirements"
            value={form.requirements}
            onChange={(v) => set("requirements", v)}
          />
          <TextArea
            label="Benefits"
            value={form.benefits}
            onChange={(v) => set("benefits", v)}
            rows={3}
          />

          <fieldset className="grid gap-x-5 gap-y-5 sm:grid-cols-2">
            <legend className="mb-4 text-base font-semibold">Details</legend>
            <Input
              label="Location"
              hint="Country code first, e.g. US, NY, New York"
              value={form.location}
              onChange={(v) => set("location", v)}
              maxLength={SHORT}
            />
            <Input
              label="Salary range"
              value={form.salary_range}
              onChange={(v) => set("salary_range", v)}
              maxLength={SHORT}
            />
            <Input
              label="Department"
              value={form.department}
              onChange={(v) => set("department", v)}
              maxLength={SHORT}
            />
            <Select
              label="Employment type"
              value={form.employment_type}
              options={CHOICES.employment_type}
              onChange={(v) => set("employment_type", v)}
            />
            <Select
              label="Experience"
              value={form.required_experience}
              options={CHOICES.required_experience}
              onChange={(v) => set("required_experience", v)}
            />
            <Select
              label="Education"
              value={form.required_education}
              options={CHOICES.required_education}
              onChange={(v) => set("required_education", v)}
            />
            <Input
              label="Industry"
              value={form.industry}
              suggestions={CHOICES.industry}
              onChange={(v) => set("industry", v)}
              maxLength={SHORT}
            />
            <Select
              label="Job function"
              value={form.function}
              options={CHOICES.function}
              onChange={(v) => set("function", v)}
            />
          </fieldset>
        </>
      )}

      <fieldset className="flex flex-col gap-3">
        <legend className="mb-3 text-base font-semibold">The ad…</legend>
        <Checkbox
          label="is for a remote job"
          checked={form.telecommuting}
          onChange={(v) => set("telecommuting", v)}
        />
        <Checkbox
          label="shows the company's logo"
          checked={form.has_company_logo}
          onChange={(v) => set("has_company_logo", v)}
        />
        <Checkbox
          label="asks screening questions when you apply"
          checked={form.has_questions}
          onChange={(v) => set("has_questions", v)}
        />
      </fieldset>

      <div className="flex flex-wrap items-center gap-4">
        <button type="submit" className="primary" disabled={disabled || submitting}>
          {submitLabel}
        </button>
        <p className="text-sm text-slate">Leave a field empty if the ad doesn&apos;t have it.</p>
      </div>
    </form>
  );
}

function Field({
  label,
  hint,
  id,
  children,
}: {
  label: string;
  hint?: string;
  id: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-semibold">
        {label}
      </label>
      {children}
      {hint && (
        <p id={`${id}-hint`} className="text-sm text-slate">
          {hint}
        </p>
      )}
    </div>
  );
}

function Input({
  label,
  hint,
  value,
  onChange,
  maxLength,
  required,
  suggestions,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (value: string) => void;
  maxLength: number;
  required?: boolean;
  suggestions?: readonly string[];
}) {
  const id = useId();
  return (
    <Field label={required ? `${label} (required)` : label} hint={hint} id={id}>
      <input
        id={id}
        className="control"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        maxLength={maxLength}
        required={required}
        aria-describedby={hint ? `${id}-hint` : undefined}
        list={suggestions ? `${id}-list` : undefined}
      />
      {suggestions && (
        <datalist id={`${id}-list`}>
          {suggestions.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      )}
    </Field>
  );
}

function TextArea({
  label,
  hint,
  value,
  onChange,
  rows = 4,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (value: string) => void;
  rows?: number;
}) {
  const id = useId();
  return (
    <Field label={label} hint={hint} id={id}>
      <textarea
        id={id}
        className="control resize-y leading-relaxed"
        rows={rows}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        maxLength={LONG}
        aria-describedby={hint ? `${id}-hint` : undefined}
      />
    </Field>
  );
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: readonly string[];
  onChange: (value: string) => void;
}) {
  const id = useId();
  return (
    <Field label={label} id={id}>
      <select id={id} className="control" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Not stated</option>
        {options.map((o) => (
          <option key={o}>{o}</option>
        ))}
      </select>
    </Field>
  );
}

function Checkbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-3">
      <input
        type="checkbox"
        className="size-4.5 accent-ink"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}
