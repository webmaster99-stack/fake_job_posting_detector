import type { ServerStatus } from "@/lib/use-server-status";

const MESSAGES: Record<ServerStatus, string> = {
  checking: "Connecting to the model server…",
  waking: "Waking up the model server. The first check after a quiet spell can take up to a minute.",
  ready: "Model server ready",
  unreachable: "The model server didn't respond.",
};

export function ServerStatusBanner({
  status,
  onRetry,
}: {
  status: ServerStatus;
  onRetry: () => void;
}) {
  return (
    <p
      role="status"
      aria-live="polite"
      data-status={status}
      className="flex items-center gap-2.5 text-sm text-slate"
    >
      <span
        aria-hidden
        className={`size-2 shrink-0 rounded-full ${
          status === "ready"
            ? "bg-real"
            : status === "unreachable"
              ? "bg-fraud"
              : "animate-pulse bg-slate motion-reduce:animate-none"
        }`}
      />
      <span>{MESSAGES[status]}</span>
      {status === "unreachable" && (
        <button type="button" onClick={onRetry} className="link">
          Try again
        </button>
      )}
    </p>
  );
}
