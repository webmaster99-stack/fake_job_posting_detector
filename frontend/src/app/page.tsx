import { Detector } from "@/components/detector";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-[90rem] flex-col gap-14 px-4 pt-12 pb-20 sm:px-8 lg:px-12 lg:pt-16">
      <header className="flex max-w-3xl flex-col gap-5">
        <h1 className="font-display text-5xl leading-[0.9] font-extrabold tracking-tight sm:text-7xl">
          Is this job ad a scam?
        </h1>
        <p className="max-w-[58ch] text-lg leading-relaxed text-slate">
          Paste a posting and a model trained on thousands of real and fraudulent job ads checks it for
          the patterns scams share: vague duties, easy money, missing company details and
          requests to get in touch off the platform.
        </p>
      </header>
      <Detector />
      <footer className="max-w-[70ch] border-t border-rule pt-6 text-sm leading-relaxed text-slate">
        Postings you check are stored to monitor the model. Leave out names, phone numbers and
        anything else personal. The verdict is a statistical estimate, not advice.
      </footer>
    </main>
  );
}
