# Frontend

Next.js (App Router) + TypeScript + Tailwind. One page: enter a posting section by section (or
paste it whole), get a verdict, the chance of fraud against the model's threshold, the signals
behind it, the posting with its suspicious words highlighted, and 👍/👎 feedback.

It calls the FastAPI backend straight from the browser (`src/lib/api.ts`), so the API's per-IP
rate limit applies to each visitor. On load it pings `/health` and shows a "waking up" state
while Render's free tier cold-starts.

```bash
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_URL, default http://localhost:8000
npm run dev                  # http://localhost:3000
npm run lint
npm run typecheck            # next typegen + tsc (LayoutProps etc. are generated types)
npm run build
```

Run the API alongside it with `uv run poe api` from the repo root. The live API at
https://fake-job-detector-api-dud2.onrender.com allows `http://localhost:3000` too, but every
check sent to it is stored in the production database.

`NEXT_PUBLIC_API_URL` is inlined at build time, so set it in the Vercel project settings, and
add the Vercel URL to the API's `ALLOWED_ORIGINS`.

Explanation labels (`src/lib/signals.ts`) and form choices (`src/lib/posting.ts`) mirror
`fjd.features.build` and EMSCAD's categories. Update them if the featurizer changes.
