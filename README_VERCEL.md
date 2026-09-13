Deploying static frontend to Vercel

1. Edit `web/config.js` and set `window.API_BASE` to your backend address, e.g.
   window.API_BASE = "https://my-camera-host.example.com";
   If you set an API key on the server (`PUBLIC_API_KEY` env var), set `window.API_KEY` to the same value.

2. Commit & push this repository (already configured for Vercel).

3. On Vercel, import the GitHub repo and deploy — it's a static site, no build step needed.

4. Ensure your backend (the machine running `app.py`) is reachable from the internet (or via tunnel like ngrok) and set `PUBLIC_API_KEY` to a secret value if you want to allow the static site to access protected endpoints without login.

Server-side notes
- `app.py` supports an optional `PUBLIC_API_KEY` env var; requests that include this key via `X-API-KEY` header or `?api_key=` query param will bypass login checks.
- CORS is enabled on the Flask app to allow cross-origin requests from Vercel.

Security caveat: using `PUBLIC_API_KEY` makes your endpoints accessible to anyone who knows the key. Use HTTPS and a strong random key.
