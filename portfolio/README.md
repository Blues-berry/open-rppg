# Open-rPPG portfolio site

This folder is a standalone static portfolio and browser-side rPPG demo. It has no backend, database, or build step.

## Deploy to Vercel

1. Push this repository to GitHub.
2. Create a Vercel account and choose **Add New → Project → Import** the repository.
3. In **Root Directory**, select `portfolio`.
4. Leave the framework preset as **Other**, and leave build/output settings empty.
5. Deploy. Add a custom domain in the project **Settings → Domains** when ready.

The live demo uses `getUserMedia`, therefore it must be served through HTTPS (Vercel provides this automatically). Camera frames and the simplified signal calculation remain in the visitor's browser.

## Limits

The browser demo deliberately uses a simplified green-channel signal and frequency estimate. It is an interactive visualization, not the repository's Python/FacePhys implementation and not a medical device.
