# NeuralScale -- React frontend

Replaces the plain HTML/JS demo with a proper Vite + React + Tailwind app:
a 3-way comparison viewer (bicubic / DIV2K model / GAN-finetuned model) with
shared zoom and pan, so you can zoom into a detail once and flip between all
three outputs without losing your position.

## Setup

```bash
cd frontend-react
npm install
npm run dev
```

Opens at http://localhost:5173. Requires the backend running separately:

```bash
# from the project root, in another terminal
uvicorn backend.app:app --reload --port 8000
```

The backend now loads BOTH checkpoints at startup (`models/best.pth` and
`models/gan_generator.pth`) and returns all three outputs plus per-model
inference timing in one response -- that's what powers the mode switch and
the readouts panel.

## Structure

```
frontend-react/
├── src/
│   ├── App.jsx                   # layout, upload handling, API call
│   ├── components/
│   │   ├── UploadZone.jsx         # drag-and-drop upload
│   │   ├── ComparisonViewer.jsx   # 3-way mode switch + zoom/pan viewport
│   │   └── MetricsPanel.jsx       # live resolution/timing readouts
│   └── index.css
├── tailwind.config.js             # design tokens (colors, fonts)
└── vite.config.js
```

## Build for deployment

```bash
npm run build
```

Outputs static files to `dist/` -- deployable to any static host (Vercel,
Netlify, GitHub Pages) if you want the frontend hosted separately from the
backend, same pattern as the CyberGuard live demo.
