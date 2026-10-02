/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: "#0B0D10",
        panel: "#14171C",
        panel2: "#1A1E24",
        border: "#262B33",
        teal: "#3DDBD9",
        amber: "#E8A33D",
        ink: "#E6E9EC",
        muted: "#8891A0",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};
