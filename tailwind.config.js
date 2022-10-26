const { inherit } = require("tailwindcss/colors");

// We are adding all color classes to safeList
const colors = [
  "slate",
  "gray",
  "zinc",
  "neutral",
  "stone",
  "red",
  "orange",
  "amber",
  "yellow",
  "lime",
  "green",
  "emerald",
  "teal",
  "cyan",
  "sky",
  "blue",
  "indigo",
  "violet",
  "purple",
  "fuchsia",
  "pink",
  "rose",
];
const scales = [
  "50",
  "100",
  "200",
  "300",
  "400",
  "500",
  "600",
  "700",
  "800",
  "900",
];
const types = ["bg", "border", "text"];

// States like hover and focus (see https://tailwindcss.com/docs/hover-focus-and-other-states)
// Add to this list as needed
const states = ["hover"];

const colorSafeList = [];
for (let i = 0; i < types.length; i++) {
  const t = types[i];

  for (let j = 0; j < colors.length; j++) {
    const c = colors[j];

    for (let k = 0; k < scales.length; k++) {
      const s = scales[k];

      colorSafeList.push(`${t}-${c}-${s}`);

      for (let l = 0; l < states.length; l++) {
        const st = states[l];
        colorSafeList.push(`${st}:${t}-${c}-${s}`);
      }
    }
  }
}

// console.log(colorSafeList);

module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./sections/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      typography: {
        starter: {
          /* prose in Hero */
          css: {
            h1: { color: "inherit", lineHeight: 1.2 },
            h2: { color: "inherit", marginBottom: "1.5rem" },
            h3: { color: "inherit" },
            h4: { color: "inherit" },
            h5: { color: "inherit" },
            li: { marginTop: "0.2em", marginBottom: "0.2em" },
            strong: { color: "inherit" },
            p: { lineHeight: "1.35", letterSpacing: "-0.02em" },
            ".color-flip": {
              a: {
                color: "#fff",
              },
              code: {
                color: "#fff",
              },
            },
            code: {
              display: "inline",
              fontSize: "0.85em",
              padding: "1px 3px",
              background: "rgba(0,0,0,.05)",
              fontWeight: 600,
              wordBreak: "break-word",
              "&:before": {
                content: "none",
              },
              "&:after": {
                content: "none",
              },
            },
            pre: {
              code: {
                display: "inline-block",
                maxWidth: "20vw",
                wordBreak: "inherit",
              },
            },
            blockquote: {
              fontStyle: "normal",
              opacity: "0.6",
            },
            ".comp_hero": {
              a: {
                color: inherit,
              },
              code: {
                color: inherit,
              },
            },
          },
        },
      },
    },
  },
  corePlugins: {
    aspectRatio: false,
  },
  plugins: [
    require("@tailwindcss/typography"),
    require("@tailwindcss/aspect-ratio"),
    require("@tailwindcss/aspect-ratio"),
  ],
  safelist: [].concat(colorSafeList),
};
