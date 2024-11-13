/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#007bff",
        secondary: "#6c757d",
        destructive: "#dc3545",
        background: "#ffffff",
        accent: "#e9ecef",
        input: "#ced4da",
        ring: "#007bff",
      },
    },
  },
  plugins: [],
}
