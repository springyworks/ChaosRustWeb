# ChaosRustWeb - 4D Wave Ripple Tank Simulation

A real-time 4D wave equation simulation in your browser, powered by Bevy 0.15 and Burn 0.20. Interactive, GPU-accelerated, and runs on desktop or mobile!

[![Live Demo](supporting-data/kazam_screencast_9s-keep.gif)](https://your-github-username.github.io/ChaosRustWeb/)


---

## Run Online (GitHub Pages)

- **Just click the GIF above** or open:  
  https://your-github-username.github.io/ChaosRustWeb/
- Works in Chrome, Firefox, Edge, and most mobile browsers with WebGL2 support.

---

## Run Locally

### Prerequisites
- [Rust](https://rustup.rs/) (latest stable)
- [Trunk](https://trunkrs.dev/) (`cargo install trunk`)
- `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)

### Build and Run

```bash
# Development server with hot reload
trunk serve --open

# Production build
trunk build --release
```

Open http://127.0.0.1:8080 in your browser.

---

# (Original README continues below)

