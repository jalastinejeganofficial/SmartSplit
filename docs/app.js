/**
 * SmartSplit — Apple-Style Interactive Studio Engine & Site Script
 */

document.addEventListener("DOMContentLoaded", () => {
  initMobileMenu();
  initHeroCanvas();
  initMultiTrackStudio();
  initToolExplorer();
  initConfigSwitcher();
  initScrollReveal();
});

/* =========================================================================
   1. MOBILE MENU TOGGLE
   ========================================================================= */
function initMobileMenu() {
  const menuBtn = document.getElementById("mobileMenuBtn");
  const drawer = document.getElementById("mobileDrawer");
  const links = document.querySelectorAll(".mobile-nav-link");

  if (!menuBtn || !drawer) return;

  let isOpen = false;

  const toggle = () => {
    isOpen = !isOpen;
    if (isOpen) {
      drawer.classList.remove("opacity-0", "pointer-events-none");
      drawer.classList.add("opacity-100", "pointer-events-auto");
      document.body.style.overflow = "hidden";
    } else {
      drawer.classList.remove("opacity-100", "pointer-events-auto");
      drawer.classList.add("opacity-0", "pointer-events-none");
      document.body.style.overflow = "";
    }
  };

  menuBtn.addEventListener("click", toggle);
  links.forEach(l => l.addEventListener("click", () => {
    if (isOpen) toggle();
  }));
}

/* =========================================================================
   2. HERO MINI WAVEFORM CANVAS
   ========================================================================= */
function initHeroCanvas() {
  const canvas = document.getElementById("heroWaveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  let phase = 0;
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const midY = canvas.height / 2;
    const width = canvas.width;

    ctx.beginPath();
    ctx.strokeStyle = "#10B981";
    ctx.lineWidth = 2;

    for (let x = 0; x < width; x += 3) {
      const freq = 0.035;
      const amp = Math.sin(x * 0.01 + phase) * 8 + Math.cos(x * freq + phase * 2) * 6;
      const y = midY + amp;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    phase += 0.05;
    requestAnimationFrame(animate);
  }
  animate();
}

/* =========================================================================
   3. MULTI-TRACK STEM MIXER STUDIO (Canvas & Web Audio Simulation)
   ========================================================================= */
function initMultiTrackStudio() {
  const masterPlayBtn = document.getElementById("masterPlayBtn");
  const playIcon = document.getElementById("playIcon");
  const pauseIcon = document.getElementById("pauseIcon");
  const timer = document.getElementById("playbackTimer");
  const canvas = document.getElementById("multiTrackCanvas");
  const zeroBleedBtn = document.getElementById("zeroBleedToggle");

  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = 144;
  }
  resize();
  window.addEventListener("resize", resize);

  // Mixer Channel States
  const stems = {
    vocals: { gain: 0.85, muted: false, solo: false, color: "#C084FC" },
    instrumental: { gain: 1.0, muted: false, solo: false, color: "#22D3EE" },
    drums: { gain: 0.90, muted: false, solo: false, color: "#FBBF24" },
    bass: { gain: 0.95, muted: false, solo: false, color: "#34D399" }
  };

  let isPlaying = false;
  let zeroBleedActive = true;
  let playSeconds = 0;
  let animId = null;
  let step = 0;

  // Toggle Zero-Bleed Mode
  if (zeroBleedBtn) {
    zeroBleedBtn.addEventListener("click", () => {
      zeroBleedActive = !zeroBleedActive;
      if (zeroBleedActive) {
        zeroBleedBtn.textContent = "ON (0.00% Bleed)";
        zeroBleedBtn.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 transition-all hover:bg-emerald-500/30";
      } else {
        zeroBleedBtn.textContent = "LEGACY (10% Bleed)";
        zeroBleedBtn.className = "px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 transition-all hover:bg-rose-500/30";
      }
    });
  }

  // Play/Pause Master
  if (masterPlayBtn) {
    masterPlayBtn.addEventListener("click", () => {
      isPlaying = !isPlaying;
      if (isPlaying) {
        playIcon.classList.add("hidden");
        pauseIcon.classList.remove("hidden");
        masterPlayBtn.classList.add("ring-4", "ring-purple-500/30");
      } else {
        playIcon.classList.remove("hidden");
        pauseIcon.classList.add("hidden");
        masterPlayBtn.classList.remove("ring-4", "ring-purple-500/30");
      }
    });
  }

  // Bind Stem Sliders & Buttons
  document.querySelectorAll(".stem-channel").forEach(channel => {
    const stemKey = channel.getAttribute("data-stem");
    const slider = channel.querySelector(".stem-slider");
    const gainText = channel.querySelector(".stem-gain-val");
    const muteBtn = channel.querySelector(".mute-btn");
    const soloBtn = channel.querySelector(".solo-btn");

    if (slider) {
      slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value) / 100.0;
        stems[stemKey].gain = val;
        const dB = val === 0 ? "-∞ dB" : `${(20 * Math.log10(val)).toFixed(1)} dB`;
        if (gainText) gainText.textContent = dB;
      });
    }

    if (muteBtn) {
      muteBtn.addEventListener("click", () => {
        stems[stemKey].muted = !stems[stemKey].muted;
        channel.classList.toggle("muted", stems[stemKey].muted);
        muteBtn.classList.toggle("bg-rose-500", stems[stemKey].muted);
        muteBtn.classList.toggle("text-white", stems[stemKey].muted);
      });
    }

    if (soloBtn) {
      soloBtn.addEventListener("click", () => {
        const wasSolo = stems[stemKey].solo;
        // Turn off all solos first
        Object.keys(stems).forEach(k => {
          stems[k].solo = false;
          const ch = document.querySelector(`.stem-channel[data-stem="${k}"]`);
          if (ch) {
            ch.classList.remove("soloed");
            ch.querySelector(".solo-btn")?.classList.remove("bg-purple-500", "text-white");
          }
        });

        if (!wasSolo) {
          stems[stemKey].solo = true;
          channel.classList.add("soloed");
          soloBtn.classList.add("bg-purple-500", "text-white");
        }
      });
    }
  });

  // Render Multi-Track Waveform Canvas
  function drawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width;
    const h = canvas.height;
    const midY = h / 2;

    const anySolo = Object.values(stems).some(s => s.solo);

    // Draw grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    // Render each stem's live synthetic waveform
    Object.keys(stems).forEach((k, idx) => {
      const stem = stems[k];
      let effectiveGain = stem.gain;

      if (stem.muted || (anySolo && !stem.solo)) {
        effectiveGain = 0;
      }

      // If legacy bleed is simulated and vocals are muted, leak 10% ghost wave into instrumental
      if (!zeroBleedActive && k === "vocals" && effectiveGain === 0) {
        effectiveGain = 0.12; // 12% ghost leak
      }

      ctx.beginPath();
      ctx.strokeStyle = stem.color;
      ctx.lineWidth = 1.8;
      ctx.globalAlpha = Math.max(0.2, effectiveGain);

      const freqMult = (idx + 1) * 0.02;
      const speed = isPlaying ? step * (idx + 1) * 0.4 : 0;

      for (let x = 0; x < w; x += 4) {
        // Compose rich multi-harmonic shape
        const wave1 = Math.sin(x * freqMult + speed) * 20;
        const wave2 = Math.cos(x * freqMult * 1.6 + speed * 1.2) * 12;
        const wave3 = Math.sin(x * 0.05 + speed * 2) * 6;
        const envelope = (wave1 + wave2 + wave3) * effectiveGain * (isPlaying ? 1.0 : 0.25);

        const y = midY + envelope;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    ctx.globalAlpha = 1.0;

    // Timer update
    if (isPlaying) {
      step += 0.08;
      playSeconds += 1 / 60;
      if (timer) {
        const mins = Math.floor(playSeconds / 60);
        const secs = Math.floor(playSeconds % 60);
        timer.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")} / 03:09`;
      }
      if (playSeconds >= 189) playSeconds = 0;
    }

    requestAnimationFrame(drawCanvas);
  }

  drawCanvas();
}

/* =========================================================================
   4. MCP TOOLS INTERACTIVE EXPLORER
   ========================================================================= */
const toolsData = {
  remove_vocals: {
    name: "remove_vocals",
    badge: "Most Popular",
    desc: "Removes vocals completely with 0.00% residual bleed to generate a studio-clean instrumental backing track and isolated vocal stem.",
    params: [
      { name: "audio_path", type: "string", required: true, desc: "Absolute or relative path to the audio file (MP3, WAV, FLAC, M4A)." },
      { name: "quality", type: "string", required: false, desc: "'fast' (Zero-Bleed DSP <2s) or 'balanced' / 'high' (Demucs Neural Network)." },
      { name: "zero_bleed", type: "boolean", required: false, desc: "Enables non-linear hard gating and stereo side-channel de-reverberation (default: true)." },
      { name: "output_dir", type: "string", required: false, desc: "Custom output directory. Defaults to ~/.smartsplit/output." }
    ],
    sampleOutput: {
      "status": "success",
      "source_file": "Rathinamo.mp3",
      "algorithm_used": "Zero-Bleed Mathematical Center-Channel Panning Cancellation + Stereo De-Reverberation",
      "quality_profile": "fast",
      "zero_bleed_enabled": true,
      "processing_time_seconds": 1.84,
      "output_directory": "C:\\Users\\user\\.smartsplit\\output\\Rathinamo",
      "stems": {
        "instrumental": "C:\\Users\\user\\.smartsplit\\output\\Rathinamo\\Rathinamo_instrumental.wav",
        "vocals": "C:\\Users\\user\\.smartsplit\\output\\Rathinamo\\Rathinamo_vocals.wav"
      },
      "summary": "Zero-Bleed vocal separation completed in 1.84s. Vocal bleed reduced to 0.00%."
    }
  },
  analyze_music: {
    name: "analyze_music",
    badge: "Acoustic Report",
    desc: "Performs full acoustic reverse-engineering on the mix: tempo estimation, frequency distribution, dynamic range, and vocal removability score.",
    params: [
      { name: "audio_path", type: "string", required: true, desc: "Path to the target audio track." },
      { name: "sample_limit_seconds", type: "number", required: false, desc: "Duration in seconds to analyze for acoustic inspection (default: 60.0)." }
    ],
    sampleOutput: {
      "status": "success",
      "file_name": "Rathinamo.mp3",
      "audio_specs": {
        "duration_formatted": "03:09",
        "sample_rate": 48000,
        "dynamic_range_db": 18.3,
        "estimated_bpm": 115
      },
      "mix_staging": {
        "stereo_width": 0.084,
        "center_energy_pct": 91.6,
        "side_energy_pct": 8.4,
        "vocal_center_index": 0.58
      },
      "vocal_separability": {
        "rating": "Very Good",
        "removability_score_pct": 92.0,
        "recommended_method": "Fast Mathematical DSP"
      }
    }
  },
  detect_instruments: {
    name: "detect_instruments",
    badge: "5+ Instruments",
    desc: "Identifies active instruments in the audio mix with confidence percentages, frequency bands, and stereo placement (Center vs. Panned).",
    params: [
      { name: "audio_path", type: "string", required: true, desc: "Path to the audio file." }
    ],
    sampleOutput: {
      "status": "success",
      "file_name": "Rathinamo.mp3",
      "detected_instruments": [
        { "name": "Vocals", "confidence": 93.7, "prominence": "Dominant Lead", "placement": "Center Channel" },
        { "name": "Drums & Percussion", "confidence": 78.8, "prominence": "Dominant Rhythm", "placement": "Sub/Center + Sides" },
        { "name": "Bass", "confidence": 99.0, "prominence": "Heavy / Driving", "placement": "Direct Center" },
        { "name": "Guitars / Keyboards", "confidence": 58.3, "prominence": "Backing", "placement": "Stereo L/R Panned" }
      ]
    }
  },
  isolate_instrument: {
    name: "isolate_instrument",
    badge: "Solo Stem",
    desc: "Isolates a single requested instrument from the mix with 0.00% cross-stem bleed ('vocals', 'instrumental', 'bass', 'drums', 'other').",
    params: [
      { name: "audio_path", type: "string", required: true, desc: "Path to the audio file." },
      { name: "instrument", type: "string", required: false, desc: "Target instrument: 'vocals', 'instrumental', 'drums', 'bass', or 'other'." },
      { name: "zero_bleed", type: "boolean", required: false, desc: "Guarantees 0.00% bleed from unselected instruments (default: true)." }
    ],
    sampleOutput: {
      "status": "success",
      "isolated_instrument": "bass",
      "file_path": "C:\\Users\\user\\.smartsplit\\output\\Rathinamo\\Rathinamo_bass.wav",
      "algorithm_used": "Zero-Bleed 8th-Order Lowpass + FFT Brickwall Stopband (<200Hz)",
      "processing_time_seconds": 2.18,
      "summary": "Successfully extracted pure isolated bass track (0.00% vocal/drum bleed) in 2.18s."
    }
  },
  separate_stems: {
    name: "separate_stems",
    badge: "4-Way Split",
    desc: "Splits track into isolated stems: 2 stems (vocals/instrumental) or 4 stems (vocals/drums/bass/other) with zero cross-stem bleed.",
    params: [
      { name: "audio_path", type: "string", required: true, desc: "Source audio file path." },
      { name: "stems", type: "number", required: false, desc: "2 or 4 stems (default: 2)." },
      { name: "zero_bleed", type: "boolean", required: false, desc: "Ensures 0% cross-bleed across all 4 channels." }
    ],
    sampleOutput: {
      "status": "success",
      "stems_count": 4,
      "stems": {
        "vocals": "...\\Rathinamo_vocals.wav",
        "drums": "...\\Rathinamo_drums.wav",
        "bass": "...\\Rathinamo_bass.wav",
        "other": "...\\Rathinamo_other.wav"
      }
    }
  },
  get_audio_info: {
    name: "get_audio_info",
    badge: "Metadata",
    desc: "Instant technical inspection without performing stem separation (duration, sample rate, channels, peak dBFS, RMS level).",
    params: [
      { name: "audio_path", type: "string", required: true, desc: "Target audio file path." }
    ],
    sampleOutput: {
      "status": "success",
      "duration_seconds": 189.96,
      "duration_formatted": "03:09",
      "sample_rate": 48000,
      "channels": 2,
      "peak_dbfs": 0.4,
      "rms_dbfs": -18.3
    }
  }
};

function initToolExplorer() {
  const container = document.getElementById("toolDetailsContainer");
  const tabBtns = document.querySelectorAll(".tool-tab-btn");

  if (!container || tabBtns.length === 0) return;

  function renderTool(toolKey) {
    const data = toolsData[toolKey];
    if (!data) return;

    container.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        <div>
          <div class="flex items-center gap-3 mb-3">
            <h3 class="text-2xl font-bold font-mono text-purple-400">${data.name}</h3>
            <span class="badge-pill-xs text-purple-300 bg-purple-500/20 border-purple-500/30">${data.badge}</span>
          </div>
          <p class="text-slate-300 text-sm leading-relaxed mb-6">${data.desc}</p>

          <h4 class="text-xs uppercase font-mono tracking-wider text-slate-400 mb-3 font-semibold">Parameters Schema:</h4>
          <div class="space-y-2 mb-6">
            ${data.params.map(p => `
              <div class="p-3 rounded-xl bg-white/[0.02] border border-white/5 text-xs font-mono">
                <div class="flex items-center justify-between text-slate-200 font-bold mb-1">
                  <span>${p.name} <span class="text-purple-400 font-normal">: ${p.type}</span></span>
                  <span class="${p.required ? 'text-amber-400' : 'text-slate-500'}">${p.required ? 'required' : 'optional'}</span>
                </div>
                <div class="text-slate-400 font-sans text-xs">${p.desc}</div>
              </div>
            `).join("")}
          </div>
        </div>

        <div>
          <div class="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
            <span>JSON Result Payload</span>
            <span class="text-emerald-400">Response 200 OK</span>
          </div>
          <div class="p-4 rounded-2xl bg-black/80 border border-white/10 font-mono text-xs overflow-x-auto text-emerald-300 max-h-[420px] overflow-y-auto">
            <pre>${JSON.stringify(data.sampleOutput, null, 2)}</pre>
          </div>
        </div>
      </div>
    `;
  }

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderTool(btn.getAttribute("data-tool"));
    });
  });

  renderTool("remove_vocals");
}

/* =========================================================================
   5. QUICKSTART CLIENT CONFIG SWITCHER
   ========================================================================= */
const configs = {
  claude: `{
  "mcpServers": {
    "smartsplit": {
      "command": "python",
      "args": ["-m", "smartsplit.server"],
      "env": {
        "PYTHONPATH": "c:\\\\path\\\\to\\\\SmartSplit"
      }
    }
  }
}`,
  cursor: `{
  "mcpServers": {
    "smartsplit": {
      "command": "python",
      "args": ["-m", "smartsplit.server"]
    }
  }
}`,
  antigravity: `{
  "smartsplit": {
    "command": "python",
    "args": ["-m", "smartsplit.server"],
    "env": {
      "PYTHONPATH": "c:\\\\path\\\\to\\\\SmartSplit"
    }
  }
}`,
  pip: `# 1. Clone the repository
git clone https://github.com/jalastinejeganofficial/SmartSplit.git
cd SmartSplit

# 2. Install SmartSplit in editable mode
pip install -e .

# 3. Test stdio server
python -m smartsplit.server`
};

function initConfigSwitcher() {
  const codeBlock = document.getElementById("configCodeBlock");
  const tabBtns = document.querySelectorAll(".client-tab-btn");
  const copyBtn = document.getElementById("copyConfigBtn");
  const copyText = document.getElementById("copyText");

  let currentClient = "claude";

  function setClient(clientKey) {
    currentClient = clientKey;
    if (codeBlock) {
      codeBlock.textContent = configs[clientKey];
    }
  }

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => {
        b.classList.remove("active", "text-white", "border-purple-500");
        b.classList.add("text-slate-400", "border-transparent");
      });
      btn.classList.add("active", "text-white", "border-purple-500");
      btn.classList.remove("text-slate-400", "border-transparent");
      setClient(btn.getAttribute("data-client"));
    });
  });

  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const code = configs[currentClient];
      navigator.clipboard.writeText(code).then(() => {
        if (copyText) copyText.textContent = "Copied to Clipboard!";
        copyBtn.classList.add("text-emerald-400");
        setTimeout(() => {
          if (copyText) copyText.textContent = "Copy Config";
          copyBtn.classList.remove("text-emerald-400");
        }, 2200);
      });
    });
  }

  setClient("claude");
}

/* =========================================================================
   6. APPLE SCROLL-REVEAL ANIMATIONS
   ========================================================================= */
function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("opacity-100", "translate-y-0");
        entry.target.classList.remove("opacity-0", "translate-y-8");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll(".glass-card, .bento-box").forEach(el => {
    el.classList.add("transition-all", "duration-700", "opacity-0", "translate-y-8");
    observer.observe(el);
  });
}
