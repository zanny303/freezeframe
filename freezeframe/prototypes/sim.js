// Shared replay engine for the prototypes. Plays the recorded demo run (data.js) in order:
// each "stats" event is one 500 ms tick, as the real server sends them; other events fire
// immediately. Prototypes only render — they never invent numbers.
(function () {
  const D = window.FF_DATA;
  const BANKS = ["A", "B", "C", "D", "E"];
  const ACTION = ["release", "hold", "freeze"];
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function fmtUsd(v) {
    if (v >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B";
    if (v >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
    return "$" + Math.round(v).toLocaleString("en-US");
  }
  const fmtMs = (v) => (v >= 100 ? Math.round(v) : v.toFixed(1)) + " ms";
  const fmtInt = (v) => Math.round(v).toLocaleString("en-US");
  const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";

  // Smoothly tween a number into an element (ease-out cubic). Cancels a running tween.
  function tween(el, to, fmt = fmtInt, dur = 450) {
    if (!el) return;
    const from = parseFloat(el.dataset.v ?? to);
    el.dataset.v = to;
    if (reduced || from === to) { el.textContent = fmt(to); return; }
    const t0 = performance.now();
    cancelAnimationFrame(el._raf);
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (to - from) * e);
      if (k < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }

  // Replay controller. handlers.onEvent(ev, state) is called for every event;
  // handlers.onReset(state) when restarted.
  function replay(handlers, opts = {}) {
    const tick = opts.tick ?? 500;
    let i = 0, timer = null, playing = false;
    const fresh = () => ({ phase: "idle", hops: {}, stats: null, skip: null, live: null,
                           heldCount: 0, tickIndex: 0 });
    let state = fresh();
    function step() {
      if (!playing) return;
      if (i >= D.events.length) { playing = false; handlers.onEnd && handlers.onEnd(state); return; }
      const ev = D.events[i++];
      if (ev.type === "stats") { state.stats = ev; state.tickIndex++; }
      else if (ev.type === "skip") { state.phase = "skip"; state.skip = ev; }
      else if (ev.type === "live") { state.phase = "live"; state.live = ev; }
      else if (ev.type === "hop") { state.hops[ev.hop] = ev; state.heldCount++; }
      else if (ev.type === "done") { state.phase = "done"; }
      handlers.onEvent && handlers.onEvent(ev, state);
      timer = setTimeout(step, ev.type === "stats" ? tick : (ev.type === "hop" ? 350 : 60));
    }
    return {
      play() { if (playing) return; if (i >= D.events.length) this.reset(); playing = true; step(); },
      pause() { playing = false; clearTimeout(timer); },
      reset() { this.pause(); i = 0; state = fresh(); handlers.onReset && handlers.onReset(state); },
      get playing() { return playing; },
      get progress() { return i / D.events.length; },
      get state() { return state; },
    };
  }

  // Convenience views of the data
  const hops = D.chain.hops.map((h) => ({
    ...h, fromBank: BANKS[h.from_member], toBank: BANKS[h.to_member],
    aloneAction: h.alone == null ? null : ACTION[h.alone],
    togetherAction: h.together == null ? null : ACTION[h.together],
    date: new Date(h.ts.replace(" ", "T")),
  }));
  const banksInChain = [...new Set(hops.flatMap((h) => [h.from_member, h.to_member]))].sort();
  const aloneSeeds = Object.values(D.results.alone_f1_by_seed);
  const summary = {
    togetherF1: D.results.together_f1,
    aloneF1Mean: aloneSeeds.reduce((a, b) => a + b, 0) / aloneSeeds.length,
    aloneF1Min: Math.min(...aloneSeeds), aloneF1Max: Math.max(...aloneSeeds),
    frozenTogether: D.decisions.together.frozen_laundering,
    frozenAloneMin: Math.min(D.decisions.alone_seed0.frozen_laundering, D.decisions.alone_seed1.frozen_laundering, D.decisions.alone_seed2.frozen_laundering),
    frozenAloneMax: Math.max(D.decisions.alone_seed0.frozen_laundering, D.decisions.alone_seed1.frozen_laundering, D.decisions.alone_seed2.frozen_laundering),
    laundering: D.decisions.together.laundering,
    legitFrozen: D.decisions.together.frozen_legit,
    aloneMissed: hops.filter((h) => h.aloneAction === "release").length,
  };

  window.FF = { D, BANKS, ACTION, hops, banksInChain, summary, replay, tween,
                fmtUsd, fmtMs, fmtInt, pct, reduced };
})();
