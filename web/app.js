/* TactiVision dashboard — multi-run viewer synced to structured analytics. */
const state = {
  matchId: null,
  matches: [],
  matchInfo: null,
  bundle: null,
  possession: [],
  events: [],
  tactics: [],
  players: [],
  teamShape: [],
  ball: [],
  availableMedia: [],
  mediaKind: "tracking",
  loadToken: 0,
  activeTab: "overview",
  playerSort: { key: "distance_yards", dir: -1 },
};

const $ = (sel) => document.querySelector(sel);
const video = $("#video");
const clock = $("#clock");
const momentFacts = $("#momentFacts");
const runMeta = $("#runMeta");
const runTags = $("#runTags");
const statusLine = $("#statusLine");
const mediaEmpty = $("#mediaEmpty");
const toastStack = $("#toastStack");

/* ---------- small utilities ---------- */

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(v, digits = 2) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : String(v);
}

function asBool(v) {
  return v === true || v === "True" || v === "true" || v === 1 || v === "1";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let n = bytes;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function tag(label, kind = "muted") {
  return `<span class="tag tag-${kind}">${escapeHtml(label)}</span>`;
}

function icon(name, size = 16) {
  return `<svg class="icon" width="${size}" height="${size}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

function panelHead(title, iconName, tags = []) {
  return `<div class="panel-head"><h2>${icon(iconName)}${escapeHtml(title)}</h2>${tags.join("")}</div>`;
}

function insufficient(msg = "Insufficient data") {
  return `<div class="empty-box"><p class="empty">${escapeHtml(msg)}</p></div>`;
}

function skeletonBlock(rows = 3) {
  return `<div class="skeleton-block">${'<div class="skeleton-row"></div>'.repeat(rows)}</div>`;
}

/* ---------- toasts ---------- */

function showToast(message, { type = "info", icon: iconName, timeout = 4200 } = {}) {
  if (!toastStack) return;
  const iconMap = { ok: "check", error: "alert", info: "activity" };
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.innerHTML = `${icon(iconName || iconMap[type] || "activity", 16)}<span>${escapeHtml(message)}</span>`;
  toastStack.appendChild(el);
  const remove = () => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 220);
  };
  const timer = setTimeout(remove, timeout);
  el.addEventListener("click", () => {
    clearTimeout(timer);
    remove();
  });
}

/* ---------- status banner ---------- */

function setStatus(msg, { error = false } = {}) {
  if (!msg) {
    statusLine.hidden = true;
    statusLine.textContent = "";
    statusLine.classList.remove("error");
    return;
  }
  statusLine.hidden = false;
  statusLine.textContent = msg;
  statusLine.classList.toggle("error", error);
}

function setMediaEmpty(show) {
  if (!mediaEmpty) return;
  mediaEmpty.hidden = !show;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${path}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

function sourceBasename(source) {
  if (!source) return "—";
  const text = String(source).replace(/\\/g, "/");
  const parts = text.split("/");
  return parts[parts.length - 1] || text;
}

function matchLabel(m) {
  const bits = [m.match_id];
  const src = sourceBasename(m.source);
  if (src && src !== m.match_id) bits.push(src);
  const dur = m.clip_seconds ?? m.duration_seconds;
  if (dur != null && dur !== "") bits.push(`${fmt(dur, 1)}s`);
  if (m.width && m.height) bits.push(`${m.width}×${m.height}`);
  if (!m.has_bundle) bits.push("no bundle");
  return bits.join(" · ");
}

function nearestByTime(rows, t, key = "timestamp") {
  if (!rows.length) return null;
  let best = rows[0];
  let bestD = Math.abs(Number(best[key]) - t);
  for (const row of rows) {
    const d = Math.abs(Number(row[key]) - t);
    if (d < bestD) {
      best = row;
      bestD = d;
    }
  }
  return bestD < 0.6 ? best : null;
}

/* ---------- segmented control sliding thumb ---------- */

function positionThumb(container) {
  if (!container) return;
  const thumb = container.querySelector(".segmented-thumb");
  const active = container.querySelector(".active");
  if (!thumb || !active) return;
  const cRect = container.getBoundingClientRect();
  const aRect = active.getBoundingClientRect();
  thumb.style.width = `${aRect.width}px`;
  thumb.style.transform = `translateX(${aRect.left - cRect.left - 2}px)`;
}

function repositionThumbs() {
  positionThumb($("#mediaSegmented"));
  positionThumb($("#tabs"));
}

window.addEventListener("resize", () => requestAnimationFrame(repositionThumbs));

/* ---------- media source ---------- */

function updateMediaButtons() {
  document.querySelectorAll(".src-btn").forEach((btn) => {
    const kind = btn.dataset.src;
    const ok = state.availableMedia.includes(kind);
    btn.disabled = !ok;
    btn.title = ok ? `Show ${kind} preview` : "Unavailable for this run";
    btn.classList.toggle("active", ok && state.mediaKind === kind);
  });
  requestAnimationFrame(() => positionThumb($("#mediaSegmented")));
}

function setVideoSource(kind) {
  if (!state.matchId) return;
  if (!state.availableMedia.includes(kind)) {
    setMediaEmpty(true);
    setStatus(`Media unavailable for this run: ${kind}`, { error: true });
    return;
  }
  state.mediaKind = kind;
  updateMediaButtons();
  setMediaEmpty(false);
  const t = video.currentTime || 0;
  video.removeAttribute("src");
  video.src = `/api/matches/${state.matchId}/media/${kind}`;
  video.onerror = () => {
    setMediaEmpty(true);
    setStatus(`Could not load ${kind} media for ${state.matchId}.`, { error: true });
  };
  video.addEventListener(
    "loadedmetadata",
    () => {
      const maxT = Number.isFinite(video.duration) ? video.duration : t;
      video.currentTime = Math.min(t, maxT || t);
      setMediaEmpty(false);
      setStatus("");
    },
    { once: true }
  );
}

/* ---------- run meta / tags ---------- */

function renderRunMeta() {
  const info = state.matchInfo || {};
  const bundle = state.bundle || {};
  const videoInfo = bundle.video || {};
  const src = sourceBasename(info.source || videoInfo.source);
  const fps = info.fps ?? videoInfo.fps;
  const w = info.width ?? videoInfo.width;
  const h = info.height ?? videoInfo.height;
  const clip = bundle.clip_seconds ?? info.clip_seconds ?? info.duration_seconds;
  const frames = info.frame_count;
  const parts = [
    state.matchId ? `<strong>${escapeHtml(state.matchId)}</strong>` : "<strong>—</strong>",
    src !== "—" ? `source ${escapeHtml(src)}` : null,
    clip != null && clip !== "" ? `${fmt(clip, 1)}s clip` : null,
    w && h ? `${w}×${h}` : null,
    fps != null && fps !== "" ? `${fmt(fps, 2)} fps` : null,
    frames != null && frames !== "" ? `${frames} frames` : null,
  ].filter(Boolean);
  runMeta.innerHTML = parts.join(" · ");

  const tags = [];
  if (state.matchId) tags.push(tag("Processed run", "ok"));
  tags.push(tag("Local", "local"));
  tags.push(tag("StatsBomb 120×80", "muted"));
  if (state.availableMedia.length) {
    tags.push(tag(`${state.availableMedia.length} media`, "estimate"));
  } else if (state.matchId) {
    tags.push(tag("No preview media", "warn"));
  }
  if (bundle?.summaries?.possession) tags.push(tag("Possession heuristic", "heuristic"));
  if (Array.isArray(state.events) && state.events.length) {
    tags.push(tag(`${state.events.length} events`, "heuristic"));
  }
  const mid = String(state.matchId || "").toLowerCase();
  if (mid.includes("snmot")) tags.push(tag("SNMOT eval context", "warn"));
  if (mid.startsWith("upload_")) tags.push(tag("Browser upload", "local"));
  runTags.innerHTML = tags.join("");
}

/* ---------- moment card ---------- */

function renderMoment(t) {
  clock.textContent = `t = ${t.toFixed(2)} s`;
  clock.setAttribute("datetime", `PT${t.toFixed(2)}S`);
  if (!state.matchId) {
    momentFacts.innerHTML = "";
    return;
  }
  const poss = nearestByTime(state.possession, t);
  const tact = nearestByTime(state.tactics, t);
  const shape = nearestByTime(state.teamShape, t);
  const ball = nearestByTime(state.ball, t);
  const nearbyEvents = state.events.filter((e) => Math.abs(Number(e.timestamp) - t) <= 0.75);

  let ballLabel = "unavailable";
  if (ball) {
    if (asBool(ball.pitch_valid)) {
      ballLabel = `(${fmt(ball.pitch_x)}, ${fmt(ball.pitch_y)})`;
    } else if (asBool(ball.observed)) {
      ballLabel = "observed (not projected)";
    } else {
      ballLabel = "not observed";
    }
  } else if (!state.ball.length) {
    ballLabel = "insufficient data";
  }

  const rows = [
    [
      "Possession (heuristic)",
      poss
        ? `${poss.possession_status}${
            poss.possession_team !== "" && poss.possession_team != null
              ? ` · team ${poss.possession_team}`
              : ""
          }`
        : state.possession.length
          ? "—"
          : "insufficient data",
    ],
    [
      "Nearest player",
      poss && poss.nearest_player_track_id !== "" && poss.nearest_player_track_id != null
        ? `id ${poss.nearest_player_track_id} · ${fmt(poss.nearest_player_distance, 2)} yd`
        : "—",
    ],
    ["Ball (pitch)", ballLabel],
    ["Tactical state", tact ? tact.tactical_state : state.tactics.length ? "—" : "insufficient data"],
    ["Team0 width", shape && asBool(shape.team0_valid) ? `${fmt(shape.team0_width)} yd` : "—"],
    ["Team1 width", shape && asBool(shape.team1_valid) ? `${fmt(shape.team1_width)} yd` : "—"],
    ["Nearby events", nearbyEvents.length ? nearbyEvents.map((e) => e.event_type).join(", ") : "none"],
  ];
  momentFacts.className = "moment-list moment-flash";
  momentFacts.innerHTML = rows
    .map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd></div>`)
    .join("");
}

/* ---------- generic renderers ---------- */

function statsGrid(items) {
  return `<div class="grid-stats">${items
    .map(
      ([k, v]) =>
        `<div class="stat"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(String(v))}</div></div>`
    )
    .join("")}</div>`;
}

function possessionTimeline(rows) {
  if (!rows.length) return insufficient("Insufficient data for estimated possession timeline.");
  const total = Number(rows[rows.length - 1].timestamp) - Number(rows[0].timestamp) || 1;
  let html = '<div class="timeline" role="img" aria-label="Estimated possession over time">';
  for (let i = 0; i < rows.length; i++) {
    const a = Number(rows[i].timestamp);
    const b = i + 1 < rows.length ? Number(rows[i + 1].timestamp) : a + 0.04;
    const w = Math.max(0.05, ((b - a) / total) * 100);
    let cls = "tx";
    if (rows[i].possession_status === "assigned" && String(rows[i].possession_team) === "0") cls = "t0";
    else if (rows[i].possession_status === "assigned" && String(rows[i].possession_team) === "1") cls = "t1";
    else if (rows[i].possession_status === "unknown") cls = "tu";
    html += `<span class="${cls}" style="width:${w}%" title="${a.toFixed(2)}s"></span>`;
  }
  html += "</div>";
  return html;
}

function possessionDonut(team0Pct, team1Pct) {
  const t0 = Math.max(0, Number(team0Pct) || 0);
  const t1 = Math.max(0, Number(team1Pct) || 0);
  const rest = Math.max(0, 100 - t0 - t1);
  const r = 40;
  const c = 2 * Math.PI * r;
  const seg = (pct) => (pct / 100) * c;
  let offset = 0;
  const arcs = [
    { pct: t0, color: "var(--team0)" },
    { pct: t1, color: "var(--team1)" },
    { pct: rest, color: "#3a4640" },
  ]
    .filter((a) => a.pct > 0)
    .map((a) => {
      const len = seg(a.pct);
      const dash = `${len} ${c - len}`;
      const circle = `<circle cx="50" cy="50" r="${r}" fill="none" stroke="${a.color}" stroke-width="14" stroke-dasharray="${dash}" stroke-dashoffset="${-offset}" transform="rotate(-90 50 50)" stroke-linecap="butt"/>`;
      offset += len;
      return circle;
    })
    .join("");
  return `
    <div class="donut-row">
      <div class="donut-wrap">
        <svg width="120" height="120" viewBox="0 0 100 100" role="img" aria-label="Estimated possession split">
          <circle cx="50" cy="50" r="${r}" fill="none" stroke="#1b2620" stroke-width="14"/>
          ${arcs}
          <text x="50" y="47" text-anchor="middle" class="donut-center-value">${fmt(t0 + t1, 0)}%</text>
          <text x="50" y="60" text-anchor="middle" class="donut-center-label">assigned</text>
        </svg>
      </div>
      <div class="donut-legend">
        <div class="donut-legend-row"><span class="swatch t0"></span> Team 0 <span class="val">${fmt(t0, 1)}%</span></div>
        <div class="donut-legend-row"><span class="swatch t1"></span> Team 1 <span class="val">${fmt(t1, 1)}%</span></div>
        <div class="donut-legend-row"><span class="swatch tu"></span> Unknown / unavailable <span class="val">${fmt(rest, 1)}%</span></div>
      </div>
    </div>`;
}

function sparkCell(value, max) {
  const v = Number(value) || 0;
  const m = Number(max) || 1;
  const pct = Math.max(2, Math.min(100, (v / m) * 100));
  return `<div class="spark-cell"><span class="spark-track"><span class="spark-fill" style="width:${pct}%"></span></span><span>${fmt(v, 1)}</span></div>`;
}

function table(headers, rows, { sortable = [] } = {}) {
  if (!rows.length) return insufficient();
  const head = headers
    .map((h, i) => {
      const key = sortable[i];
      if (!key) return `<th scope="col">${escapeHtml(h)}</th>`;
      const active = state.playerSort.key === key;
      const arrow = active ? (state.playerSort.dir === 1 ? "▲" : "▼") : "";
      return `<th scope="col" class="sortable" data-sort-key="${key}">${escapeHtml(h)}${
        arrow ? `<span class="sort-arrow">${arrow}</span>` : ""
      }</th>`;
    })
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function bindSeek(root) {
  root.querySelectorAll("[data-seek]").forEach((btn) => {
    btn.addEventListener("click", () => {
      video.currentTime = Number(btn.dataset.seek);
      video.play().catch(() => {});
    });
  });
}

/* ---------- tab renderers ---------- */

function renderOverview() {
  const s = state.bundle?.summaries || {};
  const poss = s.possession || {};
  const ball = s.ball || {};
  const events = s.events || {};
  const form = s.formation || {};
  const hasPoss = Number(poss.total_frames || 0) > 0 || state.possession.length > 0;
  const hasBall = Number(ball.total_frames || 0) > 0 || state.ball.length > 0;
  const timeline =
    state.availableMedia.includes("possession_timeline") && state.matchId
      ? `<div class="viz-row"><img src="/api/matches/${state.matchId}/media/possession_timeline" alt="Estimated possession timeline" onerror="this.replaceWith(Object.assign(document.createElement('p'),{className:'empty',textContent:'Possession timeline unavailable.'}))" /></div>`
      : insufficient("Possession timeline image unavailable for this run.");

  $("#tab-overview").innerHTML = `
    ${panelHead("Overview", "chart", [tag("Model-derived", "estimate"), tag("Not official stats", "warn")])}
    <p class="panel-lead">Selected run <strong>${escapeHtml(state.matchId || "—")}</strong> · clip ${fmt(
      state.bundle?.clip_seconds,
      1
    )}s · pitch frame StatsBomb 120×80 yards.</p>
    ${
      hasPoss
        ? possessionDonut(poss.team0_possession_percent_of_all_frames, poss.team1_possession_percent_of_all_frames)
        : ""
    }
    ${
      hasPoss || hasBall
        ? statsGrid([
            ["Estimated possession coverage", hasPoss ? `${fmt(poss.possession_coverage_percent, 2)}%` : "insufficient data"],
            ["Ball observed-frame coverage", hasBall ? `${fmt(ball.pitch_coverage_percent, 2)}%` : "insufficient data"],
            ["Heuristic events", events.total_events ?? (state.events.length || "—")],
            ["Estimated line signature T0", form.modal_shape_by_team?.["0"]?.shape_signature ?? "—"],
            ["Estimated line signature T1", form.modal_shape_by_team?.["1"]?.shape_signature ?? "—"],
          ])
        : insufficient("Insufficient analytics for this run.")
    }
    <p class="disclaimer">Percentages are heuristic / model-derived labels on this processed clip, not broadcast or Opta possession. Line signatures are spatial band counts, not named formations.</p>
    ${timeline}
  `;
}

function renderPossession() {
  const s = state.bundle?.summaries?.possession || {};
  if (!state.possession.length && !s.total_frames) {
    $("#tab-possession").innerHTML =
      panelHead("Possession", "ball", [tag("Heuristic", "heuristic")]) +
      insufficient("Insufficient data for estimated proximity possession.");
    return;
  }
  $("#tab-possession").innerHTML = `
    ${panelHead("Estimated proximity possession", "ball", [tag("Heuristic", "heuristic"), tag("Not Opta", "warn")])}
    <p class="panel-lead">Radius ${s.possession_radius_yards ?? "—"} yd · debounce ${
      s.debounce_frames ?? "—"
    } frames · proximity labels, not ground truth.</p>
    <div class="legend" aria-hidden="true">
      <span><i class="swatch t0"></i> Team 0</span>
      <span><i class="swatch t1"></i> Team 1</span>
      <span><i class="swatch tu"></i> Unknown</span>
      <span><i class="swatch tx"></i> Unavailable</span>
    </div>
    ${possessionTimeline(state.possession)}
    ${statsGrid([
      ["Assigned frames", s.frames_assigned ?? "—"],
      ["Unknown", s.frames_unknown ?? "—"],
      ["Unavailable", s.frames_unavailable ?? "—"],
      ["Transitions", s.possession_transitions ?? "—"],
      ["Median nearest (assigned)", fmt(s.median_nearest_player_distance_when_assigned, 2) + " yd"],
    ])}
  `;
}

function renderEvents() {
  if (!state.events.length) {
    $("#tab-events").innerHTML =
      panelHead("Events", "activity", [tag("Heuristic", "heuristic")]) +
      insufficient("No heuristic event candidates for this run.");
    return;
  }
  const rows = state.events.map((e) => {
    const teamCls = e.team === 0 ? "team0" : e.team === 1 ? "team1" : "";
    return `<tr>
      <td><button type="button" data-seek="${e.timestamp}">${fmt(e.timestamp, 2)}s</button></td>
      <td>${escapeHtml(e.event_type)}</td>
      <td class="${teamCls}">${e.team ?? "—"}</td>
      <td>${e.player_track_id ?? "—"}</td>
      <td>${fmt(e.confidence, 2)}</td>
    </tr>`;
  });
  $("#tab-events").innerHTML = `
    ${panelHead("Heuristic event candidates", "activity", [tag("Heuristic", "heuristic"), tag("Seekable", "local")])}
    <p class="panel-lead">Click a time to seek the preview. Confidence is model-derived, not official eventing.</p>
    ${table(["Time", "Type", "Team", "Track", "Conf"], rows)}
  `;
  bindSeek($("#tab-events"));
}

function renderShape() {
  if (!state.teamShape.length) {
    $("#tab-shape").innerHTML =
      panelHead("Team shape", "shield", [tag("Model-derived", "estimate")]) +
      insufficient("Insufficient team-shape data for this run.");
    return;
  }
  const rows = state.teamShape.slice(0, 80).map(
    (r) => `<tr>
    <td><button type="button" data-seek="${r.timestamp}">${fmt(r.timestamp, 2)}s</button></td>
    <td>${r.team0_n_players || "—"}</td>
    <td>${fmt(r.team0_width, 1)}</td>
    <td>${fmt(r.team0_compactness, 1)}</td>
    <td>${r.team1_n_players || "—"}</td>
    <td>${fmt(r.team1_width, 1)}</td>
    <td>${fmt(r.team1_compactness, 1)}</td>
  </tr>`
  );
  $("#tab-shape").innerHTML = `
    ${panelHead("Team shape", "shield", [tag("Model-derived", "estimate"), tag("Not named formations", "warn")])}
    <p class="panel-lead">Centroid / width / length / compactness on visible filtered tracks.</p>
    ${table(["t", "T0 n", "T0 width", "T0 compact", "T1 n", "T1 width", "T1 compact"], rows)}
  `;
  bindSeek($("#tab-shape"));
}

function renderPlayers() {
  if (!state.players.length) {
    $("#tab-players").innerHTML =
      panelHead("Players", "users", [tag("Track IDs", "warn")]) +
      insufficient("Insufficient player metrics for this run. Track IDs are not player names.");
    return;
  }
  const { key, dir } = state.playerSort;
  const sorted = [...state.players].sort((a, b) => (Number(a[key]) - Number(b[key])) * dir);
  const maxDist = Math.max(...state.players.map((r) => Number(r.distance_yards) || 0), 1);
  const rows = sorted.slice(0, 40).map(
    (r) => `<tr>
    <td>${escapeHtml(String(r.track_id))}</td>
    <td>${sparkCell(r.distance_yards, maxDist)}</td>
    <td>${fmt(r.mean_speed_yards_per_second, 2)}</td>
    <td>${fmt(r.max_speed_yards_per_second, 2)}</td>
    <td>${fmt(r.duration_seconds, 2)}</td>
    <td>${r.valid_samples}</td>
  </tr>`
  );
  $("#tab-players").innerHTML = `
    ${panelHead("Player tracks", "users", [tag("Model-derived", "estimate"), tag("IDs ≠ names", "warn")])}
    <p class="panel-lead">Distance/speed on smoothed pitch tracks. IDs are fragmented track IDs, not named players. Click a column header to sort.</p>
    ${table(
      ["Track", "Distance yd", "Mean yd/s", "Max yd/s", "Duration s", "Samples"],
      rows,
      {
        sortable: [
          "track_id",
          "distance_yards",
          "mean_speed_yards_per_second",
          "max_speed_yards_per_second",
          "duration_seconds",
          "valid_samples",
        ],
      }
    )}
  `;
  $("#tab-players")
    .querySelectorAll("th.sortable")
    .forEach((th) => {
      th.addEventListener("click", () => {
        const k = th.dataset.sortKey;
        if (state.playerSort.key === k) {
          state.playerSort.dir *= -1;
        } else {
          state.playerSort = { key: k, dir: -1 };
        }
        renderPlayers();
      });
    });
}

function renderBall() {
  const s = state.bundle?.summaries?.ball || {};
  const hasData = Number(s.total_frames || 0) > 0 || state.ball.length > 0;
  if (!hasData) {
    $("#tab-ball").innerHTML =
      panelHead("Ball", "ball", [tag("Model-derived", "estimate")]) +
      insufficient("Insufficient ball data for this run (observed-frame coverage unavailable).");
    return;
  }
  $("#tab-ball").innerHTML = `
    ${panelHead("Ball", "ball", [tag("Model-derived", "estimate"), tag("Coverage ≠ accuracy", "warn")])}
    <p class="panel-lead">Observed-frame coverage and continuity — not ball tracking accuracy.</p>
    ${statsGrid([
      ["Observed-frame coverage", `${fmt(s.pitch_coverage_percent, 2)}%`],
      ["Segments", s.segments ?? "—"],
      ["Longest segment", s.longest_segment_seconds != null ? `${s.longest_segment_seconds}s` : "—"],
      ["Missing intervals", s.missing_intervals ?? "—"],
      ["Median speed (observed)", s.median_speed_yards_per_second != null ? `${s.median_speed_yards_per_second} yd/s` : "—"],
      ["Max speed (observed)", s.max_speed_yards_per_second != null ? `${s.max_speed_yards_per_second} yd/s` : "—"],
    ])}
    <p class="disclaimer">Speeds use consecutive pitch-valid frames; gaps are not filled.</p>
  `;
}

function renderTactics() {
  const s = state.bundle?.summaries?.tactics || {};
  const counts = s.state_counts || {};
  const rows = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${v}</td></tr>`);
  if (!rows.length && !state.tactics.length) {
    $("#tab-tactics").innerHTML =
      panelHead("Tactics", "compass", [tag("Heuristic", "heuristic")]) +
      insufficient("Insufficient tactical-state data for this run.");
    return;
  }
  $("#tab-tactics").innerHTML = `
    ${panelHead("Tactical states", "compass", [tag("Heuristic", "heuristic"), tag("Thresholded", "muted")])}
    <p class="panel-lead">States from possession, ball thirds, and compactness thresholds.</p>
    ${table(["State", "Frames"], rows)}
  `;
}

function clearAsk() {
  $("#askOut").innerHTML = `<p class="empty">Ask uses structured analytics for <strong>${escapeHtml(
    state.matchId || "the selected run"
  )}</strong> only (model-derived / heuristic).</p>`;
  $("#askInput").value = "";
}

function clearPanels() {
  ["tab-overview", "tab-possession", "tab-events", "tab-shape", "tab-players", "tab-ball", "tab-tactics"].forEach(
    (id) => {
      $(`#${id}`).innerHTML = skeletonBlock(4);
    }
  );
  momentFacts.innerHTML = "";
  clearAsk();
}

/* ---------- run loading ---------- */

async function loadMatch(matchId) {
  const token = ++state.loadToken;
  setStatus(`Loading ${matchId}…`);
  clearPanels();
  state.matchId = matchId;
  state.matchInfo = state.matches.find((m) => m.match_id === matchId) || null;
  state.bundle = null;
  state.possession = [];
  state.events = [];
  state.tactics = [];
  state.players = [];
  state.teamShape = [];
  state.ball = [];
  state.availableMedia = state.matchInfo?.available_media || [];
  state.mediaKind = "tracking";
  video.removeAttribute("src");
  video.load();
  setMediaEmpty(!state.availableMedia.length);
  updateMediaButtons();
  renderRunMeta();

  try {
    const bundle = await api(`/api/matches/${matchId}`);
    if (token !== state.loadToken) return;
    const [poss, events, tactics, players, shape, ball] = await Promise.all([
      api(`/api/matches/${matchId}/possession`).catch(() => ({ rows: [] })),
      api(`/api/matches/${matchId}/events`).catch(() => ({ rows: [] })),
      api(`/api/matches/${matchId}/tactics`).catch(() => ({ rows: [] })),
      api(`/api/matches/${matchId}/players`).catch(() => ({ rows: [] })),
      api(`/api/matches/${matchId}/team-shape`).catch(() => ({ rows: [] })),
      api(`/api/matches/${matchId}/ball?limit=800`).catch(() => ({ rows: [] })),
    ]);
    if (token !== state.loadToken) return;

    state.bundle = bundle;
    state.possession = poss.rows || [];
    state.events = events.rows || [];
    state.tactics = tactics.rows || [];
    state.players = players.rows || [];
    state.teamShape = shape.rows || [];
    state.ball = ball.rows || [];
    if (state.matchInfo?.available_media) {
      state.availableMedia = state.matchInfo.available_media;
    }

    renderRunMeta();
    renderOverview();
    renderPossession();
    renderEvents();
    renderShape();
    renderPlayers();
    renderBall();
    renderTactics();
    clearAsk();
    renderMoment(0);
    activateTab(state.activeTab);

    const preferred = ["tracking", "ball", "trajectories", "radar"].find((k) => state.availableMedia.includes(k));
    if (preferred) {
      setVideoSource(preferred);
      setStatus("");
    } else {
      updateMediaButtons();
      setMediaEmpty(true);
      setStatus("No preview media available for this run.", { error: true });
    }
  } catch (err) {
    if (token !== state.loadToken) return;
    setStatus(String(err), { error: true });
    setMediaEmpty(true);
    $("#tab-overview").innerHTML = insufficient(`Failed to load run: ${err}`);
  }
}

/* ---------- tabs ---------- */

function activateTab(name) {
  state.activeTab = name;
  document.querySelectorAll(".tabs button").forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
    b.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll(".panel").forEach((p) => {
    const on = p.id === `tab-${name}`;
    p.classList.toggle("active", on);
    p.hidden = !on;
  });
  requestAnimationFrame(() => positionThumb($("#tabs")));
}

function wireTabKeyboard() {
  const tablist = $("#tabs");
  if (!tablist) return;
  tablist.addEventListener("keydown", (ev) => {
    const tabs = [...tablist.querySelectorAll('[role="tab"]')];
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    let next = null;
    if (ev.key === "ArrowRight" || ev.key === "ArrowDown") next = tabs[(i + 1) % tabs.length];
    else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") next = tabs[(i - 1 + tabs.length) % tabs.length];
    else if (ev.key === "Home") next = tabs[0];
    else if (ev.key === "End") next = tabs[tabs.length - 1];
    if (!next) return;
    ev.preventDefault();
    next.focus();
    activateTab(next.dataset.tab);
  });
}

/* ---------- match select ---------- */

async function refreshMatchList({ selectId = null } = {}) {
  const { matches } = await api("/api/matches");
  state.matches = matches || [];
  const select = $("#matchSelect");
  const usable = state.matches.filter((m) => m.has_bundle);
  select.innerHTML = usable
    .map((m) => `<option value="${escapeHtml(m.match_id)}">${escapeHtml(matchLabel(m))}</option>`)
    .join("");
  if (selectId && usable.some((m) => m.match_id === selectId)) {
    select.value = selectId;
  }
  return usable;
}

/* ---------- upload: dropzone + progress ---------- */

function setUploadStatus(msg, { error = false, ok = false } = {}) {
  const el = $("#uploadStatus");
  if (!el) return;
  el.textContent = msg;
  el.classList.toggle("error", error);
  el.classList.toggle("ok", ok);
}

function showDropzoneFile(file) {
  const dropzone = $("#dropzone");
  const prompt = $("#dzPrompt");
  const fileRow = $("#dzFile");
  if (!file) {
    dropzone.classList.remove("has-file");
    prompt.hidden = false;
    fileRow.hidden = true;
    return;
  }
  dropzone.classList.add("has-file");
  prompt.hidden = true;
  fileRow.hidden = false;
  $("#dzFileName").textContent = file.name;
  $("#dzFileSize").textContent = formatBytes(file.size);
}

function wireDropzone() {
  const dropzone = $("#dropzone");
  const input = $("#uploadFile");
  if (!dropzone || !input) return;

  input.addEventListener("change", () => showDropzoneFile(input.files?.[0] || null));

  ["dragenter", "dragover"].forEach((evName) =>
    dropzone.addEventListener(evName, (ev) => {
      ev.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "dragend"].forEach((evName) =>
    dropzone.addEventListener(evName, () => dropzone.classList.remove("dragover"))
  );
  dropzone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    dropzone.classList.remove("dragover");
    const file = ev.dataTransfer?.files?.[0];
    if (!file) return;
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch (_) {
      /* older browsers: fall back to native picker */
    }
    showDropzoneFile(file);
  });

  $("#dzFileClear")?.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    input.value = "";
    showDropzoneFile(null);
  });
}

function setUploadProgress({ visible, pct = null, indeterminate = false } = {}) {
  const wrap = $("#uploadProgress");
  const bar = $("#uploadProgressBar");
  if (!wrap || !bar) return;
  wrap.hidden = !visible;
  wrap.classList.toggle("indeterminate", indeterminate);
  if (!indeterminate && pct != null) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  if (!visible) bar.style.width = "0%";
}

function wireUploadForm() {
  const form = $("#uploadForm");
  if (!form) return;
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const fileInput = $("#uploadFile");
    const file = fileInput?.files?.[0];
    const btn = $("#uploadBtn");
    const limit = $("#uploadLimit")?.value || "30";
    if (!file) {
      setUploadStatus("Choose a video file first.", { error: true });
      showToast("Choose a video file first.", { type: "error" });
      return;
    }
    btn.disabled = true;
    setUploadStatus("Uploading video…");
    setStatus("Uploading video…");
    setUploadProgress({ visible: true, pct: 0 });

    const body = new FormData();
    body.append("video", file, file.name);
    body.append("max_seconds", limit);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/runs");

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable) return;
      const pct = (e.loaded / e.total) * 100;
      setUploadProgress({ visible: true, pct });
      if (pct >= 100) {
        setUploadStatus(
          `Processing video locally (limit: ${
            limit === "full" ? "full, capped at 5 min" : `${limit}s`
          }). This request waits until the run finishes…`
        );
        setStatus("Processing video… building analytics…");
        setUploadProgress({ visible: true, indeterminate: true });
      }
    });

    xhr.addEventListener("load", async () => {
      let payload = null;
      try {
        payload = JSON.parse(xhr.responseText);
      } catch (_) {
        payload = null;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail = (payload && (payload.detail || payload.message)) || `${xhr.status} upload failed`;
        const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
        setUploadStatus(msg, { error: true });
        setStatus(msg, { error: true });
        showToast(msg, { type: "error" });
        setUploadProgress({ visible: false });
        btn.disabled = false;
        return;
      }
      const runId = payload.run_id;
      setUploadStatus(`Completed — run ready: ${runId}`, { ok: true });
      setStatus(`Run ready: ${runId}`);
      showToast(`Run ready: ${runId}`, { type: "ok" });
      setUploadProgress({ visible: false });
      try {
        await refreshMatchList({ selectId: runId });
        await loadMatch(runId);
      } finally {
        btn.disabled = false;
      }
    });

    xhr.addEventListener("error", () => {
      const msg = "Upload failed — network error.";
      setUploadStatus(msg, { error: true });
      setStatus(msg, { error: true });
      showToast(msg, { type: "error" });
      setUploadProgress({ visible: false });
      btn.disabled = false;
    });

    xhr.send(body);
  });
}

/* ---------- init ---------- */

async function init() {
  setStatus("Discovering processed runs…");
  wireUploadForm();
  wireDropzone();
  wireTabKeyboard();

  const select = $("#matchSelect");
  select.addEventListener("change", () => {
    if (select.value) loadMatch(select.value);
  });

  document.querySelectorAll(".src-btn").forEach((btn) =>
    btn.addEventListener("click", () => {
      if (!btn.disabled) setVideoSource(btn.dataset.src);
    })
  );
  document.querySelectorAll(".tabs button").forEach((btn) =>
    btn.addEventListener("click", () => activateTab(btn.dataset.tab))
  );
  video.addEventListener("timeupdate", () => renderMoment(video.currentTime));

  $("#askForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const question = $("#askInput").value.trim();
    if (!question || !state.matchId) return;
    const askedFor = state.matchId;
    $("#askOut").innerHTML = `<p class="empty">Querying structured analytics for ${escapeHtml(askedFor)}…</p>`;
    try {
      const ans = await api(`/api/matches/${askedFor}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (askedFor !== state.matchId) return;
      $("#askOut").innerHTML = `
        <p class="ask-scope">Answer grounded in <strong>${escapeHtml(askedFor)}</strong> (model-derived / heuristic analytics).</p>
        <p>${escapeHtml(ans.narrative || "")}</p>
        <h3>Facts</h3><ul>${(ans.facts || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
        <h3>Interpretation</h3><ul>${(ans.interpretation || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
        <h3>Uncertainty</h3><ul>${(ans.uncertainty || []).map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
      `;
    } catch (err) {
      if (askedFor !== state.matchId) return;
      $("#askOut").innerHTML = `<p class="empty">${escapeHtml(String(err))}</p>`;
    }
  });

  const usable = await refreshMatchList();
  if (!usable.length) {
    setStatus("No processed runs yet — upload a short clip to create one.", { error: false });
    setMediaEmpty(true);
    renderRunMeta();
    $("#tab-overview").innerHTML =
      panelHead("Overview", "chart", [tag("Waiting for a run", "muted")]) +
      insufficient("No processed matches found. Upload a short football clip above, or run scripts/run_pipeline.py.");
    return;
  }

  const preferred = usable[0];
  select.value = preferred.match_id;
  await loadMatch(preferred.match_id);
  repositionThumbs();
  if (document.fonts?.ready) document.fonts.ready.then(repositionThumbs);
}

init().catch((err) => {
  setStatus(String(err), { error: true });
  $("#tab-overview").textContent = `Failed to load: ${err}`;
});
