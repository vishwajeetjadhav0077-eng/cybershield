// ================================================================
// CYBERSHIELD EXTENSION  — content.js
// Works on: YouTube · Instagram · Twitter/X · Reddit
// ================================================================
// IMPORTANT: Change API_URL to your deployed server URL before use.
// ================================================================

const API_URL = "http://127.0.0.1:5000";

// ── Severity → visual style map ──────────────────────────────
const SEVERITY_STYLE = {
  high:   { color: "#f87171", blur: "6px", label: "🚨 Cyberbullying" },
  medium: { color: "#fb923c", blur: "5px", label: "⚠️ Harmful Content" },
  low:    { color: "#fbbf24", blur: "3px", label: "⚡ Possibly Harmful" },
  none:   null,
};

// ── Cache: avoid re-checking the same text ───────────────────
const cache = new Map();

async function checkComment(text) {
  const key = text.trim().toLowerCase().slice(0, 200);
  if (cache.has(key)) return cache.get(key);

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Source": "extension" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    cache.set(key, data);
    return data;
  } catch (err) {
    console.warn("[CyberShield] API error:", err.message);
    return { result: 0, severity: "none" };
  }
}

// ── Apply blur + warning badge to an element ─────────────────
function applyWarning(el, data) {
  const style = SEVERITY_STYLE[data.severity];
  if (!style) return;

  // Blur
  el.style.filter     = `blur(${style.blur})`;
  el.style.transition = "filter 0.3s";
  el.style.outline    = `2px solid ${style.color}`;
  el.style.borderRadius = "6px";
  el.style.padding    = "4px";

  // Badge
  const parent = el.parentElement;
  if (parent && !parent.querySelector(".cs-badge")) {
    const badge = document.createElement("div");
    badge.className = "cs-badge";
    badge.textContent = `${style.label} · ${data.category?.replace(/_/g," ")} · ${data.confidence}%`;
    badge.style.cssText = `
      color:${style.color};font-size:11px;font-weight:700;
      margin-bottom:3px;cursor:pointer;user-select:none;
    `;
    badge.title = "Click to reveal comment";
    badge.addEventListener("click", () => {
      el.style.filter  = "none";
      el.style.outline = "none";
      badge.textContent = "👁 Revealed — " + badge.textContent;
      badge.style.color = "#94a3b8";
    });
    parent.insertBefore(badge, el);
  }

  // Hover to peek
  el.addEventListener("mouseenter", () => { el.style.filter = "none"; });
  el.addEventListener("mouseleave", () => {
    if (el.style.outline !== "none") el.style.filter = `blur(${style.blur})`;
  });
}

// ── Site-specific comment selectors ──────────────────────────
const SITE_SELECTORS = {
  "youtube.com": [
    "ytd-comment-thread-renderer #content-text",
    "ytd-comment-renderer #content-text",
  ],
  "instagram.com": [
    "._a9zs span",      // comment text spans
    "._aacl._aaco",     // caption
  ],
  "twitter.com": [
    "[data-testid='tweetText']",
  ],
  "x.com": [
    "[data-testid='tweetText']",
  ],
  "reddit.com": [
    ".md p",
    "[data-testid='comment'] p",
    "shreddit-comment .md p",
  ],
};

function getSelectorsForSite() {
  const host = window.location.hostname.replace("www.", "");
  for (const [domain, selectors] of Object.entries(SITE_SELECTORS)) {
    if (host.includes(domain)) return selectors;
  }
  return [];
}

// ── Main scan function ────────────────────────────────────────
async function scanPage() {
  const selectors = getSelectorsForSite();
  if (!selectors.length) return;

  const elements = document.querySelectorAll(selectors.join(", "));

  for (const el of elements) {
    if (el.dataset.csChecked) continue;
    el.dataset.csChecked = "true";

    const text = el.innerText?.trim();
    if (!text || text.length < 10) continue;

    const data = await checkComment(text);
    if (data.result === 1) applyWarning(el, data);
  }
}

// ── Run on page load + watch for new comments (infinite scroll) ──
scanPage();

const observer = new MutationObserver(() => scanPage());
observer.observe(document.body, { childList: true, subtree: true });

// Fallback poll for slow-loading pages (YouTube)
setInterval(scanPage, 5000);

console.log("[CyberShield] Extension active on", window.location.hostname);
