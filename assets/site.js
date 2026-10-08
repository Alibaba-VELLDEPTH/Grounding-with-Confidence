"use strict";

const threshold = document.querySelector("#threshold");
const candidates = [...document.querySelectorAll(".candidate-row")];
const presets = [...document.querySelectorAll("[data-threshold]")];

function updateThreshold() {
  const cutoff = Number(threshold.value);
  const selected = [];
  for (const row of candidates) {
    const keep = Number(row.dataset.score) >= cutoff;
    row.classList.toggle("is-dropped", !keep);
    row.querySelector(".candidate-state").textContent = keep ? "Keep" : "Drop";
    if (keep) selected.push(row.dataset.interval);
  }
  document.querySelector("#threshold-value").textContent = cutoff.toFixed(2);
  document.querySelector("#selected-count").textContent = `${selected.length} / ${candidates.length} intervals`;
  document.querySelector("#selected-json").textContent = `[${selected.join(", ")}]`;
  document.querySelector("#selection-message").textContent = selected.length === 0
    ? "No interval passes. Return an empty set."
    : `${selected.length === 1 ? "One interval" : selected.length === 2 ? "Two intervals" : "All three intervals"} retained. No new decoding.`;
  threshold.setAttribute("aria-valuetext", `${cutoff.toFixed(2)}, retaining ${selected.length} intervals`);
  threshold.style.setProperty("--progress", `${cutoff * 100}%`);
  for (const button of presets) button.setAttribute("aria-pressed", String(Number(button.dataset.threshold) === cutoff));
}

threshold.addEventListener("input", updateThreshold);
for (const button of presets) button.addEventListener("click", () => {
  threshold.value = button.dataset.threshold;
  updateThreshold();
});
updateThreshold();

const budgetResults = {
  10: [9.95, 9.41, 11.00, 14.42],
  25: [26.48, 22.37, 26.31, 31.12],
  50: [51.34, 43.05, 47.72, 52.82],
  75: [64.76, 61.00, 62.45, 65.30]
};
const seriesNames = ["generation order", "token likelihood", "external verifier", "our confidence"];
const budgetButtons = [...document.querySelectorAll("[data-budget]")];
for (const button of budgetButtons) button.addEventListener("click", () => {
  const budget = Number(button.dataset.budget);
  const values = budgetResults[budget];
  for (const row of document.querySelectorAll("[data-series]")) {
    const value = values[Number(row.dataset.series)];
    row.querySelector(".bar-fill").style.setProperty("--bar", `${value / 75 * 100}%`);
    row.querySelector(".bar-value").textContent = value.toFixed(2);
  }
  for (const option of budgetButtons) option.setAttribute("aria-pressed", String(option === button));
  document.querySelector(".bar-chart").setAttribute("aria-label", `At ${budget}% budget: ${values.map((value, i) => `${seriesNames[i]} ${value.toFixed(2)}%`).join(", ")}.`);
  document.querySelector("#budget-insight").textContent = `+${(values[3] - values[0]).toFixed(2)} percentage points over generation order at a ${budget}% budget.`;
});

function activateTab(button, buttons) {
  for (const tab of buttons) {
    const selected = tab === button;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  }
}

for (const tablist of document.querySelectorAll('[role="tablist"]')) {
  const buttons = [...tablist.querySelectorAll('[role="tab"]')];
  for (const button of buttons) {
    button.addEventListener("click", () => activateTab(button, buttons));
    button.addEventListener("keydown", event => {
      const index = buttons.indexOf(button);
      const keys = { ArrowRight: (index + 1) % buttons.length, ArrowDown: (index + 1) % buttons.length,
        ArrowLeft: (index - 1 + buttons.length) % buttons.length, ArrowUp: (index - 1 + buttons.length) % buttons.length,
        Home: 0, End: buttons.length - 1 };
      if (!(event.key in keys)) return;
      event.preventDefault();
      const next = buttons[keys[event.key]];
      activateTab(next, buttons);
      next.focus();
    });
  }
}

const menu = document.querySelector(".menu-button");
const navigation = document.querySelector("#navigation");
function closeMenu() {
  menu.setAttribute("aria-expanded", "false");
  navigation.classList.remove("is-open");
  menu.querySelector("span").textContent = "+";
}
menu.addEventListener("click", () => {
  const open = menu.getAttribute("aria-expanded") !== "true";
  menu.setAttribute("aria-expanded", String(open));
  navigation.classList.toggle("is-open", open);
  menu.querySelector("span").textContent = open ? "−" : "+";
});
for (const link of navigation.querySelectorAll("a")) link.addEventListener("click", closeMenu);
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && menu.getAttribute("aria-expanded") === "true") {
    closeMenu();
    menu.focus();
  }
});
window.matchMedia("(min-width: 761px)").addEventListener("change", event => {
  if (event.matches) closeMenu();
});

const dialog = document.querySelector("#figure-dialog");
const dialogImage = document.querySelector("#dialog-image");
for (const button of document.querySelectorAll("[data-zoom]")) button.addEventListener("click", () => {
  dialogImage.src = button.dataset.zoom;
  dialogImage.alt = button.closest("figure").querySelector("img").alt;
  document.querySelector("#dialog-caption").textContent = button.dataset.caption;
  dialog.showModal();
  document.body.style.overflow = "hidden";
});
document.querySelector("#close-figure").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => { document.body.style.overflow = ""; });
dialog.addEventListener("click", event => {
  const rect = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
});

const copyButton = document.querySelector("#copy-code");
let copyTimer;
copyButton.addEventListener("click", async () => {
  const code = document.querySelector(".code-panel:not([hidden]) pre code");
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(code.textContent);
    copyButton.firstChild.textContent = "Copied ";
    status.textContent = "Command copied to clipboard.";
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { copyButton.firstChild.textContent = "Copy command "; }, 2200);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Clipboard unavailable. The command is selected; press Control+C or Command+C to copy.";
    copyButton.firstChild.textContent = "Text selected ";
  }
});

const sectionObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    for (const link of navigation.querySelectorAll("a")) {
      const active = link.getAttribute("href") === `#${entry.target.id}`;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  }
}, { rootMargin: "-15% 0px -65% 0px" });
for (const section of document.querySelectorAll("#idea, #results, #method, #examples, #code")) sectionObserver.observe(section);
