// The job board. Everything here is rendered with DOM APIs (textContent),
// never innerHTML: company names and roles come from people's posts.
"use strict";

const STATE = { q: "", remote: "", newgrad: false, visa: false, hassalary: false, minsal: "", seniority: "", stack: [], sort: "new" };
let JOBS = [];

const $ = (id) => document.getElementById(id);

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  STATE.q = p.get("q") || "";
  STATE.remote = p.get("remote") || "";
  STATE.newgrad = p.get("newgrad") === "1";
  STATE.visa = p.get("visa") === "1";
  STATE.hassalary = p.get("salary") === "1";
  STATE.minsal = p.get("min") || "";
  STATE.seniority = p.get("seniority") || "";
  STATE.stack = (p.get("stack") || "").split(",").filter(Boolean);
  STATE.sort = p.get("sort") === "salary" ? "salary" : "new";
}

function writeHash() {
  const p = new URLSearchParams();
  if (STATE.q) p.set("q", STATE.q);
  if (STATE.remote) p.set("remote", STATE.remote);
  if (STATE.newgrad) p.set("newgrad", "1");
  if (STATE.visa) p.set("visa", "1");
  if (STATE.hassalary) p.set("salary", "1");
  if (STATE.minsal) p.set("min", STATE.minsal);
  if (STATE.seniority) p.set("seniority", STATE.seniority);
  if (STATE.stack.length) p.set("stack", STATE.stack.join(","));
  if (STATE.sort !== "new") p.set("sort", STATE.sort);
  const h = p.toString();
  history.replaceState(null, "", h ? "#" + h : location.pathname);
}

function money(n, cur) {
  if (n == null) return "";
  const v = n >= 1000 ? Math.round(n / 1000) + "k" : String(Math.round(n));
  const sym = { USD: "$", EUR: "€", GBP: "£", INR: "₹" }[cur];
  return sym ? sym + v : v + (cur ? " " + cur : "");
}

function salaryText(j) {
  if (j.salary_min == null && j.salary_max == null) return "";
  if (j.salary_min != null && j.salary_max != null) {
    return j.salary_min === j.salary_max ? money(j.salary_min, j.currency) : money(j.salary_min, j.currency) + "–" + money(j.salary_max, j.currency).replace(/^[$€£₹]/, "");
  }
  return j.salary_min != null ? money(j.salary_min, j.currency) + "+" : "up to " + money(j.salary_max, j.currency);
}

function matches(j) {
  if (STATE.remote && j.remote !== STATE.remote) return false;
  if (STATE.newgrad && j.new_grad !== "yes") return false;
  if (STATE.visa && j.visa !== "yes") return false;
  if (STATE.hassalary && j.salary_min == null && j.salary_max == null) return false;
  if (STATE.seniority && j.seniority !== STATE.seniority) return false;
  if (STATE.minsal) {
    const floor = Number(STATE.minsal);
    const top = j.salary_max ?? j.salary_min;
    if (j.currency !== "USD" || top == null || top < floor) return false;
  }
  for (const s of STATE.stack) if (!j.stack.includes(s)) return false;
  if (STATE.q) {
    const hay = [j.company, j.role, j.location].filter(Boolean).join(" ").toLowerCase();
    for (const w of STATE.q.toLowerCase().split(/\s+/).filter(Boolean)) if (!hay.includes(w)) return false;
  }
  return true;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function card(j) {
  const li = el("li", "job");
  const head = el("header");
  const h = el("h2");
  h.append(el("span", "company", j.company || "Unnamed company"));
  if (j.role) h.append(document.createTextNode(" · "), el("span", "role", j.role));
  head.append(h);
  const sal = salaryText(j);
  if (sal) head.append(el("span", "salary", sal));
  li.append(head);

  const where = [j.location, { yes: "Remote", hybrid: "Hybrid", no: "Onsite" }[j.remote]].filter(Boolean).join(" · ");
  if (where) li.append(el("p", "where", where));

  const tags = el("ul", "tags");
  if (j.new_grad === "yes") tags.append(el("li", "good", "New grads welcome"));
  if (j.visa === "yes") tags.append(el("li", "good", "Visa sponsorship"));
  const sen = { intern: "Intern", new_grad: "New grad / junior", mid: "Mid-level", senior: "Senior+" }[j.seniority];
  if (sen) tags.append(el("li", null, sen));
  for (const s of j.stack.slice(0, 8)) tags.append(el("li", null, s));
  if (tags.childElementCount) li.append(tags);

  const foot = el("footer");
  foot.append(el("span", null, monthName(j.month)));
  const a = el("a", null, "Read the post on HN →");
  a.href = "https://news.ycombinator.com/item?id=" + Number(j.id);
  a.rel = "noopener";
  foot.append(a);
  li.append(foot);
  return li;
}

function monthName(m) {
  const [y, mo] = m.split("-").map(Number);
  return new Date(Date.UTC(y, mo - 1, 1)).toLocaleString(undefined, { month: "long", year: "numeric", timeZone: "UTC" }) + " thread";
}

function render() {
  const list = JOBS.filter(matches);
  if (STATE.sort === "salary") {
    const top = (j) => (j.currency === "USD" ? j.salary_max ?? j.salary_min ?? -1 : -1);
    list.sort((a, b) => top(b) - top(a));
  }
  const ol = $("jobs");
  ol.replaceChildren(...list.slice(0, 300).map(card));
  if (!list.length) ol.append(el("li", "empty", "No posts match these filters."));
  $("count").textContent = `${list.length.toLocaleString()} of ${JOBS.length.toLocaleString()} posts` + (list.length > 300 ? " (showing 300)" : "");
  writeHash();
}

function syncControls() {
  $("q").value = STATE.q;
  for (const b of $("remote").children) b.setAttribute("aria-pressed", String(b.dataset.v === STATE.remote));
  $("newgrad").checked = STATE.newgrad;
  $("visa").checked = STATE.visa;
  $("hassalary").checked = STATE.hassalary;
  $("minsal").value = STATE.minsal;
  $("seniority").value = STATE.seniority;
  $("sort").value = STATE.sort;
  for (const b of $("stack").children) b.setAttribute("aria-pressed", String(STATE.stack.includes(b.dataset.v)));
}

function buildStackChips() {
  const counts = new Map();
  for (const j of JOBS) for (const s of j.stack) counts.set(s, (counts.get(s) || 0) + 1);
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 24);
  $("stack").replaceChildren(...top.map(([s, n]) => {
    const b = el("button", null, s);
    b.type = "button";
    b.dataset.v = s;
    b.append(el("small", null, String(n)));
    b.addEventListener("click", () => {
      STATE.stack = STATE.stack.includes(s) ? STATE.stack.filter((x) => x !== s) : [...STATE.stack, s];
      syncControls();
      render();
    });
    return b;
  }));
}

function wire() {
  $("q").addEventListener("input", (e) => { STATE.q = e.target.value.trim(); render(); });
  for (const b of $("remote").children) b.addEventListener("click", () => { STATE.remote = b.dataset.v; syncControls(); render(); });
  for (const id of ["newgrad", "visa", "hassalary"]) $(id).addEventListener("change", (e) => { STATE[id] = e.target.checked; render(); });
  $("minsal").addEventListener("input", (e) => { STATE.minsal = e.target.value; render(); });
  $("seniority").addEventListener("change", (e) => { STATE.seniority = e.target.value; render(); });
  $("sort").addEventListener("change", (e) => { STATE.sort = e.target.value; render(); });
  $("reset").addEventListener("click", () => { Object.assign(STATE, { q: "", remote: "", newgrad: false, visa: false, hassalary: false, minsal: "", seniority: "", stack: [], sort: "new" }); syncControls(); render(); });
}

async function main() {
  readHash();
  wire();
  try {
    const res = await fetch("data/jobs.json", { cache: "no-cache" });
    const data = await res.json();
    JOBS = data.jobs;
    const when = new Date(data.generated_at).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    $("meta").textContent = `Updated ${when} from the ${data.months.map(monthName).join(", ")}. Fields extracted by ${data.extractor}.`;
  } catch (e) {
    $("count").textContent = "Could not load the posts.";
    return;
  }
  buildStackChips();
  syncControls();
  render();
}

main();
