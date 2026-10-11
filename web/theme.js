// Light/dark toggle. Loaded in <head> so the saved choice is applied before the
// page paints (no flash). With no saved choice, the device's setting decides.
(function () {
  var KEY = "trainò_theme";
  function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function save(v) { try { localStorage.setItem(KEY, v); } catch (e) {} }
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  var s = saved();
  if (s === "light" || s === "dark") document.documentElement.setAttribute("data-theme", s);

  var SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/></svg>';

  function makeToggle() {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "themebtn";
    function paint() {
      b.innerHTML = isDark() ? SUN : MOON;
      b.setAttribute("aria-label", isDark() ? "Switch to light mode" : "Switch to dark mode");
      b.title = b.getAttribute("aria-label");
    }
    b.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      save(next);
      paint();
    });
    paint();
    return b;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var nav = document.querySelector(".pagenav");
    if (nav) {
      var bar = document.createElement("div");
      bar.className = "masthead";
      bar.innerHTML = '<a class="wordmark" href="index.html">Trainò<span>.</span></a>';
      bar.appendChild(makeToggle());
      nav.parentNode.insertBefore(bar, nav);
    }
    var login = document.querySelector("body.login .card");
    if (login) {
      var w = document.createElement("a");
      w.className = "wordmark"; w.href = "#"; w.innerHTML = "Trainò<span>.</span>";
      var h = login.querySelector("h1");
      if (h) h.replaceWith(w); else login.insertBefore(w, login.firstChild);
      var tb = makeToggle();
      tb.style.cssText = "position:fixed;top:16px;right:16px";
      document.body.appendChild(tb);
    }
  });
})();
