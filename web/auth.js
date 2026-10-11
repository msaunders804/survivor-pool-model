// Shared by every page. One job: make sure a valid token is in
// localStorage before the page's own fetches run, and wrap fetch so a
// 401 anywhere sends you back to login instead of showing a broken page.
// The API serves these pages, so calls go to the same address the page came from.
// Only a page opened straight from disk (file://) needs the local dev server's address.
const API = location.protocol === "file:" ? "http://localhost:8000" : "";

function getToken() {
  return localStorage.getItem("trainò_token");
}

function requireAuthOrRedirect() {
  if (!getToken()) {
    window.location.href = "login.html";
    throw new Error("not logged in"); // stop the calling page's script here
  }
}

async function authedFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    ...options,
    headers: { ...(options.headers || {}), Authorization: `Bearer ${getToken()}` },
  });
  if (res.status === 401) {
    localStorage.removeItem("trainò_token");
    window.location.href = "login.html";
    throw new Error("session expired");
  }
  return res;
}

// The league's current week, set on update.html. Pages use this instead of a hardcoded week.
async function getCurrentWeek(league) {
  const res = await authedFetch(`/leagues/${league}/config`);
  return (await res.json()).current_week;
}
