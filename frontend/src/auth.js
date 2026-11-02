const COOKIE = "facecheck_token";

export function getToken() {
  const entry = document.cookie
    .split("; ")
    .find((row) => row.startsWith(`${COOKIE}=`));
  if (!entry) {
    return "";
  }
  return decodeURIComponent(entry.slice(COOKIE.length + 1));
}

export function setToken(token) {
  document.cookie = `${COOKIE}=${encodeURIComponent(token)}; Path=/; SameSite=Lax`;
}

export function clearToken() {
  document.cookie = `${COOKIE}=; Path=/; Max-Age=0; SameSite=Lax`;
}
