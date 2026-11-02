import { clearToken, getToken, setToken } from "./auth";

const base = import.meta.env.VITE_API_URL ?? "";

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg).join(" ");
    }
  } catch {
    return "Não foi possível concluir a operação";
  }
  return "Não foi possível concluir a operação";
}

export async function registerUser(name, email, image) {
  const body = new FormData();
  body.append("name", name);
  body.append("email", email);
  body.append("image", image);
  const response = await fetch(`${base}/api/auth/register`, {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export async function loginUser(email, image) {
  const body = new FormData();
  body.append("email", email);
  body.append("image", image);
  const response = await fetch(`${base}/api/auth/login`, {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  const data = await response.json();
  setToken(data.access_token);
  return data;
}

export async function fetchMe() {
  const response = await fetch(`${base}/api/users/me`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (response.status === 401) {
    clearToken();
    const error = new Error("Não autenticado");
    error.status = 401;
    throw error;
  }
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}
