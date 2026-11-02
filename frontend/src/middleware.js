import { redirect } from "react-router";
import { getToken } from "./auth";

export async function requireAuth(_args, next) {
  if (!getToken()) {
    throw redirect("/login");
  }
  return next();
}
