import { createBrowserRouter } from "react-router";
import { requireAuth } from "./middleware";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";

export const router = createBrowserRouter(
  [
    {
      path: "/",
      middleware: [requireAuth],
      Component: HomePage,
    },
    {
      path: "/login",
      Component: LoginPage,
    },
    {
      path: "/register",
      Component: RegisterPage,
    },
  ],
  { future: { v8_middleware: true } },
);
