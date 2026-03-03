import { createBrowserRouter } from "react-router";
import Root from "./components/Root";
import Landing from "./components/Landing";
import Login from "./components/Login";
import Register from "./components/Register";
import Dashboard from "./components/Dashboard";
import MemoriesView from "./components/MemoriesView";
import Contact from "./components/Contact";
import NotFound from "./components/NotFound";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: Landing },
      { path: "login", Component: Login },
      { path: "register", Component: Register },
      { path: "dashboard", Component: Dashboard },
      { path: "memories", Component: MemoriesView },
      { path: "contact", Component: Contact },
      { path: "*", Component: NotFound },
    ],
  },
]);
