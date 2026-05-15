/* main.tsx - Punto de entrada de la aplicación React
 *
 * Monta el componente App en el elemento HTML con id "root".
 * StrictMode ayuda a detectar problemas en tiempo de desarrollo.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);