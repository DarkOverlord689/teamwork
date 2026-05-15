/* App.tsx - Componente principal de la aplicación SMATC-UPAO
 *
 * Configura el enrutador, el tema visual y el store de Redux.
 * Renderiza las cuatro páginas principales: Dashboard, Subir Video,
 * Reportes y Detalle de Grupo.
 */

import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Provider } from "react-redux";
import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";

import { store } from "./store";
import Layout from "./components/layout/Layout";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Reports from "./pages/Reports";
import GroupDetail from "./pages/GroupDetail";

/* Tema personalizado de Material UI con colores primario (azul) y secundario (rojo) */
const theme = createTheme({
  palette: {
    primary: { main: "#1976d2" },
    secondary: { main: "#dc004e" },
  },
});

function App() {
  return (
    /* Provider expone el store de Redux a todos los componentes hijos */
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/groups/:groupId" element={<GroupDetail />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </ThemeProvider>
    </Provider>
  );
}

export default App;