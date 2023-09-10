import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@mui/material/styles";
import { Toaster } from "react-hot-toast";
import { theme } from "./theme";
import "./App.css";

import { AuthSharedLayout, DashboardLayout, ChatLayout } from "./layouts";
import {
  LogIn,
  Register,
  Home,
  ProductPage,
  CheckoutPage,
  ShowItem,
  ChatAI,
  SuggPage,
} from "./pages";

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <Toaster />
        <Routes>
          <Route element={<AuthSharedLayout />}>
            <Route path="/login" element={<LogIn />} />
            <Route path="/register" element={<Register />} />
          </Route>
          <Route path="/home" element={<DashboardLayout />}>
            <Route index element={<Home />} />
            <Route path="category" element={<ProductPage />} />
            <Route path="item" element={<ShowItem />} />
            <Route path="checkout" element={<CheckoutPage />} />
          </Route>
          <Route path="/" element={<ChatLayout />}>
            <Route path="chat" element={<ChatAI />} />
            <Route path="recommended" element={<SuggPage />} />
          </Route>
        </Routes>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;
