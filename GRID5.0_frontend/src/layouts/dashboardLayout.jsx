import { Box } from "@mui/material";
import { Footer, Header, ProtectedRoute } from "../components";

const DashboardLayout = () => {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh " }}>
      <Header />
      <ProtectedRoute  />
      <Footer />
    </Box>
  );
};

export default DashboardLayout;
