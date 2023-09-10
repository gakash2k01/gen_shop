import { Box } from "@mui/material";
import { Header, ProtectedRoute } from "../components";

const ChatLayout = () => {
  return (
    <Box sx={{ height: "100vh" }}>
      <Header />
      <ProtectedRoute />
    </Box>
  );
};

export default ChatLayout;
