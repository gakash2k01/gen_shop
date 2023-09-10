import { Avatar, Box, Typography } from "@mui/material";
import moment from "moment";

const AIMessageBox = ({ text, timestamp }) => {
  return (
    <Box sx={{ my: "1em", display: "flex", gap: "1em" }}>
      <Avatar sx={{ bgcolor: "#ff5722" }}>AI</Avatar>
      <Box>
        <Box
          sx={{
            backgroundColor: "white",
            borderRadius: "0% 10px 10px 10px",
            padding: "0.25em 1em",
          }}
        >
          {text}
        </Box>
        <Typography variant="caption">{moment(timestamp).format('Do MMMM, h:mm a')}</Typography>
      </Box>
    </Box>
  );
};

export default AIMessageBox;
