import { Avatar, Box, Typography } from "@mui/material";
import moment from "moment";

const ReceiverMessageBox = ({ text, timestamp }) => {
  return (
    <Box
      sx={{
        my: "1em",
        mr: "1em",
        display: "flex",
        justifyContent: "flex-end",
        gap: "1em",
      }}
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Box
          sx={{
            backgroundColor: "white",
            borderRadius: "10px 0 10px 10px",
            padding: "0.25em 1em",
          }}
        >
          {text}
        </Box>
        <Typography variant="caption" align="right">
          {moment(timestamp).format("Do MMMM, h:mm a")}
        </Typography>
      </Box>
      <Avatar sx={{ bgcolor: "#f80dec" }}>R</Avatar>
    </Box>
  );
};

export default ReceiverMessageBox;
