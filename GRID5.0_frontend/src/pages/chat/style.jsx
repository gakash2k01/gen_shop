import styled from "@emotion/styled";
import { Container } from "@mui/material";

export const ChatBox = styled(Container)({
  //   backgroundColor: "#F1F1F1",
  width: "100%",

  ".left-box": {
    // heigh: "80%",
    display: "flex",
    flexDirection: "column",
    width: "40%",
    // overflow: "hidden",
    // border: "1px solid red",
  },

  ".right-box": {
    width: "60%",
    padding: "1em",
    borderRadius: "10px",
    backgroundColor: "#f1f1f1af",
    display: "flex",
    justifyContent: "flex-end",
    alignItems: "center",
    flexDirection: "column",

    ".chat-box": {
      width: "100%",
      height: "100%",
      border: "px solid red",
      scrollbarWidth: "thin",
      overflowX: "hidden",
      overflowY: "scroll",
    },

    ".btn": {
      backgroundColor: "rgba(196, 224, 245, 0.514)",
    },

    ".message-input": {
      // border: "5px solid green",
      display: "flex",
      gap: "1em",
      width: "100%",
    },
  },
});
