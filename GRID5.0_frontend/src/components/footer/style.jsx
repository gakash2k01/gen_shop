import styled from "@emotion/styled";
import Box from "@mui/material/Box";
import BGImg from "../../assets/images/bg-footer.png";

export const Wrapper = styled(Box)({
  // border: "1px solid red",
  minHeight: "250px",
  bottom: '0',
  width: "100%",
  backgroundImage: `url(${BGImg})`,
  backgroundRepeat: "no-repeat",
  backgroundPosition: "bottom",
  backgroundSize: "contain",
  backgroundColor: "#0d0e43",
  color: "white",
  padding: "0 2rem",
  paddingTop: "3rem",
  paddingBottom: "1rem",
  flexDirection: "column",

  ".top-box": {
    display: "flex",
    justifyContent: "center",

    ".row-title": {
      marginBottom: "1.5em",
    },

    li: {
      marginBottom: "1em",
      cursor: "pointer",

      "&:hover": {
        color: "#e87d01",
      },

      a: {
        all: "unset",
      },
    },

    "@media (max-width: 450px)": {
      flexWrap: "wrap",
    },
  },
  ".bottom-box": {
    minHeight: "100px",
    display: "flex",
    justifyContent: "center",
    alignItems: "flex-end",
    gap: "1em",
  },
});
