import { Box } from "@mui/material";
import styled from "@emotion/styled";

export const CardPriceBox = styled(Box)({
  display: "flex",
  alignItems: "baseline",
  gap: "0.5em",
  // margin: "0.5em 0",
  ".price": {
    color: "#212121",
    fontWeight: "bold",
    fontsize: "0.9em",
  },
  ".actual-price": {
    textDecoration: "line-through",
    fontSize: "0.9em",
    color: "#787777",
},
".discount": {
    color: "#388e3c",
    fontSize: "0.8em",
    letterSpacing: "-.2px",
    fontWeight: "500",
  },
});
