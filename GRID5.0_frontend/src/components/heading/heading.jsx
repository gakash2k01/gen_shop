import { Typography } from "@mui/material";

const Heading = ({ children, siz }) => {
  return <Typography variant={siz}>{children}</Typography>;
};

export default Heading;
