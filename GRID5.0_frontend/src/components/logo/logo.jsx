import LogoImg from "../../assets/logo/logo-transparent.svg";
import { Box } from "@mui/material";

const Logo = () => {
  return (
    <Box sx={{ p: 2}}>
      <img src={LogoImg} alt="GenShop" width="175px" />
    </Box>
  );
};

export default Logo;
