import Box from "@mui/material/Box";
import null_img from "../../assets/images/null_image1.png";

const NoImage = () => {
  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
      }}
    >
      <img src={null_img} />
    </Box>
  );
};

export default NoImage;
