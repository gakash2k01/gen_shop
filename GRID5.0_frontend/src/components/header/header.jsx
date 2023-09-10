import { useEffect, useState } from "react";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import Menu from "@mui/material/Menu";
import Container from "@mui/material/Container";
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import InputBase from "@mui/material/InputBase";
import Tooltip from "@mui/material/Tooltip";
import MenuItem from "@mui/material/MenuItem";
import Badge from "@mui/material/Badge";
import {
  Collapse,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
} from "@mui/material";
import { logoutUser } from "../../features/user/userSlice";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate, useLocation, Link } from "react-router-dom";
import toast from "react-hot-toast";

import {
  SearchIcon,
  ShoppingCartCheckoutIcon,
  PersonIcon,
  ChatIcon,
} from "../../icons";
import { styled } from "@mui/material/styles";
import Logo from "../logo/logo";

const Search = styled(Box)({
  display: "flex",
  position: "relative",
  borderRadius: "10px",
  backgroundColor: "#f0f5ff",
  marginLeft: "1em",
  width: "100%",
  maxWidth: "400px",
  paddingTop: "0.2em",
  paddingBottom: "0.2em",
  height: "min-content",
  color: "grey",
});

const SearchIconWrapper = styled("div")(({ theme }) => ({
  padding: theme.spacing(0, 2),
  height: "100%",
  position: "absolute",
  pointerEvents: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
  color: "rgb(0, 0, 0)",
  width: "100%",
  "& .MuiInputBase-input": {
    padding: theme.spacing(1, 1, 1, 0),
    paddingLeft: `calc(1em + ${theme.spacing(4)})`,
    transition: theme.transitions.create("width"),
    width: "100%",
  },
}));

const Header = () => {
  const cartLen = useSelector((state) => state.cart.cart.length);
  const [cartLength, setCartLength] = useState(cartLen);
  const [openOption, setOpenOption] = useState(false);
  const location = useLocation();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [anchorElUser, setAnchorElUser] = useState(null);

  const handleOpenUserMenu = (event) => {
    setAnchorElUser(event.currentTarget);
  };

  const handleCloseUserMenu = () => {
    setAnchorElUser(null);
  };

  const handleLogout = () => {
    handleCloseUserMenu();
    dispatch(logoutUser());
    navigate("/");
    toast.success("User logged out successfully !");
  };

  useEffect(() => {
    setCartLength(cartLen);
  }, [cartLen]);

  function stringToColor(string) {
    let hash = 0;
    let i;

    /* eslint-disable no-bitwise */
    for (i = 0; i < string.length; i += 1) {
      hash = string.charCodeAt(i) + ((hash << 5) - hash);
    }

    let color = "#";

    for (i = 0; i < 3; i += 1) {
      const value = (hash >> (i * 8)) & 0xff;
      color += `00${value.toString(16)}`.slice(-2);
    }
    /* eslint-enable no-bitwise */

    return color;
  }

  function stringAvatar(name) {
    if (!name) name = "B";
    return {
      sx: {
        bgcolor: stringToColor(name),
      },
      children: `${name.split(" ")[0][0]}`,
    };
  }

  return (
    <AppBar position="sticky" sx={{ mb: 2, backgroundColor: "#fff" }}>
      <Container maxWidth="xl">
        <Toolbar
          disableGutters
          sx={{ flexDirection: { md: "row", xs: "column" } }}
        >
          <Box onClick={() => navigate("/home")} sx={{ cursor: "pointer" }}>
            <Logo />
          </Box>
          <Box
            className="align-vertical"
            sx={{ flexGrow: 1, fontSize: "large" }}
          >
            {/* ADD EXTRA LINKS */}
            <Search>
              <SearchIconWrapper>
                <SearchIcon />
              </SearchIconWrapper>
              <StyledInputBase
                placeholder="Search for Products, Brands and More"
                inputProps={{ "aria-label": "search" }}
              />
            </Search>
          </Box>

          <Box
            sx={{
              display: "flex",
              flexGrow: 0,
              fontSize: "large",
              mr: "2em",
            }}
          >
            <Button
              sx={{ color: "grey", mr: "1em" }}
              startIcon={<ChatIcon />}
              onClick={() => navigate("/chat")}
            >
              Start new chat
            </Button>

            <Button
              sx={{ color: "grey", mr: "1em" }}
              startIcon={<ShoppingCartCheckoutIcon />}
              onClick={() => navigate("/home/checkout")}
            >
              <Badge badgeContent={cartLength} color="secondary">
                Cart
              </Badge>
            </Button>
            <Box
              sx={{ position: "relative" }}
              onMouseEnter={() => setOpenOption(!openOption)}
              onMouseLeave={() => setOpenOption(!openOption)}
            >
              <Button
                sx={{ color: "grey", mr: "1em" }}
                startIcon={<PersonIcon />}
              >
                User
              </Button>
              <Collapse in={openOption} timeout="auto" unmountOnExit>
                <List
                  component="div"
                  disablePadding
                  sx={{
                    borderRadius: "10px",
                    position: "absolute",
                    top: "40px",
                    // border: "1px solid red",
                    right: "0",
                    color: "black",
                    backgroundColor: "#f9f9f9",
                  }}
                > 
                  <ListItemButton
                    sx={{ pl: 4 }}
                    onClick={handleLogout}
                  >
                    <ListItemText primary="Logout" />
                  </ListItemButton>
                </List>
              </Collapse>
            </Box>
            <Menu
              sx={{ mt: "45px" }}
              id="menu-appbar"
              anchorEl={anchorElUser}
              anchorOrigin={{
                vertical: "top",
                horizontal: "right",
              }}
              keepMounted
              transformOrigin={{
                vertical: "top",
                horizontal: "right",
              }}
              open={Boolean(anchorElUser)}
              onClose={handleCloseUserMenu}
            >
              <MenuItem onClick={handleLogout} sx={{ minWidth: "150px" }}>
                <Typography textAlign="left">Logout</Typography>
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;
