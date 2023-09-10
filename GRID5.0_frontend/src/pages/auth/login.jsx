import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import customFetch from "../../utils/axios";
import { saveUser } from "../../features/user/userSlice";
import toast from "react-hot-toast";

import {
  Button,
  CssBaseline,
  TextField,
  Grid,
  Container,
  Typography,
  Box,
  InputAdornment,
  Avatar,
} from "@mui/material";
import { Link } from "react-router-dom";
import { EmailIcon, LockOutlinedIcon, NoEncryptionIcon } from "../../icons";

const LogIn = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    try {
      e.preventDefault();
      const data = new FormData(e.currentTarget);
      const response = {
        email: data.get("email"),
        password: data.get("password"),
      };
      // console.log(response);
      const resp = await customFetch.post("/login", response);
      const {token, user} = resp.data.data;
      dispatch(saveUser({token, user: user.data}));
      toast.success(`Welcome back ${user?.data?.name}`);
      navigate("/home")
    } catch (e) {
      console.log(e);
      toast.error(e.response.data.msg);
    }
  };

  return (
    <>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 12,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <Avatar sx={{ m: 1, bgcolor: "secondary.main" }}>
            <LockOutlinedIcon />
          </Avatar>
          <Typography
            component="h1"
            variant="h5"
            sx={{ textTransform: "uppercase" }}
          >
            Sign in to GEN SHOP
          </Typography>
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              type="email"
              label="Email Address"
              name="email"
              color="secondary"
              autoFocus
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <EmailIcon />
                  </InputAdornment>
                ),
              }}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              variant="outlined"
              name="password"
              label="Password"
              type="password"
              color="secondary"
              id="password"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <NoEncryptionIcon />
                  </InputAdornment>
                ),
              }}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              color="secondary"
            >
              Sign In
            </Button>
            <Grid container>
              <Grid item xs>
                <Link href="#" variant="body2">
                  Forgot password?
                </Link>
              </Grid>
              <Grid item>
                <Link to="/register" variant="body2">
                  Register now
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>
      </Container>
    </>
  );
};

export default LogIn;
