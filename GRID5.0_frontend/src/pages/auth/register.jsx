import { useState } from "react";
import customFetch from "../../utils/axios";
import toast from "react-hot-toast";
import { useDispatch } from "react-redux";
import { saveUser } from "../../features/user/userSlice";
import { useNavigate } from "react-router-dom";

import {
  Avatar,
  Button,
  CssBaseline,
  TextField,
  Grid,
  Box,
  Typography,
  Container,
  InputAdornment,
  InputLabel,
  MenuItem,
  FormControl,
  Select,
} from "@mui/material";
import { Link } from "react-router-dom";
import {
  PersonIcon,
  EmailIcon,
  LockOutlinedIcon,
  NoEncryptionIcon,
  PhoneIcon,
  LocationOnIcon,
} from "../../icons";

export default function Register() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [gender, setGender] = useState("male");

  const handleSubmit = async (event) => {
    try {
      event.preventDefault();
      const data = new FormData(event.currentTarget);
      const response = {
        name: data.get("fullname"),
        gender,
        age: data.get("age"),
        email: data.get("email"),
        password: data.get("password"),
        phone_number: data.get("phone_number"),
      };
      console.log(response);
      const resp = await customFetch.post("/register", response);
      console.log(resp);
      const {token, message, user} = resp.data.data;
      dispatch(saveUser({token, user: user.data}));
      toast.success(message);
      navigate("/home")
    } catch (e) {
      toast.error("Something went wrong while registering !");
      console(e.response.data.errors);
    }
  };

  const handleGenderChange = (event) => {
    setGender(event.target.value);
  };

  return (
    <>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 4,
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
            register to Bal Asha
          </Typography>
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 3 }}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  size="small"
                  name="fullname"
                  type="text"
                  required
                  fullWidth
                  id="fullName"
                  label="Full Name"
                  color="secondary"
                  autoFocus
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <PersonIcon fontSize="small" />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  size="small"
                  name="age"
                  type="number"
                  required
                  id="age"
                  label="Age"
                  color="secondary"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth size="small" required>
                  <InputLabel id="gender">Gender</InputLabel>
                  <Select
                    color="secondary"
                    labelId="gender"
                    // id="gender"
                    value={gender}
                    label="Role"
                    name="gender"
                    onChange={handleGenderChange}
                  >
                    <MenuItem value="male">Male</MenuItem>
                    <MenuItem value="femail">Female</MenuItem>
                    <MenuItem value="other">Other</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  size="small"
                  required
                  fullWidth
                  id="email"
                  type="email"
                  label="Email Address"
                  name="email"
                  color="secondary"
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <EmailIcon fontSize="small" />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  size="small"
                  required
                  fullWidth
                  type="password"
                  name="password"
                  label="Password"
                  id="password"
                  color="secondary"
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <NoEncryptionIcon fontSize="small" />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  size="small"
                  name="phone_number"
                  type="tel"
                  fullWidth
                  id="phone_number"
                  label="Phone Number"
                  color="secondary"
                  inputProps={{ inputMode: "numeric", pattern: "[0-9]*" }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <PhoneIcon fontSize="small" />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
            </Grid>
            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="secondary"
              sx={{ mt: 1, mb: 1 }}
            >
              Sign Up
            </Button>
            <Grid container justifyContent="flex-end">
              <Grid item>
                <Link to="/login" sx={{ color: "#cd366b" }}>
                  Already have an account? Log in
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>
      </Container>
    </>
  );
}
