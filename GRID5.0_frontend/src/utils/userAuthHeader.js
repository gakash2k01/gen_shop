const authHeader = (token) => ({
  headers: {
    authorization: `Bearer ${token}`,
  },
});

export default authHeader;
