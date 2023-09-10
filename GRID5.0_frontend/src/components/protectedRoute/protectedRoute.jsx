import { useSelector } from "react-redux";
import { Outlet, Navigate } from "react-router-dom";

const ProtectedRoute = () => {
  const isUser = useSelector((state) => state.user);
  return isUser ? <Outlet /> : <Navigate to="/home" />;
};

export default ProtectedRoute;
