import React, { useContext } from 'react';
import AuthContext from './AuthContext';
import StyledRawApp from './RawApp';
import Login from './Login';


const AuthWrapper: React.FC = () => {
  const { user, setUser } = useContext(AuthContext);

  if (!user) {
    return <Login />;
  }
  return <StyledRawApp />;
};

export default AuthWrapper;
