import { createContext } from 'react';

export type AuthContextType = {
  user: string | null;
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setUser: React.Dispatch<React.SetStateAction<string | null>>;
};

const AuthContext = createContext<AuthContextType>({
  user: null,
  setUser: () => {},
});

export const AuthProvider = AuthContext.Provider;

export default AuthContext;
